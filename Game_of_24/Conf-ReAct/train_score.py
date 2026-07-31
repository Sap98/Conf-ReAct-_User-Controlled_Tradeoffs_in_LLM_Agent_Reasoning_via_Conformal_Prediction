"""
Training script for the Game-of-24 ScoreFunction model.

Loads training_data_2_game24_train.pkl, precomputes BERT embeddings, builds
(state, action, label) pairs, and trains the score function.

SET-COVERAGE (game24): positives are the WHOLE correct-action set, not a single
optimal action. build_records stores `correct_idxs`; the dataset labels every
correct action 1.0; Recall@K is set-aware (top-K contains ANY correct action).

Usage:
  python train_score.py
  python train_score.py --epochs 50 --batch_size 128 --lr 0.001
"""

import os
import pickle
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from score_model import (
    ScoreFunction, BertEmbeddingCache,
    BERT_DIM, SOFTMAX_BINS,
)

# ── Defaults ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_DATA_PATH = "training_data_2_game24_train.pkl"
SAVE_DIR = "trained_models"
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-5
VAL_SPLIT = 0.2
D_PROJ = 256
HIDDEN = 128


# ═══════════════════════════════════════════════════════════════════════════════
# 1. COLLECT ALL TEXTS FOR BERT PRECOMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def collect_all_texts(training_data: Dict) -> List[str]:
    texts = set()
    for game_file, states in training_data.items():
        for state in states:
            texts.add(state['task_name'])
            texts.add(state['location'])
            for a in state.get('prev_actions', []):
                texts.add(a)
            for a in state.get('admissible_actions', []):
                texts.add(a)
    return list(texts)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BUILD PRECOMPUTED RECORDS
# ═══════════════════════════════════════════════════════════════════════════════

def build_records(training_data: Dict, bert_cache: BertEmbeddingCache) -> List[Dict]:
    """
    Convert training_data_2_game24.pkl into flat records with precomputed tensors.

    Each record:
      state_raw:    [2304]  (task || loc || mean_prev)
      action_embs:  [K, 768]
      softmax_bins: [K, 10]
      optimal_idx:  int          (one correct action — back-compat)
      correct_idxs: list[int]    (ALL correct actions — set-coverage target)
    """
    records = []
    skipped = 0

    for game_file, states in training_data.items():
        for state in states:
            task = state['task_name']
            location = state['location']
            prev_actions = state.get('prev_actions', [])
            admissible = state.get('admissible_actions', [])
            softmax_dict = state.get('softmax_bin_distributions', {})
            optimal = state.get('optimal_action', '')
            correct = state.get('correct_actions', []) or ([optimal] if optimal else [])

            if not admissible or not optimal or optimal not in admissible:
                skipped += 1
                continue

            # State embedding
            task_emb = bert_cache.get(task)
            loc_emb = bert_cache.get(location)
            if prev_actions:
                prev_pool = bert_cache.get_batch(prev_actions).mean(dim=0)
            else:
                prev_pool = torch.zeros(BERT_DIM)
            state_raw = torch.cat([task_emb, loc_emb, prev_pool])  # [2304]

            # Action embeddings + softmax bins
            action_embs = bert_cache.get_batch(admissible)  # [K, 768]
            bins = [softmax_dict.get(a, [0] * SOFTMAX_BINS) for a in admissible]
            softmax_bins = torch.tensor(bins, dtype=torch.float32)  # [K, 10]

            optimal_idx = admissible.index(optimal)
            # SET-COVERAGE (patch #1): indices of EVERY correct action present
            correct_idxs = [admissible.index(a) for a in correct if a in admissible]
            if not correct_idxs:
                correct_idxs = [optimal_idx]

            records.append({
                'game_file': game_file,
                'state_raw': state_raw,
                'action_embs': action_embs,
                'softmax_bins': softmax_bins,
                'optimal_idx': optimal_idx,
                'correct_idxs': correct_idxs,
            })

    print(f"  Built {len(records)} records ({skipped} skipped)")
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class StateActionPairDataset(Dataset):
    """
    Expands records into flat (state_raw, action_emb, softmax_bin, label) pairs.
    SET-COVERAGE: label = 1.0 for EVERY correct action (not just one optimal).
    """

    def __init__(self, records: List[Dict]):
        self.pairs = []
        for rec in records:
            state_raw = rec['state_raw']
            action_embs = rec['action_embs']
            softmax_bins = rec['softmax_bins']
            correct_idxs = set(rec['correct_idxs'])

            K = action_embs.size(0)
            for j in range(K):
                self.pairs.append((
                    state_raw,
                    action_embs[j],
                    softmax_bins[j],
                    1.0 if j in correct_idxs else 0.0,
                ))
        print(f"  Dataset: {len(self.pairs)} (state, action) pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        state_raw, action_emb, sm_bin, label = self.pairs[idx]
        return state_raw, action_emb, sm_bin, torch.tensor(label, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING AND VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for state_raw, action_emb, sm_bin, labels in loader:
        state_raw = state_raw.to(device)
        action_emb = action_emb.to(device)
        sm_bin = sm_bin.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(state_raw, action_emb, sm_bin).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for state_raw, action_emb, sm_bin, labels in loader:
        state_raw = state_raw.to(device)
        action_emb = action_emb.to(device)
        sm_bin = sm_bin.to(device)
        labels = labels.to(device)

        logits = model(state_raw, action_emb, sm_bin).squeeze(-1)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RECALL@K EVALUATION (set-aware)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_recall_at_k(model, records: List[Dict], device, ks=(1, 2, 3, 5)):
    """
    SET-COVERAGE Recall@K: a state is a hit at K if the top-K lowest-nonconformity
    actions contain AT LEAST ONE correct action.
    """
    model.eval()
    hits = {k: 0 for k in ks}
    total = 0
    for rec in records:
        state_raw = rec['state_raw'].to(device)
        action_embs = rec['action_embs'].to(device)
        softmax_bins = rec['softmax_bins'].to(device)
        correct_idxs = set(rec.get('correct_idxs', [rec['optimal_idx']]))

        nc_scores = model.compute_nonconformity_scores(state_raw, action_embs, softmax_bins)
        ranked = torch.argsort(nc_scores).cpu().tolist()  # ascending = best first

        for k in ks:
            if correct_idxs & set(ranked[:k]):
                hits[k] += 1
        total += 1
    return {k: 100.0 * hits[k] / total for k in ks}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Game-of-24 ScoreFunction")
    parser.add_argument("--data", type=str, default=TRAINING_DATA_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--d_proj", type=int, default=D_PROJ)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--val_split", type=float, default=VAL_SPLIT)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the puzzle-level train/val split (within the train file)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Training Game-of-24 ScoreFunction")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Data:   {args.data}")

    print("\nLoading training data ...")
    with open(args.data, 'rb') as f:
        training_data = pickle.load(f)
    total_states = sum(len(s) for s in training_data.values())
    print(f"  {len(training_data)} puzzles, {total_states} states")

    print("\nPrecomputing BERT embeddings ...")
    bert_cache = BertEmbeddingCache(device=DEVICE)
    all_texts = collect_all_texts(training_data)
    print(f"  {len(set(all_texts))} unique texts")
    bert_cache.precompute(all_texts)

    print("\nBuilding dataset ...")
    records = build_records(training_data, bert_cache)

    # Train / val split at PUZZLE (game_file) level — states within a puzzle
    # share task + trajectory prefixes, so a pair-level split would leak.
    game_files = sorted({rec['game_file'] for rec in records})
    rng = random.Random(args.seed)
    rng.shuffle(game_files)
    n_val_games = max(1, int(len(game_files) * args.val_split))
    val_games = set(game_files[:n_val_games])

    train_records = [r for r in records if r['game_file'] not in val_games]
    val_records   = [r for r in records if r['game_file'] in val_games]
    print(f"  Puzzles: {len(game_files)}  "
          f"(train {len(game_files) - n_val_games} / val {n_val_games}, seed={args.seed})")

    train_ds = StateActionPairDataset(train_records)
    val_ds = StateActionPairDataset(val_records)
    print(f"  Train pairs: {len(train_ds)}  |  Val pairs: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = ScoreFunction(d_proj=args.d_proj, hidden=args.hidden).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print("\n" + "=" * 70)
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "best_score_model.pt")
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        tag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'd_proj': args.d_proj,
                'hidden': args.hidden,
            }, save_path)
            tag = " *saved*"

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%{tag}")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")

    # Loss curve
    plot_path = os.path.join(SAVE_DIR, "loss_curve.png")
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, args.epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(plot_path); plt.close()
    print(f"Loss curve saved to: {plot_path}")

    # Recall@K with best model
    print("\n" + "=" * 70)
    print("  Set-aware Recall@K (best saved model, val puzzles)")
    print("=" * 70)
    checkpoint = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    recall = evaluate_recall_at_k(model, val_records, DEVICE, ks=(1, 2, 3, 5))
    for k, v in sorted(recall.items()):
        print(f"  Recall@{k}: {v:.1f}%")

    cache_path = os.path.join(SAVE_DIR, "bert_cache.pkl")
    bert_cache.save(cache_path)
    print(f"\nBERT cache saved to: {cache_path}")


if __name__ == "__main__":
    main()
