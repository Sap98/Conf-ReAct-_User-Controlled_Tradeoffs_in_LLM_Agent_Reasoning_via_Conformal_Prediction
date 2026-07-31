"""
Training script for the ScoreFunction model.

Loads training_data_2.pkl, precomputes BERT embeddings, builds
(state, action, label) pairs, and trains the score function.

Usage:
  python train_score.py
  python train_score.py --epochs 50 --batch_size 128 --lr 0.001
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import matplotlib.pyplot as plt

from score_model import (
    ScoreFunction, BertEmbeddingCache,
    BERT_DIM, SOFTMAX_BINS,
)

# ── Defaults ─────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_DATA_PATH = os.path.join(_HERE, "training_data_2_sciworld.pkl")
SAVE_DIR = os.path.join(_HERE, "trained_models")
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

def collect_all_texts(training_data: Dict) -> List[str]:  # returns list of all unique texts
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
    Convert training_data_*.pkl into flat records with precomputed tensors.

    Multi-label aware: each record may have multiple optimal actions. Reads
      - state['optimal_actions']: list[str]   (new merged-pickle schema)
      - state['optimal_action']:  str         (legacy single-label schema)
    and stores them as a list of indices into `admissible_actions`.

    Each record:
      state_raw:    [2304]  (task || loc || mean_prev)
      action_embs:  [K, 768]
      softmax_bins: [K, 10]
      optimal_idxs: list[int]   (one or more positives among the K actions)
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

            # Pull optimals as a list, accepting either schema.
            if 'optimal_actions' in state:
                optimals = list(state['optimal_actions'])
            else:
                opt = state.get('optimal_action', '')
                optimals = [opt] if opt else []

            # Keep only optimals that actually appear in admissible.
            optimal_idxs = [admissible.index(o) for o in optimals if o in admissible]

            if not admissible or not optimal_idxs:   # need ≥1 positive label in admissible
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

            records.append({
                'state_raw': state_raw,
                'action_embs': action_embs,
                'softmax_bins': softmax_bins,
                'optimal_idxs': sorted(set(optimal_idxs)),
                # Human-readable metadata (ignored by training; used to populate
                # the conformal calibration pool's cal_metadata for diagnostics).
                'meta': {
                    'task_name':          task,
                    'location':           location,
                    'prev_actions':       list(prev_actions),
                    'admissible_actions': list(admissible),
                    'optimal_actions':    [admissible[i] for i in sorted(set(optimal_idxs))],
                },
            })

    n_multi = sum(1 for r in records if len(r['optimal_idxs']) > 1)
    print(f"  Built {len(records)} records ({skipped} skipped, {n_multi} multi-label)")
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class StateActionPairDataset(Dataset):
    """
    Expands records into flat (state_raw, action_emb, softmax_bin, label) pairs.
    For each state with K admissible actions: K pairs, with label=1 for every
    action in the optimal set and 0 otherwise. Multi-label states produce
    multiple positives per state.
    """

    def __init__(self, records: List[Dict]):
        self.pairs = []
        n_pos = 0

        for rec in records:
            state_raw = rec['state_raw']        # [2304]
            action_embs = rec['action_embs']    # [K, 768]
            softmax_bins = rec['softmax_bins']  # [K, 10]
            optimal_set = set(rec['optimal_idxs'])

            K = action_embs.size(0)
            for j in range(K):
                label = 1.0 if j in optimal_set else 0.0
                if label == 1.0:
                    n_pos += 1
                self.pairs.append((
                    state_raw,
                    action_embs[j],
                    softmax_bins[j],
                    label,
                ))

        print(f"  Dataset: {len(self.pairs)} (state, action) pairs  "
              f"(positives: {n_pos})")

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
        labels = labels.to(device)    #putting all embeddings on gpu/cpu

        optimizer.zero_grad()
        logits = model(state_raw, action_emb, sm_bin).squeeze(-1)  # forward() is called
        loss = criterion(logits, labels)  # loss is calculated.
        loss.backward()     # loss is back-propagated
        optimizer.step()    # Update all model parameters using their computed gradients.

        total_loss += loss.item() * labels.size(0)  # loss.item() is the avg loss per sample in a batch; labels.size = batch size, it stores the total batch loss
        preds = (torch.sigmoid(logits) > 0.5).float()  # converts logits to probability/softmax values
        correct += (preds == labels).sum().item() # stors how many predictions match with the labels.
        total += labels.size(0)  # Add the number of samples in this batch to the running total.

    return total_loss / total, 100.0 * correct / total   # returns avg loss, accuracy percentage


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

    return total_loss / total, 100.0 * correct / total # returns avg loss, accuracy percentage


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RECALL@K EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_recall_at_k(model, records: List[Dict], device, ks=(1, 2, 3, 5)):
    """
    Per-state Recall@K: for each state, rank admissible actions by
    nonconformity score (ascending = best first), check if ANY action in the
    optimal set is within the top K. With multi-label states this is a
    set-membership test (a state is a hit if at least one valid optimal
    appears in the top-K).
    """
    model.eval()
    hits = {k: 0 for k in ks}
    total = 0

    for rec in records:
        state_raw = rec['state_raw'].to(device)
        action_embs = rec['action_embs'].to(device)
        softmax_bins = rec['softmax_bins'].to(device)
        optimal_set = set(rec['optimal_idxs'])

        nc_scores = model.compute_nonconformity_scores(
            state_raw, action_embs, softmax_bins
        )  # [K]
        ranked_indices = torch.argsort(nc_scores).cpu().tolist()

        for k in ks:
            if any(idx in optimal_set for idx in ranked_indices[:k]):
                hits[k] += 1
        total += 1

    results = {k: 100.0 * hits[k] / total for k in ks}
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train ScoreFunction")
    parser.add_argument("--data", type=str, default=TRAINING_DATA_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--d_proj", type=int, default=D_PROJ)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--val_split", type=float, default=VAL_SPLIT)
    args = parser.parse_args()

    print("=" * 70)
    print("  Training ScoreFunction")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Data:   {args.data}")

    # ── Load training data ──
    print("\nLoading training data ...")
    with open(args.data, 'rb') as f:
        training_data = pickle.load(f)
    total_states = sum(len(s) for s in training_data.values())
    print(f"  {len(training_data)} games, {total_states} states")

    # ── Precompute BERT embeddings ──
    print("\nPrecomputing BERT embeddings ...")
    bert_cache = BertEmbeddingCache(device=DEVICE)
    all_texts = collect_all_texts(training_data)  # collects list of all unique texts.
    print(f"  {len(set(all_texts))} unique texts")
    bert_cache.precompute(all_texts)   # collects embeddings of all unique texts

    # ── Build dataset ──
    print("\nBuilding dataset ...")
    records = build_records(training_data, bert_cache)   # retuns all embeddings(state, actions, softmax, optimmal idx)
    dataset = StateActionPairDataset(records)  # dataset contains all states with each admissible_action, softmax bin and whether its optimal or not

    # ── Train / val split ──
    n_val = int(len(dataset) * args.val_split)  # val_split = 0.2
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])  # splits into training 80 % and validation 20 %
    print(f"  Train: {n_train}  |  Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # ── Model ──
    model = ScoreFunction(d_proj=args.d_proj, hidden=args.hidden).to(DEVICE)  #d_proj = 256
    n_params = sum(p.numel() for p in model.parameters())  # n_params stores the total number of scalar parameters (weights + biases) in your model.
    print(f"\nModel parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()  # Loss=BCE(σ(z),y); z is the logits from score.head; apply sigmoid over it and then calculate loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # ── Training loop ──
    print("\n" + "=" * 70)
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "best_score_model.pt")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss) # Here is the current validation loss. Decide whether to reduce the learning rate. Scheduler does that

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

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%{tag}"
        )

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")

    # ── Plot and save loss curves ──
    plot_path = os.path.join(SAVE_DIR, "loss_curve_100_epochs.png")
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, args.epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss curve saved to: {plot_path}")

    # ── Load best model and evaluate Recall@K ──
    print("\n" + "=" * 70)
    print("  Evaluating Recall@K (using best saved model)")
    print("=" * 70)

    checkpoint = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    recall = evaluate_recall_at_k(model, records, DEVICE, ks=(1, 2, 3, 5))
    for k, v in sorted(recall.items()):
        print(f"  Recall@{k}: {v:.1f}%")

    # ── Save BERT cache for later inference ──
    cache_path = os.path.join(SAVE_DIR, "bert_cache.pkl")
    bert_cache.save(cache_path)
    print(f"\nBERT cache saved to: {cache_path}")


if __name__ == "__main__":
    main()
