"""
Training script for the WebShop ScoreFunction model.

Loads WebShop_ScoreFunc_Train_Data.csv, precomputes BERT embeddings, builds
(state, action, label) pairs, and trains the score function.

Usage:
  python train_score_webshop.py
  python train_score_webshop.py --epochs 100 --batch_size 64 --lr 0.001
"""
import random
import ast
import csv
import math
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import matplotlib.pyplot as plt

from score_model_webshop import (
    ScoreFunctionWebShop, BertEmbeddingCache,
    BERT_DIM, SOFTMAX_BINS, _logprobs_to_softmax_bins,
)

# ── Defaults ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "WebShop_ScoreFunc_Train_Data.csv")
SAVE_DIR = "trained_models_webshop"
BATCH_SIZE = 64
EPOCHS = 150
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
D_PROJ = 64
HIDDEN = 32
EARLY_STOPPING_PATIENCE = 15


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD AND PARSE CSV DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_webshop_data(csv_path: str) -> List[Dict]:
    """
    Parse WebShop_ScoreFunc_Train_Data.csv into a list of clean dicts.

    Each returned dict:
      instruction:        task description string
      prev_actions:       list of previous actions (excluding 'reset')
      admissible_actions: list of K candidate action strings
      logits_from_llm:    list of K lists; each inner list is token log-probs
                          for the corresponding admissible action
      optimal_action:     ground-truth best action string

    Rows are skipped when:
      - admissible_actions is empty
      - optimal_action is not present in admissible_actions
      - CSV fields fail to parse
    """
    data = []
    skipped = 0

    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            # Skip placeholder / empty rows
            raw_adm = row['Admissiable_action'].strip()
            if not raw_adm or raw_adm == '[]':
                skipped += 1
                continue

            try:
                admissible = ast.literal_eval(raw_adm)
                logits = ast.literal_eval(row['Logits_from_LLM'])
                prev_raw = ast.literal_eval(row['Sequence_of_previous_action'])
            except (SyntaxError, ValueError):
                skipped += 1
                continue

            instruction = row['Original_instruction'].strip()
            optimal = row['optimal_action'].strip()

            if not instruction or not admissible or not optimal:
                skipped += 1
                continue

            # If the optimal action is missing from the candidates, inject it.
            # Assign an empty logit list (→ all-zero bins) since the LLM never
            # generated it — it had zero probability in the LLM's output.
            if optimal not in admissible:
                admissible.append(optimal)
                logits.append([])

            # Filter out the 'reset' pseudo-action from history
            prev_actions = [a for a in prev_raw if a != 'reset']

            data.append({
                'episode_id': row['id'].strip(),
                'instruction': instruction,
                'prev_actions': prev_actions,
                'admissible_actions': admissible,
                'logits_from_llm': logits,
                'optimal_action': optimal,
            })

    print(f"  Loaded {len(data)} valid rows ({skipped} skipped)")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 2. COLLECT ALL TEXTS FOR BERT PRECOMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def collect_all_texts(data: List[Dict]) -> List[str]:
    """Return a deduplicated list of every text string that will be BERT-embedded."""
    texts = set()
    for item in data:
        texts.add(item['instruction'])
        for a in item['prev_actions']:
            texts.add(a)
        for a in item['admissible_actions']:
            texts.add(a)
    return list(texts)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BUILD PRECOMPUTED RECORDS
# ═══════════════════════════════════════════════════════════════════════════════

def build_records(data: List[Dict], bert_cache: BertEmbeddingCache) -> List[Dict]:
    """
    Convert parsed data into flat records with precomputed tensors.

    Each record:
      state_raw:    [1536]   concat(instruction_emb, prev_pool)
      action_embs:  [K, 768]
      softmax_bins: [K, 10]  10-bin softmax histogram per action
      optimal_idx:  int
    """
    records = []

    for item in data:
        instruction = item['instruction']
        prev_actions = item['prev_actions']
        admissible = item['admissible_actions']
        logits_from_llm = item['logits_from_llm']
        optimal = item['optimal_action']

        # ── State embedding ──
        instr_emb = bert_cache.get(instruction)              # [768]
        if prev_actions:
            prev_pool = bert_cache.get_batch(prev_actions).mean(dim=0)  # [768]
        else:
            prev_pool = torch.zeros(BERT_DIM)
        state_raw = torch.cat([instr_emb, prev_pool])        # [1536]

        # ── Action embeddings ──
        action_embs = bert_cache.get_batch(admissible)       # [K, 768]

        # Convert token log-probs → 10-bin softmax histogram per action.
        # For injected optimal actions (empty logit list), use the mean of the
        # other actions' bins instead of all-zeros to avoid a spurious shortcut
        # where the model simply learns "zero bins → optimal".
        bins = [_logprobs_to_softmax_bins(lp) for lp in logits_from_llm]
        existing = [b for b, lp in zip(bins, logits_from_llm) if lp]
        mean_bin = [sum(b[i] for b in existing) / len(existing)
                    for i in range(SOFTMAX_BINS)] if existing else [0.0] * SOFTMAX_BINS
        bins = [mean_bin if not lp else b for b, lp in zip(bins, logits_from_llm)]
        softmax_bins = torch.tensor(bins, dtype=torch.float32)  # [K, 10]

        optimal_idx = admissible.index(optimal)

        records.append({
            'episode_id': item['episode_id'],
            'state_raw': state_raw,
            'action_embs': action_embs,
            'softmax_bins': softmax_bins,
            'optimal_idx': optimal_idx,
        })

    print(f"  Built {len(records)} records")
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class StateActionPairDataset(Dataset):
    """
    Expands records into flat (state_raw, action_emb, softmax_bin, label) pairs.
    For each state with K admissible actions: K pairs, one with label=1 (optimal).
    """

    def __init__(self, records: List[Dict]):
        self.pairs = []

        for rec in records:
            state_raw    = rec['state_raw']      # [1536]
            action_embs  = rec['action_embs']    # [K, 768]
            softmax_bins = rec['softmax_bins']   # [K, 10]
            optimal_idx  = rec['optimal_idx']

            K = action_embs.size(0)
            for j in range(K):
                self.pairs.append((
                    state_raw,
                    action_embs[j],    # [768]
                    softmax_bins[j],   # [10]
                    1.0 if j == optimal_idx else 0.0,
                ))

        print(f"  Dataset: {len(self.pairs)} (state, action) pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        state_raw, action_emb, sm_bin, label = self.pairs[idx]
        return state_raw, action_emb, sm_bin, torch.tensor(label, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING AND VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for state_raw, action_emb, sm_bin, labels in loader:
        state_raw  = state_raw.to(device)
        action_emb = action_emb.to(device)
        sm_bin     = sm_bin.to(device)
        labels     = labels.to(device)

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
        state_raw  = state_raw.to(device)
        action_emb = action_emb.to(device)
        sm_bin     = sm_bin.to(device)
        labels     = labels.to(device)

        logits = model(state_raw, action_emb, sm_bin).squeeze(-1)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


# ═══════════════════════════════════════════════════════════════════════════════
# 6. RECALL@K EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_recall_at_k(model, records: List[Dict], device, ks=(1, 2, 3, 5)):
    """
    Per-state Recall@K: rank admissible actions by nonconformity score (ascending
    = best first), check if optimal action is within the top K.

    Returns:
        dict {k: recall_percentage}
    """
    model.eval()
    hits = {k: 0 for k in ks}
    total = 0

    for rec in records:
        state_raw    = rec['state_raw'].to(device)
        action_embs  = rec['action_embs'].to(device)
        softmax_bins = rec['softmax_bins'].to(device)
        optimal_idx  = rec['optimal_idx']

        nc_scores = model.compute_nonconformity_scores(
            state_raw, action_embs, softmax_bins
        )  # [K]

        ranked = torch.argsort(nc_scores).cpu().tolist()  # ascending = best first

        for k in ks:
            if optimal_idx in ranked[:k]:
                hits[k] += 1
        total += 1

    return {k: 100.0 * hits[k] / total for k in ks}


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train WebShop ScoreFunction")
    parser.add_argument("--data",       type=str,   default=TRAINING_DATA_PATH)
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--d_proj",     type=int,   default=D_PROJ)
    parser.add_argument("--hidden",     type=int,   default=HIDDEN)
    parser.add_argument("--val_split",  type=float, default=VAL_SPLIT)
    parser.add_argument("--dropout",    type=float, default=0.4)
    parser.add_argument("--patience",   type=int,   default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Seed for the episode-level train/val split")
    args = parser.parse_args()

    print("=" * 70)
    print("  Training WebShop ScoreFunction")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Data:   {args.data}")

    # ── Load and parse CSV ──
    print("\nLoading training data ...")
    data = load_webshop_data(args.data)

    # ── Precompute BERT embeddings ──
    print("\nPrecomputing BERT embeddings ...")
    bert_cache = BertEmbeddingCache(device=DEVICE)
    all_texts = collect_all_texts(data)
    print(f"  {len(all_texts)} unique texts")
    bert_cache.precompute(all_texts)

    # ── Build records and split at EPISODE level ──
    # All states from the same WebShop goal (csv id) must land on the same
    # side: states within an episode share the instruction and trajectory
    # prefixes, so a state-level split still leaks across train/val.
    print("\nBuilding records ...")

    records = build_records(data, bert_cache)

    episode_ids = sorted({rec['episode_id'] for rec in records})
    rng = random.Random(args.seed)
    rng.shuffle(episode_ids)
    n_val_ep = max(1, int(len(episode_ids) * args.val_split))
    val_ep   = set(episode_ids[:n_val_ep])

    train_records = [r for r in records if r['episode_id'] not in val_ep]
    val_records   = [r for r in records if r['episode_id'] in val_ep]
    print(f"  Episodes: {len(episode_ids)}  "
          f"(train {len(episode_ids) - n_val_ep} / val {n_val_ep}, seed={args.seed})")
    print(f"  Train states: {len(train_records)}  |  Val states: {len(val_records)}")

    print("\nBuilding datasets ...")
    train_ds = StateActionPairDataset(train_records)
    val_ds   = StateActionPairDataset(val_records)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Model ──
    model = ScoreFunctionWebShop(d_proj=args.d_proj, hidden=args.hidden, dropout=args.dropout).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # ── Training loop ──
    print("\n" + "=" * 70)
    os.makedirs(args.data.replace(os.path.basename(args.data), SAVE_DIR), exist_ok=True)
    save_dir  = os.path.join(os.path.dirname(args.data), SAVE_DIR)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_score_model_webshop.pt")

    best_val_loss    = float('inf')
    patience_counter = 0
    train_losses     = []
    val_losses       = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, DEVICE)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        tag = ""
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc':  val_acc,
                'd_proj':   args.d_proj,
                'hidden':   args.hidden,
            }, save_path)
            tag = " *saved*"
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%{tag}"
        )

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")

    # ── Loss curve ──
    plot_path = os.path.join(save_dir, "loss_curve_webshop.png")
    plt.figure()
    epochs_run = len(train_losses)
    plt.plot(range(1, epochs_run + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs_run + 1), val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("WebShop ScoreFunction — Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss curve saved to: {plot_path}")

    # ── Recall@K on full dataset with best model ──
    print("\n" + "=" * 70)
    print("  Evaluating Recall@K (best saved model)")
    print("=" * 70)
    checkpoint = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    recall = evaluate_recall_at_k(model, val_records, DEVICE, ks=(1, 2, 3, 5))
    for k, v in sorted(recall.items()):
        print(f"  Recall@{k}: {v:.1f}%")

    # ── Save BERT cache ──
    cache_path = os.path.join(save_dir, "bert_cache_webshop.pkl")
    bert_cache.save(cache_path)
    print(f"\nBERT cache saved to: {cache_path}")


if __name__ == "__main__":
    main()
