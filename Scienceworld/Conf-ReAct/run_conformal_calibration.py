"""
Stage 6 — One-time calibration of the ConformalPredictor on ScienceWorld.

Loads training_data_2_sciworld.pkl, reproduces the same val split used in
eval_recall.py (fixed seed), computes nonconformity scores of the optimal
action on each val state, and saves them to
calibrated_models/conformal_scores.pkl.

Reused unchanged (env-agnostic) from the ALFWorld pipeline; only the
DATA_PATH / MODEL_PATH defaults differ.

Usage:
    python run_conformal_calibration.py
    python run_conformal_calibration.py --alpha 0.05 --seed 42 --val_split 0.2
"""

import os
import random
import argparse
import pickle
import torch

from score_model import ScoreFunction, BertEmbeddingCache
from train_score import collect_all_texts, build_records
from conformal_predictor import ConformalPredictor

_HERE           = os.path.dirname(os.path.abspath(__file__))
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH       = os.path.join(_HERE, "training_data_2_sciworld.pkl")
MODEL_PATH      = os.path.join(_HERE, "trained_models", "best_score_model.pt")
BERT_CACHE_PATH = os.path.join(_HERE, "trained_models", "bert_cache.pkl")
SAVE_PATH       = os.path.join(_HERE, "calibrated_models", "conformal_scores.pkl")


def main():
    parser = argparse.ArgumentParser(description="Calibrate ConformalPredictor on ScienceWorld")
    parser.add_argument("--alpha",     type=float, default=0.1)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--data",      type=str,   default=DATA_PATH)
    parser.add_argument("--model",     type=str,   default=MODEL_PATH)
    parser.add_argument("--save",      type=str,   default=SAVE_PATH)
    args = parser.parse_args()

    print(f"Device: {DEVICE}   α={args.alpha}   seed={args.seed}")

    # ── Load training data ──────────────────────────────────────────────────
    print(f"\nLoading {args.data} ...")
    with open(args.data, 'rb') as f:
        training_data = pickle.load(f)
    total = sum(len(v) for v in training_data.values())
    print(f"  {len(training_data)} games, {total} states")

    # ── BERT cache ──────────────────────────────────────────────────────────
    bert_cache = BertEmbeddingCache(device=DEVICE)
    if os.path.exists(BERT_CACHE_PATH):
        print(f"Loading BERT cache from {BERT_CACHE_PATH} ...")
        bert_cache.load(BERT_CACHE_PATH)
    else:
        print("Precomputing BERT embeddings ...")
        bert_cache.precompute(collect_all_texts(training_data))

    # ── Build records (pre-computed tensors) ───────────────────────────────
    print("\nBuilding records ...")
    records = build_records(training_data, bert_cache)

    # ── Reproduce the same val split as eval_recall.py ─────────────────────
    random.seed(args.seed)
    indices = list(range(len(records)))
    random.shuffle(indices)
    n_val       = int(len(records) * args.val_split)
    val_indices = indices[:n_val]
    val_records = [records[i] for i in val_indices]
    print(f"  Total records: {len(records)}   Val records: {len(val_records)}")

    # ── Load trained ScoreFunction ──────────────────────────────────────────
    print(f"\nLoading model from {args.model} ...")
    ckpt  = torch.load(args.model, map_location=DEVICE)
    model = ScoreFunction(d_proj=ckpt['d_proj'], hidden=ckpt['hidden']).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # ── Calibrate ───────────────────────────────────────────────────────────
    predictor = ConformalPredictor(model, bert_cache, alpha=args.alpha)

    print("\nCalibrating ...")
    predictor.calibrate_from_records(val_records, DEVICE)

    # ── Save ────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    predictor.save(args.save)
    print(f"\nDone.  Calibration saved to {args.save}")


if __name__ == "__main__":
    main()
