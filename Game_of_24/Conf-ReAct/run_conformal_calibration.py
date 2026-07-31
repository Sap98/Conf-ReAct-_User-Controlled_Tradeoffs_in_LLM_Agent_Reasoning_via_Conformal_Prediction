"""
One-time calibration for the Game-of-24 ConformalPredictor.

Loads training_data_2_game24_cal.pkl (the held-out CALIBRATION puzzle split,
disjoint from train), computes the set-coverage nonconformity score for every
calibration state (min over the correct-action set — patch #2 lives in
conformal_predictor.calibrate_from_records), and saves the pool to
calibrated_models/conformal_scores.pkl.

Because the cal file is ALREADY a disjoint puzzle split, we calibrate on ALL of
it (no internal val split needed).

Usage:
    python run_conformal_calibration.py --alpha 0.1
    python run_conformal_calibration.py --alpha 0.2
    python run_conformal_calibration.py --alpha 0.3
"""

import os
import argparse
import pickle
import torch

from score_model import ScoreFunction, BertEmbeddingCache
from train_score import collect_all_texts, build_records
from conformal_predictor import ConformalPredictor

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH       = "training_data_2_game24_cal.pkl"
MODEL_PATH      = "trained_models/best_score_model.pt"
BERT_CACHE_PATH = "trained_models/bert_cache.pkl"
SAVE_PATH       = "calibrated_models/conformal_scores.pkl"


def main():
    parser = argparse.ArgumentParser(description="Calibrate Game-of-24 ConformalPredictor")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--data",  type=str, default=DATA_PATH)
    parser.add_argument("--out",   type=str, default=SAVE_PATH)
    args = parser.parse_args()

    print(f"Device: {DEVICE}   alpha={args.alpha}")
    print(f"Calibration data: {args.data}")

    with open(args.data, 'rb') as f:
        training_data = pickle.load(f)
    total = sum(len(v) for v in training_data.values())
    print(f"  {len(training_data)} calibration puzzles, {total} states")

    # BERT cache (reuse the one saved during training so embeddings match)
    bert_cache = BertEmbeddingCache(device=DEVICE)
    if os.path.exists(BERT_CACHE_PATH):
        print(f"Loading BERT cache from {BERT_CACHE_PATH} ...")
        bert_cache.load(BERT_CACHE_PATH)
        # add any cal-only texts not seen during training
        bert_cache.precompute(collect_all_texts(training_data))
    else:
        print("Precomputing BERT embeddings ...")
        bert_cache.precompute(collect_all_texts(training_data))

    print("\nBuilding records ...")
    cal_records = build_records(training_data, bert_cache)   # carries correct_idxs
    print(f"  Calibration records: {len(cal_records)}")

    print(f"\nLoading model from {MODEL_PATH} ...")
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    model = ScoreFunction(d_proj=ckpt['d_proj'], hidden=ckpt['hidden']).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    predictor = ConformalPredictor(model, bert_cache, alpha=args.alpha)
    print("\nCalibrating (set-coverage: min nonconformity over correct set) ...")
    predictor.calibrate_from_records(cal_records, DEVICE)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    predictor.save(args.out)
    print(f"\nDone. Calibration saved to {args.out}")


if __name__ == "__main__":
    main()
