"""Dump PER-STATE (set_size, n_admissible, set_frac, covered) for the offline
conformal eval — ScienceWorld, so we can plot set-size distribution + coverage
(the game-24 'plot_setsize_abs_distribution_coverage' analog).

Same split / calibration / predictor as eval_coverage_offline_*.py (seed 42,
val 0.2, cal_frac 0.5, dual_lp, k=50). Run this FROM each model dir so the
score_model / train_score / conformal_predictor imports and the default data
resolve to that model:

  cd conformal_prediction            && python dump_setsize_coverage_sciworld.py \
       --data training_data_merged_sciworld.pkl --label Qwen3-8B   --out setsize_cov_qwen3.csv
  cd conformal_prediction_qwen25_3b  && python ../conformal_prediction/dump_setsize_coverage_sciworld.py \
       --data training_data_2_sciworld.pkl      --label Qwen2.5-3B --out setsize_cov_qwen25.csv
  cd conformal_prediction_gemma      && python ../conformal_prediction/dump_setsize_coverage_sciworld.py \
       --data training_data_2_sciworld.pkl      --label Gemma-4-12B --out setsize_cov_gemma.csv

CSV columns: model,alpha,set_size,n_admissible,set_frac,covered
"""
import os
import csv
import random
import argparse
import pickle
import torch

from score_model import ScoreFunction, BertEmbeddingCache
from train_score import collect_all_texts, build_records
from conformal_predictor import ConformalPredictor

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHAS    = (0.1, 0.2, 0.3)
SEED      = 42
VAL_SPLIT = 0.2
CAL_FRAC  = 0.5
METHOD    = "dual_lp"
K         = 50
N_COMP    = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training_data_merged_sciworld.pkl")
    ap.add_argument("--model", default=os.path.join("trained_models", "best_score_model.pt"))
    ap.add_argument("--label", default="Qwen3-8B", help="model name written into the CSV")
    ap.add_argument("--out", default="setsize_cov_qwen3.csv")
    ap.add_argument("--max_test", type=int, default=0, help="cap test states (0=all)")
    ap.add_argument("--method", default=METHOD, choices=["dual_lp", "quantile"])
    ap.add_argument("--k", type=int, default=K)
    args = ap.parse_args()

    print(f"[{args.label}] device={DEVICE}  data={args.data}  model={args.model}", flush=True)

    ckpt  = torch.load(args.model, map_location=DEVICE)
    model = ScoreFunction(d_proj=ckpt['d_proj'], hidden=ckpt['hidden']).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()

    with open(args.data, 'rb') as f:
        training_data = pickle.load(f)
    print(f"  states : {sum(len(v) for v in training_data.values())} "
          f"in {len(training_data)} episodes", flush=True)

    print("  recomputing BERT embeddings (RAM-light) ...", flush=True)
    bert_cache = BertEmbeddingCache(device=DEVICE)
    bert_cache.precompute(collect_all_texts(training_data))
    records = build_records(training_data, bert_cache)

    # seed-42 record-level val split, then val -> cal'/test' (matches eval_coverage_offline)
    random.seed(SEED)
    idx = list(range(len(records))); random.shuffle(idx)
    val_records = [records[i] for i in idx[:int(len(records) * VAL_SPLIT)]]
    random.seed(SEED + 1)
    vidx = list(range(len(val_records))); random.shuffle(vidx)
    n_cal = int(len(val_records) * CAL_FRAC)
    cal_records  = [val_records[i] for i in vidx[:n_cal]]
    test_records = [val_records[i] for i in vidx[n_cal:]]
    if args.max_test > 0:
        test_records = test_records[:args.max_test]
    print(f"  val={len(val_records)}  cal'={len(cal_records)}  test'={len(test_records)}", flush=True)

    rows = []
    for alpha in ALPHAS:
        predictor = ConformalPredictor(model, bert_cache, alpha=alpha)
        predictor.calibrate_from_records(cal_records, DEVICE)
        covered = 0
        for rec in test_records:
            meta = rec['meta']
            admissible = meta['admissible_actions']
            sm = {admissible[i]: rec['softmax_bins'][i].tolist() for i in range(len(admissible))}
            pred_set, *_ = predictor.get_prediction_set(
                task=meta['task_name'], previous_actions=meta['prev_actions'],
                location=meta['location'], admissible_actions=admissible,
                softmax_values=sm, method=args.method, select='topk',
                k=args.k, n_components=N_COMP)
            golds = set(meta['optimal_actions'])
            cov = 1 if (golds & set(pred_set)) else 0
            covered += cov
            n_adm = max(1, len(admissible))
            rows.append((args.label, alpha, len(pred_set), len(admissible),
                         len(pred_set) / n_adm, cov))
        n = len(test_records)
        print(f"  alpha={alpha}  coverage={covered}/{n}={100*covered/max(n,1):.1f}%", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "alpha", "set_size", "n_admissible", "set_frac", "covered"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.6f}", r[5]])
    print(f"Saved -> {args.out}  ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
