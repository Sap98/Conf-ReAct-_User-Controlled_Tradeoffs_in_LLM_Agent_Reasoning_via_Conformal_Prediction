"""
Stage 3 — Convert raw candidate token-logprobs into 10-bin softmax histograms.

Analog of ALFWorld's `data_generation_for_scr_func_2.py`, but MUCH simpler:
ScienceWorld has no expert planner to query, so the optimal action was already
filled in during Stage 2 (gold-path teacher forcing: optimal = gold[t]). This
stage therefore only:

  1. converts `log_probs_of_admissible_actions` (per-candidate token-logprob
     lists) into `softmax_bin_distributions` (per-candidate 10-bin histograms),
     using the SAME binning as the play-time code
     (bfs_conformal_alfworld.py:logprobs_to_softmax_bins, num_bins=10), and
  2. drops the raw logprobs.

Input : training_data_sciworld.pkl     (Stage 2 output)
Output: training_data_2_sciworld.pkl   (schema consumed by train_score.py /
        build_records / run_conformal_calibration.py)

Per-state output schema:
    {task_name, prev_actions, location, admissible_actions,
     softmax_bin_distributions: {action: [10 ints]}, optimal_action}

Usage:
    python add_optimal_and_bins_sciworld.py
    python add_optimal_and_bins_sciworld.py --in training_data_sciworld.pkl \
                                            --out training_data_2_sciworld.pkl
"""

import os
import math
import pickle
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))

SOFTMAX_BINS = 10   # must match score_model.SOFTMAX_BINS


def logprobs_to_softmax_bins(token_logprobs, num_bins=SOFTMAX_BINS):
    """Histogram exp(logprob) of each action token into num_bins buckets over
    [0, 1]. Identical to the play-time binning so train/play features match."""
    bins = [0] * num_bins
    for lp in token_logprobs:
        idx = min(int(math.exp(lp) * num_bins), num_bins - 1)
        bins[idx] += 1
    return bins


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3 — softmax-bin conversion for ScienceWorld conformal data")
    p.add_argument("--in", dest="in_path",
                   default=os.path.join(_HERE, "training_data_sciworld.pkl"))
    p.add_argument("--out", dest="out_path",
                   default=os.path.join(_HERE, "training_data_2_sciworld.pkl"))
    p.add_argument("--drop-unlabeled", action="store_true",
                   help="Drop states whose optimal_action is empty (off-gold-path).")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 72)
    print("  Stage 3 — softmax-bin conversion")
    print("=" * 72)
    print(f"  in  : {args.in_path}")
    print(f"  out : {args.out_path}")

    with open(args.in_path, 'rb') as f:
        training_data = pickle.load(f)

    n_states = n_dropped = n_actions = n_empty_lp = 0

    for key, states in training_data.items():
        kept = []
        for state in states:
            optimal = state.get('optimal_action', '')
            if args.drop_unlabeled and not optimal:
                n_dropped += 1
                continue

            log_probs = state.get('log_probs_of_admissible_actions', {})
            bin_dist = {}
            for action in state.get('admissible_actions', []):
                token_lps = log_probs.get(action, [])
                if token_lps:
                    bin_dist[action] = logprobs_to_softmax_bins(token_lps)
                else:
                    bin_dist[action] = [0] * SOFTMAX_BINS
                    n_empty_lp += 1
                n_actions += 1

            state['softmax_bin_distributions'] = bin_dist
            state.pop('log_probs_of_admissible_actions', None)
            kept.append(state)
            n_states += 1

        training_data[key] = kept

    with open(args.out_path, 'wb') as f:
        pickle.dump(training_data, f)

    print(f"\n  episodes        : {len(training_data)}")
    print(f"  states kept     : {n_states}   dropped(unlabeled): {n_dropped}")
    print(f"  candidate slots : {n_actions}   (with empty logprobs: {n_empty_lp})")
    print(f"  saved           : {args.out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
