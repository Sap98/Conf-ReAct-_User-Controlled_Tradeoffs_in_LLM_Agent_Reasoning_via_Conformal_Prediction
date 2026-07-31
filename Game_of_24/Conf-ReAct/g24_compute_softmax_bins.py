"""
Stage 2 for Game of 24 — add softmax-bin distributions to the stage-1 pickle.

Mirrors compute_softmax_bins_10.py (AlfWorld), but:
  * NO env / expert replay is needed — `optimal_action` and the full
    `correct_actions` set already come from the oracle in stage 1.
  * It PRESERVES `correct_actions` (the complete set of solvability-preserving
    moves), `optimal_action`, `oracle_labels`, `solvable`, `step` so the
    set-coverage conformal variant can use them downstream.

For each state it adds:
    'softmax_bin_distributions' : {action: [n0, n1, ..., n_{B-1}]}
        a B-bin histogram (default B=10) over per-token softmax probs exp(lp),
        one histogram per admissible action — exactly what the score function /
        conformal predictor consume as `softmax_values`.

Invariants checked (so calibration won't break):
  - optimal_action ∈ admissible_actions
  - every correct action ∈ admissible_actions
  - every admissible action has a bin histogram

Run:
    python3 g24_compute_softmax_bins.py \
        --in training_data_game24.pkl --out training_data_2_game24.pkl --bins 10
"""
import math
import pickle
import argparse


def logprobs_to_softmax_bins(token_logprobs, num_bins):
    """Per-token softmax prob exp(lp) ∈ (0,1], histogrammed into num_bins."""
    bin_width = 1.0 / num_bins
    bins = [0] * num_bins
    for lp in token_logprobs:
        val = math.exp(lp)
        idx = int(val / bin_width)
        if idx >= num_bins:
            idx = num_bins - 1
        bins[idx] += 1
    return bins


def parse_args():
    p = argparse.ArgumentParser(description="Game-of-24 stage-2 softmax binning")
    p.add_argument("--in", dest="inp", default="training_data_game24.pkl")
    p.add_argument("--out", default="training_data_2_game24.pkl")
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--keep-logprobs", action="store_true",
                   help="keep raw log_probs_of_admissible_actions in the output")
    p.add_argument("--require-correct", action="store_true",
                   help="drop states whose correct_actions set is empty "
                        "(unsolvable / dead-end states from --policy llm)")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.inp, "rb") as f:
        training_data = pickle.load(f)
    print(f"Loaded {args.inp}: {len(training_data)} episodes")

    n_states = n_dropped = n_states_kept = 0
    warn_optimal = warn_correct = warn_nolp = 0

    for key, states in training_data.items():
        kept = []
        for st in states:
            n_states += 1
            admissible = st.get("admissible_actions", [])
            correct = st.get("correct_actions", []) or []
            optimal = st.get("optimal_action")

            if args.require_correct and not correct:
                n_dropped += 1
                continue

            # ── integrity checks ────────────────────────────────────────────
            if optimal is not None and optimal not in admissible:
                admissible.append(optimal)            # guarantee positive present
                warn_optimal += 1
            for a in correct:
                if a not in admissible:
                    admissible.append(a)
                    warn_correct += 1
            st["admissible_actions"] = admissible

            # keep correct_actions as a de-duplicated set (order preserved)
            st["correct_actions"] = list(dict.fromkeys(correct))

            # ── softmax bins per admissible action ──────────────────────────
            log_probs = st.get("log_probs_of_admissible_actions", {})
            bin_dists = {}
            for a in admissible:
                lps = log_probs.get(a, [])
                if lps:
                    bin_dists[a] = logprobs_to_softmax_bins(lps, args.bins)
                else:
                    bin_dists[a] = [0] * args.bins
                    warn_nolp += 1
            st["softmax_bin_distributions"] = bin_dists

            if not args.keep_logprobs:
                st.pop("log_probs_of_admissible_actions", None)

            kept.append(st)
            n_states_kept += 1
        training_data[key] = kept

    with open(args.out, "wb") as f:
        pickle.dump(training_data, f)

    print(f"  states seen      : {n_states}")
    print(f"  states kept      : {n_states_kept}   dropped(empty correct): {n_dropped}")
    print(f"  fixups           : optimal∉adm {warn_optimal}, correct∉adm {warn_correct}, "
          f"actions w/o logprobs {warn_nolp}")
    print(f"  saved -> {args.out}  (bins={args.bins})")


if __name__ == "__main__":
    main()
