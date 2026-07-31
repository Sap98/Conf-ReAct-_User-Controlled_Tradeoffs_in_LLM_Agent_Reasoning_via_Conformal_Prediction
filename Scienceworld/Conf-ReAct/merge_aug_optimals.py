"""
Merge gold pickle + augmentation pickle into a single training pickle whose
records carry a set of optimal actions per state instead of a single label.

For every (task_name, tuple(prev_actions)) state seen across both files we
union the labels:
  - gold pickle contributes:        optimal_action = gold[t]
  - augmentation 'A' records add:   optimal_action = X  (alt-optimal at gold[:t])
  - augmentation 'B' records pass through unchanged (off-gold state, single label)

Output schema (per record):
  {
    'task_name':         str,
    'prev_actions':      list[str],
    'location':          str,
    'admissible_actions':list[str],
    'log_probs_of_admissible_actions': dict[str -> list[float]],
    'optimal_actions':   list[str],    # <-- NEW: list, may have len>=1
    'record_type':       'gold' | 'aug-A' | 'aug-B',
    'source_key':        str,          # original pickle key for debugging
  }

Usage:
  python merge_aug_optimals.py \
      --gold training_data_2_sciworld.pkl \
      --aug  training_data_aug_mpo_sciworld.pkl \
      --out  training_data_merged_sciworld.pkl
"""
import os
import sys
import argparse
import pickle
from collections import defaultdict


def _state_key(rec):
    return (rec['task_name'], tuple(rec['prev_actions']))


def merge(gold_path, aug_path, out_path):
    with open(gold_path, 'rb') as f:
        gold = pickle.load(f)
    with open(aug_path, 'rb') as f:
        aug = pickle.load(f)

    # alt_optimals[(task, prev_actions_tuple)] -> set of extra optimals (X's)
    alt_optimals = defaultdict(set)
    aug_b_records = []   # off-gold B-records: keep as-is, single label
    aug_a_count = 0
    aug_b_count = 0

    for ep_key, recs in aug.items():
        for r in recs:
            rtype = r.get('record_type')
            if rtype == 'A':
                alt_optimals[_state_key(r)].add(r['optimal_action'])
                aug_a_count += 1
            elif rtype == 'B':
                aug_b_records.append((ep_key, r))
                aug_b_count += 1
            else:
                # Legacy records without record_type — treat as B (off-gold).
                aug_b_records.append((ep_key, r))
                aug_b_count += 1

    out = {}
    promoted = 0
    for ep_key, recs in gold.items():
        new_recs = []
        for r in recs:
            sk = _state_key(r)
            opts = {r['optimal_action']}
            if sk in alt_optimals:
                opts |= alt_optimals[sk]
                promoted += 1
            nr = dict(r)
            nr.pop('optimal_action', None)
            nr['optimal_actions'] = sorted(opts)
            nr['record_type'] = 'gold'
            nr['source_key'] = ep_key
            new_recs.append(nr)
        out[ep_key] = new_recs

    # Append B-records as their own entries (off-gold states, single label list)
    for ep_key, r in aug_b_records:
        nr = dict(r)
        nr.pop('optimal_action', None)
        nr['optimal_actions'] = [r['optimal_action']]
        nr['record_type'] = 'aug-B'
        nr['source_key'] = ep_key
        out.setdefault(ep_key, []).append(nr)

    with open(out_path, 'wb') as f:
        pickle.dump(out, f)

    n_gold_recs = sum(len(v) for v in gold.values())
    n_out_recs  = sum(len(v) for v in out.values())
    print(f"gold episodes      : {len(gold)}  ({n_gold_recs} records)")
    print(f"aug episodes       : {len(aug)}   (A={aug_a_count}, B={aug_b_count})")
    print(f"states promoted    : {promoted}   (gold record gained ≥1 alt-optimal)")
    print(f"output episodes    : {len(out)}   ({n_out_recs} records)")
    print(f"wrote              : {out_path}")


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Merge gold + augmentation pickles "
                                "into a single multi-optimal pickle")
    p.add_argument("--gold", default=os.path.join(here, "training_data_2_sciworld.pkl"))
    p.add_argument("--aug",  default=os.path.join(here, "training_data_aug_mpo_sciworld.pkl"))
    p.add_argument("--out",  default=os.path.join(here, "training_data_merged_sciworld.pkl"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge(args.gold, args.aug, args.out)
