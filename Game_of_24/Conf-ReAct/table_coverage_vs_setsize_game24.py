"""Empirical coverage vs absolute set-size bins (0-2, 2-4, 4-16) — Game of 24,
as a TABLE, for Conf-ReAct (alpha=0.1/0.2/0.3) AND the baselines.

Rows per model:
  Conf-ReAct a=0.1/0.2/0.3 : per-state (pred-set size, covered) from the BFS
      conformal logs (same extraction as the distribution plot).
  ReAct / ReflAct          : single-action agents -> set size 1 (bin 0-2).
      covered = the printed move's in-log oracle label is correct/win;
      denominator = all non-think (non-neutral) labeled moves.
  Rollback                 : its log has no oracle tags, so committed moves
      ('Act k:' lines; numbering k gives the rollback depth: traj=traj[:k-1]+mv)
      are re-labeled OFFLINE with g24_oracle.label_action, replaying each
      episode's state from the puzzle numbers. covered = correct/win.
  ReAct/Reflexion T1, Reflexion T2/T3 : the self-sized-set baseline
      (reflexion_k3_trial{t}_<model>.csv).

Cells: coverage% (n states). Writes coverage_vs_setsize_game24.txt/.csv.

Usage:  python table_coverage_vs_setsize_game24.py
"""
import csv
import re
import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g24_oracle import label_action  # noqa: E402

# reuse the plot's conformal extraction ([BFS Node]/Pred set/[STATEMETRIC])
PRED_RE = re.compile(r"Pred set \((\d+)/(\d+)\)")
FB_RE   = re.compile(r"\[fallback\] Empty pred set")
SM_RE   = re.compile(r"\[STATEMETRIC\]\s+set_size_pct=[0-9.]+\s+covered=([01])")
ORACLE_RE = re.compile(r"\[oracle: ([a-z_]+)\]")
PUZZLE_RE = re.compile(r"^\[(\d+)/100\] puzzle='([^']+)'")
ID_RE = re.compile(r"^id:(\d+),")
ACT_COLON_RE = re.compile(r"^\s*Act\s+(\d+)\s*:\s*(.+?)\s*$")

COVERED_LABELS = {"correct", "win"}
NEUTRAL_LABELS = {"neutral"}


def extract_conformal(log_path):
    out, k = [], None
    for line in open(log_path, errors="ignore"):
        if line.startswith("[BFS Node"):
            k = None
        m = PRED_RE.search(line)
        if m:
            k = int(m.group(1))
        elif FB_RE.search(line):
            k = 1
        sm = SM_RE.search(line)
        if sm:
            if k is not None and k > 0:
                out.append((k, int(sm.group(1))))
            k = None
    return out


def extract_oracle_singleaction(log_path):
    """ReAct/ReflAct: every non-neutral [oracle: X] tag is one size-1 'set'."""
    out = []
    for line in open(log_path, errors="ignore"):
        m = ORACLE_RE.search(line)
        if m and m.group(1) not in NEUTRAL_LABELS:
            out.append((1, 1 if m.group(1) in COVERED_LABELS else 0))
    return out


def load_puzzles(conformal_log):
    """id (1-based) -> puzzle numbers string, from '[k/100] puzzle=..' summaries."""
    puzzles = {}
    for line in open(conformal_log, errors="ignore"):
        m = PUZZLE_RE.match(line)
        if m:
            puzzles[int(m.group(1))] = m.group(2)
    return puzzles


def extract_rollback(log_path, puzzles):
    """Re-label rollback's committed moves offline. 'Act k:' numbering encodes
    the depth after rollbacks: trajectory = traj[:k-1] + move."""
    out = []
    traj = []            # list of (move, child_state_list)
    ep = 1
    for line in open(log_path, errors="ignore"):
        idm = ID_RE.match(line)
        if idm:
            ep = int(idm.group(1)) + 1     # next episode
            traj = []
            continue
        m = ACT_COLON_RE.match(line)
        if not m or ep not in puzzles:
            continue
        k, move = int(m.group(1)), m.group(2)
        if k == 0 or move.lower().startswith("think"):
            continue
        traj = traj[:k - 1]
        state = puzzles[ep].split() if not traj else traj[-1][1]
        if state is None:
            continue
        ref = puzzles[ep].split()
        try:
            lab = label_action(state, move, reference=ref)
        except Exception:
            continue
        if lab.get("label") in NEUTRAL_LABELS:
            continue
        out.append((1, 1 if lab.get("label") in COVERED_LABELS else 0))
        if lab.get("kind") == "intermediate" and lab.get("child"):
            traj.append((move, lab["child"]))
        else:
            traj.append((move, state))     # answer/illegal: state unchanged
    return out


def load_trial_csv(path):
    if not Path(path).exists():
        return None
    return [(int(r["set_size"]), int(r["covered"]))
            for r in csv.DictReader(open(path))]


def bucket(samples, edges):
    n = len(edges) - 1
    s = np.zeros(n); c = np.zeros(n, dtype=int)
    for size, cov in samples:
        b = int(np.clip(np.searchsorted(edges, size, side="left") - 1, 0, n - 1))
        if size <= edges[0]:
            b = 0
        s[b] += cov; c[b] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(c > 0, 100.0 * s / c, np.nan)
    return avg, c


NICE = {"qwen25_3b": "Qwen2.5-3B", "qwen3_8b": "Qwen3-8B", "gemma4_12b": "Gemma-4-12B"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen25_3b", "qwen3_8b", "gemma4_12b"])
    ap.add_argument("--bin_edges", nargs="+", type=float, default=[0, 2, 4, 16])
    ap.add_argument("--out_txt", default="coverage_vs_setsize_game24.txt")
    ap.add_argument("--out_csv", default="coverage_vs_setsize_game24.csv")
    args = ap.parse_args()
    root = Path(__file__).parent

    edges = np.array(sorted(args.bin_edges), dtype=float)
    n_bins = len(edges) - 1
    bin_labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(n_bins)]

    lines = []
    csv_rows = [["model", "method", "overall_coverage%", "avg_set_size", "n"] +
                [f"cov_{bl}" for bl in bin_labels] + [f"n_{bl}" for bl in bin_labels]]

    def out(s=""):
        print(s); lines.append(s)

    for model in args.models:
        d = root / f"compare_logs_100_{model}"
        series = []
        for a, fn in [(0.1, "m5_alpha01.txt"), (0.2, "m5_alpha02.txt"), (0.3, "m5_alpha03.txt")]:
            if (d / fn).exists():
                series.append((f"Conf-ReAct a={a}", extract_conformal(d / fn)))
        for name, fn in [("ReAct", "base_react.txt"), ("ReflAct", "base_reflact.txt")]:
            if (d / fn).exists():
                series.append((name, extract_oracle_singleaction(d / fn)))
        if (d / "base_rollback.txt").exists() and (d / "m5_alpha01.txt").exists():
            series.append(("Rollback", extract_rollback(d / "base_rollback.txt",
                                                        load_puzzles(d / "m5_alpha01.txt"))))
        for t in [1, 2, 3]:
            samp = load_trial_csv(root / f"reflexion_k3_trial{t}_{model}.csv")
            if samp is not None:
                nm = "ReAct/Reflexion T1" if t == 1 else f"Reflexion T{t}"
                series.append((nm, samp))

        out("=" * 96)
        out(f"{NICE.get(model, model)}   (Game of 24 — per-bin empirical coverage over "
            f"absolute set size [# actions kept])")
        out("=" * 96)
        out(f"  {'method':<20}" + "".join(f"{bl:>14}" for bl in bin_labels)
            + f"{'OVERALL':>12}{'avg size':>10}    (cell = coverage% (n))")
        for name, samples in series:
            if not samples:
                out(f"  {name:<20}  (no data)"); continue
            avg, cnt = bucket(samples, edges)
            overall = 100.0 * np.mean([c for _, c in samples])
            mean_sz = np.mean([s for s, _ in samples])
            cells = ""
            for b in range(n_bins):
                n = int(cnt[b]); v = avg[b]
                cells += (f"{v:>7.0f}% ({n:>4d})" if n > 0 else f"{'-':>7} ({0:>4d})")
            out(f"  {name:<20}" + cells + f"{overall:>10.1f}% {mean_sz:>9.2f}")
            csv_rows.append([NICE.get(model, model), name, f"{overall:.2f}",
                             f"{mean_sz:.3f}", len(samples)]
                            + [f"{avg[i]:.1f}" if not np.isnan(avg[i]) else ""
                               for i in range(n_bins)]
                            + [int(cnt[i]) for i in range(n_bins)])
        out()

    with open(root / args.out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(root / args.out_csv, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"Saved -> {root / args.out_txt}\nSaved -> {root / args.out_csv}")


if __name__ == "__main__":
    main()
