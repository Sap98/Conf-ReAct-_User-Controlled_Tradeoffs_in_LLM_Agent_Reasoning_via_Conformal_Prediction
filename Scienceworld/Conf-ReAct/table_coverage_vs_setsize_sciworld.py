"""Coverage vs Set Size for ScienceWorld — as a TABLE (no plot), including the
reflexion self-set baselines (T1/T2/T3) alongside conformal (alpha).

For each model, rows = method (alpha=0.1/0.2/0.3, ReAct/Reflexion T1, Reflexion
T2/T3), columns = absolute set-size bins (0-2, 2-4, 4-16) + OVERALL + avg size.
Cells = coverage% (n states): fraction of states in that bin whose prediction /
proposed set contains a gold action.

Conformal data: setsize_cov_<stem>.csv  (dump_setsize_coverage_sciworld.py)
Baseline data : reflexion_selfset_trial{1,2,3}_<stem>.csv (reflexion_selfset_sciworld.py)

Writes coverage_vs_setsize_sciworld.txt + .csv.

Usage:  python table_coverage_vs_setsize_sciworld.py
"""
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_conformal(path):
    """{alpha: [(set_size, covered)]}"""
    by_alpha = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            by_alpha[float(row["alpha"])].append((int(row["set_size"]), int(row["covered"])))
    return dict(sorted(by_alpha.items()))


def load_trial(path):
    """[(set_size, covered)] from a reflexion_selfset_trial CSV, or None."""
    if not Path(path).exists():
        return None
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            out.append((int(row["set_size"]), int(row["covered"])))
    return out


def bucket(samples, edges):
    """Absolute-size buckets, same convention as the distribution bars."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", default=[
        "setsize_cov_qwen25.csv", "setsize_cov_qwen3.csv", "setsize_cov_gemma.csv"])
    ap.add_argument("--labels", nargs="+", default=["Qwen2.5-3B", "Qwen3-8B", "Gemma-4-12B"])
    ap.add_argument("--bin_edges", nargs="+", type=float, default=[0, 2, 4, 16])
    ap.add_argument("--out_txt", default="coverage_vs_setsize_sciworld.txt")
    ap.add_argument("--out_csv", default="coverage_vs_setsize_sciworld.csv")
    args = ap.parse_args()

    edges = np.array(sorted(args.bin_edges), dtype=float)
    n_bins = len(edges) - 1
    bin_labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(n_bins)]

    lines = []
    csv_rows = [["model", "method", "overall_coverage%", "avg_set_size", "n"] +
                [f"cov_{bl}" for bl in bin_labels] + [f"n_{bl}" for bl in bin_labels]]

    def out(s=""):
        print(s); lines.append(s)

    for label, csvp in zip(args.labels, args.csvs):
        if not Path(csvp).exists():
            out(f"[skip] {label}: {csvp} not found"); continue
        stem = Path(csvp).stem.replace("setsize_cov_", "")

        # build ordered (method_name, samples) list: conformal alphas then reflexion trials
        series = [(f"alpha={a}", s) for a, s in load_conformal(csvp).items()]
        for t in [1, 2, 3]:
            samp = load_trial(Path(csvp).parent / f"reflexion_selfset_trial{t}_{stem}.csv")
            if samp is not None:
                name = "ReAct/Reflexion T1" if t == 1 else f"Reflexion T{t}"
                series.append((name, samp))

        out("=" * 96)
        out(f"{label}   (ScienceWorld — per-bin coverage over absolute set size [# actions kept])")
        out("=" * 96)
        hdr = f"  {'method':<20}" + "".join(f"{bl:>14}" for bl in bin_labels) \
              + f"{'OVERALL':>12}{'avg size':>10}    (cell = coverage% (n))"
        out(hdr)
        for name, samples in series:
            avg, cnt = bucket(samples, edges)
            overall = 100.0 * np.mean([c for _, c in samples]) if samples else float("nan")
            mean_sz = np.mean([s for s, _ in samples]) if samples else float("nan")
            cells = ""
            for b in range(n_bins):
                n = int(cnt[b]); v = avg[b]
                cells += (f"{v:>7.0f}% ({n:>4d})" if n > 0 else f"{'-':>7} ({0:>4d})")
            out(f"  {name:<20}" + cells + f"{overall:>10.1f}% {mean_sz:>9.2f}")
            csv_rows.append([label, name, f"{overall:.2f}", f"{mean_sz:.3f}", len(samples)]
                            + [f"{avg[i]:.1f}" if not np.isnan(avg[i]) else "" for i in range(n_bins)]
                            + [int(cnt[i]) for i in range(n_bins)])
        out()

    with open(args.out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(args.out_csv, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"Saved -> {args.out_txt}\nSaved -> {args.out_csv}")


if __name__ == "__main__":
    main()
