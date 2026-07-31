"""
Prediction-set-size DISTRIBUTION (ABSOLUTE size) — Game of 24. AAAI-formatted.

Bars = per-α set-size distribution, in solid deep colors. Coverage lines are OFF
by default (use --coverage for the secondary-axis line version); per-bin coverage
labels on top of bars are OFF by default (use --bar_labels to show them).

Ground truth is in-log:
  * absolute set size = k from "Pred set (k/N)" (fallback 'Empty pred set' → 1).
  * coverage          = the pipeline's own "[STATEMETRIC] ... covered=0/1".

Usage:
    python fraction_of_states_vs_prediction_set_size.py
    python fraction_of_states_vs_prediction_set_size.py --coverage
    python fraction_of_states_vs_prediction_set_size.py --bar_labels
    python fraction_of_states_vs_prediction_set_size.py --wide
"""
import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── AAAI camera-ready style ───────────────────────────────────────────────────
AAAI_COL_WIDTH  = 3.3
AAAI_FULL_WIDTH = 7.0
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6, "legend.title_fontsize": 6,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.0, "lines.markersize": 2.5,
    "grid.linewidth": 0.4, "grid.linestyle": ":",
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "0.7",
    "legend.borderpad": 0.3, "legend.handlelength": 1.4,
    "legend.handletextpad": 0.4, "legend.labelspacing": 0.25,
    "legend.columnspacing": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})

# Conf-ReAct: one warm family, light→dark with stricter target (0.1 light orange,
# 0.2 deep orange, 0.3 red) so the conformal bars pop as a group.
ALPHA_COLORS = {0.1: "#fdae6b", 0.2: "#f16913", 0.3: "#d62728"}
FALLBACK_COLORS = ["#fdae6b", "#f16913", "#d62728", "#ff7f0e", "#9467bd"]
MARKERS = ["o", "s", "^", "D", "v"]

PRED_RE = re.compile(r"Pred set \((\d+)/(\d+)\)")
FB_RE   = re.compile(r"\[fallback\] Empty pred set")
SM_RE   = re.compile(r"\[STATEMETRIC\]\s+set_size_pct=[0-9.]+\s+covered=([01])")


def extract_matched(log_path):
    """Return list of (abs_set_size, covered 0/1) per scored state."""
    out = []
    k = None
    for line in open(log_path):
        if line.startswith("[BFS Node"):
            k = None
        m = PRED_RE.search(line)
        if m:
            k = int(m.group(1))
        elif FB_RE.search(line):
            k = 1
        sm = SM_RE.search(line)
        if sm:
            covered = int(sm.group(1))
            if k is not None and k > 0:
                out.append((k, covered))
            k = None
    print(f"  {log_path}: matched {len(out)} states")
    return out


def main():
    ap = argparse.ArgumentParser()
    default_logs = [
        "compare_logs_100_qwen3_8b/m5_alpha01.txt",
        "compare_logs_100_qwen3_8b/m5_alpha02.txt",
        "compare_logs_100_qwen3_8b/m5_alpha03.txt",
    ]
    ap.add_argument("--logs", nargs="+", default=default_logs)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    ap.add_argument("--max_size", type=int, default=0,
                    help="cap x-axis at this absolute size (0 = observed max).")
    ap.add_argument("--bin_width", type=int, default=2,
                    help="uniform bin width; used only if --bin_edges is cleared.")
    ap.add_argument("--bin_edges", nargs="+", type=float,
                    default=[0, 2, 4, 16],
                    help="bin edges; default 0 2 4 16 → bins 0-2, 2-4, 4-16.")
    ap.add_argument("--counts", action="store_true",
                    help="bars = raw counts (default: fraction of that α's states).")
    ap.add_argument("--wide", action="store_true",
                    help="two-column spanning figure (7.0in) instead of 3.3in.")
    ap.add_argument("--height", type=float, default=None)
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--legend_loc", default="upper right",
                    help="matplotlib legend loc.")
    ap.add_argument("--legend_fontsize", type=float, default=6.0)
    ap.add_argument("--legend_bbox", nargs=2, type=float, default=None,
                    metavar=("X", "Y"))
    ap.add_argument("--baseline_csv", nargs="+",
                    default=["reflexion_k3_trial1_qwen3_8b.csv",
                             "reflexion_k3_trial2_qwen3_8b.csv",
                             "reflexion_k3_trial3_qwen3_8b.csv"],
                    help="one or more CSVs (location,set_size,covered) as baseline series "
                         "(default: reflexion T1/T2/T3, Qwen3-8B).")
    ap.add_argument("--baseline_label", nargs="+",
                    default=["ReAct/Reflexion T1", "reflexion T2", "reflexion T3"])
    ap.add_argument("--coverage", action="store_true",
                    help="draw coverage LINES on a secondary axis (default: OFF).")
    ap.add_argument("--bar_labels", action="store_true",
                    help="show per-bin coverage %% labels on top of bars (default: off).")
    ap.add_argument("--out", default="setsize_abs_distribution_coverage_game24.pdf")
    args = ap.parse_args()
    normalize = not args.counts
    show_cov_line = args.coverage          # lines are opt-IN
    show_bar_labels = args.bar_labels      # on-bar labels are opt-IN

    if len(args.logs) != len(args.alphas):
        raise SystemExit("--logs and --alphas must have equal length.")

    runs = []
    observed_max = 1
    for idx, (log, alpha) in enumerate(zip(args.logs, args.alphas)):
        print(f"\n=== α = {alpha} ===")
        matched = extract_matched(log)
        if not matched:
            continue
        sizes = np.array([s for s, _ in matched])
        covs  = np.array([c for _, c in matched])
        observed_max = max(observed_max, int(sizes.max()))
        color = ALPHA_COLORS.get(alpha, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])
        runs.append({"alpha": alpha, "color": color,
                     "marker": MARKERS[idx % len(MARKERS)],
                     "label": f"Conf-ReAct $\\alpha$={alpha}",
                     "sizes": sizes, "covs": covs, "n": len(matched),
                     "overall_cov": 100.0 * covs.mean()})
    if not runs:
        raise SystemExit("No matched states.")

    if args.baseline_csv:
        import csv as _csv
        # Reflexion trials: one cool family, light→dark green (T1 light, T2 deep,
        # T3 dark) contrasting with the warm Conf-ReAct family.
        BL_COLORS  = ["#a1d99b", "#41ab5d", "#006d2c", "#17becf", "#e377c2"]
        BL_MARKERS = ["D", "P", "X", "*", "v"]
        labels = args.baseline_label or [f"baseline{i+1}" for i in range(len(args.baseline_csv))]
        for bi, (csvp, lab) in enumerate(zip(args.baseline_csv, labels)):
            bs, bc = [], []
            with open(csvp) as f:
                for row in _csv.DictReader(f):
                    bs.append(int(row["set_size"])); bc.append(int(row["covered"]))
            if not bs:
                continue
            bsizes, bcovs = np.array(bs), np.array(bc)
            observed_max = max(observed_max, int(bsizes.max()))
            runs.append({"alpha": lab, "color": BL_COLORS[bi % len(BL_COLORS)],
                         "marker": BL_MARKERS[bi % len(BL_MARKERS)],
                         "label": f"{lab}",
                         "sizes": bsizes, "covs": bcovs, "n": len(bs),
                         "overall_cov": 100.0 * bcovs.mean()})
            print(f"\n=== baseline: {lab}  n={len(bs)}  "
                  f"mean_size={bsizes.mean():.2f}  coverage={100*bcovs.mean():.1f}% ===")

    max_size = args.max_size if args.max_size > 0 else observed_max
    if args.bin_edges:
        edges = np.array(sorted(args.bin_edges), dtype=float)
    else:
        bw = max(1, args.bin_width)
        edges = np.arange(0, max_size + bw, bw, dtype=float)
    n_bins = len(edges) - 1
    xs = np.arange(n_bins)
    bin_labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(n_bins)]

    for r in runs:
        # size s falls in bin (edges[i], edges[i+1]]; sizes past the last edge
        # are clipped into the final bin.
        bins = np.clip(np.searchsorted(edges, r["sizes"], side="left") - 1, 0, n_bins - 1)
        total = np.array([(bins == b).sum() for b in range(n_bins)])
        cov_c = np.array([r["covs"][bins == b].sum() for b in range(n_bins)])
        r["total"] = total
        r["dist"] = total / r["n"] if normalize else total.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            r["cov_line"] = np.where(total > 0, 100.0 * cov_c / total, np.nan)

    # ── console tables ────────────────────────────────────────────────────────
    unit = "frac" if normalize else "count"
    print(f"\n=== SET-SIZE DISTRIBUTION TABLE ({unit} of states per bin) ===")
    print("  " + f"{'series':<16}" + f"{'n':>6}" + "".join(f"{lbl:>9}" for lbl in bin_labels))
    for r in runs:
        lbl = r['label'].replace('$', '').replace('\\alpha', 'alpha')
        vals = "".join((f"{d:>9.3f}" if normalize else f"{int(d):>9d}") for d in r["dist"])
        print("  " + f"{lbl:<16}" + f"{r['n']:>6}" + vals)

    print(f"\n=== PER-BIN COVERAGE TABLE (coverage%% ; n states in that bin) ===")
    print("  " + f"{'series':<16}" + "".join(f"{lbl:>14}" for lbl in bin_labels) + f"{'overall':>10}")
    for r in runs:
        lbl = r['label'].replace('$', '').replace('\\alpha', 'alpha')
        cells = ""
        for b in range(n_bins):
            cnt = int(r["total"][b]); cov = r["cov_line"][b]
            cells += (f"{cov:>7.0f}% ({cnt:>3d})" if cnt > 0 else f"{'-':>7} ({0:>3d})")
        print("  " + f"{lbl:<16}" + cells + f"{r['overall_cov']:>9.1f}%")

    # ── plot ──────────────────────────────────────────────────────────────────
    n_alpha = len(runs)
    bar_w = 0.8 / n_alpha
    fig_w = AAAI_FULL_WIDTH if args.wide else AAAI_COL_WIDTH
    fig_h = args.height if args.height else (2.6 if args.wide else 2.1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax2 = ax.twinx() if show_cov_line else None

    for j, r in enumerate(runs):
        offset = (j - (n_alpha - 1) / 2) * bar_w
        xpos = xs + offset
        # SOLID deep color — no alpha, thin white edge to separate touching bars
        ax.bar(xpos, r["dist"], bar_w * 0.9, color=r["color"],
               edgecolor="white", linewidth=0.4, label=r["label"])

        if show_bar_labels:
            for xi, h, cov, cnt in zip(xpos, r["dist"], r["cov_line"], r["total"]):
                if cnt == 0 or np.isnan(cov):
                    continue
                ax.text(xi, h + 0.012, f"{cov:.0f}", ha="center", va="bottom",
                        fontsize=5.0, color=r["color"], fontweight="bold",
                        rotation=90 if not args.wide else 0)

        if show_cov_line:
            mask = ~np.isnan(r["cov_line"])
            ax2.plot(xpos[mask], r["cov_line"][mask], marker=r["marker"],
                     markersize=2.5, linewidth=1.0, color=r["color"],
                     markeredgewidth=0.4, markeredgecolor="white",
                     label=f"{r['label']} cov")

    ax.set_xticks(xs)
    ax.set_xticklabels(bin_labels, rotation=0 if args.wide else 20,
                       ha="center" if args.wide else "right")
    ax.set_xlabel("Prediction set size (# actions kept)", labelpad=2)
    ax.set_ylabel("Fraction of states" if normalize else "Number of states", labelpad=2)
    if args.model:
        ax.set_title(args.model, pad=4)
    ax.set_ylim(0, 1.0 if normalize else max(r["dist"].max() for r in runs) * 1.22)

    if show_cov_line:
        ax2.set_ylabel("Coverage %", labelpad=3)
        ax2.set_ylim(0, 105)
        ax2.spines["top"].set_visible(False)
        ax2.tick_params(pad=1.5)

    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.tick_params(pad=1.5)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = (ax2.get_legend_handles_labels() if show_cov_line else ([], []))
    leg = ax.legend(h1 + h2, l1 + l2, loc=args.legend_loc,
                    bbox_to_anchor=tuple(args.legend_bbox) if args.legend_bbox else None,
                    ncol=1, borderaxespad=0.3, fontsize=args.legend_fontsize,
                    handlelength=1.1, handletextpad=0.3, labelspacing=0.2,
                    borderpad=0.25, columnspacing=0.6)
    leg.get_frame().set_linewidth(0.4)

    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(Path(args.out).with_suffix(".png"), dpi=600,
                bbox_inches="tight", pad_inches=0.02)
    print(f"\nSaved plot → {args.out}")


if __name__ == "__main__":
    main()