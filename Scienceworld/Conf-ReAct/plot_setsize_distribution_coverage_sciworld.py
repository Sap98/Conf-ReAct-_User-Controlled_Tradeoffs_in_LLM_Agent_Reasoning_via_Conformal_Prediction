"""Set-size distribution + coverage plot for ScienceWorld conformal — the
game-24 'plot_setsize_abs_distribution_coverage_game24.py' analog, for all three
models (Qwen2.5-3B, Qwen3-8B, Gemma-4-12B).

Input: the per-state CSVs written by dump_setsize_coverage_sciworld.py
       (columns: model,alpha,set_size,n_admissible,set_frac,covered), one per
       model, produced from the OFFLINE conformal eval on held-out gold states
       (the same evaluation coverage_<model>.json reports) — because online BFS
       play almost never lands on a gold state, coverage is only measurable here.

Per model it makes one figure (bars = set-size distribution per alpha; optional
coverage line) + prints a distribution table and a per-bin coverage table.

x-axis: --x frac (default) bins the set size as a FRACTION of admissible actions
        (ScienceWorld action spaces vary per state); --x abs uses absolute size.

Usage:
    python plot_setsize_distribution_coverage_sciworld.py
    python plot_setsize_distribution_coverage_sciworld.py --coverage
    python plot_setsize_distribution_coverage_sciworld.py --x abs --bin_edges 0 2 4 8 30
"""
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── AAAI camera-ready style (matches the game-24 script) ──────────────────────
AAAI_COL_WIDTH, AAAI_FULL_WIDTH = 3.3, 7.0
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
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})
ALPHA_COLORS = {0.1: "#fdae6b", 0.2: "#f16913", 0.3: "#d62728"}  # light→deep orange→red
MARKERS = ["o", "s", "^", "D", "v"]


def load_csv(path, x_mode):
    """Return {alpha: (sizes_array, covs_array)} where 'size' is abs or pct."""
    by_alpha = defaultdict(lambda: ([], []))
    with open(path) as f:
        for row in csv.DictReader(f):
            a = float(row["alpha"])
            if x_mode == "abs":
                s = int(row["set_size"])
            else:                       # fraction -> percent 0..100
                s = 100.0 * float(row["set_frac"])
            by_alpha[a][0].append(s)
            by_alpha[a][1].append(int(row["covered"]))
    return {a: (np.array(v[0], dtype=float), np.array(v[1]))
            for a, v in sorted(by_alpha.items())}


def plot_model(label, csv_path, args):
    x_is_frac = (args.x == "frac")
    data = load_csv(csv_path, args.x)
    if not data:
        print(f"  [{label}] no rows in {csv_path}"); return

    if args.bin_edges:
        edges = np.array(sorted(args.bin_edges), dtype=float)
    elif x_is_frac:
        edges = np.array([0, 20, 40, 60, 80, 100], dtype=float)
    else:
        edges = np.array([0, 2, 4, 16], dtype=float)     # game24-style abs bins
    n_bins = len(edges) - 1
    xs = np.arange(n_bins)
    unit = "%" if x_is_frac else ""
    bin_labels = [f"{int(edges[i])}-{int(edges[i+1])}{unit}" for i in range(n_bins)]

    def make_run(sizes, covs, label, color, marker):
        bins = np.clip(np.searchsorted(edges, sizes, side="left") - 1, 0, n_bins - 1)
        bins = np.where(sizes <= edges[0], 0, bins)          # left-edge -> bin 0
        total = np.array([(bins == b).sum() for b in range(n_bins)])
        cov_c = np.array([covs[bins == b].sum() for b in range(n_bins)])
        with np.errstate(invalid="ignore", divide="ignore"):
            cov_line = np.where(total > 0, 100.0 * cov_c / total, np.nan)
        return {"label": label, "color": color, "marker": marker,
                "n": len(sizes), "total": total,
                "dist": total / len(sizes) if not args.counts else total.astype(float),
                "cov_line": cov_line, "overall_cov": 100.0 * covs.mean() if len(covs) else np.nan,
                "mean_size": sizes.mean() if len(sizes) else 0.0}

    runs = []
    for i, (alpha, (sizes, covs)) in enumerate(data.items()):
        runs.append(make_run(sizes, covs, f"Conf-ReAct $\\alpha$={alpha}",
                              ALPHA_COLORS.get(alpha, "#7f7f7f"), MARKERS[i % len(MARKERS)]))

    # reflexion self-set baselines: reflexion_selfset_trial{t}_<stem>.csv (per model)
    BL_COLORS = ["#a1d99b", "#41ab5d", "#006d2c", "#17becf", "#e377c2"]  # light→deep→dark green
    BL_MARKERS = ["D", "P", "X", "*", "v"]
    stem = Path(csv_path).stem.replace("setsize_cov_", "")
    for bi, t in enumerate([1, 2, 3]):
        blp = Path(csv_path).parent / f"reflexion_selfset_trial{t}_{stem}.csv"
        if not blp.exists():
            continue
        bs, bc = [], []
        with open(blp) as f:
            for row in csv.DictReader(f):
                bs.append(int(row["set_size"])); bc.append(int(row["covered"]))
        if bs:
            bl_label = "ReAct/Reflexion T1" if t == 1 else f"Reflexion T{t}"
            runs.append(make_run(np.array(bs, dtype=float), np.array(bc),
                                 bl_label, BL_COLORS[bi % len(BL_COLORS)],
                                 BL_MARKERS[bi % len(BL_MARKERS)]))

    # ── console tables ────────────────────────────────────────────────────────
    print(f"\n{'='*72}\n{label}   (ScienceWorld, offline held-out gold states)\n{'='*72}")
    print(f"=== SET-SIZE DISTRIBUTION ({'fraction' if not args.counts else 'count'} per bin) ===")
    print("  " + f"{'series':<12}{'n':>6}{'mean':>7}" + "".join(f"{l:>10}" for l in bin_labels))
    for r in runs:
        lbl = r['label'].replace('$', '').replace('\\alpha', 'alpha')
        vals = "".join((f"{d:>10.3f}" if not args.counts else f"{int(d):>10d}") for d in r["dist"])
        print("  " + f"{lbl:<12}{r['n']:>6}{r['mean_size']:>7.1f}" + vals)

    print(f"\n=== PER-BIN COVERAGE (coverage% ; n states in bin) ===")
    print("  " + f"{'series':<12}" + "".join(f"{l:>15}" for l in bin_labels) + f"{'overall':>10}")
    for r in runs:
        lbl = r['label'].replace('$', '').replace('\\alpha', 'alpha')
        cells = ""
        for b in range(n_bins):
            cnt = int(r["total"][b]); cov = r["cov_line"][b]
            cells += (f"{cov:>8.0f}% ({cnt:>3d})" if cnt > 0 else f"{'-':>8} ({0:>3d})")
        print("  " + f"{lbl:<12}" + cells + f"{r['overall_cov']:>9.1f}%")

    # ── plot ──────────────────────────────────────────────────────────────────
    n_a = len(runs); bar_w = 0.8 / n_a
    fig_w = AAAI_FULL_WIDTH if args.wide else AAAI_COL_WIDTH
    fig, ax = plt.subplots(figsize=(fig_w, args.height or (2.6 if args.wide else 2.1)))
    ax2 = ax.twinx() if args.coverage else None

    for j, r in enumerate(runs):
        xpos = xs + (j - (n_a - 1) / 2) * bar_w
        ax.bar(xpos, r["dist"], bar_w * 0.9, color=r["color"],
               edgecolor="white", linewidth=0.4, label=r["label"])
        if args.coverage:
            mask = ~np.isnan(r["cov_line"])
            ax2.plot(xpos[mask], r["cov_line"][mask], marker=r["marker"], markersize=2.5,
                     linewidth=1.0, color=r["color"], markeredgewidth=0.4,
                     markeredgecolor="white", label=f"{r['label']} cov")

    ax.set_xticks(xs)
    ax.set_xticklabels(bin_labels, rotation=0 if args.wide else 20,
                       ha="center" if args.wide else "right")
    xlab = "Prediction set size (% of admissible actions)" if args.x == "frac" \
           else "Prediction set size (# actions kept)"
    ax.set_xlabel(xlab, labelpad=2)
    ax.set_ylabel("Number of states" if args.counts else "Fraction of states", labelpad=2)
    ax.set_ylim(0, (max(r["dist"].max() for r in runs) * 1.22) if args.counts else 1.0)
    if args.coverage:
        ax2.set_ylabel("Coverage %", labelpad=3); ax2.set_ylim(0, 105)
        ax2.spines["top"].set_visible(False); ax2.tick_params(pad=1.5)
    ax.set_title(label, pad=3)
    ax.grid(axis="y", alpha=0.5); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.tick_params(pad=1.5)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = (ax2.get_legend_handles_labels() if args.coverage else ([], []))
    ax.legend(h1 + h2, l1 + l2, loc=args.legend_loc, ncol=1, borderaxespad=0.3,
              fontsize=6, handlelength=1.1, handletextpad=0.3, labelspacing=0.2)

    out = args.out_prefix + Path(csv_path).stem.replace("setsize_cov_", "") + f"_{args.x}"
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  saved -> {out}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", default=[
        "setsize_cov_qwen25.csv", "setsize_cov_qwen3.csv", "setsize_cov_gemma.csv"])
    ap.add_argument("--labels", nargs="+", default=["Qwen2.5-3B", "Qwen3-8B", "Gemma-4-12B"])
    ap.add_argument("--x", choices=["frac", "abs"], default="abs")
    ap.add_argument("--bin_edges", nargs="+", type=float, default=None)
    ap.add_argument("--counts", action="store_true")
    ap.add_argument("--coverage", action="store_true", help="overlay coverage line (opt-in)")
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--height", type=float, default=None)
    ap.add_argument("--legend_loc", default="upper right")
    ap.add_argument("--out_prefix", default="setsize_distribution_coverage_sciworld_")
    args = ap.parse_args()

    for label, csvp in zip(args.labels, args.csvs):
        if not Path(csvp).exists():
            print(f"  [skip] {label}: {csvp} not found"); continue
        plot_model(label, csvp, args)


if __name__ == "__main__":
    main()
