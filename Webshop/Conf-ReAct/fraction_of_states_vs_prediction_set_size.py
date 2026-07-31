"""
Prediction-set-size DISTRIBUTION (ABSOLUTE size) with coverage as an overlaid line.
AAAI-formatted output.

Companion to plot_setsize_distribution_coverage_webshop.py, but:
  * x-axis = ABSOLUTE prediction set size = number of actions kept
             (len(pred_set) from the log's "Pred set (k/N)"), NOT % buckets.
  * bars   = distribution: fraction of that alpha's states whose set has that size
             (grouped by alpha).  --counts for raw counts.
  * a COVERAGE LINE per alpha (0.1/0.2/0.3) is drawn on a secondary y-axis:
             at each size s, coverage = covered / total among size-s states.

This file also carries the shared BFS-log / training-CSV parsing helpers
(CSV_PATH, load_csv_lookup, parse_bfs_log, in_action_set, strip_price_clause,
etc.) used by the other Conf-ReAct evaluation scripts
(table_coverage_vs_setsize_webshop.py, webshop_reflexion_selfset.py,
webshop_reflexion_selfset_3trials.py) -- formerly a separate module.

Usage:
    python fraction_of_states_vs_prediction_set_size.py
    python fraction_of_states_vs_prediction_set_size.py --counts --max_size 8
    python fraction_of_states_vs_prediction_set_size.py --wide
"""
import re
import csv
import ast
import argparse
import os
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOGS = os.path.join(_HERE, "logs")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = os.path.join(_HERE, "WebShop_ScoreFunc_Train_Data.csv")

# ── shared BFS-log / training-CSV parsing helpers ─────────────────────────────
# (used by this script and by table_coverage_vs_setsize_webshop.py,
#  webshop_reflexion_selfset.py, webshop_reflexion_selfset_3trials.py)
EPISODE_RE    = re.compile(r"^\s*EPISODE\s+(\d+)\s*/\s*\d+\b")
ENV_INSTR_RE  = re.compile(r"^\s*Env instruction\s*:\s*(.+?)\s*$")
CSV_INSTR_RE  = re.compile(r"^\s*CSV instruction\s*:\s*(.+?)\s*$")
INSTR_RE      = re.compile(r"^\s*Instruction\s*:\s*(.+?)\s*$")
NODE_HDR_RE   = re.compile(r"^\[BFS Node \d+\]")
TRAJ_RE       = re.compile(r"^\s*Traj\s*:\s*(.+)$")
REAL_RE       = re.compile(r"^\s*Real\s*\(\s*(\d+)\s*\)\s*:\s*(.+)$")
SCORE_LINE_RE = re.compile(r"^\s*([0-9.]+)\s+(.+?)\s*$")
SCORES_HDR    = "Scores [lower = more likely optimal]"
IN_SET_TAG    = "[IN SET]"

_PRICE_CLAUSE_RE = re.compile(
    r",?\s*(?:and\s+)?price\s+(?:lower|less)\s+than\s+\$?\d+(?:\.\d+)?\s*dollars?\.?\s*$",
    re.IGNORECASE,
)


def strip_price_clause(s: str) -> str:
    """Drop trailing 'price lower than X dollars' so price-only variants compare equal."""
    s = _PRICE_CLAUSE_RE.sub("", s).strip().rstrip(",").strip()
    return s.lower()


# ── Action equality (fuzzy for search[...]) ───────────────────────────────────
_ACTION_RE   = re.compile(r"^(search|click|think)\[(.*)\]\s*$", re.IGNORECASE | re.DOTALL)
_TOKEN_RE    = re.compile(r"[a-z0-9]+")
_PRICE_NOISE = {
    "price", "prices", "lower", "less", "than", "under", "below", "around",
    "dollars", "dollar",
}
_STOPWORDS   = {
    "a", "an", "the", "and", "or", "in", "with", "of", "for", "to", "on", "at",
    "is", "are", "be", "by", "from", "i", "me", "my", "we", "our",
    "want", "wants", "need", "needs", "looking", "would", "like", "give", "find", "get",
    "some", "any",
}


def _split_action(x: str):
    m = _ACTION_RE.match(x.strip())
    if m:
        return m.group(1).lower(), m.group(2)
    return None, x.strip()


def _content_tokens(text: str) -> set:
    toks = set(_TOKEN_RE.findall(text.lower()))
    toks -= _PRICE_NOISE
    toks -= _STOPWORDS
    return toks


def action_match(a: str, b: str, jaccard_thresh: float = 0.8) -> bool:
    """
    True iff two actions are 'the same'.
    - search[q1] == search[q2]  if Jaccard(content_tokens) >= jaccard_thresh
    - click[X] / think[X]: case-insensitive exact match
    - everything else: case-insensitive exact match
    """
    if a.strip().lower() == b.strip().lower():
        return True
    pa, ba = _split_action(a)
    pb, bb = _split_action(b)
    if pa is None or pa != pb:
        return False
    if pa != "search":
        return ba.strip().lower() == bb.strip().lower()
    ta = _content_tokens(ba)
    tb = _content_tokens(bb)
    if not ta or not tb:
        return False
    return (len(ta & tb) / len(ta | tb)) >= jaccard_thresh


def in_action_set(target: str, actions, jaccard_thresh: float = 0.8) -> bool:
    return any(action_match(target, a, jaccard_thresh) for a in actions)


def load_csv_lookup(path: str):
    """
    Build dict[(norm_instruction, traj_tuple)] = (optimal_action, admissible_size, instruction).

    norm_instruction = original instruction with the trailing
    "price lower than X dollars" clause stripped and lowercased, so two
    instructions that differ only in price collide on the same key.

    State key = prev_actions with 'reset' stripped from front and the
    trailing duplicate optimal_action stripped from the back (CSV stores
    the optimal as the last entry of prev_actions).
    """
    lookup    = {}
    inst_keys = set()
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                prev  = ast.literal_eval(row['Sequence_of_previous_action'])
                admis = ast.literal_eval(row['Admissiable_action'])
            except (SyntaxError, ValueError):
                continue
            optimal     = row['optimal_action'].strip()
            instruction = row['Original_instruction'].strip()
            if not optimal or not instruction or not isinstance(prev, list) or not isinstance(admis, list):
                continue
            real = [a for a in prev if a != 'reset']
            if real and real[-1] == optimal:
                real = real[:-1]
            inst_key = strip_price_clause(instruction)
            key = (inst_key, tuple(real))
            lookup[key] = (optimal, len(admis), instruction)
            inst_keys.add(inst_key)
    print(f"CSV lookup: {len(lookup)} (norm_instruction, traj) keys  |  "
          f"{len(inst_keys)} unique price-stripped instructions")
    return lookup, inst_keys


def _safe_literal_list(raw: str):
    raw = raw.strip()
    if raw == '(start)':
        return []
    try:
        v = ast.literal_eval(raw)
        return list(v) if isinstance(v, (list, tuple)) else None
    except (SyntaxError, ValueError):
        return None


def parse_bfs_log(path: Path):
    """
    Walk the BFS log. Yield (norm_instruction, traj_tuple, real_count, pred_set_actions,
    raw_instruction) for every parseable BFS Node block.

    For each episode we capture an instruction string from one of (in priority order):
      "Env instruction:"   (preferred; matches what the WebShop env returned)
      "Instruction:"       (legacy single-instruction header)
      "CSV instruction:"   (fallback if env line is missing)
    The norm form has the trailing price clause stripped and is lowercased.
    """
    samples       = []
    cur_inst_raw  = None
    cur_inst_norm = None
    inst_from_log = {}    # episode_idx (0-based) -> raw instruction (best available)
    cur_episode   = None
    with open(path) as f:
        lines = f.readlines()

    i, n = 0, len(lines)
    while i < n:
        m = EPISODE_RE.match(lines[i])
        if m:
            cur_episode   = int(m.group(1))
            cur_inst_raw  = None
            cur_inst_norm = None
            i += 1
            continue

        em = ENV_INSTR_RE.match(lines[i])
        cm = CSV_INSTR_RE.match(lines[i])
        im = INSTR_RE.match(lines[i])
        if em or cm or im:
            inst = (em or cm or im).group(1).strip()
            # Env > legacy Instruction > CSV.  Env always wins; CSV only fills in if nothing else.
            if em is not None:
                cur_inst_raw  = inst
                cur_inst_norm = strip_price_clause(inst)
            elif im is not None and cur_inst_raw is None:
                cur_inst_raw  = inst
                cur_inst_norm = strip_price_clause(inst)
            elif cm is not None and cur_inst_raw is None:
                cur_inst_raw  = inst
                cur_inst_norm = strip_price_clause(inst)
            if cur_episode is not None and cur_inst_raw is not None:
                inst_from_log[cur_episode - 1] = cur_inst_raw
            i += 1
            continue

        if NODE_HDR_RE.match(lines[i]):
            traj       = None
            real_count = None
            candidates = []
            in_set     = []
            j = i + 1
            while j < n and not NODE_HDR_RE.match(lines[j]) and not EPISODE_RE.match(lines[j]):
                tm = TRAJ_RE.match(lines[j])
                if tm:
                    parsed = _safe_literal_list(tm.group(1))
                    if parsed is not None:
                        traj = tuple(parsed)
                rm = REAL_RE.match(lines[j])
                if rm:
                    real_count = int(rm.group(1))

                if SCORES_HDR in lines[j]:
                    k = j + 1
                    while k < n and not NODE_HDR_RE.match(lines[k]) and not EPISODE_RE.match(lines[k]):
                        s = lines[k].rstrip()
                        if not s.strip():
                            k += 1
                            continue
                        ls = s.lstrip()
                        if ls.startswith("→") or ls.startswith("'→") or "INVALID" in ls:
                            break
                        is_in_set = IN_SET_TAG in s
                        body = s.rstrip()[: -len(IN_SET_TAG)].rstrip() if is_in_set else s
                        sm = SCORE_LINE_RE.match(body)
                        if not sm:
                            break
                        action = sm.group(2).strip()
                        candidates.append(action)
                        if is_in_set:
                            in_set.append(action)
                        k += 1
                    j = k
                    continue
                j += 1
            i = j
            if cur_inst_norm is not None and traj is not None:
                samples.append((cur_inst_norm, traj, real_count, in_set, candidates, cur_inst_raw))
        else:
            i += 1
    return samples, inst_from_log        # returns


# ── AAAI camera-ready style ───────────────────────────────────────────────────
# Sizes are the ON-PAGE sizes: figures are saved at final width and included
# with \includegraphics[width=\columnwidth] (NO scaling), so nothing shrinks.
AAAI_COL_WIDTH  = 3.3   # inches, single column
AAAI_FULL_WIDTH = 7.0   # inches, spanning both columns

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          7,
    "axes.labelsize":     8,
    "axes.titlesize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    6,
    "legend.title_fontsize": 6,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "lines.linewidth":    1.0,
    "lines.markersize":   2.5,
    "grid.linewidth":     0.4,
    "grid.linestyle":     ":",
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.7",
    "legend.borderpad":   0.3,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.4,
    "legend.labelspacing": 0.25,
    "legend.columnspacing": 0.8,
    "pdf.fonttype":       42,   # TrueType, required for camera-ready
    "ps.fonttype":        42,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.01,
})

ALPHA_COLORS = {0.1: "#fdae6b", 0.2: "#f16913", 0.3: "#d62728"}  # light→deep orange→red
FALLBACK_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
# distinct markers so the three coverage lines separate in grayscale print
MARKERS = ["o", "s", "^", "D", "v"]


def extract_matched(log_path, csv_lookup, jaccard, restrict):
    """Return list of (abs_set_size, covered 0/1) per CSV-matched BFS node."""
    bfs_nodes, _ = parse_bfs_log(Path(log_path))
    out = []
    for norm_instr, traj, real_count, pred_set, candidates, _raw in bfs_nodes:
        key = (norm_instr, traj)
        if key not in csv_lookup:
            continue
        optimal = csv_lookup[key][0]
        if restrict and not in_action_set(optimal, candidates, jaccard):
            continue
        size = len(pred_set)
        if size <= 0:
            continue
        covered = 1 if in_action_set(optimal, pred_set, jaccard) else 0
        out.append((size, covered))
    print(f"  {log_path}: matched {len(out)} states")
    return out


def load_baseline_csv(path):
    """Return (sizes[np], covs[np]) from a CSV with set_size,covered columns."""
    import csv as _csv
    bs, bc = [], []
    with open(path) as f:
        for row in _csv.DictReader(f):
            bs.append(int(row["set_size"])); bc.append(int(row["covered"]))
    return np.array(bs), np.array(bc)


def main():
    ap = argparse.ArgumentParser()
    default_logs = [
        os.path.join(_LOGS, "bfs_webshop_qwen_3_8b_0.1.txt"),
        os.path.join(_LOGS, "bfs_webshop_qwen_3_8b_0.2.txt"),
        os.path.join(_LOGS, "bfs_webshop_qwen_3_8b_0.3.txt"),
    ]
    ap.add_argument("--logs", nargs="+", default=default_logs)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--jaccard", type=float, default=0.8)
    ap.add_argument("--max_size", type=int, default=0,
                    help="cap x-axis at this absolute size (0 = use observed max).")
    ap.add_argument("--bin_width", type=int, default=2,
                    help="uniform absolute-size bin width (e.g. 2 → 0-2, 2-4, …).")
    ap.add_argument("--bin_edges", nargs="+", type=float, default=[0, 2, 4, 16],
                    help="explicit (variable-width) bin edges (overrides --bin_width); "
                         "default 0 2 4 16 → bins 0-2, 2-4, 4-16.")
    ap.add_argument("--counts", action="store_true",
                    help="bars = raw state counts (default: fraction of that α's states).")
    ap.add_argument("--no_restrict", action="store_true",
                    help="disable the 'optimal must be in candidates' filter.")
    default_baseline_csv = [
        os.path.join(_LOGS, "webshop_reflexion_k3_trial1_qwen3_8b.csv"),
        os.path.join(_LOGS, "webshop_reflexion_k3_trial2_qwen3_8b.csv"),
        os.path.join(_LOGS, "webshop_reflexion_k3_trial3_qwen3_8b.csv"),
    ]
    ap.add_argument("--baseline_csv", nargs="+", default=default_baseline_csv,
                    help="one or more CSVs (…,set_size,covered) added as baseline series "
                         "(default: reflexion T1/T2/T3, Qwen3-8B).")
    ap.add_argument("--baseline_label", nargs="+",
                    default=["ReAct/Reflexion T1", "reflexion T2", "reflexion T3"],
                    help="legend label(s) for the baseline series (same count as --baseline_csv).")
    ap.add_argument("--coverage", action="store_true",
                    help="show coverage lines / secondary axis (default: bars-only; tables always printed).")
    ap.add_argument("--wide", action="store_true",
                    help="Size for a two-column spanning figure (7.0in) instead of 3.3in.")
    ap.add_argument("--height", type=float, default=None,
                    help="Figure height in inches (default 2.1 single-col, 2.6 wide).")
    ap.add_argument("--legend_loc", default="upper right")
    ap.add_argument("--legend_fontsize", type=float, default=5.0)
    ap.add_argument("--legend_bbox", nargs=2, type=float, default=None, metavar=("X", "Y"))
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--out", default="setsize_abs_distribution_coverage_webshop.png")
    args = ap.parse_args()
    normalize = not args.counts
    restrict = not args.no_restrict
    show_cov = args.coverage

    if len(args.logs) != len(args.alphas):
        raise SystemExit("--logs and --alphas must have equal length.")
    csv_lookup, _ = load_csv_lookup(args.csv)

    runs = []
    observed_max = 1
    for idx, (log, alpha) in enumerate(zip(args.logs, args.alphas)):
        print(f"\n=== α = {alpha} ===")
        matched = extract_matched(log, csv_lookup, args.jaccard, restrict)
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

    # optional baseline series (reflexion trials, sample-10, …) from CSVs
    if args.baseline_csv:
        BL_COLORS  = ["#a1d99b", "#41ab5d", "#006d2c", "#17becf", "#e377c2"]  # light→deep→dark green
        BL_MARKERS = ["D", "P", "X", "*", "v"]
        labels = args.baseline_label or [f"baseline{i+1}" for i in range(len(args.baseline_csv))]
        for bi, (csvp, lab) in enumerate(zip(args.baseline_csv, labels)):
            bsizes, bcovs = load_baseline_csv(csvp)
            if len(bsizes) == 0:
                continue
            observed_max = max(observed_max, int(bsizes.max()))
            runs.append({"alpha": lab, "color": BL_COLORS[bi % len(BL_COLORS)],
                         "marker": BL_MARKERS[bi % len(BL_MARKERS)], "label": lab,
                         "sizes": bsizes, "covs": bcovs, "n": len(bsizes),
                         "overall_cov": 100.0 * bcovs.mean()})
            print(f"\n=== baseline: {lab}  n={len(bsizes)}  "
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
        bins = np.clip(np.searchsorted(edges, r["sizes"], side="left") - 1, 0, n_bins - 1)
        total = np.array([(bins == b).sum() for b in range(n_bins)])
        cov_c = np.array([r["covs"][bins == b].sum() for b in range(n_bins)])
        r["total"] = total
        r["dist"]  = total / r["n"] if normalize else total.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            r["cov_line"] = np.where(total > 0, 100.0 * cov_c / total, np.nan)

    # ── tables ─────────────────────────────────────────────────────────────────
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

    # ── plot: bars (+ optional coverage lines) ─────────────────────────────────
    n_alpha = len(runs)
    bar_w = 0.8 / n_alpha
    fig_w = AAAI_FULL_WIDTH if args.wide else AAAI_COL_WIDTH
    fig_h = args.height if args.height else (2.6 if args.wide else 2.1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax2 = ax.twinx() if show_cov else None

    for j, r in enumerate(runs):
        offset = (j - (n_alpha - 1) / 2) * bar_w
        xpos = xs + offset
        # SOLID deep color — no alpha, thin white edge to separate touching bars
        ax.bar(xpos, r["dist"], bar_w * 0.9, color=r["color"],
               edgecolor="white", linewidth=0.4, label=r["label"])
        if show_cov:
            mask = ~np.isnan(r["cov_line"])
            ax2.plot(xpos[mask], r["cov_line"][mask], marker=r["marker"],
                     markersize=2.5, linewidth=1.0, color=r["color"],
                     markeredgewidth=0.4, markeredgecolor="white",
                     label=f"{r['label']} cov ({r['overall_cov']:.0f}%)")

    ax.set_xticks(xs)
    ax.set_xticklabels(bin_labels, rotation=0 if args.wide else 20,
                       ha="center" if args.wide else "right")
    ax.set_xlabel("Prediction set size (# actions kept)", labelpad=2)
    ax.set_ylabel("Fraction of states" if normalize else "Number of states", labelpad=2)
    if args.model:
        ax.set_title(args.model, pad=4)
    if show_cov:
        ax2.set_ylabel("Coverage %", labelpad=3)
        ax2.set_ylim(0, 105)
        ax2.spines["top"].set_visible(False)
        ax2.tick_params(pad=1.5)
    ax.set_ylim(0, 1.0 if normalize else max(r["dist"].max() for r in runs) * 1.22)
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.tick_params(pad=1.5)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = (ax2.get_legend_handles_labels() if show_cov else ([], []))
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