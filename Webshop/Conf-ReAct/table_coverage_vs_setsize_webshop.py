"""Empirical coverage vs absolute set-size bins (0-2, 2-4, 4-16) — WebShop,
as a TABLE, for Conf-ReAct (alpha=0.1/0.2/0.3) AND the baselines.

Rows per model:
  Conf-ReAct a=0.1/0.2/0.3 : per-state (pred-set size, covered) from the BFS
      conformal logs (same extraction/restriction as the distribution plot:
      CSV-matched states whose candidate pool contains the optimal).
  ReAct / ReflAct          : single-action agents -> set size 1 (bin 0-2).
      Their trajectory states are matched against the SAME CSV optimal lookup
      (key = price-stripped instruction + action prefix); covered = the printed
      action fuzzy-matches the CSV optimal (jaccard 0.8). States that never
      align with a CSV-known prefix are unmeasurable and skipped.
  Rollback                 : same, but 'Action k:' numbering encodes rollback
      depth (traj = traj[:k-1] + action); replayed actions inside the
      '****Analysis****' blocks are ignored; each (instr, traj) state is scored
      once. (No Qwen3-8B rollback log exists.)
  ReAct/Reflexion T1, Reflexion T2/T3 : the self-sized-set baseline CSVs.

Cells: coverage% (n states). Writes coverage_vs_setsize_webshop.txt/.csv.

Usage:  python table_coverage_vs_setsize_webshop.py
"""
import csv
import re
import sys
import argparse
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
BASE = _HERE / "logs"
GPT = BASE
sys.path.insert(0, str(_HERE))

from fraction_of_states_vs_prediction_set_size import (  # noqa: E402
    CSV_PATH, load_csv_lookup, strip_price_clause, in_action_set, extract_matched,
)

INSTR_HDR_RE = re.compile(r"^Instruction:\s*$")
EP_SEP = "-----------------"
ACTION_RE = re.compile(r"^Action: (.+)$")                 # react/reflact style
RB_ACTION_RE = re.compile(r"^Action (\d+): (.+)$")        # rollback style
ANALYSIS_RE = re.compile(r"\*+Analysis\*+")
STARS_RE = re.compile(r"^\s*\*{4,}\s*$")


def extract_react_style(log_path, csv_lookup, jaccard):
    """ReAct/ReflAct: 1 action/state; match (instr_norm, traj-prefix) to CSV."""
    out = []
    instr_norm, traj = None, []
    lines = Path(log_path).read_text(errors="ignore").splitlines()
    for j, line in enumerate(lines):
        if line == EP_SEP and j + 1 < len(lines) and lines[j + 1].strip().isdigit():
            instr_norm, traj = None, []
            continue
        if INSTR_HDR_RE.match(line) and j + 1 < len(lines):
            instr_norm = strip_price_clause(lines[j + 1].strip())
            traj = []
            continue
        m = ACTION_RE.match(line)
        if m and instr_norm is not None:
            act = m.group(1).strip()
            if act == "reset" or act.startswith("think["):
                continue
            key = (instr_norm, tuple(traj))
            if key in csv_lookup:
                covered = 1 if in_action_set(csv_lookup[key][0], [act], jaccard) else 0
                out.append((1, covered))
            traj.append(act)
    return out


def extract_rollback(log_path, csv_lookup, jaccard):
    """Rollback: 'Action k:' numbering gives depth after rollback; skip replayed
    actions inside Analysis blocks; score each (instr, traj) state once."""
    out, seen = [], set()
    instr_norm, traj = None, []
    in_analysis = False
    lines = Path(log_path).read_text(errors="ignore").splitlines()
    for j, line in enumerate(lines):
        if ANALYSIS_RE.search(line):
            in_analysis = True
            continue
        if in_analysis and STARS_RE.match(line):
            in_analysis = False
            continue
        if INSTR_HDR_RE.match(line) or line.strip() == "Instruction:":
            if j + 1 < len(lines) and not in_analysis:
                instr_norm = strip_price_clause(lines[j + 1].strip())
            continue
        m = RB_ACTION_RE.match(line)
        if not m or in_analysis or instr_norm is None:
            continue
        k, act = int(m.group(1)), m.group(2).strip()
        if act == "reset":
            traj = []           # new episode (Action 0: reset)
            continue
        if act.startswith("think["):
            continue
        traj = traj[:k - 1]
        key = (instr_norm, tuple(traj))
        if key in csv_lookup and (key, act) not in seen:
            seen.add((key, act))
            covered = 1 if in_action_set(csv_lookup[key][0], [act], jaccard) else 0
            out.append((1, covered))
        traj.append(act)
    return out


def load_trial_csv(path):
    if not Path(path).exists():
        return None
    return [(int(r["set_size"]), int(r["covered"]))
            for r in csv.DictReader(open(path)) if r.get("set_size") is not None]


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


MODELS = {
    "Qwen2.5-3B": dict(
        conformal=[BASE / f"bfs_webshop_conformal_0.{a}.txt" for a in "123"],
        react=GPT / "react_webshop_qwen_25.txt",
        reflact=GPT / "reflact_webshop_qwen_25.txt",
        rollback=GPT / "web_rollback_qwen_25.txt",
        trials=[BASE / f"webshop_reflexion_k3_trial{t}_qwen25_3b.csv" for t in (1, 2, 3)],
    ),
    "Qwen3-8B": dict(
        conformal=[BASE / f"bfs_webshop_qwen_3_8b_0.{a}.txt" for a in "123"],
        react=GPT / "react_webshop_qwen_3.txt",
        reflact=BASE / "reflact_webshop.txt",
        rollback=None,
        trials=[BASE / f"webshop_reflexion_k3_trial{t}_qwen3_8b.csv" for t in (1, 2, 3)],
    ),
    "Gemma-4-12B": dict(
        conformal=[BASE / f"bfs_webshop_gemma4_conformal_0.{a}.txt" for a in "123"],
        react=GPT / "react_webshop_gemma4_12b.txt",
        reflact=GPT / "reflact_webshop_gemma4_12b.txt",
        rollback=GPT / "web_rollback_gemma4_12b.txt",
        trials=[BASE / f"webshop_refl_trial{t}_gemma4_12b.csv" for t in (1, 2, 3)],
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin_edges", nargs="+", type=float, default=[0, 2, 4, 16])
    ap.add_argument("--jaccard", type=float, default=0.8)
    ap.add_argument("--out_txt", default="coverage_vs_setsize_webshop.txt")
    ap.add_argument("--out_csv", default="coverage_vs_setsize_webshop.csv")
    args = ap.parse_args()

    csv_lookup, _ = load_csv_lookup(CSV_PATH)
    edges = np.array(sorted(args.bin_edges), dtype=float)
    n_bins = len(edges) - 1
    bin_labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(n_bins)]

    lines = []
    csv_rows = [["model", "method", "overall_coverage%", "avg_set_size", "n"] +
                [f"cov_{bl}" for bl in bin_labels] + [f"n_{bl}" for bl in bin_labels]]

    def out(s=""):
        print(s); lines.append(s)

    for model, fm in MODELS.items():
        series = []
        for a, lg in zip((0.1, 0.2, 0.3), fm["conformal"]):
            if Path(lg).exists():
                series.append((f"Conf-ReAct a={a}",
                               extract_matched(lg, csv_lookup, args.jaccard, True)))
        if Path(fm["react"]).exists():
            series.append(("ReAct", extract_react_style(fm["react"], csv_lookup, args.jaccard)))
        if Path(fm["reflact"]).exists():
            series.append(("ReflAct", extract_react_style(fm["reflact"], csv_lookup, args.jaccard)))
        if fm["rollback"] is not None and Path(fm["rollback"]).exists():
            series.append(("Rollback", extract_rollback(fm["rollback"], csv_lookup, args.jaccard)))
        else:
            series.append(("Rollback", None))
        for t, trial in zip((1, 2, 3), fm["trials"]):
            samp = load_trial_csv(trial)
            if samp is not None:
                nm = "ReAct/Reflexion T1" if t == 1 else f"Reflexion T{t}"
                series.append((nm, samp))

        out("=" * 96)
        out(f"{model}   (WebShop — per-bin empirical coverage over absolute set size "
            f"[# actions kept])")
        out("=" * 96)
        out(f"  {'method':<20}" + "".join(f"{bl:>14}" for bl in bin_labels)
            + f"{'OVERALL':>12}{'avg size':>10}    (cell = coverage% (n))")
        for name, samples in series:
            if samples is None:
                out(f"  {name:<20}  (no log for this model)"); continue
            if not samples:
                out(f"  {name:<20}  (no CSV-matchable states)"); continue
            avg, cnt = bucket(samples, edges)
            overall = 100.0 * np.mean([c for _, c in samples])
            mean_sz = np.mean([s for s, _ in samples])
            cells = ""
            for b in range(n_bins):
                n = int(cnt[b]); v = avg[b]
                cells += (f"{v:>7.0f}% ({n:>4d})" if n > 0 else f"{'-':>7} ({0:>4d})")
            out(f"  {name:<20}" + cells + f"{overall:>10.1f}% {mean_sz:>9.2f}")
            csv_rows.append([model, name, f"{overall:.2f}", f"{mean_sz:.3f}", len(samples)]
                            + [f"{avg[i]:.1f}" if not np.isnan(avg[i]) else ""
                               for i in range(n_bins)]
                            + [int(cnt[i]) for i in range(n_bins)])
        out()

    with open(_HERE / args.out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(_HERE / args.out_csv, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"Saved -> {_HERE / args.out_txt}\nSaved -> {_HERE / args.out_csv}")


if __name__ == "__main__":
    main()
