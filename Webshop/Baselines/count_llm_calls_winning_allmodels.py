"""WebShop LLM calls on WINNING (score=1.0) episodes — for ALL THREE models
(Qwen2.5-3B, Qwen3-8B, Gemma-4-12B), same format as count_llm_calls_winning.py,
now WITH a Rollback row.

Reuses the react/reflact/reflexion/bfs parsers from count_llm_calls_winning.py.
Adds parse_rollback_webshop:
  - episode boundary = 'Action 0: reset'; win = 'Your score ...: 1.0'.
  - one LLM call per '****Analysis****' banner (error detection) + one per
    newly-generated action ('Action N:', N>=1). Replayed history is dropped:
    action lines inside an analysis block ('# Next Task / ### Trajectory') are
    skipped, and a line identical to the previously counted action (the
    '---rollback happened---' re-print) is de-duplicated.

Qwen3-8B has no WebShop rollback log -> Rollback row shows '(log missing)'.

Usage:  python count_llm_calls_winning_allmodels.py
"""
import re
from pathlib import Path

from count_llm_calls_winning import (
    parse_react_style, parse_reflexion, parse_bfs_conformal,
    report, report_bfs_distribution,
)

BASE = "/home/saptarshi/transfer/alfworld/ReAct/4th_SEM_MTP/S_T_Webshop"
GPT = f"{BASE}/baselines_gpt_4.1"

ANALYSIS_RE = re.compile(r'\*+Analysis\*+')          # opening banner (has the word)
STARS_RE    = re.compile(r'^\s*\*{4,}\s*$')          # closing banner (stars only)
ACTION_RE   = re.compile(r'^\s*Action\s+(\d+)\s*:\s*(.*)$')
SCORE_RE    = re.compile(r'Your score \(min 0\.0, max 1\.0\): ([\d.]+)')


def parse_rollback_webshop(path):
    """Returns [(llm_calls, won)] per episode, dropping replayed history."""
    if not Path(path).exists():
        return None
    episodes, calls, won = [], 0, False
    in_analysis, last_act = False, None
    for line in Path(path).read_text(errors='ignore').splitlines():
        if ANALYSIS_RE.search(line):
            calls += 1; in_analysis = True; continue
        if in_analysis and STARS_RE.match(line):
            in_analysis = False; continue
        m = ACTION_RE.match(line)
        if m:
            n, text = int(m.group(1)), m.group(2).strip()
            if n == 0 and text == 'reset':                 # episode boundary
                episodes.append((calls, won))
                calls, won, in_analysis, last_act = 0, False, False, None
            elif not in_analysis and text != last_act:      # new (non-replayed) action
                calls += 1; last_act = text
            continue
        s = SCORE_RE.search(line)
        if s and float(s.group(1)) == 1.0:
            won = True
    episodes.append((calls, won))
    return [e for e in episodes if e != (0, False)] if episodes else episodes


# per-model file map: (react, reflact, reflexion, rollback, [bfs 0.1/0.2/0.3])
MODELS = {
    "Qwen2.5-3B": dict(
        react=f"{GPT}/react_webshop_qwen_25.txt",
        reflact=f"{GPT}/reflact_webshop_qwen_25.txt",
        reflexion=f"{GPT}/reflexion_webshop_qwen_25.txt",
        rollback=f"{GPT}/web_rollback_qwen_25.txt",
        bfs=[f"{BASE}/bfs_webshop_conformal_0.{a}.txt" for a in "123"],
    ),
    "Qwen3-8B": dict(
        react=f"{GPT}/react_webshop_qwen_3.txt",
        reflact=f"{BASE}/reflact_webshop.txt",
        reflexion=f"{BASE}/webshop_reflexion.txt",
        rollback=None,                                   # no qwen3 webshop rollback log
        bfs=[f"{BASE}/bfs_webshop_qwen_3_8b_0.{a}.txt" for a in "123"],
    ),
    "Gemma-4-12B": dict(
        react=f"{GPT}/react_webshop_gemma4_12b.txt",
        reflact=f"{GPT}/reflact_webshop_gemma4_12b.txt",
        reflexion=f"{GPT}/reflexion_webshop_gemma4_12b.txt",
        rollback=f"{GPT}/web_rollback_gemma4_12b.txt",
        bfs=[f"{BASE}/bfs_webshop_gemma4_conformal_0.{a}.txt" for a in "123"],
    ),
}


if __name__ == '__main__':
    for model, fm in MODELS.items():
        print("\n" + "=" * 74)
        print(f"{model}   (WebShop)")
        print("=" * 74)
        print("LLM calls spent on WINNING (score=1.0) episodes  "
              "[Reflexion: cumulative to win]")

        react_eps = parse_react_style(fm["react"])
        reflact_eps = parse_react_style(fm["reflact"])
        reflexion_eps = [(v['calls'], v['won']) for v in parse_reflexion(fm["reflexion"]).values()]
        report("ReAct", react_eps)
        report("ReflAct", reflact_eps)
        report("Reflexion", reflexion_eps)
        if fm["rollback"] is None:
            print(f"{'Rollback':10s}  (log missing — no WebShop rollback run for this model)")
        else:
            report("Rollback", parse_rollback_webshop(fm["rollback"]))

        for alpha, log_path in zip(("0.1", "0.2", "0.3"), fm["bfs"]):
            bfs_eps, n_samples = parse_bfs_conformal(log_path)
            print(f"\nBFS conformal (α={alpha}): one request per BFS node, "
                  f"n_samples={n_samples} generations each")
            report("BFS req", bfs_eps)
            report("BFS gen", [(c * n_samples, w) for c, w in bfs_eps])
            report_bfs_distribution(bfs_eps)
