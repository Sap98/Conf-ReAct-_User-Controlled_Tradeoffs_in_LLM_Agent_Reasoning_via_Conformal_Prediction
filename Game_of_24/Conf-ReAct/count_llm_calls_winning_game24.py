"""Count LLM calls spent on WINNING (solved) puzzles — Game-of-24 output logs.
Same methodology / output format as WebShop's count_llm_calls_winning.py:
report LLM calls on the solved episodes and the average per solved episode,
plus BFS conformal split into "req" (1 per node) and "gen" (nodes*n_samples).

Counting rules (per game-24 runner):
  - react / reflact : one LLM call per 'Act N>' action step (think + move).
      episode boundary = 'id:N, success rate:'; win = '[oracle] solved=True'.
  - reflexion (cumulative-to-win): trial-major log. Per puzzle, sum the action
      calls of trials 0..t where t is the FIRST trial that solved it (win =
      'Obs N> Success!'); un-solved puzzles are simply not wins.
  - rollback : real committed actions are 'Act N:' (colon); 'Act N>' lines are
      replayed history and are NOT calls. Each '****Analysis****' banner is one
      error-detection LLM call. win = '[oracle] solved=True'.
  - BFS conformal : each '[BFS Node]' = one completions.create request drawing
      n_samples generations. Per-episode summary '[k/100] ... solved=.. nodes=N'
      gives nodes and win directly. echo-scoring / score-model NOT counted.

Usage:  python count_llm_calls_winning_game24.py
"""
import re
from pathlib import Path

ID_RE       = re.compile(r'^id:(\d+),')
RATE_RE     = re.compile(r'^id:(\d+),\s*success rate:([\d.]+)')
SOLVED_RE   = re.compile(r'\[oracle\]\s+solved=(True|False)')
SUCCESS_RE  = re.compile(r'Obs\s+\d+>\s*Success!')
ACT_GT_RE   = re.compile(r'^\s*Act\s+\d+\s*>')          # react/reflact/reflexion real step
ACT_COLON_RE= re.compile(r'^\s*Act\s+\d+\s*:')          # rollback committed action
ANALYSIS_RE = re.compile(r'\*+Analysis\*+')             # rollback analysis banner
STARS_RE    = re.compile(r'^\s*\*{4,}\s*$')             # analysis-block closing banner
ACTION_RE   = re.compile(r'^\s*Action\s+(\d+)\s*:\s*(.*)$')  # webshop-style (unused here)
TRIAL_EP_RE = re.compile(r'\[Trial #(\d+)\] Episode (\d+)/\d+ completed')
BFS_SUMMARY_RE = re.compile(r'^\[(\d+)/\d+\]\s+puzzle=.*\bsolved=(True|False)\b.*\bnodes=(\d+)')


def parse_react_style(path):
    """react/reflact: episodes split by 'id:N'; calls = 'Act N>'; win = solved=True."""
    if not Path(path).exists():
        return None
    episodes, calls, won = [], 0, False
    for line in Path(path).read_text(errors='ignore').splitlines():
        if ACT_GT_RE.match(line):
            calls += 1
        m = SOLVED_RE.search(line)
        if m:
            won = (m.group(1) == 'True')
        if ID_RE.match(line):
            episodes.append((calls, won)); calls, won = 0, False
    return episodes


def parse_reflexion(path):
    """trial-major reflexion; cumulative-to-win per puzzle."""
    if not Path(path).exists():
        return None
    per = {}                       # k -> {t: (calls, won)}
    cur_calls, cur_won = 0, False
    for line in Path(path).read_text(errors='ignore').splitlines():
        if ACT_GT_RE.match(line):
            cur_calls += 1
        if SUCCESS_RE.search(line):
            cur_won = True
        m = TRIAL_EP_RE.search(line)
        if m:
            t, k = int(m.group(1)), int(m.group(2))
            per.setdefault(k, {})[t] = (cur_calls, cur_won)
            cur_calls, cur_won = 0, False
    episodes = []
    for k, trials in per.items():
        cost, won = 0, False
        for t in sorted(trials):
            c, w = trials[t]
            cost += c
            if w:
                won = True; break
        episodes.append((cost, won))
    return episodes


def parse_rollback(path):
    """rollback: calls = committed 'Act N:' + '****Analysis****' banners.
    win = puzzle N raised the cumulative solved count (running success-rate
    increment) — the rollback log prints no per-episode 'solved=' line."""
    if not Path(path).exists():
        return None
    episodes, calls, prev_cum = [], 0, 0
    for line in Path(path).read_text(errors='ignore').splitlines():
        if ACT_COLON_RE.match(line):
            calls += 1
        elif ANALYSIS_RE.search(line):
            calls += 1
        m = RATE_RE.match(line)
        if m:
            n, rate = int(m.group(1)), float(m.group(2))
            cum = round(rate * n)
            episodes.append((calls, cum > prev_cum))
            prev_cum = cum; calls = 0
    return episodes


def parse_bfs_conformal(path, n_samples=10):
    """BFS conformal: per-episode (nodes, won) from the '[k/100] ... nodes=N' summary."""
    if not Path(path).exists():
        return None, n_samples
    episodes = []
    for line in Path(path).read_text(errors='ignore').splitlines():
        m = BFS_SUMMARY_RE.match(line)
        if m:
            episodes.append((int(m.group(3)), m.group(2) == 'True'))
    return episodes, n_samples


def report(name, per_episode):
    wins = [(c, w) for c, w in per_episode if w]
    n_win = len(wins)
    total = sum(c for c, _ in wins)
    avg = total / n_win if n_win else 0.0
    print(f"{name:10s}  episodes={len(per_episode):4d}  wins={n_win:4d}  "
          f"LLM calls (winning)={total:6d}  avg/win={avg:.2f}")


G24_NODE_BUCKETS = [("1-2", 1, 2), ("3-5", 3, 5), ("6-10", 6, 10),
                    ("11-20", 11, 20), ("21-50", 21, 50)]


def report_bfs_distribution(per_episode, buckets=G24_NODE_BUCKETS):
    win_nodes = sorted(c for c, w in per_episode if w)
    n_win = len(win_nodes)
    total_req = sum(win_nodes)
    if n_win == 0:
        print("  (no winning episodes — distribution skipped)"); return
    print(f"\nDistribution over the {n_win} wins (nodes = completions requests):")
    print("+------------+------------+-----------------------+")
    print("| Nodes used |    Wins    | Share of all requests |")
    print("+------------+------------+-----------------------+")
    counted = req_counted = 0
    for label, lo, hi in buckets:
        sel = [c for c in win_nodes if lo <= c <= hi]
        w, req = len(sel), sum(sel)
        counted += w; req_counted += req
        print(f"| {label:10s} | {w:2d} ({100.0*w/n_win:4.1f}%) | "
              f"{100.0*req/total_req if total_req else 0:20.1f}% |")
        print("+------------+------------+-----------------------+")
    outside = [c for c in win_nodes if not any(lo <= c <= hi for _, lo, hi in buckets)]
    if outside:
        w, req = len(outside), sum(outside)
        counted += w; req_counted += req
        print(f"| {'other':10s} | {w:2d} ({100.0*w/n_win:4.1f}%) | "
              f"{100.0*req/total_req if total_req else 0:20.1f}% |")
        print("+------------+------------+-----------------------+")
    print(f"Totals: wins={counted} (of {n_win}), requests={req_counted} (of {total_req})")


NICE = {"qwen25_3b": "Qwen2.5-3B", "qwen3_8b": "Qwen3-8B", "gemma4_12b": "Gemma-4-12B"}


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    for model in ["qwen25_3b", "qwen3_8b", "gemma4_12b"]:
        d = root / f"compare_logs_100_{model}"
        print("\n" + "=" * 74)
        print(f"{NICE.get(model, model)}   (Game-of-24, 100 test puzzles)")
        print("=" * 74)
        print("LLM calls spent on WINNING (solved) puzzles  "
              "[Reflexion: cumulative to win]")
        for name, parser, fn in [("ReAct", parse_react_style, "base_react.txt"),
                                 ("ReflAct", parse_react_style, "base_reflact.txt"),
                                 ("Reflexion", parse_reflexion, "base_reflexion.txt"),
                                 ("Rollback", parse_rollback, "base_rollback.txt")]:
            eps = parser(d / fn)
            if eps is None:
                print(f"{name:10s}  (log missing)")
            else:
                report(name, eps)

        for alpha, fn in [("0.1", "m5_alpha01.txt"), ("0.2", "m5_alpha02.txt"),
                          ("0.3", "m5_alpha03.txt")]:
            eps, ns = parse_bfs_conformal(d / fn)
            if eps is None:
                print(f"\nBFS conformal (α={alpha}): log missing"); continue
            print(f"\nBFS conformal (α={alpha}): one request per BFS node, "
                  f"n_samples={ns} generations each")
            report("BFS req", eps)
            report("BFS gen", [(c * ns, w) for c, w in eps])
            report_bfs_distribution(eps)
