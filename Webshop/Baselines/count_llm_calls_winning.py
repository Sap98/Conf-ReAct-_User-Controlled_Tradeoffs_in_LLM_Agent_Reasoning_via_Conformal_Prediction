"""Count LLM calls spent on WINNING (score=1.0) episodes in the WebShop baseline
output logs (ReAct / ReflAct / Reflexion), and the average per winning episode.

Counting rules (verified against each script's source):
  - Every printed "Action: ..." line is one LLM call, EXCEPT:
      * "Action: reset" (environment reset, no LLM),
      * one call per "[auto-redirect]" marker: the inserted click[Back to Search]
        and the deferred search[] that follows are two printed actions produced
        by a single LLM call.
  - think[...] / reflection[...] lines and actions answered with
    "Invalid action!" ARE LLM calls.
  - Reflexion (cumulative-to-win): for an env whose trial t won, count the
    action calls of ALL its printed trials (failed trials 0..t-1 + winning
    trial t) plus one call per "[Reflexion] Generating reflection..." line.

Usage:
  python count_llm_calls_winning.py
"""

import re
import sys

SCORE_RE = re.compile(r'Your score \(min 0\.0, max 1\.0\): ([\d.]+)')
EP_SEP = '-----------------'


def parse_react_style(path):
    """react/reflact logs: episodes separated by '-----------------' + index line.
    Returns list of (llm_calls, won) per episode."""
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()

    episodes = []
    cur = None  # [calls, won]
    for i, line in enumerate(lines):
        if line == EP_SEP and i + 1 < len(lines) and lines[i + 1].strip().isdigit():
            if cur is not None:
                episodes.append(tuple(cur))
            cur = [0, False]
            continue
        if cur is None:
            continue
        if line.startswith('Action: '):
            if line != 'Action: reset':
                cur[0] += 1
        elif '[auto-redirect]' in line:
            cur[0] -= 1          # 2 printed actions came from 1 LLM call
        else:
            m = SCORE_RE.search(line)
            if m and float(m.group(1)) == 1.0:
                cur[1] = True
    if cur is not None:
        episodes.append(tuple(cur))
    return episodes


ENV_RE = re.compile(r'^\[Env \d+/\d+\] Trial (\d+)\s+session=(fixed_\d+)')
WON_RE = re.compile(r'reward=([\d.]+)\s+won=(True|False)')


def parse_reflexion(path):
    """reflexion log: trial blocks '[Env k/N] Trial t  session=fixed_i' scattered
    across TRIAL sections. Returns {session: {'calls': int, 'won': bool}} with
    cumulative calls (all printed trials + reflection generations)."""
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()

    envs = {}
    cur = None
    for line in lines:
        m = ENV_RE.match(line)
        if m:
            session = m.group(2)
            cur = envs.setdefault(session, {'calls': 0, 'won': False})
            continue
        if line.startswith('=' * 20):   # section header ends any open block
            cur = None
            continue
        if cur is None:
            continue
        if line.startswith('Action: '):
            if line != 'Action: reset':
                cur['calls'] += 1
        elif '[auto-redirect]' in line:
            cur['calls'] -= 1
        elif line.startswith('[Reflexion] Generating reflection'):
            cur['calls'] += 1
        else:
            m = WON_RE.search(line)
            if m and m.group(2) == 'True':
                cur['won'] = True
    return envs


EPISODE_HDR_RE = re.compile(r'^\s*EPISODE (\d+) / \d+\s+\(')
BFS_NODE_RE = re.compile(r'^\[BFS Node \d+\]')
BFS_REWARD_RE = re.compile(r'^\s*Reward\s*:\s*([\d.]+)')
BFS_NSAMPLES_RE = re.compile(r'^\s*n_samples\s*:\s*(\d+)')


def parse_bfs_conformal(path):
    """BFS conformal log: each '[BFS Node ...]' is ONE completions.create request
    that draws n_samples generations. Wins: 'Reward : 1.0000' in the episode
    summary. Returns (episodes=[(nodes, won)], n_samples)."""
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()

    n_samples = None
    episodes = []
    cur = None  # [nodes, won]
    for line in lines:
        if n_samples is None:
            m = BFS_NSAMPLES_RE.match(line)
            if m:
                n_samples = int(m.group(1))
        if EPISODE_HDR_RE.match(line):
            if cur is not None:
                episodes.append(tuple(cur))
            cur = [0, False]
            continue
        if cur is None:
            continue
        if BFS_NODE_RE.match(line):
            cur[0] += 1
        else:
            m = BFS_REWARD_RE.match(line)
            if m and float(m.group(1)) == 1.0:
                cur[1] = True
    if cur is not None:
        episodes.append(tuple(cur))
    return episodes, n_samples


def report(name, per_episode):
    wins = [(c, w) for c, w in per_episode if w]
    total_eps = len(per_episode)
    n_win = len(wins)
    total_calls = sum(c for c, _ in wins)
    avg = total_calls / n_win if n_win else 0.0
    print(f"{name:10s}  episodes={total_eps:4d}  wins={n_win:4d}  "
          f"LLM calls (winning)={total_calls:6d}  avg/win={avg:.2f}")
    return n_win, total_calls, avg


# (label, lo, hi) inclusive node-count buckets for the BFS-conformal win breakdown.
BFS_NODE_BUCKETS = [
    ("3-5",   3,   5),
    ("6-10",  6,  10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-99", 51, 99),
]


def report_bfs_distribution(per_episode, buckets=BFS_NODE_BUCKETS):
    """Bucket WINNING BFS episodes by nodes used (= completions requests).

    For each bucket prints wins (count + % of all wins) and the share of all
    winning requests those wins consumed. Non-cumulative: each win lands in
    exactly one bucket, so the columns sum to the totals.
    """
    win_nodes = sorted(c for c, w in per_episode if w)
    n_win = len(win_nodes)
    total_req = sum(win_nodes)
    if n_win == 0:
        print("  (no winning episodes — distribution skipped)")
        return

    print("\nDistribution over the {} wins (nodes = completions requests):".format(n_win))
    print("+------------+------------+-----------------------+")
    print("| Nodes used |    Wins    | Share of all requests |")
    print("+------------+------------+-----------------------+")
    counted = 0
    req_counted = 0
    for label, lo, hi in buckets:
        sel = [c for c in win_nodes if lo <= c <= hi]
        w = len(sel)
        req = sum(sel)
        counted += w
        req_counted += req
        win_pct = 100.0 * w / n_win
        req_pct = 100.0 * req / total_req if total_req else 0.0
        print(f"| {label:10s} | {w:2d} ({win_pct:4.1f}%) | {req_pct:20.1f}% |")
        print("+------------+------------+-----------------------+")

    # Anything outside the predefined ranges (defensive — should be 0 for this log).
    outside = [c for c in win_nodes if not any(lo <= c <= hi for _, lo, hi in buckets)]
    if outside:
        w = len(outside)
        req = sum(outside)
        counted += w
        req_counted += req
        print(f"| {'other':10s} | {w:2d} ({100.0*w/n_win:4.1f}%) | "
              f"{100.0*req/total_req if total_req else 0.0:20.1f}% |")
        print("+------------+------------+-----------------------+")

    print(f"Totals: wins={counted} (of {n_win}), requests={req_counted} (of {total_req})")


if __name__ == '__main__':
    base = '/home/saptarshi/transfer/alfworld/ReAct/4th_SEM_MTP/S_T_Webshop'
    react_eps = parse_react_style(f'{base}/baselines_gpt_4.1/react_webshop_qwen_3.txt')
    reflact_eps = parse_react_style(f'{base}/reflact_webshop.txt')
    reflexion_envs = parse_reflexion(f'{base}/webshop_reflexion.txt')
    reflexion_eps = [(v['calls'], v['won']) for v in reflexion_envs.values()]

    print("LLM calls spent on WINNING (score=1.0) episodes  "
          "[Reflexion: cumulative to win]")
    report("ReAct", react_eps)
    report("ReflAct", reflact_eps)
    report("Reflexion", reflexion_eps)

    bfs_logs = [
        ("0.1", f'{base}/bfs_webshop_conformal_0.1.txt'),
        ("0.2", f'{base}/bfs_webshop_conformal_0.2.txt'),
        ("0.3", f'{base}/bfs_webshop_conformal_0.3.txt'),
    ]
    for alpha, log_path in bfs_logs:
        bfs_eps, n_samples = parse_bfs_conformal(log_path)
        print(f"\nBFS conformal (α={alpha}): one request per BFS node, "
              f"n_samples={n_samples} generations each")
        report("BFS req", bfs_eps)
        report("BFS gen", [(c * n_samples, w) for c, w in bfs_eps])
        report_bfs_distribution(bfs_eps)
