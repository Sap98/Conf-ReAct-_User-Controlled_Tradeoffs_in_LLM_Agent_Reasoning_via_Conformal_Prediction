"""
Count LLM calls spent on WINNING episodes for ScienceWorld methods, and the
average per winning episode — the ScienceWorld analog of the WebShop
count_llm_calls_winning.py.

ScienceWorld runs write structured JSONL, so we count from fields (not text):

Baselines (ReAct / ReflAct / Reflexion):
  * each episode record carries `n_llm_calls` (LLM calls made in that episode).
  * win = record's `won` is True (equivalently score >= 100).
  * ReAct / ReflAct: one record per episode → calls = n_llm_calls.
  * Reflexion (cumulative-to-win): records are per TRIAL (trial_idx) per
    (sw_name, var). For an env that first won at trial t, count n_llm_calls of
    ALL its trials 0..t (failed trials + the winning trial). Envs that never
    win are not wins.

BFS conformal:
  * `n_nodes` = number of BFS nodes expanded. Per node the play makes:
        - 1 SAMPLING request  (completions.create, n=n_samples generations), and
        - 1 echo probe + K echo-scoring requests (K = #candidates, batch_size=1).
    The JSONL doesn't store K, so we report two clean views (matching WebShop's
    "requests" vs "generations") from n_nodes, and separately the echo overhead
    parsed from the run logs when available:
        sampling_requests = n_nodes
        generations       = n_nodes * n_samples
  * win = record's `won` is True.

Excludes the 5 leaking identify-life-stages (task,var) pairs so the winning set
matches the clean 206-episode evaluation.

Usage:
    python count_llm_calls_winning_sciworld.py
"""
import os
import re
import json
import glob
import collections

_HERE = os.path.dirname(os.path.abspath(__file__))
N_SAMPLES = 10          # bfs_conformal_211.py default --n_samples

LEAK = {('identify-life-stages-1', 9), ('identify-life-stages-2', 6),
        ('identify-life-stages-2', 7), ('identify-life-stages-2', 8),
        ('identify-life-stages-2', 9)}


def _sw(r):
    return r.get('sw_name') or r.get('task') or ''


def _won(r):
    return bool(r.get('won')) or r.get('score', 0) >= 100


def _leaky(r):
    return (_sw(r), r.get('var')) in LEAK


# ── Baselines ────────────────────────────────────────────────────────────────

def react_style_calls(path):
    """ReAct / ReflAct: one record per episode. Returns [(calls, won)]."""
    if not path or not os.path.exists(path):
        return None
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if _leaky(r):
            continue
        out.append((r.get('n_llm_calls', 0), _won(r)))
    return out


def reflexion_calls(path):
    """Reflexion: per-trial records. Cumulative calls up to & including the first
    winning trial per (sw_name, var). Returns [(calls, won)] per env."""
    if not path or not os.path.exists(path):
        return None
    trials = collections.defaultdict(list)   # (sw,var) -> [(trial_idx, calls, won)]
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if _leaky(r):
            continue
        trials[(_sw(r), r.get('var'))].append(
            (r.get('trial_idx', 0), r.get('n_llm_calls', 0), _won(r)))
    out = []
    for _, ts in trials.items():
        ts.sort(key=lambda x: x[0])
        cum = 0
        won = False
        for _, calls, w in ts:
            cum += calls
            if w:
                won = True
                break          # stop at first winning trial (cumulative-to-win)
        out.append((cum, won))
    return out


# ── Conformal ────────────────────────────────────────────────────────────────

def conformal_calls(jsonl_glob):
    """Per alpha: [(n_nodes, won)] over clean episodes. Returns {alpha: [...]}"""
    per = collections.defaultdict(list)
    for f in glob.glob(jsonl_glob):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if _leaky(r):
                continue
            per[r.get('alpha')].append((r.get('n_nodes', 0), _won(r)))
    return per


# ── Reporting ────────────────────────────────────────────────────────────────

def report(name, per_episode):
    if per_episode is None:
        print(f"  {name:28s}  (no data / not run yet)")
        return
    wins = [(c, w) for c, w in per_episode if w]
    n_win = len(wins)
    total = sum(c for c, _ in wins)
    avg = total / n_win if n_win else 0.0
    print(f"  {name:28s}  episodes={len(per_episode):4d}  wins={n_win:3d}  "
          f"calls_on_wins={total:6d}  avg/win={avg:6.1f}")


def report_conformal(name, per_alpha):
    for a in sorted(per_alpha):
        eps = per_alpha[a]
        wins = [(nodes, w) for nodes, w in eps if w]
        n_win = len(wins)
        nodes_on_wins = sum(n for n, _ in wins)
        gen_on_wins = nodes_on_wins * N_SAMPLES
        avg_nodes = nodes_on_wins / n_win if n_win else 0.0
        print(f"  {name} α={a}: episodes={len(eps):3d}  wins={n_win:3d}  "
              f"| sampling_requests_on_wins={nodes_on_wins:5d} (avg/win={avg_nodes:5.1f})  "
              f"| generations_on_wins={gen_on_wins:6d} (avg/win={avg_nodes*N_SAMPLES:6.1f})")


MODELS = [
    ("Qwen3-8B", {
        'react':     f"{_HERE}/results_react_211.jsonl",
        'reflact':   None,     # ReflAct Qwen3 is in scienceworld_reflact.txt (text), skip here
        'reflexion': f"{_HERE}/results/results_reflexion_211.jsonl",
        'conformal': f"{_HERE}/conformal_prediction/results/qwen_211/qwen_a*_shard*.jsonl",
    }),
    ("Gemma-4-12B", {
        'react':     f"{_HERE}/gemma_baseline_results/react_gemma_211.jsonl",
        'reflact':   f"{_HERE}/gemma_baseline_results/reflact_gemma_211.jsonl",
        'reflexion': f"{_HERE}/gemma_baseline_results/reflexion_gemma_211.jsonl",
        'conformal': f"{_HERE}/conformal_prediction_gemma/results/gemma_211/gemma_a*_shard*.jsonl",
    }),
    ("Qwen2.5-3B", {
        'react':     f"{_HERE}/qwen25_baseline_results/react_qwen25_211.jsonl",
        'reflact':   f"{_HERE}/qwen25_baseline_results/reflact_qwen25_211.jsonl",
        'reflexion': f"{_HERE}/qwen25_baseline_results/reflexion_qwen25_211.jsonl",
        'conformal': f"{_HERE}/conformal_prediction_qwen25_3b/results/qwen25_211/qwen25_a*_shard*.jsonl",
    }),
]


def main():
    print("=" * 78)
    print("  ScienceWorld — LLM calls spent on WINNING episodes (clean 206, leaks excluded)")
    print("=" * 78)
    for model, paths in MODELS:
        print(f"\n### {model}")
        report("ReAct",     react_style_calls(paths['react']))
        report("ReflAct",   react_style_calls(paths['reflact']))
        report("Reflexion (cumulative-to-win)", reflexion_calls(paths['reflexion']))
        conf = conformal_calls(paths['conformal'])
        if conf:
            report_conformal("BFS conformal", conf)
        else:
            print("  BFS conformal                 (no data)")
    print("\nNotes:")
    print(f"  * Baseline calls = recorded n_llm_calls per episode.")
    print(f"  * Conformal 'sampling_requests' = BFS nodes (1 completions.create/node,")
    print(f"    n_samples={N_SAMPLES} generations each). 'generations' = nodes*n_samples.")
    print(f"  * Conformal also issues ~(1 probe + K echo-scoring) requests/node for the")
    print(f"    logprob features (K=#candidates); not included above — parse run logs for it.")


if __name__ == "__main__":
    main()
