"""
Game-of-24-style BFS conformal play for ScienceWorld.

Port of `bfs_conformal_game24.py` (GA-Rollback/test_game24_smoke/conformal/):
a conformal-pruned FIFO BFS over the action tree, run on the first
`--max_episodes` (default 100) entries of the MPO test set.

Differences from `bfs_conformal_sciworld_new.py` (the priority-queue variant):
  * plain FIFO BFS (collections.deque) — true breadth-first, like game24/webshop;
  * per-state SET-COVERAGE metric.  Game24 has an oracle (`correct_actions`)
    at every state; ScienceWorld's only oracle is the static gold action
    sequence from the episode start, so coverage is measured exactly on
    ON-GOLD-PATH nodes (prev_actions == gold[:t], correct action = gold[t]) —
    the same states calibration was built on.  Off-path nodes are still
    expanded but are not scored for coverage (analog of game24 skipping
    unsolvable states, where coverage is undefined).
    Per eligible state we emit the machine-parseable line the plotter reads:
        [STATEMETRIC] set_size_pct=.. covered=0/1
  * `--oracle_candidates` (default on, like game24): union gold[t] into the
    LLM candidate pool at on-path nodes so coverage is achievable and the
    pool matches calibration (data generation unioned gold[t] too).
  * game24-style FINAL RESULTS block + count/% distributions + histograms of
    search nodes (all episodes) and winning path length (solved episodes).

LLM protocol (sampling `running + "Thought:"`, echo-scoring
`running + "Action:"`, 10-bin softmax) is IDENTICAL to
data_generation_sciworld.py / bfs_conformal_sciworld_new.py so the trained
score model and calibration pool line up.

Usage:
    python bfs_conformal_sciworld_game24_style.py --alpha 0.1 --max_episodes 100 \
        --results results/bfs_conformal_g24style_a01.jsonl > bfs_g24style_a01.txt
"""

import os
import sys
import json
import math
import time
import random
import pickle
import argparse
import collections
from collections import deque

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openai import OpenAI
from scienceworld import ScienceWorldEnv

# ── Path bootstrap ───────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.abspath(__file__))      # .../conformal_prediction
_REACT = os.path.dirname(_HERE)                          # .../react
_REPO  = os.path.dirname(_REACT)                         # .../ScienceWorld
sys.path.insert(0, _REACT)
sys.path.insert(0, os.path.join(_REPO, "examples"))
sys.path.insert(0, _HERE)

from react_sciworld import (
    parse_thought_action, normalize_obs, PRESETS,
    load_mpo_testset, aggregate,
)
from scienceworld_react_prompt import SCIENCEWORLD_REACT_PROMPT

from score_model import ScoreFunction, BertEmbeddingCache, SOFTMAX_BINS
from train_score import build_records
from conformal_predictor import ConformalPredictor


# ── Paths / defaults ─────────────────────────────────────────────────────────
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH       = os.path.join(_HERE, "training_data_2_sciworld.pkl")
MODEL_PATH      = os.path.join(_HERE, "trained_models", "best_score_model.pt")
BERT_CACHE_PATH = os.path.join(_HERE, "trained_models", "bert_cache.pkl")

MODEL_NAME    = "google/gemma-4-12B-it"
VLLM_BASE_URL = "http://10.5.18.73:8003/v1"
client = None     # set in main() once --base_url is known

# ── Location tracking — must match data_generation_sciworld.py exactly ──────
INITIAL_LOCATION = "the starting location"
_MOVE_PREFIXES   = ("teleport to ", "go to ", "move to ")


def update_location(location, action):
    a = action.strip()
    low = a.lower()
    for pref in _MOVE_PREFIXES:
        if low.startswith(pref):
            return a[len(pref):].strip()
    return location


# ═══════════════════════════════════════════════════════════════════════════════
# LLM HELPERS — protocol identical to data_generation_sciworld.py
# ═══════════════════════════════════════════════════════════════════════════════

def logprobs_to_softmax_bins(token_logprobs, num_bins=SOFTMAX_BINS):
    bins = [0] * num_bins
    for lp in token_logprobs:
        idx = min(int(math.exp(lp) * num_bins), num_bins - 1)
        bins[idx] += 1
    return bins


def sample_candidate_actions(sampling_prompt, n, temperature, max_tokens=200):
    """Sample n completions, parse each as (thought, action), dedupe actions."""
    resp = client.completions.create(
        model=MODEL_NAME,
        prompt=sampling_prompt,
        n=n,
        max_tokens=max_tokens,
        temperature=temperature,
        echo=False,
    )
    candidates, seen, action_think = [], set(), {}
    for choice in resp.choices:
        thought, action = parse_thought_action(choice.text)
        if not action:
            continue
        a = action.strip()
        if a in seen:
            continue
        seen.add(a)
        candidates.append(a)
        if thought:
            action_think[a] = thought
    return candidates, action_think


def get_action_logprobs(scoring_prompt, actions, batch_size=1):
    """Echo-score each action: scoring_prompt + ' ' + action → {action: [lps]}."""
    if not actions:
        return {}
    probe = client.completions.create(
        model=MODEL_NAME, prompt=scoring_prompt,
        max_tokens=1, echo=True, logprobs=1,
    )
    n_prompt = len(probe.choices[0].logprobs.tokens) - 1

    results = {}
    bs = max(1, batch_size)
    for i in range(0, len(actions), bs):
        chunk = actions[i:i + bs]
        full_prompts = [scoring_prompt + " " + a for a in chunk]
        resp = client.completions.create(
            model=MODEL_NAME, prompt=full_prompts,
            max_tokens=0, echo=True, logprobs=1,
        )
        for choice, action in zip(resp.choices, chunk):
            lps = [lp for lp in choice.logprobs.token_logprobs[n_prompt:]
                   if lp is not None]
            results[action] = lps
    return results


def build_softmax_values(scoring_prompt, candidates, batch_size=1):
    if not candidates:
        return {}
    raw = get_action_logprobs(scoring_prompt, candidates, batch_size=batch_size)
    return {a: logprobs_to_softmax_bins(raw.get(a, [])) for a in candidates}


# ═══════════════════════════════════════════════════════════════════════════════
# ENV STATE RESTORE  (ScienceWorld has no snapshot API → replay)
# ═══════════════════════════════════════════════════════════════════════════════

def replay_to_state(env, sw_name, var_idx, simpl_str, prev_actions):
    env.load(sw_name, var_idx, simpl_str)
    env.reset()
    obs, score, done, info = "", 0, False, {}
    for a in prev_actions:
        obs, _r, done, info = env.step(a)
        score = info.get('score', score)
        if done:
            break
    return obs, score, done, info


def get_gold_sequence(env):
    """Gold action sequence from the episode start, or [] if unavailable."""
    try:
        gold = env.get_gold_action_sequence()
    except Exception:
        return []
    if not gold or (len(gold) == 1 and str(gold[0]).startswith("ERROR")):
        return []
    return [str(a).strip() for a in gold]


# ═══════════════════════════════════════════════════════════════════════════════
# GAME24-STYLE DISTRIBUTION PRINT / PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def _print_dist(label, values):
    if not values:
        print(f"\n  DISTRIBUTION ({label})\n  (no data)")
        return
    dist = collections.Counter(values)
    total = len(values)
    print(f"\n  DISTRIBUTION ({label})")
    print(f"  {'Value':>6}  {'Count':>6}  {'%':>7}")
    for v in range(0, max(values) + 1):
        c = dist.get(v, 0)
        if c == 0:
            continue
        print(f"  {v:>6}  {c:>6}  {c/total*100:>6.1f}%")


def _plot_dist(label, values, out_path, color, xlabel, succ):
    if not values:
        print(f"  [plot] no data for {label}, skipping {out_path}")
        return
    dist = collections.Counter(values)
    xs = sorted(dist.keys())
    counts = [dist[x] for x in xs]
    total = len(values)
    pcts = [c / total * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(xs, counts, width=0.8, edgecolor="black", color=color)
    for bar, c, p in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{c}\n({p:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"ScienceWorld — {label}  ({total} episodes)\n"
                 f"success rate: {succ*100:.1f}%")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of episodes")
    ax.set_xticks(xs)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved -> {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFORMAL-PRUNED FIFO BFS — one episode
# ═══════════════════════════════════════════════════════════════════════════════

# on_gold: prev_actions is exactly gold[:len(prev_actions)] — the states where
# the correct next action (gold[t]) is known and coverage is defined.
BFSNode = collections.namedtuple(
    'BFSNode', ['prev_actions', 'location', 'running', 'on_gold'])


def run_bfs_episode(env, predictor, sw_name, var_idx, simpl_str, task_desc,
                    gold, args, max_depth, max_nodes):
    """Returns (solved, best_score, best_traj, nodes, n_states, n_covered, win_len,
    n_cond_states, n_cond_covered).  n_states/n_covered = unconditional coverage;
    n_cond_states/n_cond_covered = conditional (gold-in-candidates only)."""
    # '/no_think' suppresses Qwen3 thinking blocks; other models don't need it.
    prefix = "/no_think\n" if "qwen3" in MODEL_NAME.lower() else ""
    base = prefix + SCIENCEWORLD_REACT_PROMPT + task_desc + "\n"
    frontier = deque([BFSNode(prev_actions=[], location=INITIAL_LOCATION,
                              running=base, on_gold=bool(gold))])
    visited = set()

    nodes = 0            # BFS nodes expanded (search effort)
    n_states = 0         # on-gold-path states scored (UNCONDITIONAL coverage denom)
    n_covered = 0        # of those, gold_next landed in the prediction set
    n_cond_states = 0    # on-gold states where gold WAS an LLM candidate (CONDITIONAL denom)
    n_cond_covered = 0   # of those, gold_next in pred_set  (isolates conformal validity)
    solved = False
    best_score = 0
    best_traj = []
    win_len = 0

    while frontier and nodes < max_nodes:
        node = frontier.popleft()
        prev_actions, location, running = \
            node.prev_actions, node.location, node.running

        key = tuple(prev_actions)
        if key in visited:
            continue
        visited.add(key)

        # ── 1. Restore env to this node's state ───────────────────────────────
        obs, score, done, info = replay_to_state(
            env, sw_name, var_idx, simpl_str, prev_actions)
        if score > best_score:
            best_score = score
            best_traj = list(prev_actions)
        if score >= 100:
            solved = True
            win_len = win_len or len(prev_actions)
            print(f"  *** full credit (score={score}) via {prev_actions} ***")
            break
        if done:
            continue
        if len(prev_actions) >= max_depth:
            continue

        nodes += 1
        gold_next = (gold[len(prev_actions)]
                     if node.on_gold and len(prev_actions) < len(gold) else None)

        print(f"\n[BFS Node {nodes}]  depth={len(prev_actions)}")
        print(f"  Task     : {task_desc}")
        print(f"  Traj     : {prev_actions if prev_actions else '(start)'}")
        print(f"  Location : {location}   env_score={score}")
        print(f"  On gold  : {node.on_gold}   gold_next={gold_next!r}")
        sys.stdout.flush()

        # ── 2. LLM samples candidate actions at this state ────────────────────
        sampling_prompt = running + "Thought:"
        try:
            candidates, action_think = sample_candidate_actions(
                sampling_prompt, n=args.n_samples, temperature=args.temperature)
        except Exception as ex:
            # Most likely the running prompt outgrew the model context
            # (deep node + long observations). Prune this branch.
            print(f"  [LLM error] {type(ex).__name__}: {str(ex)[:200]} — node pruned")
            sys.stdout.flush()
            continue
        print(f"  [LLM sampled actions] ({len(candidates)}) {candidates}")

        # Candidate pool: union with the gold next action at on-path nodes so
        # coverage is achievable and the pool matches calibration (game24's
        # --oracle_candidates). --no_oracle_candidates → LLM-only pool.
        if gold_next and args.oracle_candidates:
            n_before = len(candidates)
            candidates = list(dict.fromkeys(candidates + [gold_next]))
            print(f"  Candidates ({len(candidates)}): {n_before} LLM "
                  f"+ {len(candidates) - n_before} gold added")

        if not candidates:
            # Fallback: one greedy generation
            try:
                resp = client.completions.create(
                    model=MODEL_NAME, prompt=sampling_prompt,
                    n=1, max_tokens=200, temperature=0.0)
                t, a = parse_thought_action(resp.choices[0].text)
            except Exception as ex:
                print(f"  [LLM error] {type(ex).__name__}: {str(ex)[:200]}")
                t, a = "", ""
            if a:
                candidates = [a.strip()]
                if t:
                    action_think[a.strip()] = t

        if not candidates:
            print("  [skip] No candidate actions.")
            if gold_next:                    # uncovered eligible state
                n_states += 1
                print("[STATEMETRIC] set_size_pct=0.00 covered=0")
            sys.stdout.flush()
            continue

        # ── 3. Echo-score candidates → 10-bin softmax distributions ───────────
        scoring_prompt = running + "Action:"
        try:
            softmax_values = build_softmax_values(
                scoring_prompt, candidates, batch_size=args.score_batch_size)
        except Exception as ex:
            print(f"  [LLM error] {type(ex).__name__}: {str(ex)[:200]} — node pruned")
            sys.stdout.flush()
            continue

        # ── 4. Conformal prediction set ───────────────────────────────────────
        pred_set, scores, S_star, n_sel, _ = predictor.get_prediction_set(
            task               = task_desc,
            previous_actions   = prev_actions,
            location           = location,
            admissible_actions = candidates,
            softmax_values     = softmax_values,
            method             = args.method,
            select             = 'topk',
            k                  = args.k,
            n_components       = args.n_components,
        )

        if not pred_set:
            pred_set = [min(scores, key=scores.__getitem__)]
            print(f"  [fallback] Empty pred set → best-score: {pred_set}")
        else:
            print(f"  Pred set ({len(pred_set)}/{len(candidates)}) "
                  f"using {n_sel} calibration states: {pred_set}")

        print("  Scores [lower = more likely optimal]:")
        for a in sorted(scores, key=scores.__getitem__):
            gtag = " (gold)" if a == gold_next else ""
            tag = " [IN SET]" if a in pred_set else ""
            print(f"    {scores[a]:.4f}  {a}{gtag}{tag}")

        # ── SET-COVERAGE metric (on-gold-path states only) ────────────────────
        # Two coverage numbers, both honouring "no oracle union":
        #   unconditional : covered over ALL on-gold states (gold-not-sampled = miss).
        #                   Reflects the whole system incl. LLM candidate-gen; reads
        #                   low because the LLM rarely proposes the exact gold action.
        #   conditional   : covered only over states where gold WAS an LLM candidate.
        #                   Isolates the conformal predictor; this is the number that
        #                   should track 1-alpha if calibration is valid.
        if gold_next:
            gold_in_cands = gold_next in candidates
            covered = gold_next in pred_set
            set_pct = 100.0 * len(pred_set) / max(1, len(candidates))
            n_states += 1
            n_covered += int(covered)
            if gold_in_cands:
                n_cond_states += 1
                n_cond_covered += int(covered)
            print(f"[STATEMETRIC] set_size_pct={set_pct:.2f} covered={int(covered)} "
                  f"gold_in_cands={int(gold_in_cands)}")

        # ── 5. Execute / expand each action in the conformal set ──────────────
        children = []
        for action in pred_set:
            # Re-restore before each branch (siblings share the parent state)
            replay_to_state(env, sw_name, var_idx, simpl_str, prev_actions)
            child_obs, _r, child_done, child_info = env.step(action)
            child_obs = normalize_obs(child_obs)
            child_score = child_info.get('score', 0)

            new_prev = prev_actions + [action]
            print(f"    → {action!r}   score={child_score}   done={child_done}")

            if child_score > best_score:
                best_score = child_score
                best_traj = list(new_prev)
            if child_score >= 100:
                solved = True
                if not win_len:
                    win_len = len(new_prev)
                print(f"  *** full credit via {new_prev} ***")

            if child_done or solved:
                continue

            thought = action_think.get(action) or (
                next(iter(action_think.values())) if action_think else "")
            children.append(BFSNode(
                prev_actions=new_prev,
                location=update_location(location, action),
                running=running + (f"Thought: {thought}\nAction: {action}\n"
                                   f"Observation: {child_obs}\n"),
                on_gold=bool(node.on_gold and action == gold_next),
            ))

        # Expansion order (pred_set is sorted best-first):
        #   bfs (default): children go to the back — breadth-first, game24-style.
        #   dfs: children go to the FRONT, best-scored first — the search follows
        #        the top action deep and falls back to conformal-set siblings on
        #        dead ends. Needed for ScienceWorld's deep tasks (10-80 steps),
        #        where breadth at ~2-3 branching exhausts any budget by depth ~5.
        if getattr(args, 'dfs', False):
            frontier.extendleft(reversed(children))
        else:
            frontier.extend(children)

        sys.stdout.flush()
        if solved:
            break

    return (solved, best_score, best_traj, nodes, n_states, n_covered, win_len,
            n_cond_states, n_cond_covered)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def _open_resume(results_path):
    """Return {(sw_name, var)} already present in the JSONL."""
    seen = set()
    if not results_path or not os.path.exists(results_path):
        return seen
    with open(results_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                seen.add((rec.get('sw_name'), int(rec.get('var', -1))))
            except Exception:
                pass
    return seen


def parse_args():
    p = argparse.ArgumentParser(
        description="Game-of-24-style BFS conformal play — ScienceWorld")
    # conformal
    p.add_argument("--alpha",        type=float, default=0.1)
    p.add_argument("--method",       choices=["dual_lp", "quantile"], default="dual_lp")
    p.add_argument("--k",            type=int,   default=50)
    p.add_argument("--n_components", type=int,   default=10)
    # BFS
    p.add_argument("--n_samples",    type=int,   default=10)
    p.add_argument("--temperature",  type=float, default=0.7)
    p.add_argument("--score_batch_size", type=int, default=1)
    p.add_argument("--max_depth",    type=int,   default=50,
                   help="Max BFS depth (also capped by the per-task MPO step limit)")
    p.add_argument("--max_nodes",    type=int,   default=100,
                   help="Max BFS nodes expanded per episode")
    p.add_argument("--max_episodes", type=int,   default=100,
                   help="Cap on MPO test-set episodes (game24's --max_puzzles)")
    p.add_argument("--dfs", action="store_true", default=False,
                   help="depth-first expansion (best child first, conformal-set "
                        "siblings as backtrack points) instead of FIFO BFS")
    p.add_argument("--oracle_candidates", action="store_true", default=False,
                   help="union gold[t] into candidates at on-path nodes "
                        "(OFF by default: LLM-only pool; coverage is then "
                        "capped by the LLM's gold hit-rate)")
    # env
    p.add_argument("--env-step-limit", type=int, default=200)
    p.add_argument("--simplifications-preset", choices=list(PRESETS), default="paper")
    # io
    p.add_argument("--results", type=str, default=None,
                   help="JSONL output (default results/bfs_conformal_g24style_a<alpha>.jsonl)")
    p.add_argument("--resume",  action="store_true")
    # calibration split (must match run_conformal_calibration.py)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--seed",      type=int,   default=42)
    # artefacts
    p.add_argument("--data",       type=str, default=DATA_PATH)
    p.add_argument("--model",      type=str, default=MODEL_PATH)
    p.add_argument("--bert_cache", type=str, default=BERT_CACHE_PATH)
    p.add_argument("--base_url",   type=str, default=VLLM_BASE_URL)
    p.add_argument("--model_name", type=str, default=MODEL_NAME,
                   help="Served model id on the vLLM endpoint")
    p.add_argument("--verbose",    action="store_true")
    return p.parse_args()


def main():
    global client, MODEL_NAME
    args = parse_args()
    MODEL_NAME = args.model_name
    a_tag = str(args.alpha).replace('.', '')
    if args.results is None:
        args.results = os.path.join(_HERE, "results",
                                    f"bfs_conformal_g24style_a{a_tag}.jsonl")

    client = OpenAI(base_url=args.base_url, api_key="EMPTY",
                    timeout=120.0, max_retries=5)

    print("=" * 72)
    print(f"  Game-of-24-style BFS Conformal — ScienceWorld  ({MODEL_NAME})")
    print("=" * 72)
    print(f"  α            : {args.alpha}  (target coverage ≥ {(1-args.alpha)*100:.0f}%)")
    print(f"  method       : {args.method}   k={args.k}   n_comp={args.n_components}")
    print(f"  n_samples    : {args.n_samples}   temperature={args.temperature}")
    print(f"  max_depth    : {args.max_depth}   max_nodes={args.max_nodes}")
    print(f"  max_episodes : {args.max_episodes}")
    print(f"  oracle_cands : {args.oracle_candidates}")
    print(f"  results      : {args.results}")
    sys.stdout.flush()

    # ── Preflight: LLM endpoint ──────────────────────────────────────────────
    try:
        served = [m.id for m in client.models.list().data]
        print(f"  LLM endpoint : OK  (serving {served})")
    except Exception as ex:
        sys.exit(f"\nLLM endpoint unreachable at {args.base_url}\n"
                 f"  {type(ex).__name__}: {ex}")

    # ── BERT cache + score model ─────────────────────────────────────────────
    print(f"\nLoading BERT cache from {args.bert_cache} ...")
    bert_cache = BertEmbeddingCache(device=DEVICE)
    bert_cache.load(args.bert_cache)

    print(f"Loading score model from {args.model} ...")
    ckpt  = torch.load(args.model, map_location=DEVICE)
    model = ScoreFunction(d_proj=ckpt['d_proj'], hidden=ckpt['hidden']).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # ── Calibration pool from the HELD-OUT val split (no train leakage) ──────
    print(f"\nWarm-starting calibration pool from held-out val split of {args.data} ...")
    with open(args.data, 'rb') as f:
        training_data = pickle.load(f)
    records = build_records(training_data, bert_cache)
    random.seed(args.seed)
    indices = list(range(len(records)))
    random.shuffle(indices)
    n_val = int(len(records) * args.val_split)
    val_records = [records[i] for i in indices[:n_val]]
    print(f"  Total records: {len(records)}   Calibration records: {len(val_records)}"
          f"   (seed={args.seed}, val_split={args.val_split})")

    predictor = ConformalPredictor(model, bert_cache, alpha=args.alpha)
    predictor.calibrate_from_records(val_records, DEVICE)
    print(f"  Pool size: {len(predictor.cal_scores)}")
    sys.stdout.flush()

    # ── ScienceWorld env + MPO test set ──────────────────────────────────────
    print(f"\nInitialising ScienceWorld env (step_limit={args.env_step_limit}) ...")
    env = ScienceWorldEnv("", envStepLimit=args.env_step_limit)
    simpl_str = PRESETS[args.simplifications_preset]
    entries = load_mpo_testset(env)
    split_total = len(entries)
    if args.max_episodes > 0:
        entries = entries[:args.max_episodes]

    seen = _open_resume(args.results) if args.resume else set()
    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    out_f = open(args.results, 'a')

    print(f"\nPlaying {len(entries)} MPO test episodes  alpha={args.alpha}  "
          f"method={args.method}")
    n_solved = tot_states = tot_covered = 0
    tot_cond_states = tot_cond_covered = 0
    solved_flags, all_scores, all_nodes, perfect_win_lens = [], [], [], []

    for i, entry in enumerate(entries, 1):
        sw_name, var_idx = entry['sw_name'], entry['var']
        if (sw_name, var_idx) in seen:
            print(f"[{i}/{len(entries)}] {entry['mpo_name']} var={var_idx} — "
                  f"already in results, skipped")
            continue

        cap_steps = entry['max_steps'] or args.max_depth
        max_depth = min(cap_steps, args.max_depth)

        print("\n" + "#" * 72)
        print(f"  EPISODE {i}/{len(entries)}  {entry['mpo_name']}  var={var_idx}  "
              f"max_depth={max_depth}  max_nodes={args.max_nodes}")
        print("#" * 72)

        env.load(sw_name, var_idx, simpl_str, generateGoldPath=True)
        env.reset()
        task_desc = env.get_task_description().strip()
        gold = get_gold_sequence(env)
        print(f"  task description: {task_desc}")
        print(f"  gold sequence ({len(gold)}): {gold if args.verbose else '(hidden, use --verbose)'}")
        sys.stdout.flush()

        t0 = time.time()
        (solved, score, traj, nodes, n_states, n_cov, win_len,
         n_cond_states, n_cond_cov) = run_bfs_episode(
            env, predictor, sw_name, var_idx, simpl_str, task_desc, gold,
            args, max_depth=max_depth, max_nodes=args.max_nodes)
        dt = time.time() - t0

        n_solved += int(solved)
        tot_states += n_states
        tot_covered += n_cov
        tot_cond_states += n_cond_states
        tot_cond_covered += n_cond_cov
        solved_flags.append(int(solved))
        all_scores.append(score)
        all_nodes.append(nodes)
        if solved:
            perfect_win_lens.append(win_len)

        cov = (n_cov / n_states) if n_states else 0.0
        print(f"[{i}/{len(entries)}] task={entry['mpo_name']!r} var={var_idx}  "
              f"solved={solved}  score={score}  states={n_states}  nodes={nodes}  "
              f"coverage={cov:.2f}  success_rate={n_solved}/{len(solved_flags)}"
              f"={n_solved/len(solved_flags):.3f}  ({dt:.1f}s)")
        sys.stdout.flush()

        out_f.write(json.dumps({
            'idx': i, 'mpo_name': entry['mpo_name'], 'sw_name': sw_name,
            'var': var_idx, 'score': int(score), 'won': bool(solved),
            'trajectory': traj, 'n_nodes': nodes, 'n_states': n_states,
            'n_covered': n_cov, 'n_cond_states': n_cond_states,
            'n_cond_covered': n_cond_cov, 'win_len': win_len, 'wall_time': dt,
        }) + "\n")
        out_f.flush()

    out_f.close()

    # ── FINAL RESULTS (game24-style) ─────────────────────────────────────────
    N = len(solved_flags)
    succ = (sum(solved_flags) / N) if N else 0.0
    avg_nodes = (sum(all_nodes) / N) if N else 0.0
    avg_win = (sum(perfect_win_lens) / len(perfect_win_lens)) if perfect_win_lens else None
    agg = aggregate(all_scores, [bool(s) for s in solved_flags]) if N else None
    print("\n" + "=" * 72)
    print(f"  FINAL RESULTS  ({N} test episodes, alpha={args.alpha})")
    print(f"  Episodes run            : {N}  (of {split_total} in MPO test set, "
          f"capped at {args.max_episodes})")
    print(f"  Success rate (score=100): {sum(solved_flags)}/{N} = {succ:.3f}")
    if agg:
        print(f"  Avg reward (env score)  : {agg['avg_reward']:.2f}")
    print(f"  Coverage (unconditional): {tot_covered}/{tot_states} = "
          f"{(tot_covered/tot_states if tot_states else 0):.3f}  "
          f"(all on-gold states; LLM-miss counted as miss)")
    print(f"  Coverage (conditional)  : {tot_cond_covered}/{tot_cond_states} = "
          f"{(tot_cond_covered/tot_cond_states if tot_cond_states else 0):.3f}  "
          f"(gold-in-candidates only; target >= {1-args.alpha:.2f})")
    print(f"  Avg BFS nodes / episode : {avg_nodes:.2f}")
    if avg_win is not None:
        print(f"  Avg winning path length : {avg_win:.2f}")

    _print_dist("SEARCH NODES (all episodes)", all_nodes)
    _print_dist("WINNING PATH LENGTH (solved episodes)", perfect_win_lens)
    print("=" * 72)

    print("\nGenerating histograms ...")
    _plot_dist("search nodes (all)", all_nodes,
               os.path.join(_HERE, f"traj_len_hist_conformal_all_sciworld_a{a_tag}.png"),
               "#4C72B0", xlabel="BFS nodes explored", succ=succ)
    _plot_dist("winning path length (solved)", perfect_win_lens,
               os.path.join(_HERE, f"traj_len_hist_conformal_perfect_sciworld_a{a_tag}.png"),
               "#55A868", xlabel="Winning path length (steps)", succ=succ)


if __name__ == "__main__":
    main()
