"""
Online conformal-prediction play for Game of 24 (the full pipeline endpoint).

For each HELD-OUT TEST puzzle (game24_splits.json["test"]) we run a
conformal-pruned BFS over the action tree:

  At each state:
    1. admissible_actions = ALL legal moves (oracle _child_moves) — guarantees
       the correct actions are in the candidate pool.
    2. echo-score each action's token log-probs in the running ReAct context
       (vLLM) -> 10-bin softmax distribution (the score function's features).
    3. pred_set = predictor.get_prediction_set(task, prev_actions, location,
                  admissible, softmax_values)   # conformal set
    4. SET-COVERAGE metric (patch #3):
           covered = (pred_set  ∩  correct_actions) != ∅
    5. expand children ONLY for actions in pred_set (conformal pruning) and keep
       searching for a leaf that equals 24.

Per state we emit a machine-parseable line for the plotter:
    [STATEMETRIC] set_size_pct=.. covered=0/1

Run:
    python conf_react_game24.py --alpha 0.1 --split test \
        --splits_file game24_splits.json --max_puzzles 50 > bfs_game24_a0.1.txt
"""
import os
import sys
import json
import re
import argparse
import collections
from collections import deque

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openai import OpenAI

from g24_oracle import to_state, fmt, solvable, correct_actions, _child_moves, label_action
from g24_compute_softmax_bins import logprobs_to_softmax_bins
from g24_data_generation import (
    get_action_logprobs, sample_candidate_actions, parse_action,
    MODEL_NAME, VLLM_BASE_URL,
)
from score_model import ScoreFunction, BertEmbeddingCache
from conformal_predictor import ConformalPredictor

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts') + os.sep
REACT_FILE = 'game24_base_react.txt'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_running(puzzle, prev_steps, base_prompt):
    """Reconstruct the running ReAct prompt (ending 'Act i>') and the current
    oracle state after applying prev_steps.

    prev_steps is a list of (thought, action) pairs. `thought` is the raw
    visible ReAct thought generated at that node ('' / None if force_think was
    off); it is replayed as a 'think:' step so the reasoning history PERSISTS
    down the trajectory (parent thoughts stay in the child's prompt).
    Returns (running, state, next_act), where next_act is the 'Act N>' number
    the running currently ends with — i.e. the slot the next action fills."""
    running = f"{base_prompt}\n# Here is the task:\nInput: {puzzle}\nAct 1>"
    state = list(to_state(puzzle))
    act = 1
    for thought, a in prev_steps:
        if thought:                                  # replay this node's thought
            running += f" think: {thought}\nObs {act}> OK.\nAct {act + 1}>"
            act += 1
        lab = label_action(state, a)
        state = list(to_state(lab['child']))
        obs = 'numbers left: ' + ' '.join(fmt(x) for x in state)
        running += f" {a}\nObs {act}> {obs}\nAct {act + 1}>"
        act += 1
    return running, state, act


def _remove_duplicate_think_prefix(text):
    """Strip a leading 'think:'/'Thought:'/'Plan:' the model may re-emit
    when we already primed the prompt with 'Act k> think:'."""
    text = text.strip()
    text = re.sub(r"^think\s*:\s*", "", text, flags=re.I).strip()
    text = re.sub(r"^Thought\s*:\s*", "", text, flags=re.I).strip()
    text = re.sub(r"^Plan\s*:\s*", "", text, flags=re.I).strip()
    return text


def generate_think(client, running, temperature=0.1, max_tokens=100):
    """Force one visible ReAct thought by priming the running context with
    'think:'. Returns the thought body (without the 'think:' prefix)."""
    resp = client.completions.create(
        model=MODEL_NAME, prompt=running + " think:", max_tokens=max_tokens,
        temperature=temperature, top_p=1, stop=['\n'], echo=False,
    )
    thought = _remove_duplicate_think_prefix(parse_action(resp.choices[0].text))
    if not thought:
        thought = "I will choose a valid operation that keeps the puzzle solvable."
    return thought


def natural_think(client, running, temperature=0.7, max_tokens=100):
    """Sample the next ReAct line WITHOUT priming 'think:'. If the model
    naturally emits a visible thought, return its body (prefix stripped);
    otherwise return '' — no forced, no canned fallback. This lets the think
    arise from the LLM exactly like it does in the linear react rollout: the
    model chooses whether to think or act, and we only keep it if it thought."""
    resp = client.completions.create(
        model=MODEL_NAME, prompt=running + " ", max_tokens=max_tokens,
        temperature=temperature, top_p=1, stop=['\n'], echo=False,
    )
    line = parse_action(resp.choices[0].text).strip()
    if line.lower().startswith(('think', 'thought', 'plan')):
        return _remove_duplicate_think_prefix(line)
    return ''                                   # model chose to act -> no think


def _print_dist(label, values):
    """Print a count/% distribution table (WebShop-style)."""
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
    """Save a histogram of `values` (WebShop-style bars with count/% labels)."""
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
    ax.set_title(f"Game of 24 — {label}  ({total} puzzles)\n"
                 f"success rate: {succ*100:.1f}%")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of puzzles")
    ax.set_xticks(xs)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved -> {out_path}")


def play_puzzle(puzzle, predictor, client, base_prompt, args):
    """Conformal-pruned BFS.
    Returns (solved, n_states, n_covered, nodes, win_len) where
      nodes   = BFS nodes scored for this puzzle (search effort), and
      win_len = length of the winning action path (steps to reach 24), 0 if unsolved.
    """
    n_states = 0
    n_covered = 0
    solved = False
    nodes = 0
    win_len = 0

    # BFS frontier of (prev_actions,) ; child states recomputed from oracle
    frontier = deque()
    frontier.append([])
    visited = set()

    while frontier and nodes < args.max_nodes:
        prev_steps = frontier.popleft()
        prev_actions = [a for _, a in prev_steps]    # arithmetic-only view
        key = tuple(prev_actions)                    # dedupe on the state (actions)
        if key in visited:
            continue
        visited.add(key)

        running, state, slot = build_running(puzzle, prev_steps, base_prompt)
        # print(f'\nRunning: ', running)
        # print("\nState: ", state)
        # input("Press Enter to continue......")
        
        # leaf?
        if len(state) == 1:
            if state[0] == 24:
                solved = True
            continue
        if len(prev_actions) >= args.max_depth:
            continue

        loc_str = ' '.join(fmt(x) for x in state)
        correct = set(correct_actions(state))   # oracle ground truth (for coverage)

        # Unsolvable state (reached by expanding a dead-end action): there is NO
        # correct action here, so coverage is undefined. Calibration was done only
        # on solvable states, so don't score/count it — just prune this branch.
        if not correct:
            if args.verbose:
                print(f"\n[skip] unsolvable state {loc_str} via {prev_actions} — pruned")
            continue

        nodes += 1
        print(f"\n[BFS Node {nodes}]  depth={len(prev_actions)}")
        print(f"  Task     : {puzzle}")
        print(f"  Traj     : {prev_actions if prev_actions else '(start)'}")
        print(f"  Location : {loc_str}")

        # ── Visible ReAct thought (forced default; natural via --think_mode) ──
        # `running` ends 'Act {slot}>'.
        #   forced : prime 'think:' so the model always emits a thought (canned
        #            fallback if empty) — every node gets one.
        #   natural: sample the next line freely and keep it only if the model
        #            actually thought (no priming, no fallback).
        # Either way a kept thought is folded into `running` so the candidate
        # sampling and echo-scoring below condition on it — but the thought is
        # never itself scored.
        forced_think = None
        node_thought = ""                            # raw thought stored in the frontier
        if args.force_think:
            if args.think_mode == 'natural':
                node_thought = natural_think(client, running,
                                             temperature=args.think_temperature)
            else:
                node_thought = generate_think(client, running,
                                              temperature=args.think_temperature)
            if node_thought:                         # a thought was produced / forced
                forced_think = f"think: {node_thought}"
                running += f" {forced_think}\nObs {slot}> OK.\nAct {slot + 1}>"
                print(f"  [think]  {forced_think}")
            else:                                    # natural mode: LLM acted directly
                print("  [think]  (none — LLM proposed an action directly)")

        # ── LLM samples candidate actions at this state ───────────────────────
        print("  ┌─ TRAJECTORY PROMPT → LLM ─────────────────────────────────")
        for _line in (running + " ").splitlines():
            print(f"  │ {_line}")
        print("  └───────────────────────────────────────────────────────────")
        sys.stdout.flush()
        sampled = sample_candidate_actions(client, running + " ",
                                           n=args.n_samples, temperature=args.temperature)
        print(f"  [LLM sampled actions] {sampled}")

        # Split the LLM samples into reflection / think / answer text vs. REAL
        # actions. Only genuine arithmetic moves become candidates, so the
        # prediction set (and hence what gets scored) can never contain a
        # reflection, think tag, or answer string — just the actual actions.
        # Seed the think list with the forced thought so it is counted/shown in
        # `Think (N):` (it is generated separately above, not among `sampled`).
        think_actions = [forced_think] if forced_think else []
        real_actions = []
        for a in sampled:
            low = a.strip().lower()
            
            if low.startswith(('think', 'answer')):
                think_actions.append(a)
            # skip anything that is a reflection / think / answer tag; these
            # are never scored regardless of what text follows them.
            if low.startswith(('reflection', 'think', 'answer')):
                continue
            lab = label_action(state, a)
            if lab['kind'] == 'intermediate' and lab['label'] in ('correct', 'dead_end'):
                if a not in real_actions:
                    real_actions.append(a)
        print(f"  Think ({len(think_actions)}): {think_actions}")
        print(f"  Real  ({len(real_actions)}):  {real_actions}")
        print(f"  All LLM actions ({len(sampled)}): {sampled}")
        print(f"  All correct actions ({len(correct)}): {sorted(correct)}")

        # Candidate pool. Calibration force-added the oracle-correct set into
        # `admissible`, so to keep the test pool consistent (and let coverage be
        # achievable) we union the LLM's real actions with the oracle-correct
        # moves. --no_oracle_candidates reverts to the LLM-only pool (realistic,
        # but coverage is then capped by the LLM's proposal hit-rate).
        if args.oracle_candidates:
            candidates = list(dict.fromkeys(real_actions + list(correct)))
            n_added = len(set(correct) - set(real_actions))
            print(f"  Candidates ({len(candidates)}): {len(real_actions)} LLM "
                  f"+ {n_added} oracle-correct added")
        else:
            candidates = list(real_actions)

        if not candidates:
            print("  [skip] No candidate actions.")
            n_states += 1                       # count as an (uncovered) state
            print(f"[STATEMETRIC] set_size_pct=0.00 covered=0")
            sys.stdout.flush()
            continue

        # echo-score candidates -> 10-bin softmax distributions
        logprobs = get_action_logprobs(client, running, candidates,
                                       batch_size=args.score_batch_size)
        softmax_values = {a: logprobs_to_softmax_bins(logprobs.get(a, []), 10)
                          for a in candidates}

        # conformal prediction set
        pred_set, scores, S_star, n_sel, _ = predictor.get_prediction_set(
            task=puzzle, previous_actions=prev_actions, location=loc_str,
            admissible_actions=candidates, softmax_values=softmax_values,
            method=args.method, select='topk', k=args.k,
            n_components=args.n_components,
        )

        if not pred_set:
            pred_set = [min(scores, key=scores.__getitem__)]
            print(f"  [fallback] Empty pred set → best-score: {pred_set}")
        else:
            print(f"  Pred set ({len(pred_set)}/{len(candidates)}) "
                  f"using {n_sel} calibration states: {pred_set}")

        print("  Scores [lower = more likely optimal]:")
        for a in sorted(scores, key=scores.__getitem__):
            ctag = " (correct)" if a in correct else ""
            tag = " [IN SET]" if a in pred_set else ""
            print(f"    {scores[a]:.4f}  {a}{ctag}{tag}")

        covered = len(set(pred_set) & correct) > 0
        set_pct = 100.0 * len(pred_set) / max(1, len(candidates))
        n_states += 1
        n_covered += int(covered)

        # ── ONLINE CALIBRATION ────────────────────────────────────────────────
        # Fold this (off-distribution) search state into the pool, labeled by its
        # best (lowest-score) correct action — the same set-coverage convention as
        # run_conformal_calibration (min nonconformity over the correct set). The
        # coverage measured just above used the pool BEFORE this add, so the metric
        # is not self-leaking. Pool accumulates across puzzles, so later states get
        # a pool increasingly matched to the BFS search distribution.
        if getattr(args, 'online_calib', False):
            correct_in_cands = [a for a in candidates if a in scores and a in correct]
            if correct_in_cands:
                best_correct = min(correct_in_cands, key=lambda a: scores[a])
                predictor.commit_step(best_correct)

        # execute / expand each action in the conformal set
        for a in pred_set:
            child = list(to_state(label_action(state, a)['child']))
            done = (len(child) == 1)
            win = done and child[0] == 24
            reward = 1.0 if win else 0.0
            print(f"    → {a!r}   reward={reward:.4f}   done={done}")
            if win:
                solved = True
                if not win_len:                       # first (shortest) winning path
                    win_len = len(prev_actions) + 1
                print(f"  *** reached 24 via {prev_actions + [a]} ***")
            if not done:
                # carry this node's thought forward so the child prompt keeps
                # the full reasoning history, not just the arithmetic actions.
                frontier.append(prev_steps + [(node_thought, a)])

        print(f"[STATEMETRIC] set_size_pct={set_pct:.2f} covered={int(covered)}")
        sys.stdout.flush()

    return solved, n_states, n_covered, nodes, win_len


def main():
    ap = argparse.ArgumentParser(description="Game-of-24 online conformal BFS")
    ap.add_argument("--alpha", type=float, default=0.1, help="label only; threshold is baked into the loaded pool")
    # Default to the fixed 200-task set saved from sample.txt so the same exact
    # puzzles are played on every run (override with --splits_file/--split).
    ap.add_argument("--splits_file", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_test_tasks.json"))
    ap.add_argument("--split", type=str, default="react_test")
    ap.add_argument("--scores", type=str, default="calibrated_models/conformal_scores.pkl")
    ap.add_argument("--model", type=str, default="trained_models/best_score_model.pt")
    ap.add_argument("--bert_cache", type=str, default="trained_models/bert_cache.pkl")
    ap.add_argument("--method", choices=['quantile', 'dual_lp'], default='dual_lp')
    ap.add_argument("--n_samples", type=int, default=10, help="LLM action samples per node")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--oracle_candidates", action="store_true", default=True,
                    help="union LLM samples with oracle-correct moves (matches calibration; default)")
    ap.add_argument("--no_oracle_candidates", dest="oracle_candidates", action="store_false",
                    help="LLM-only candidate pool (realistic; coverage capped by LLM hit-rate)")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--n_components", type=int, default=10)
    ap.add_argument("--max_depth", type=int, default=10)
    ap.add_argument("--max_nodes", type=int, default=100)
    ap.add_argument("--max_puzzles", type=int, default=100)
    ap.add_argument("--score_batch_size", type=int, default=1)
    ap.add_argument("--base_url", type=str, default=VLLM_BASE_URL)
    ap.add_argument("--force_think", action="store_true", default=True,
                    help="include a 'think:' step before actions; kept in the trajectory but never scored (default)")
    ap.add_argument("--no_force_think", dest="force_think", action="store_false",
                    help="disable the think step entirely (action-only sampling)")
    ap.add_argument("--think_mode", choices=['forced', 'natural'], default='natural',
                    help="'forced' (default): prime 'think:' every node with a canned fallback; "
                         "'natural': let the LLM decide, keep a think only if it emits one")
    ap.add_argument("--think_temperature", type=float, default=0.7,
                    help="sampling temperature for the think line (natural mode often wants higher, e.g. 0.7)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--online_calib", action="store_true", default=False,
                    help="ONLINE CALIBRATION: fold each scored solvable search state "
                         "into the calibration pool (labeled by its best/lowest-score "
                         "correct action, matching the set-coverage convention). Grows "
                         "the pool with search-distribution states as BFS drifts "
                         "off-distribution — attacks the calibration↔BFS-test shift.")
    ap.add_argument("--no_online_calib", dest="online_calib", action="store_false")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=120.0, max_retries=5)
    with open(os.path.join(FOLDER, REACT_FILE)) as f:
        base_prompt = f.read()
    with open(args.splits_file) as f:
        puzzles = json.load(f)[args.split]
    split_total = len(puzzles)                       # full split size before capping
    if args.max_puzzles > 0:
        puzzles = puzzles[:args.max_puzzles]

    # load score model + calibrated pool
    ckpt = torch.load(args.model, map_location=DEVICE)
    model = ScoreFunction(d_proj=ckpt['d_proj'], hidden=ckpt['hidden']).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    bert_cache = BertEmbeddingCache(device=DEVICE)
    bert_cache.load(args.bert_cache)
    predictor = ConformalPredictor(model, bert_cache, alpha=args.alpha)
    predictor.load(args.scores)
    # load() restores the pool's baked alpha; re-apply the CLI alpha so it is a
    # LIVE knob. Calibration scores are alpha-independent (alpha only sets the
    # conformal quantile level), so no recalibration is needed to change it.
    predictor.alpha = args.alpha

    print(f"Playing {len(puzzles)} TEST puzzles  alpha={args.alpha}  method={args.method}")
    n_solved = tot_states = tot_covered = 0
    solved_flags = []                 # 1/0 per puzzle (success)
    all_nodes = []                    # BFS nodes scored per puzzle (search effort)
    perfect_win_lens = []             # winning path length per solved puzzle
    for i, puzzle in enumerate(puzzles, 1):
        solved, n_states, n_cov, nodes, win_len = play_puzzle(
            puzzle, predictor, client, base_prompt, args)
        n_solved += int(solved)
        tot_states += n_states
        tot_covered += n_cov
        solved_flags.append(int(solved))
        all_nodes.append(nodes)
        if solved:
            perfect_win_lens.append(win_len)
        cov = (n_cov / n_states) if n_states else 0.0
        success_rate = n_solved / i
        print(f"[{i}/{len(puzzles)}] puzzle={puzzle!r}  solved={solved}  "
              f"states={n_states}  nodes={nodes}  coverage={cov:.2f}  "
              f"success_rate={n_solved}/{i}={success_rate:.3f}")
        sys.stdout.flush()

    # ── FINAL RESULTS ──────────────────────────────────────────────────── #
    N = len(solved_flags)
    succ = (sum(solved_flags) / N) if N else 0.0
    avg_nodes = (sum(all_nodes) / N) if N else 0.0
    avg_win = (sum(perfect_win_lens) / len(perfect_win_lens)) if perfect_win_lens else None
    print("\n" + "=" * 72)
    print(f"  FINAL RESULTS  ({N} test puzzles, alpha={args.alpha})")
    print(f"  Puzzles run             : {N}  (of {split_total} in '{args.split}' split"
          f"{f', capped at {args.max_puzzles}' if args.max_puzzles > 0 else ''})")
    print(f"  Success rate (solved)   : {sum(solved_flags)}/{N} = {succ:.3f}")
    print(f"  State coverage          : {tot_covered}/{tot_states} = "
          f"{(tot_covered/tot_states if tot_states else 0):.3f}  (target >= {1-args.alpha:.2f})")
    print(f"  Avg BFS nodes / puzzle  : {avg_nodes:.2f}")
    if avg_win is not None:
        print(f"  Avg winning path length : {avg_win:.2f}")

    _print_dist("SEARCH NODES (all puzzles)", all_nodes)
    _print_dist("WINNING PATH LENGTH (solved puzzles)", perfect_win_lens)
    print("=" * 72)

    print("\nGenerating histograms ...")
    a = str(args.alpha).replace('.', '')
    otag = "_online" if args.online_calib else ""
    _plot_dist("search nodes (all)", all_nodes,
               f"traj_len_hist_conformal_all_game24_a{a}{otag}.png", "#4C72B0",
               xlabel="BFS nodes explored", succ=succ)
    _plot_dist("winning path length (solved)", perfect_win_lens,
               f"traj_len_hist_conformal_perfect_game24_a{a}{otag}.png", "#55A868",
               xlabel="Winning path length (steps)", succ=succ)


if __name__ == "__main__":
    main()
