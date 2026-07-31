"""
Game-of-24 training-data generation (mirrors the ScienceWorld / WebShop pipeline).

For each puzzle we let the LLM PLAY (sample candidate actions at every state and
echo-score their token log-probs), and at each visited state we record a training
example in the SAME schema as data_generation_sciworld.py:

    {
        'task_name'         : "1 2 3 4",          # the original puzzle
        'prev_actions'      : ["1 + 2 = 3", ...], # actions committed so far
        'location'          : "3 3 4",            # the STATE = numbers left
        'state_numbers'     : ["3", "3", "4"],
        'admissible_actions': [...],              # LLM samples  ∪  oracle-correct moves
        'log_probs_of_admissible_actions': {a: [tok_logprobs]},
        'oracle_labels'     : {a: 'correct'|'dead_end'|'illegal'|'wrong_arithmetic'},
        'correct_actions'   : [...],              # COMPLETE set of solvability-preserving
                                                  #   moves from this state (the ground truth)
        'optimal_action'    : "6 * 4 = 24",       # one correct action (convenience label)
        'solvable'          : True,
        'step'              : t,
    }

Unlike ScienceWorld (no oracle -> bushwhack to find correct actions), Game of 24
has a perfect oracle: `correct_actions(state)` returns EVERY solvability-preserving
move directly, so `correct_actions` here is complete and exact.

Two commit policies decide which action advances the episode:
    --policy oracle : teacher-force along an optimal path (every recorded state is
                      solvable, so positives are guaranteed). Mirrors gold-forcing.
    --policy llm    : commit the LLM's own greedy (max-logprob) move -- a true
                      rollout that may walk into dead ends (recorded as such).

Run:
    python3 g24_data_generation.py --num_samples 20 --seed 42 \
        --n_samples 10 --policy oracle --out training_data_game24.pkl
"""
import os
import re
import sys
import time
import json
import pickle
import argparse

import numpy as np
import pandas as pd
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError

from g24_oracle import (
    to_state, fmt, solvable, correct_actions, label_action,
)

# ── vLLM endpoint (same as the rest of the pipeline) ───────────────────────── #
MODEL_NAME = "Qwen/Qwen3-8B"
VLLM_BASE_URL = "http://10.5.18.73:8002/v1"

CSV_PATH = '24.csv'
FOLDER = '../../prompts/game24/'   # prompts live at GA-Rollback/prompts/game24/
REACT_FILE = 'game24_base_react.txt'


_TRANSIENT_EXC = (APIConnectionError, APITimeoutError, RateLimitError)


def _retry(fn, *args, _label="call", _tries=4, _backoff=2.0, **kwargs):
    last = None
    for i in range(_tries):
        try:
            return fn(*args, **kwargs)
        except _TRANSIENT_EXC as ex:
            last = ex
            wait = _backoff ** i
            print(f"    [{_label}] transient {type(ex).__name__} "
                  f"(try {i+1}/{_tries}); sleeping {wait:.1f}s", flush=True)
            time.sleep(wait)
    raise last


# ═══════════════════════════════════════════════════════════════════════════ #
# LLM helpers (candidate sampling + echo-scored log-probs)
# ═══════════════════════════════════════════════════════════════════════════ #
def parse_action(text: str) -> str:
    """First non-empty line of a completion is the action (e.g. '1 + 2 = 3')."""
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def sample_candidate_actions(client, sampling_prompt, n, temperature, max_tokens=64):
    """Sample n completions; return deduped list of action strings (order kept)."""
    resp = _retry(
        client.completions.create, _label="cand",
        model=MODEL_NAME, prompt=sampling_prompt, n=n,
        max_tokens=max_tokens, temperature=temperature, stop=['\n'], echo=False,
    )
    cands, seen = [], set()
    for choice in resp.choices:
        a = parse_action(choice.text)
        if a and a not in seen:
            seen.add(a)
            cands.append(a)
    return cands


def get_action_logprobs(client, scoring_prompt, actions, batch_size=1):
    """
    Echo-score each action under the LLM (no generation): {action: [tok_logprobs]}.
    `scoring_prompt` ends with 'Act i>' so that scoring_prompt + ' ' + action
    reconstructs the natural continuation '...Act i> 1 + 2 = 3'.
    """
    if not actions:
        return {}
    probe = _retry(
        client.completions.create, _label="probe",
        model=MODEL_NAME, prompt=scoring_prompt, max_tokens=1, echo=True, logprobs=1,
    )
    n_prompt = len(probe.choices[0].logprobs.tokens) - 1

    results = {}
    for i in range(0, len(actions), max(1, batch_size)):
        chunk = actions[i:i + max(1, batch_size)]
        full = [scoring_prompt + " " + a for a in chunk]
        resp = _retry(
            client.completions.create, _label="score",
            model=MODEL_NAME, prompt=full, max_tokens=0, echo=True, logprobs=1,
        )
        for choice, a in zip(resp.choices, chunk):
            lps = [lp for lp in choice.logprobs.token_logprobs[n_prompt:]
                   if lp is not None]
            results[a] = lps
    return results


def _mean(xs):
    return sum(xs) / len(xs) if xs else float('-inf')


# ═══════════════════════════════════════════════════════════════════════════ #
# PER-PUZZLE EPISODE
# ═══════════════════════════════════════════════════════════════════════════ #
def generate_for_puzzle(client, puzzle, base_prompt, args):
    """Play one puzzle, recording a training state at every visited state."""
    state = list(to_state(puzzle))
    prev_actions = []
    # running buffer ends with 'Act i>' (no trailing space; helpers add it)
    init_prefix = f"{base_prompt}\n# Here is the task:\nInput: {puzzle}\nAct 1>"
    running = init_prefix

    records = []
    t = 0
    note = "ran out of steps"
    while len(state) > 1 and t < args.max_steps:
        state_str = ' '.join(fmt(x) for x in state)
        is_solvable = solvable(tuple(state))
        correct_set = correct_actions(state)        # COMPLETE ground-truth set

        # ── LLM plays: sample candidates at this state ──────────────────── #
        sampling_prompt = running + " "
        cand = sample_candidate_actions(
            client, sampling_prompt, n=args.n_samples, temperature=args.temperature,
        )
        # union with oracle-correct moves so positives are always present
        candidates = list(dict.fromkeys(cand + correct_set))

        # ── echo-score every candidate ─────────────────────────────────── #
        logprobs = get_action_logprobs(
            client, running, candidates, batch_size=args.score_batch_size,
        )
        labels = {a: label_action(state, a)['label'] for a in candidates}

        records.append({
            'task_name':          puzzle,
            'prev_actions':       list(prev_actions),
            'location':           state_str,
            'state_numbers':      [fmt(x) for x in state],
            'admissible_actions': list(candidates),
            'log_probs_of_admissible_actions':
                {a: logprobs.get(a, []) for a in candidates},
            'oracle_labels':      labels,
            'correct_actions':    list(correct_set),
            'optimal_action':     correct_set[0] if correct_set else None,
            'solvable':           is_solvable,
            'step':               t,
            'split':              args.split,
        })

        if args.verbose:
            print(f"    step {t:2d}  loc={state_str!r}  cands={len(candidates)}  "
                  f"#correct={len(correct_set)}  llm_sampled={len(cand)}", flush=True)

        # ── choose the action to COMMIT (advance the episode) ───────────── #
        if args.policy == 'oracle' and correct_set:
            commit = correct_set[0]
        else:  # llm greedy: highest mean-logprob among the LLM's own samples
            scored = [(a, _mean(logprobs.get(a, []))) for a in cand] or \
                     [(a, _mean(logprobs.get(a, []))) for a in candidates]
            commit = max(scored, key=lambda kv: kv[1])[0] if scored else None

        if commit is None:
            note = "no action to commit"
            break

        lab = label_action(state, commit)
        if lab['label'] not in ('correct', 'dead_end'):
            note = f"committed {lab['label']} action: {commit!r}"
            break
        state = list(to_state(lab['child']))
        obs = 'numbers left: ' + ' '.join(fmt(x) for x in state)
        running += f" {commit}\nObs {t+1}> {obs}\nAct {t+2}>"
        prev_actions.append(commit)
        t += 1

    solved = (len(state) == 1 and state[0] == 24)
    if solved:
        note = None
    return records, solved, note


# ═══════════════════════════════════════════════════════════════════════════ #
# MAIN
# ═══════════════════════════════════════════════════════════════════════════ #
def select_puzzles(args):
    """Pick the puzzles to generate data for.

    --split {train,cal,test} reads the disjoint puzzle list from --splits_file
    (produced by g24_splits.py) so training data never overlaps test puzzles.
    --split all falls back to the seeded sample (no train/test separation).
    """
    if args.split != 'all':
        with open(args.splits_file) as f:
            splits = json.load(f)
        puzzles = splits[args.split]
        if args.num_samples and 0 < args.num_samples < len(puzzles):
            puzzles = puzzles[:args.num_samples]   # split is pre-shuffled
        return puzzles
    data = pd.read_csv(CSV_PATH)['Puzzles']
    np.random.seed(args.seed)
    return list(data.sample(frac=1, random_state=args.seed).head(args.num_samples))


def parse_args():
    p = argparse.ArgumentParser(description="Game-of-24 conformal training-data generation")
    p.add_argument("--num_samples", type=int, default=20, help="number of puzzles")
    p.add_argument("--seed", type=int, default=42, help="puzzle-sampling seed")
    p.add_argument("--n_samples", type=int, default=10, help="LLM candidate samples per state")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--policy", choices=['oracle', 'llm'], default='oracle',
                   help="oracle = teacher-force optimal path; llm = greedy rollout")
    p.add_argument("--max_steps", type=int, default=6)
    p.add_argument("--score_batch_size", type=int, default=1)
    p.add_argument("--model_name", type=str, default=MODEL_NAME)
    p.add_argument("--base_url", type=str, default=VLLM_BASE_URL)
    p.add_argument("--split", choices=['all', 'train', 'cal', 'test'], default='all',
                   help="which disjoint puzzle split to generate (from --splits_file)")
    p.add_argument("--splits_file", type=str, default="game24_splits.json")
    p.add_argument("--out", type=str, default="training_data_game24.pkl")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key="EMPTY",
                    timeout=120.0, max_retries=5)

    prompt_file = REACT_FILE 
    with open(os.path.join(FOLDER, prompt_file)) as f:
        base_prompt = f.read()

    print("=" * 72)
    print("  Game-of-24 training-data generation")
    print("=" * 72)
    print(f"  policy      : {args.policy}")
    print(f"  n_samples   : {args.n_samples}   temperature: {args.temperature}")
    print(f"  out         : {args.out}")
    try:
        served = [m.id for m in client.models.list().data]
        print(f"  LLM endpoint: OK  (serving {served})")
    except Exception as ex:
        sys.exit(f"\nLLM endpoint unreachable at {args.base_url}: {type(ex).__name__}: {ex}")

    puzzles = select_puzzles(args)
    end = len(puzzles) if args.end < 0 else min(args.end, len(puzzles))
    puzzles = puzzles[args.start:end]
    print(f"  puzzles     : {len(puzzles)}  (range {args.start}:{end})")

    training_data = {}
    n_states = n_solved = 0
    for i, puzzle in enumerate(puzzles, start=1):
        t0 = time.time()
        print(f"\n[{i}/{len(puzzles)}] puzzle={puzzle!r}")
        sys.stdout.flush()
        try:
            records, solved, note = generate_for_puzzle(client, puzzle, base_prompt, args)
        except Exception as ex:
            print(f"  CRASH: {type(ex).__name__}: {ex}")
            continue

        training_data[f"{puzzle}__seed{args.seed}"] = records
        n_states += len(records)
        n_solved += int(solved)
        print(f"  states={len(records)}  solved={solved}  "
              f"note={note or 'ok'}  {time.time()-t0:.1f}s")

        if i % args.save_every == 0:
            with open(args.out, 'wb') as f:
                pickle.dump(training_data, f)

    with open(args.out, 'wb') as f:
        pickle.dump(training_data, f)

    print("\n" + "=" * 72)
    print(f"  DONE — puzzles={len(puzzles)}  solved={n_solved}  "
          f"recorded states={n_states}")
    print(f"  saved -> {args.out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
