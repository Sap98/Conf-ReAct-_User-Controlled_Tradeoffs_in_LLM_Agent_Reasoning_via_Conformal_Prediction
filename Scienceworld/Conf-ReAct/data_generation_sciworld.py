"""
Stage 2 — Training-data generation for the ScienceWorld conformal pipeline.

Analog of ALFWorld's `data_generation_for_scr_func.py`, but adapted to
ScienceWorld's three differences (see PLAN.md):

  * No per-state replanner. The optimal action is obtained ONLY via
    gold-path teacher forcing: we roll the env forward along
    `env.get_gold_action_sequence()` and, at the gold prefix of length t, the
    optimal action is gold[t]. Every recorded state is therefore an
    on-gold-path state.

  * `get_valid_action_object_combinations()` is huge, so we do NOT logprob-score
    all of it. Candidates come from LLM sampling (WebShop style): sample n
    completions, parse one action each, dedupe — then UNION with gold[t] so the
    positive label is guaranteed present in the candidate set.

  * Candidate log-probs are obtained by echo-scoring each candidate appended to
    a prompt that ends with "Action:" (think-free), mirroring the play-time
    protocol in bfs_conformal_alfworld.py:get_action_logprobs so that the
    softmax-bin feature is computed identically at train and play time.

Output: a pickle keyed by f"{task}__var{var}" -> list of per-state dicts with
schema:
    {task_name, prev_actions, location, admissible_actions,
     log_probs_of_admissible_actions: {action: [token_logprobs]},
     optimal_action}
Stage 3 (`add_optimal_and_bins_sciworld.py`) converts the raw token-logprobs
into 10-bin softmax histograms (`softmax_bin_distributions`) and drops the raw
logprobs, producing the schema that train_score.py / build_records expects.

Usage:
    # One task, its first 20 variations
    python data_generation_sciworld.py --task-num 13 --num-variations 20

    # All 30 tasks, 10 variations each
    python data_generation_sciworld.py --task-num -1 --num-variations 10
"""

import os
import sys
import math
import pickle
import argparse

# ── Make sibling modules importable regardless of CWD ────────────────────────
_HERE  = os.path.dirname(os.path.abspath(__file__))   # .../react/conformal_prediction
_REACT = os.path.dirname(_HERE)                        # .../react
_REPO  = os.path.dirname(_REACT)                       # .../ScienceWorld
sys.path.insert(0, _REACT)
sys.path.insert(0, os.path.join(_REPO, "examples"))

from openai import OpenAI
from scienceworld import ScienceWorldEnv

# Reuse the exact prompt + parsing + obs-normalisation used by stage-1 ReAct play,
# so training data is drawn from the same distribution the agent will play in.
from react_sciworld import parse_thought_action, normalize_obs, PRESETS
from scienceworld_react_prompt import SCIENCEWORLD_REACT_PROMPT

# ── vLLM (OpenAI-compatible) client ──────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-8B"
client = OpenAI(base_url="http://10.5.30.29:8001/v1", api_key="EMPTY")

# Location feature tracking. Stage 7 (BFS play) MUST use this same init + helper
# so the BERT state embedding matches between train and play.
INITIAL_LOCATION = "the starting location"
_MOVE_PREFIXES   = ("teleport to ", "go to ", "move to ")


def update_location(location: str, action: str) -> str:
    """Update the tracked room from a movement action; unchanged otherwise."""
    a = action.strip()
    low = a.lower()
    for pref in _MOVE_PREFIXES:
        if low.startswith(pref):
            return a[len(pref):].strip()
    return location


# ═══════════════════════════════════════════════════════════════════════════════
# LLM HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def sample_candidate_actions(sampling_prompt: str, n: int, temperature: float,
                             max_tokens: int = 200):
    """
    Sample n completions from a prompt ending with "Thought:", parse each into a
    (thought, action) pair via the stage-1 parser, and return:
        candidates   — deduped list of action strings (sampling order preserved)
        action_think — {action: first thought seen for it} (for prompt growth)
    """
    resp = client.completions.create(
        model=MODEL_NAME,
        prompt=sampling_prompt,
        n=n,
        max_tokens=max_tokens,
        temperature=temperature,
        echo=False,
    )

    candidates, seen = [], set()
    action_think = {}
    for choice in resp.choices:
        thought, action = parse_thought_action(choice.text)
        if not action:
            continue
        action = action.strip()
        if action not in seen:
            seen.add(action)
            candidates.append(action)
            if thought:
                action_think[action] = thought
    return candidates, action_think


def get_action_logprobs(scoring_prompt: str, actions, batch_size: int = 1):
    """
    Echo-score each action under the LLM (no generation), returning
    {action: [token_logprobs]}. `scoring_prompt` ends with "Action:" so that
    scoring_prompt + " " + action == "...Action: <action>".

    Mirrors bfs_conformal_alfworld.py:get_action_logprobs (which ends with '\\n>'),
    but scores candidates in chunks of `batch_size`. `echo=True` makes vLLM
    materialise prompt_logprobs for EVERY token of EVERY prompt in the request;
    over ScienceWorld's long ReAct prompts a wide batch can OOM the GPU, so we
    keep the batch small (default 1). The returned log-probs are identical
    regardless of batch_size — chunking only bounds peak server memory.
    """
    if not actions:
        return {}

    # Count prompt tokens (subtract 1 for the single forced generation token).
    # scoring_prompt is the same for every candidate, so probe once.
    probe = client.completions.create(
        model=MODEL_NAME, prompt=scoring_prompt,
        max_tokens=1, echo=True, logprobs=1,
    )
    n_prompt = len(probe.choices[0].logprobs.tokens) - 1

    results = {}
    for i in range(0, len(actions), max(1, batch_size)):
        chunk = actions[i:i + max(1, batch_size)]
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


# ═══════════════════════════════════════════════════════════════════════════════
# PER-EPISODE GOLD-PATH ROLLOUT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_for_episode(env, task, var, simpl, args):
    """
    Roll the env forward along the gold action sequence for (task, var).
    At each gold prefix, record a training state. Returns (states, final_score,
    note) where note is None on success or a short skip/warn reason.
    """
    env.load(task, var, simpl, generateGoldPath=True)
    env.reset()
    task_desc = env.get_task_description().strip()
    gold = env.get_gold_action_sequence()

    if not gold or (len(gold) == 1 and str(gold[0]).startswith("ERROR")):
        return [], 0, f"no gold path ({gold})"

    base    = "/no_think\n" + SCIENCEWORLD_REACT_PROMPT + task_desc + "\n"
    running = base                       # grows with committed Thought/Action/Observation
    prev_actions = []
    location = INITIAL_LOCATION
    states   = []
    score    = 0

    for t, gold_action in enumerate(gold):
        gold_action = gold_action.strip()

        # ── Candidates: LLM samples ∪ {gold[t]} (gold guarantees a positive) ──
        sampling_prompt = running + "Thought:"
        cand_actions, action_think = sample_candidate_actions(
            sampling_prompt, n=args.n_samples, temperature=args.temperature,
        )
        # dict.fromkeys dedupes while preserving order; gold appended last.
        candidates = list(dict.fromkeys(cand_actions + [gold_action]))

        # ── Score every candidate's token log-probs (think-free "Action:") ──
        scoring_prompt = running + "Action:"
        logprobs = get_action_logprobs(
            scoring_prompt, candidates, batch_size=args.score_batch_size,
        )

        states.append({
            'task_name':          task_desc,
            'prev_actions':       list(prev_actions),
            'location':           location,
            'admissible_actions': list(candidates),
            'log_probs_of_admissible_actions':
                {a: logprobs.get(a, []) for a in candidates},
            'optimal_action':     gold_action,
        })

        if args.verbose:
            in_set = "yes" if gold_action in cand_actions else "no (added)"
            print(f"    step {t:2d}  loc={location!r}  "
                  f"cands={len(candidates)}  gold_sampled={in_set}  gold={gold_action!r}")

        # ── Commit the gold step; grow the prompt with a (best-effort) thought ──
        thought = action_think.get(gold_action) or (
            next(iter(action_think.values())) if action_think else "")
        obs, reward, done, info = env.step(gold_action)
        obs = normalize_obs(obs)
        score = info.get('score', score)

        running += f"Thought: {thought}\nAction: {gold_action}\nObservation: {obs}\n"
        prev_actions.append(gold_action)
        location = update_location(location, gold_action)

        if done:
            break

    note = None if score >= 100 else f"gold path ended at score={score} (<100)"
    return states, score, note


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Stage 2 — ScienceWorld conformal training-data generation")
    p.add_argument("--task-num", type=int, default=13,
                   help="Task index 0-29, or -1 for all tasks.")
    p.add_argument("--var-start", type=int, default=0, help="First variation index.")
    p.add_argument("--num-variations", type=int, default=20,
                   help="Number of variations per task (capped by get_max_variations).")
    p.add_argument("--n-samples", type=int, default=10, help="LLM candidate samples per state.")
    p.add_argument("--score-batch-size", type=int, default=1,
                   help="Candidates per echo-scoring request. 1 = lowest GPU memory "
                        "(echo computes prompt_logprobs for every token of every prompt "
                        "in a request). Raise for speed if the server has headroom.")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature for candidate diversity.")
    p.add_argument("--simplifications-preset", choices=list(PRESETS), default="paper",
                   help="Must match stage-1 / play. 'paper' = teleportAction,openContainers.")
    p.add_argument("--env-step-limit", type=int, default=200,
                   help="Env step limit; set high enough to replay full gold paths.")
    p.add_argument("--out", type=str,
                   default=os.path.join(_HERE, "training_data_sciworld.pkl"),
                   help="Output pickle path.")
    p.add_argument("--verbose", action="store_true", help="Per-step logging.")
    return p.parse_args()


def main():
    args = parse_args()
    simpl = PRESETS[args.simplifications_preset]

    print("=" * 72)
    print("  Stage 2 — ScienceWorld conformal training-data generation")
    print("=" * 72)
    print(f"  simplifications : {simpl or '(none)'}")
    print(f"  n_samples       : {args.n_samples}")
    print(f"  score_batch     : {args.score_batch_size}")
    print(f"  temperature     : {args.temperature}")
    print(f"  out             : {args.out}")

    # ── Preflight: fail fast (with a clear message) if the LLM endpoint is down,
    #    instead of crashing once per episode with a vague connection error. ──
    try:
        models = client.models.list()
        served = [m.id for m in models.data]
        print(f"  LLM endpoint    : OK  (serving {served})")
    except Exception as ex:
        sys.exit(
            f"\nLLM endpoint unreachable at {client.base_url}\n"
            f"  {type(ex).__name__}: {ex}\n"
            f"  → Is the vLLM server up?  Check:  curl -s {str(client.base_url).rstrip('/')}/models"
        )

    env = ScienceWorldEnv("", envStepLimit=args.env_step_limit)
    task_names = env.get_task_names()

    if args.task_num >= 0:
        tasks = [task_names[args.task_num]]
    else:
        tasks = task_names
    print(f"  tasks           : {len(tasks)}  ({tasks if len(tasks) <= 5 else '...'})")

    training_data = {}
    n_states = 0
    n_ok = n_warn = n_skip = 0

    for task in tasks:
        max_var = env.get_max_variations(task)
        v_end   = min(args.var_start + args.num_variations, max_var)
        for var in range(args.var_start, v_end):
            key = f"{task}__var{var}"
            print(f"\n[{task}  var={var}]  (max_var={max_var})")
            try:
                states, score, note = generate_for_episode(env, task, var, simpl, args)
            except Exception as ex:
                print(f"  CRASH: {type(ex).__name__}: {ex}")
                n_skip += 1
                continue

            if not states:
                print(f"  SKIP: {note}")
                n_skip += 1
                continue

            training_data[key] = states
            n_states += len(states)
            if note:
                print(f"  WARN: {note}  ({len(states)} states recorded)")
                n_warn += 1
            else:
                print(f"  OK: score={score}  {len(states)} states")
                n_ok += 1

            # Incremental save so long runs survive interruption.
            with open(args.out, 'wb') as f:
                pickle.dump(training_data, f)

    print("\n" + "=" * 72)
    print(f"  DONE  episodes: ok={n_ok}  warn={n_warn}  skip={n_skip}")
    print(f"  total states  : {n_states}  across {len(training_data)} episodes")
    print(f"  saved         : {args.out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
