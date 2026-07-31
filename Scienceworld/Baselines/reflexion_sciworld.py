"""Reflexion (Shinn et al., NeurIPS 2023) on ScienceWorld, layered on our ReAct setup.

Pipeline mirrors `noahshinn/reflexion/alfworld_runs/`:
  - Outer loop: per env, up to --num-trials trials. After each FAILED trial,
    generate a verbal "reflection" / new plan, append to that env's memory
    (capped at last 3), and retry. On success, skip remaining trials.
  - Inner trial: one ReAct rollout using examples/scienceworld_react_prompt.py
    with a "Your memory for the task below:" block (last 3 reflections)
    inserted before "Now here is your task." in the prompt
    (mirrors EnvironmentHistory._get_base_query).
  - Evaluator (heuristic): success iff `score >= 100` at episode end.
    Trial fails on: per-task step cap exceeded, episode terminated with
    score<100, OR 3 consecutive identical executed actions (Reflexion's
    exhaustion heuristic).
  - Reflection: a separate /no_think LLM call with the failed trajectory +
    past reflections + the 2 few-shot examples in reflexion_few_shot_examples.txt;
    returns a "New plan: ..." string.

Reuses helpers from react/react_sciworld.py.

Examples:
  # Single-task smoke (3 trials of task 29):
  python react/reflexion_sciworld.py --task-num 29 --num-episodes 1 --num-trials 3 --max-steps 30

  # Full 211 MPO/ReflAct test split with 10 outer trials each:
  python react/reflexion_sciworld.py --mpo-testset --num-trials 10 \\
      --results react/results_reflexion_211.jsonl --resume
"""

import os
import sys
import json
import time
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "examples"))

# Reuse ReAct helpers
from react_sciworld import (
    client, MODEL_NAME, PRESETS,
    llm_infer, parse_thought_action, normalize_obs,
    load_mpo_testset, aggregate, print_trajectory_length_distribution,
)
from scienceworld_react_prompt import SCIENCEWORLD_REACT_PROMPT
from scienceworld import ScienceWorldEnv
import react_sciworld

# '/no_think' is a Qwen3 control token; blanked for non-Qwen models in main().
NOTHINK_PREFIX = "/no_think\n"

# Reflection few-shot examples
REFLEXION_FEW_SHOT_PATH = os.path.join(_HERE, "reflexion_few_shot_examples.txt")
with open(REFLEXION_FEW_SHOT_PATH) as f:
    REFLEXION_FEW_SHOT = f.read()

# How many past reflections to include in the prompt (Reflexion's Ω)
MAX_MEMORY = 3

# Exhaustion: end the trial after this many consecutive identical executed actions.
MAX_CONSECUTIVE_REPEATS = 3

# Cap the trajectory length when serialising it into the reflection prompt
# (avoids blowing past the 8192-token context on long runs).
REFLECTION_TRAJ_LAST_N = 40

# Anchor where the memory block is injected in the ReAct prompt.
_NOW_HERE_TOKEN = "Now here is your task."


def build_prompt_with_memory(task_desc: str, memory) -> str:
    """Build the full LLM prompt for a Reflexion trial:
      /no_think + [SCIENCEWORLD_REACT_PROMPT with the memory block inserted
                   immediately BEFORE "Now here is your task."]
      + task_desc + "\\nThought:".
    """
    head, _, tail = SCIENCEWORLD_REACT_PROMPT.partition(_NOW_HERE_TOKEN)
    assert tail, "SCIENCEWORLD_REACT_PROMPT must contain the 'Now here is your task.' anchor"

    mem_block = ""
    if memory:
        mem_block = "Your memory for the task below:\n"
        for i, m in enumerate(memory):
            mem_block += f"Trial {i}:\n{m.strip()}\n"
        mem_block += "\n"

    return (
        NOTHINK_PREFIX
        + head + mem_block + _NOW_HERE_TOKEN + tail
        + task_desc + "\nThought:"
    )


def reflexion_trial(env, sw_name: str, var_idx: int, simpl_str: str,
                    memory, max_steps: int, to_print: bool = True):
    """Run one ReAct rollout with `memory` prepended.
    Returns (score, won, n_llm_calls, n_env_steps, transcript, exhausted, task_desc).
    `won` = `score >= 100`; the env's `done` flag is a termination signal, not
    a success signal (also fires on terminal failure / timeout / score<0).
    """
    env.load(sw_name, var_idx, simpl_str)
    env.reset()
    task_desc = env.get_task_description().strip()

    prompt = build_prompt_with_memory(task_desc, memory)

    if to_print:
        print(f"Task description: {task_desc}")
        if memory:
            print(f"(Using {len(memory)} reflection(s) in memory)")
        sys.stdout.flush()

    score = 0
    done = False
    n_env_steps = 0
    transcript = []
    last_exec = None
    consec_same = 0
    exhausted = False
    i = 0  # in case max_steps is 0

    for i in range(1, max_steps + 1):
        reply = llm_infer(prompt)
        thought, action = parse_thought_action(reply)
        if action is None:
            if to_print:
                print("No Action parsed from the LLM response; ending trial.")
            transcript.append({"step": i, "note": "no action parsed; ending trial"})
            break

        observation, reward, done, info = env.step(action)
        score = info['score']
        observation = normalize_obs(observation)
        n_env_steps += 1
        transcript.append({
            "step":        i,
            "thought":     thought,
            "action":      action,
            "observation": observation,
            "score":       score,
            "done":        bool(done),
        })

        if to_print:
            print(f"Step {i}")
            print(f"Thought: {thought}")
            print(f"Action: {action}")
            print(f"Observation: {observation}")
            print(f"Score: {score}   done: {done}")
            sys.stdout.flush()

        prompt += f" {thought}\nAction: {action}\nObservation: {observation}\nThought:"

        # Exhaustion heuristic: 3 consecutive identical executed actions.
        if action == last_exec:
            consec_same += 1
        else:
            consec_same = 1
            last_exec = action
        if consec_same >= MAX_CONSECUTIVE_REPEATS:
            exhausted = True
            if to_print:
                print(f"[exhaustion] action {action!r} repeated {consec_same}x; ending trial.")
            break

        if done:
            break

    return score, score >= 100, i, n_env_steps, transcript, exhausted, task_desc


def _serialize_trajectory(task_desc: str, transcript, status: str = "FAIL") -> str:
    """Serialise a (failed) trial's trajectory for the reflection prompt,
    in the same format as the few-shot examples. Truncates to the most
    recent REFLECTION_TRAJ_LAST_N step entries to keep within context."""
    body = transcript[-REFLECTION_TRAJ_LAST_N:]
    lines = []
    head_line = task_desc if task_desc.lower().startswith("your task is to") \
                          else f"Your task is to {task_desc}"
    lines.append(head_line)
    if len(transcript) > REFLECTION_TRAJ_LAST_N:
        lines.append(f"(... earlier {len(transcript) - REFLECTION_TRAJ_LAST_N} steps truncated ...)")
    for entry in body:
        if "note" in entry:
            continue
        lines.append(f"Thought: {entry['thought']}")
        lines.append(f"Action: {entry['action']}")
        lines.append(f"Observation: {entry['observation']}")
    lines.append(f"STATUS: {status}")
    return "\n".join(lines)


def generate_reflection(task_desc: str, transcript, memory) -> str:
    """Reflexion-style reflection query: failed trajectory + past reflections
    + 2 few-shot examples; LLM returns the 'New plan' text."""
    scenario = _serialize_trajectory(task_desc, transcript, status="FAIL")
    query = (
        NOTHINK_PREFIX +
        "You will be given the history of a past experience in which you were placed in an "
        "environment and given a task to complete. You were unsuccessful in completing the task. "
        "Do not summarize your environment, but rather think about the strategy and path you "
        "took to attempt to complete the task. Devise a concise, new plan of action that "
        "accounts for your mistake with reference to specific actions that you should have "
        "taken. For example, if you tried A and B but forgot C, then devise a plan to achieve "
        "C with environment-specific actions. You will need this later when you are solving "
        "the same task. Give your plan after \"New plan\". Here are two examples:\n\n"
        f"{REFLEXION_FEW_SHOT}\n\n"
        f"{scenario}"
    )
    if memory:
        query += "\n\nPlans from past attempts:"
        for i, m in enumerate(memory):
            query += f"\nTrial #{i}: {m}"
    query += "\n\nNew plan:"

    resp = client.completions.create(
        model=MODEL_NAME,
        prompt=query,
        n=1,
        temperature=0.1,
        max_tokens=256,
        echo=False,
    )
    text = resp.choices[0].text.replace("</think>", " ").replace("<think>", " ").strip()
    # Stop at obvious continuation markers if the model rambles past its plan.
    for stop in ("\nYour task is to", "\nSTATUS:", "\nTrial #", "\n\nNew plan"):
        idx = text.find(stop)
        if idx >= 0:
            text = text[:idx]
    return text.strip()


def _load_resume(results_path):
    """Read existing JSONL records to restore per-(env_key) state.
    Returns {key: {'is_success': bool, 'memory': [reflection, ...]}}."""
    state = {}
    if not (results_path and os.path.exists(results_path)):
        return state
    with open(results_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            key = (rec.get("mpo_name") or rec.get("sw_name"), rec.get("var"))
            entry = state.setdefault(key, {"is_success": False, "memory": []})
            if rec.get("won"):
                entry["is_success"] = True
            if rec.get("reflection"):
                entry["memory"].append(rec["reflection"])
    for v in state.values():
        if len(v["memory"]) > MAX_MEMORY:
            v["memory"] = v["memory"][-MAX_MEMORY:]
    return state


def run_reflexion(env, env_configs, num_trials, simpl_str, results_path, to_print=True):
    """Outer Reflexion loop. Writes one JSONL record per (env, trial)."""
    out = None
    if results_path:
        os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
        out = open(results_path, "a")

    final_won_by_key   = {}
    final_score_by_key = {}
    win_step_by_key    = {}   # for the winning-trajectory histogram

    t_start = time.time()
    n_envs = len(env_configs)

    for trial_idx in range(num_trials):
        already = sum(1 for c in env_configs if c["is_success"])
        print("\n" + "=" * 80)
        print(f"TRIAL #{trial_idx}  (already succeeded: {already}/{n_envs})")
        print("=" * 80)
        sys.stdout.flush()

        for z, cfg in enumerate(env_configs):
            if cfg["is_success"]:
                continue
            key = (cfg.get("mpo_name") or cfg["sw_name"], cfg["var"])
            print(f"\n[Env {z + 1}/{n_envs}] Trial {trial_idx}  "
                  f"{cfg.get('mpo_name') or cfg['sw_name']}  var={cfg['var']}  step_cap={cfg['max_steps']}")
            sys.stdout.flush()

            t0 = time.time()
            try:
                score, won, n_calls, n_env, transcript, exhausted, task_desc = reflexion_trial(
                    env, cfg["sw_name"], cfg["var"], simpl_str,
                    cfg["memory"][-MAX_MEMORY:], cfg["max_steps"],
                    to_print=to_print,
                )
                err = None
            except Exception as ex:
                score, won, n_calls, n_env = -1, False, 0, 0
                transcript, exhausted, task_desc = [], False, ""
                err = f"{type(ex).__name__}: {ex}"
            dt = time.time() - t0

            reflection = None
            if not won and err is None and transcript:
                if to_print:
                    print("\n[Reflexion] Generating reflection...")
                    sys.stdout.flush()
                try:
                    reflection = generate_reflection(
                        task_desc, transcript, cfg["memory"][-MAX_MEMORY:]
                    )
                    if to_print and reflection:
                        print(f"[Reflection] {reflection}")
                        sys.stdout.flush()
                except Exception as ex:
                    print(f"  [warn] reflection generation failed: {type(ex).__name__}: {ex}")
                    reflection = None

            memory_used_snapshot = list(cfg["memory"][-MAX_MEMORY:])
            if won:
                cfg["is_success"] = True
                win_step_by_key[key] = n_env
            elif reflection:
                cfg["memory"].append(reflection)
                cfg["memory"] = cfg["memory"][-MAX_MEMORY:]

            final_won_by_key[key]   = cfg["is_success"]
            final_score_by_key[key] = score

            rec = {
                "trial_idx":   trial_idx,
                "mpo_name":    cfg.get("mpo_name"),
                "sw_name":     cfg["sw_name"],
                "var":         cfg["var"],
                "score":       score,
                "won":         bool(won),
                "exhausted":   bool(exhausted),
                "n_llm_calls": n_calls,
                "n_env_steps": n_env,
                "step_cap":    cfg["max_steps"],
                "seconds":     round(dt, 2),
                "error":       err,
                "memory_used": memory_used_snapshot,
                "reflection":  reflection,
                "transcript":  transcript,
            }
            if out:
                out.write(json.dumps(rec) + "\n")
                out.flush()

            n_succ = sum(1 for c in env_configs if c["is_success"])
            print(f"  score={score}  won={won}  exhausted={exhausted}  "
                  f"steps={n_env}  ({dt:.1f}s)  |  overall {n_succ}/{n_envs} "
                  f"({n_succ / n_envs * 100:.1f}%)")
            sys.stdout.flush()

        if all(c["is_success"] for c in env_configs):
            print("\n[Reflexion] All envs solved; stopping trials early.")
            break

    if out:
        out.close()

    # Final aggregate over envs (last result per env)
    total_dt = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"DONE  ({total_dt / 60:.1f} min)")
    print("=" * 80)
    final_scores = list(final_score_by_key.values())
    final_wons   = list(final_won_by_key.values())
    if final_scores:
        agg = aggregate(final_scores, final_wons)
        print(f"  Final OVERALL  n={agg['n']}  avg_reward={agg['avg_reward']:.2f}  "
              f"SR={agg['success_rate']*100:.2f}%")
    print_trajectory_length_distribution(list(win_step_by_key.values()))


def parse_args():
    p = argparse.ArgumentParser(description="Reflexion on ScienceWorld (multi-trial ReAct with verbal reflection).")
    p.add_argument("--task-num",     type=int, default=13, help="Single-task mode: task index 0-29.")
    p.add_argument("--var-num",      type=int, default=0,  help="Single-task: starting variation index.")
    p.add_argument("--num-episodes", type=int, default=1,  help="Single-task: number of variations (envs).")
    p.add_argument("--env-step-limit", type=int, default=100, help="ScienceWorld env step limit.")
    p.add_argument("--max-steps",    type=int, default=50, help="Single-task: per-trial step cap.")
    p.add_argument("--num-trials",   type=int, default=10, help="Reflexion: outer trials per env (paper uses ~12).")
    p.add_argument("--simplifications-preset", choices=list(PRESETS), default="paper",
                   help="Default: paper (teleportAction+openContainers).")
    p.add_argument("--mpo-testset",  action="store_true",
                   help="Run the 211 MPO/ReflAct test pairs; uses per-task step caps from max_steps.json.")
    p.add_argument("--results",      type=str, default=None,
                   help="JSONL output path. Required when --mpo-testset is set; optional in single-task mode (no JSONL written if omitted).")
    p.add_argument("--resume",       action="store_true",
                   help="Restore per-env memory + is_success from an existing --results JSONL.")
    p.add_argument("--base_url", type=str, default=None, help="Override vLLM endpoint.")
    p.add_argument("--model_name", type=str, default=None, help="Override served model id.")
    return p.parse_args()


def apply_model_override(args):
    """Point both this module's LLM calls AND the reused react_sciworld helpers
    (llm_infer) at the override endpoint/model, and set the think-prefix."""
    global client, MODEL_NAME, NOTHINK_PREFIX
    react_sciworld.apply_model_override(args)      # fixes llm_infer's globals
    client         = react_sciworld.client
    MODEL_NAME     = react_sciworld.MODEL_NAME
    NOTHINK_PREFIX = react_sciworld.NOTHINK_PREFIX


def main():
    args = parse_args()
    apply_model_override(args)
    print(f"ScienceWorld Reflexion (multi-trial ReAct + verbal self-reflection, {MODEL_NAME})")
    simpl_str = PRESETS[args.simplifications_preset]
    print(f"Simplifications: {simpl_str or '(none)'}")

    env = ScienceWorldEnv("", envStepLimit=args.env_step_limit)

    if args.mpo_testset:
        entries = load_mpo_testset(env)
        env_configs = [
            {
                "name":       e["mpo_name"],
                "mpo_name":   e["mpo_name"],
                "sw_name":    e["sw_name"],
                "var":        e["var"],
                "max_steps":  e["max_steps"] or args.max_steps,
                "memory":     [],
                "is_success": False,
            }
            for e in entries
        ]
        if args.results is None:
            sys.exit("Error: --results <path.jsonl> is required when --mpo-testset is set.")
    else:
        task_name = env.get_task_names()[args.task_num]
        max_var   = env.get_max_variations(task_name)
        env_configs = [
            {
                "name":       f"{task_name}_v{(args.var_num + ep) % max_var}",
                "mpo_name":   None,
                "sw_name":    task_name,
                "var":        (args.var_num + ep) % max_var,
                "max_steps":  args.max_steps,
                "memory":     [],
                "is_success": False,
            }
            for ep in range(args.num_episodes)
        ]

    if args.resume:
        state = _load_resume(args.results)
        if state:
            for cfg in env_configs:
                key = (cfg.get("mpo_name") or cfg["sw_name"], cfg["var"])
                if key in state:
                    cfg["is_success"] = state[key]["is_success"]
                    cfg["memory"]     = state[key]["memory"][-MAX_MEMORY:]
            n_done = sum(1 for c in env_configs if c["is_success"])
            n_mem  = sum(1 for c in env_configs if c["memory"])
            print(f"[Resume] restored: {n_done} already-succeeded envs + {n_mem} envs with memory.")

    print(f"Envs: {len(env_configs)}  |  Trials per env: {args.num_trials}")
    if args.results:
        print(f"Results appended to: {args.results}")
    else:
        print("Results: (no JSONL output — --results not set)")

    run_reflexion(env, env_configs, args.num_trials, simpl_str, args.results, to_print=True)


if __name__ == "__main__":
    main()
