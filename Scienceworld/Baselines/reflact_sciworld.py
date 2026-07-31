"""Single-response ReflAct agent on ScienceWorld, using the ReflAct paper's prompt.

ReflAct counterpart of react_sciworld.py. The ONLY changes vs that file are the
ReflAct changes (mirroring the WebShop reflact port's `think[] -> reflection[]`):

  1. The few-shot prompt is the ReflAct one (scienceworld_reflact_prompt.py):
     each reasoning step is rewritten from ReAct's next-action *planning* thought
     into a ReflAct *goal-state reflection* (ground the current state -> relate
     it to the task goal), and the "Thought:" tag becomes "Reflection:".
  2. Parsing, prompt building, the transcript, and printing key on "Reflection:"
     instead of "Thought:".
Actions, observations, env stepping, the MPO test-set loop, and the reporting
are all identical to the ReAct version.

Prompt: examples/scienceworld_reflact_prompt.py (Reflection / Action / Observation
format). Each LLM generation emits one Reflection + one Action; we step the env,
append the observation, repeat.

LLM: vLLM Qwen3-8B via the OpenAI-compatible client (no local GPU needed).
A leading "/no_think" disables Qwen3's long chain-of-thought.

Two run modes:
  - Default: --task-num + --num-episodes, loops variations of one task.
  - --mpo-testset: iterates the 211 (task, variation) pairs from MPO/ReflAct's
    test split (react/mpo_data/test_indices.json), using per-task step caps
    from react/mpo_data/max_steps.json. Writes per-episode results to JSONL.

No data storage, no log-probs, no conformal anything.

Examples:
  # Single task (the original run mode)
  python react/reflact_sciworld.py --task-num 29 --var-num 0 --num-episodes 1 --max-steps 30

  # Full 211-task ReflAct/MPO test set
  python react/reflact_sciworld.py --mpo-testset --results react/results_reflact_211.jsonl
"""

import os
import re
import sys
import json
import time
import argparse

# Make `scienceworld_reflact_prompt` importable regardless of CWD (it lives in ../examples).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "examples"))

from openai import OpenAI
from scienceworld import ScienceWorldEnv
from scienceworld_reflact_prompt import SCIENCEWORLD_REFLACT_PROMPT

# ── vLLM (OpenAI-compatible) client ──────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-8B"
client = OpenAI(base_url="http://10.5.30.30:8001/v1", api_key="EMPTY")
NOTHINK_PREFIX = "/no_think\n"   # Qwen3 control token; blanked for non-Qwen in main()

# Simplifications presets. 'paper' matches the assumptions baked into the
# ReflAct paper's prompt (teleportAction + openContainers).
PRESETS = {
    "paper": "teleportAction,openContainers",
    "easy":  "teleportAction,selfWateringFlowerPots,openDoors,noElectricalAction",
    "none":  "",
}

MPO_DATA_DIR    = os.path.join(_HERE, "mpo_data")
MPO_TEST_FILE   = os.path.join(MPO_DATA_DIR, "test_indices.json")
MPO_STEPS_FILE  = os.path.join(MPO_DATA_DIR, "max_steps.json")


def llm_infer(prompt, max_tokens=200):
    """Single greedy-ish completion from the vLLM Qwen3-8B endpoint."""
    resp = client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        n=1,
        temperature=0.1,
        max_tokens=max_tokens,
        echo=False,
    )
    return resp.choices[0].text


def parse_reflection_action(reply: str):
    """Extract (reflection, action) from one LLM response.

    The prompt ends with 'Reflection:', so the model continues with the
    reflection body followed by an 'Action: <command>' line. Strips Qwen3
    <think>/</think> markers; ignores any hallucinated Observation or follow-up
    Reflection.
    """
    reply = reply.replace("</think>", " ").replace("<think>", " ").strip()
    reflection_lines = []
    action = None
    for raw in reply.split("\n"):
        stripped = raw.strip()
        low = stripped.lower()
        if low.startswith("action:"):
            action = stripped[len("action:"):].strip()
            break
        if low.startswith("observation:"):
            break
        if low.startswith("reflection:") and reflection_lines:
            break
        if low.startswith("reflection:"):
            stripped = stripped[len("reflection:"):].strip()
        if stripped:
            reflection_lines.append(stripped)
    reflection = " ".join(reflection_lines).strip()
    return reflection, action


def normalize_obs(obs: str) -> str:
    """Flatten ScienceWorld's tab-indented observation to match the paper's
    few-shot format (no leading whitespace per line)."""
    return "\n".join(line.lstrip("\t ").rstrip() for line in obs.split("\n"))


def mpo_to_sciworld_name(mpo_name: str) -> str:
    """Translate an MPO task name like 'task-10-measure-melting-point-(known-substance)'
    to the ScienceWorld API name 'measure-melting-point-known-substance'."""
    s = re.sub(r"^task-\d+[a-z]?-", "", mpo_name)
    return s.replace("(", "").replace(")", "")


def load_mpo_testset(env):
    """Return (entries, max_steps_lookup).

    entries: list of dicts {mpo_name, sw_name, var, max_steps}
    Validates every translated name against env.get_task_names(); raises if any
    name is unknown (so we catch mismatches at startup, not mid-run).
    """
    with open(MPO_TEST_FILE) as f:
        pairs = json.load(f)              # [[mpo_name, var], ...]
    with open(MPO_STEPS_FILE) as f:
        max_steps_by_mpo = json.load(f)   # {mpo_name: int}

    sw_known = set(env.get_task_names())
    entries = []
    unknown = []
    for mpo_name, var in pairs:
        sw_name = mpo_to_sciworld_name(mpo_name)
        if sw_name not in sw_known:
            unknown.append((mpo_name, sw_name))
        entries.append({
            "mpo_name": mpo_name,
            "sw_name":  sw_name,
            "var":      int(var),
            "max_steps": int(max_steps_by_mpo.get(mpo_name, 0)),
        })
    if unknown:
        raise RuntimeError(
            "MPO test set contains names that don't translate to known ScienceWorld task names:\n  "
            + "\n  ".join(f"{m!r} -> {s!r}" for m, s in unknown)
        )
    return entries


def scienceworld_run(env, sw_name, var_idx, simpl_str, max_steps, to_print=True):
    """Play one episode with the paper's ReflAct loop.
    Returns (score, won, n_llm_calls, n_env_steps, transcript). `won` is
    `score >= 100` — the env's `done` flag is a termination signal (also fires
    on terminal failure / timeout / negative score), not a success signal."""
    env.load(sw_name, var_idx, simpl_str)
    env.reset()  # initial obs is discarded; the agent will 'look around' itself
    task_desc = env.get_task_description().strip()

    prompt = NOTHINK_PREFIX + SCIENCEWORLD_REFLACT_PROMPT + task_desc + "\nReflection:"

    if to_print:
        print(f"Task description: {task_desc}\n")
        sys.stdout.flush()

    score = 0
    done = False
    n_env_steps = 0
    transcript = []
    for i in range(1, max_steps + 1):
        reply = llm_infer(prompt)
        reflection, action = parse_reflection_action(reply)
        if action is None:
            if to_print:
                print("No Action parsed from the LLM response; ending episode.")
            transcript.append({"step": i, "note": "no action parsed; ending episode"})
            break

        observation, reward, done, info = env.step(action)
        score = info['score']
        observation = normalize_obs(observation)
        n_env_steps += 1
        transcript.append({
            "step": i,
            "reflection": reflection,
            "action": action,
            "observation": observation,
            "score": score,
            "done": bool(done),
        })

        if to_print:
            print(f"Step {i}")
            print(f"Reflection: {reflection}")
            print(f"Action: {action}")
            print(f"Observation: {observation}")
            print(f"Score: {score}   done: {done}")
            sys.stdout.flush()

        prompt += f" {reflection}\nAction: {action}\nObservation: {observation}\nReflection:"
        if done:
            break

    return score, score >= 100, i, n_env_steps, transcript


def parse_args():
    p = argparse.ArgumentParser(description="Single-response ReflAct on ScienceWorld (ReflAct paper prompt, vLLM Qwen3-8B).")
    p.add_argument("--task-num", type=int, default=13, help="Task index 0-29 (ignored in --mpo-testset mode).")
    p.add_argument("--var-num", type=int, default=0, help="Starting variation index (ignored in --mpo-testset).")
    p.add_argument("--num-episodes", type=int, default=134, help="Number of episodes (ignored in --mpo-testset).")
    p.add_argument("--env-step-limit", type=int, default=100, help="ScienceWorld env step limit.")
    p.add_argument("--max-steps", type=int, default=50, help="Max ReflAct (LLM) steps per episode (ignored in --mpo-testset; uses per-task caps).")
    p.add_argument("--simplifications-preset", choices=list(PRESETS), default="paper",
                   help="Simplifications preset. 'paper' = teleportAction+openContainers (matches the prompt). Default: paper.")
    p.add_argument("--mpo-testset", action="store_true",
                   help="Run the full 211-task MPO/ReflAct test split with per-task step caps.")
    p.add_argument("--results", type=str, default=None,
                   help="Path to JSONL results file (one line per episode). Required when --mpo-testset is set.")
    p.add_argument("--resume", action="store_true",
                   help="Skip (task, var) pairs already present in --results. Use to resume an interrupted run.")
    p.add_argument("--base_url", type=str, default=None, help="Override vLLM endpoint.")
    p.add_argument("--model_name", type=str, default=None, help="Override served model id.")
    return p.parse_args()


def apply_model_override(args):
    global client, MODEL_NAME, NOTHINK_PREFIX
    if getattr(args, "model_name", None):
        MODEL_NAME = args.model_name
    if getattr(args, "base_url", None):
        client = OpenAI(base_url=args.base_url, api_key="EMPTY",
                        timeout=120.0, max_retries=5)
    NOTHINK_PREFIX = "/no_think\n" if "qwen" in MODEL_NAME.lower() else ""


def save_trajectory_length_plot(won_lengths, save_path: str, title_suffix: str = "") -> None:
    """Save a 5-step-bin bar chart of winning trajectory lengths to a PNG.
    Same binning as print_trajectory_length_distribution()."""
    if not won_lengths:
        print(f"  (no wins to plot)")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_len  = max(won_lengths)
    bin_size = 5
    n_bins   = (max_len // bin_size) + 1
    bins     = [0] * n_bins
    for l in won_lengths:
        bins[(l - 1) // bin_size] += 1
    labels = [f"{b*bin_size+1}-{b*bin_size+bin_size}" for b in range(n_bins)]

    fig, ax = plt.subplots(figsize=(max(6, n_bins * 0.6), 4))
    bars = ax.bar(labels, bins, color="#3a76c8", edgecolor="#1f4e8f")
    ax.set_xlabel("Trajectory length (env steps)")
    ax.set_ylabel("Number of winning episodes")
    ax.set_title(f"Winning trajectory length distribution{title_suffix}")
    for bar, c in zip(bars, bins):
        if c > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    str(c), ha="center", va="bottom", fontsize=9)
    fig.text(
        0.99, 0.01,
        f"n={len(won_lengths)}  min={min(won_lengths)}  max={max_len}  "
        f"mean={sum(won_lengths)/len(won_lengths):.1f}",
        ha="right", va="bottom", fontsize=8, color="#555",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  trajectory length plot saved: {save_path}")


def print_trajectory_length_distribution(won_lengths):
    """Print a 5-step-bin histogram of winning-trajectory env step counts.
    Mirrors the block at the end of bfs_conformal_alfworld.py."""
    if not won_lengths:
        print("\n  Winning trajectory length distribution: (no wins)")
        return
    max_len  = max(won_lengths)
    bin_size = 5
    n_bins   = (max_len // bin_size) + 1
    bins     = [0] * n_bins
    for l in won_lengths:
        bins[(l - 1) // bin_size] += 1
    print("\n  Winning trajectory length distribution:")
    for b in range(n_bins):
        lo  = b * bin_size + 1
        hi  = lo + bin_size - 1
        pct = bins[b] / len(won_lengths) * 100
        print(f"    {lo:>2}-{hi:<2}  {bins[b]:>3}  ({pct:5.1f}%)")
    print(f"  (min={min(won_lengths)}  max={max_len}  "
          f"mean={sum(won_lengths)/len(won_lengths):.1f}  n={len(won_lengths)})")


def aggregate(scores, wons=None):
    """Compute summary stats from per-episode scores + win flags.

    Win = `score >= 100`. `wons` is accepted for compatibility but ignored;
    success rate is always derived from the scores themselves."""
    if not scores:
        return {"n": 0, "avg_reward": 0.0, "success_rate": 0.0}
    pos = [s for s in scores if s >= 0]
    wins = [s >= 100 for s in scores]
    return {
        "n":            len(scores),
        "avg_reward":   sum(pos) / len(pos) if pos else 0.0,
        "success_rate": sum(1 for w in wins if w) / len(wins),
    }


def run_mpo_testset(env, args):
    simpl_str = PRESETS[args.simplifications_preset]
    print(f"Simplifications: {simpl_str or '(none)'}")

    entries = load_mpo_testset(env)
    print(f"MPO test set: {len(entries)} episodes across {len({e['sw_name'] for e in entries})} task types")

    if args.results is None:
        sys.exit("Error: --results <path.jsonl> is required when --mpo-testset is set.")
    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    print(f"Writing per-episode results to: {args.results}")

    # Resume: skip already-done (mpo_name, var) pairs.
    done_pairs = set()
    if args.resume and os.path.exists(args.results):
        with open(args.results) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_pairs.add((rec["mpo_name"], rec["var"]))
                except Exception:
                    pass
        print(f"Resume: {len(done_pairs)} episodes already in results file; skipping those.")

    per_type_scores = {}
    per_type_wons   = {}
    all_scores = []
    all_wons   = []
    won_lengths = []   # n_env_steps for winning episodes (winning-only)
    t_start = time.time()

    with open(args.results, "a") as out:
        for i, e in enumerate(entries, start=1):
            key = (e["mpo_name"], e["var"])
            if key in done_pairs:
                continue

            cap = e["max_steps"] or args.max_steps
            print(f"\n[{i}/{len(entries)}]  {e['mpo_name']}  var={e['var']}  step_cap={cap}")
            sys.stdout.flush()

            t0 = time.time()
            try:
                score, won, n_calls, n_env, transcript = scienceworld_run(
                    env, e["sw_name"], e["var"], simpl_str, cap, to_print=True,
                )
                err = None
            except Exception as ex:
                score, won, n_calls, n_env, transcript = -1, False, 0, 0, []
                err = f"{type(ex).__name__}: {ex}"

            dt = time.time() - t0
            rec = {
                "mpo_name":  e["mpo_name"],
                "sw_name":   e["sw_name"],
                "var":       e["var"],
                "score":     score,
                "won":       bool(won),
                "n_llm_calls": n_calls,
                "n_env_steps": n_env,
                "step_cap":  cap,
                "seconds":   round(dt, 2),
                "error":     err,
                "transcript": transcript,
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()

            all_scores.append(score)
            all_wons.append(bool(won))
            per_type_scores.setdefault(e["sw_name"], []).append(score)
            per_type_wons.setdefault(e["sw_name"], []).append(bool(won))
            if won:
                won_lengths.append(n_env)

            agg = aggregate(all_scores, all_wons)
            print(f"  score={score}  won={won}  steps={n_env}  ({dt:.1f}s)  | running avg_reward={agg['avg_reward']:.1f}  SR={agg['success_rate']*100:.1f}%  ({agg['n']}/{len(entries)})")
            sys.stdout.flush()

    total_dt = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"DONE  ({total_dt/60:.1f} min)")
    print("=" * 80)
    print("Per-task-type results:")
    for sw_name in sorted(per_type_scores):
        ss = per_type_scores[sw_name]
        ws = per_type_wons[sw_name]
        a = aggregate(ss, ws)
        print(f"  {sw_name:50s}  n={a['n']:3d}  avg_reward={a['avg_reward']:5.1f}  SR={a['success_rate']*100:5.1f}%")
    agg = aggregate(all_scores, all_wons)
    print("-" * 80)
    print(f"  OVERALL  n={agg['n']}  avg_reward={agg['avg_reward']:.2f}  SR={agg['success_rate']*100:.2f}%")
    print_trajectory_length_distribution(won_lengths)
    if args.results:
        plot_path = os.path.splitext(args.results)[0] + "_trajlen.png"
        save_trajectory_length_plot(won_lengths, plot_path, title_suffix=" (MPO test set)")


def run_single_task(env, args):
    simpl_str = PRESETS[args.simplifications_preset]
    print(f"Simplifications: {simpl_str or '(none)'}")
    task_name = env.get_task_names()[args.task_num]
    max_var = env.get_max_variations(task_name)
    scores = []
    wons   = []
    won_lengths = []   # n_env_steps for winning episodes (winning-only)
    for ep in range(args.num_episodes):
        var_idx = (args.var_num + ep) % max_var
        print("\n" + "=" * 80)
        print(f"EPISODE {ep + 1}/{args.num_episodes}")
        print("=" * 80)
        print(f"Task {args.task_num} ({task_name})  variation {var_idx}")
        score, won, _, n_env, _ = scienceworld_run(env, task_name, var_idx, simpl_str, args.max_steps, to_print=True)
        scores.append(score)
        wons.append(bool(won))
        if won:
            won_lengths.append(n_env)
        print(f"\nEpisode {ep + 1} final score: {score}   won: {won}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"  Task {args.task_num}: {task_name}")
    print(f"  Episode scores: {scores}")
    if scores:
        agg = aggregate(scores, wons)
        print(f"  Average score: {agg['avg_reward']:.1f}")
        print(f"  Solved (score>=100): {sum(wons)}/{len(wons)}")
    print_trajectory_length_distribution(won_lengths)
    plot_path = (os.path.splitext(args.results)[0] + "_trajlen.png") if args.results \
        else os.path.join(_HERE, f"trajlen_task{args.task_num}_{time.strftime('%Y%m%d-%H%M%S')}.png")
    save_trajectory_length_plot(won_lengths, plot_path, title_suffix=f" (task {args.task_num})")
    print("=" * 80)


def main():
    args = parse_args()
    apply_model_override(args)
    print(f"ScienceWorld ReflAct (single-response, {MODEL_NAME}, ReflAct prompt)")

    env = ScienceWorldEnv("", envStepLimit=args.env_step_limit)
    if args.mpo_testset:
        run_mpo_testset(env, args)
    else:
        run_single_task(env, args)


if __name__ == "__main__":
    main()
