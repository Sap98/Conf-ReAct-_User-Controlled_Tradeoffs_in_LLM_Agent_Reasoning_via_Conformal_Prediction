"""
ReflAct on Game of 24 — a faithful port of g24_react.py that swaps ReAct's
forward-planning `think:` steps for ReflAct `reflection:` steps.

Changes vs g24_react.py (and ONLY these):
  1. Loads the ReflAct few-shot prompt (game24_base_reflact.txt) instead of the
     ReAct one. Its reasoning steps GROUND the current state (which numbers are
     left) and RELATE it to the goal (reaching 24), rather than just planning
     the next move.
  2. `reflection:` actions are intercepted in the runner as pure no-ops that
     return `Obs> OK.` — the same short-circuit reflact_webshop.py applies to
     `reflection[`. The shared env (game24.py) and oracle (g24_oracle.py), which
     only recognize `think`, are left untouched: a reflection never touches the
     env's step counter / value cache, and never gets an oracle label (it
     carries no ground-truth signal, exactly like a think step).
  3. Mode is 'reflact'; the per-step loop is widened to accommodate the extra
     reflection turns interleaved between the (at most 4) real moves.

Run:
  python g24_reflact.py                       # 100 samples, seed 42
  python g24_reflact.py --num_samples 50
"""
import os
import sys
import json
import argparse
import collections

from openai import OpenAI
from typing import Any, List, Dict, Tuple
from game24 import Game24Task
from g24_oracle import TrajectoryScorer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Fixed 200-task set saved from sample.txt so the same exact puzzles play every run.
_HERE = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(_HERE, 'sample_test_tasks.csv')
FOLDER = os.path.join(_HERE, 'prompts') + os.sep
ACTIONLY_FILE = 'game24_base_actionly.txt'
REACT_FILE = 'game24_base_react.txt'
REFLACT_FILE = 'game24_base_reflact.txt'

MODEL_NAME = "Qwen/Qwen3-8B"
VLLM_BASE_URL = "http://10.5.18.73:8002/v1"

# max per-episode LLM turns (real moves + interleaved reflections + final answer)
MAX_TURNS = 11


class VLLM_llm:
    def __init__(self, model_name=MODEL_NAME, base_url=VLLM_BASE_URL):
        self.name = model_name.split('/')[-1]
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY",
            timeout=120.0,        # tolerate slow generations
            max_retries=5,        # retry transient APIConnectionError / 5xx
        )

    def _generate(self, prompt: str, stop=None):
        # Raw text completion (not chat) so the few-shot "Act N> " prompt is
        # continued verbatim and Qwen3's chat/thinking template is NOT applied.
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            temperature=0.1,
            max_tokens=100,
            top_p=1,
            stop=stop,
        )
        action = response.choices[0].text
        # stop is applied server-side, but keep the split as a safety net
        if stop and stop in action:
            action = action.split(stop)[0]

        # strip FIRST: with no trailing space in the prompt the model emits a
        # leading space (e.g. " reflection:"), so normalize before lowercasing
        # the first character, otherwise action[0] would be the space.
        action = action.strip()
        if len(action) > 1:
            action = action[0].lower() + action[1:]

        return action


def _print_dist(label, values):
    """Print a count/% distribution table (WebShop-style)."""
    if not values:
        print(f"\n  DISTRIBUTION ({label})\n  (no data)")
        return
    dist = collections.Counter(values)
    total = len(values)
    print(f"\n  DISTRIBUTION ({label})")
    print(f"  {'Steps':>6}  {'Count':>6}  {'%':>7}")
    for v in range(0, max(values) + 1):
        c = dist.get(v, 0)
        if c == 0:
            continue
        print(f"  {v:>6}  {c:>6}  {c/total*100:>6.1f}%")


def _plot_dist(label, values, out_path, color, succ):
    """Save a labeled bar histogram of trajectory lengths (WebShop-style)."""
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
    ax.set_title(f"Game of 24 ReflAct — trajectory length ({label})  "
                 f"({total} puzzles)\nsuccess rate: {succ*100:.1f}%")
    ax.set_xlabel("Trajectory length (turns taken)")
    ax.set_ylabel("Number of puzzles")
    ax.set_xticks(xs)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved -> {out_path}")


def game24_run(task_input, env, prompt, llm, to_print=True, oracle_stats=None):
    ob = '\n# Here is the task:\nInput: ' + task_input
    init_prompt = prompt + ob
    prompt = ''
    scorer = TrajectoryScorer(task_input)   # ground-truth per-state labels
    if to_print:
        print(ob)
        sys.stdout.flush()
    reward = 0
    i = 0
    for i in range(1, MAX_TURNS + 1):
        # Feed the generation boundary as "Act i>" with NO trailing space
        # (mirrors g24_reflexion.py) so the model reproduces the few-shot
        # "Act i> reflection:" pattern; the recorded history keeps the
        # "Act i> " space as the separator before the action.
        action = llm._generate(init_prompt + prompt + f'\nAct {i}>', stop='\n')

        # ReflAct reflection: pure no-op — ground the state, don't change it.
        # Skip both the env and the oracle (mirrors how `think` carries no
        # ground-truth signal), just echo OK. and advance the turn counter.
        if action.startswith('reflection'):
            observation = 'OK.'
            if to_print:
                print(f'Act {i}> {action}\nObs {i}> {observation}')
                sys.stdout.flush()
            prompt += f'\nAct {i}> {action}\nObs {i}> {observation}'
            continue

        lab = scorer.step(action)           # label this proposal vs. true state
        response = env.step(action)

        reward, observation = response['r'], response['ob']
        if to_print:
            print(f'Act {i}> {action}    [oracle: {lab["label"]}]\nObs {i}> {observation}')
            sys.stdout.flush()
        prompt += f'\nAct {i}> {action}\nObs {i}> {observation}'
        if reward or 'Exceeded' in observation:
            break

    summ = scorer.summary()
    if oracle_stats is not None:
        oracle_stats.append(summ)
    if to_print:
        print(f'[oracle] solved={summ["solved"]} '
              f'solvability_preserved_rate={summ["solvability_preserved_rate"]} '
              f'(correct {summ["correct_moves"]}/{summ["decidable_moves"]}) '
              f'illegal={summ["illegal"]} wrong_arith={summ["wrong_arithmetic"]} '
              f'dead_end={summ["dead_end"]}')
        sys.stdout.flush()
    return reward, i      # i = number of turns taken (trajectory length)


def main(args):
    env = Game24Task(file_path, args.seed, args.num_samples)
    if args.mode == 'reflact':
        with open(os.path.join(FOLDER, REFLACT_FILE), 'r') as f:
            BASE_PROMPT = f.read()
    elif args.mode == 'react':
        with open(os.path.join(FOLDER, REACT_FILE), 'r') as f:
            BASE_PROMPT = f.read()
    elif args.mode == 'act':
        with open(os.path.join(FOLDER, ACTIONLY_FILE), 'r') as f:
            BASE_PROMPT = f.read()
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Mode must be 'reflact', 'react' or 'act'")

    llm = VLLM_llm(args.model_name, args.base_url)
    results = []
    oracle_stats = []
    all_traj_lengths = []        # trajectory length per puzzle
    perfect_traj_lengths = []    # trajectory length for solved (winning) puzzles

    for i in range(args.num_samples):
        print('---------------------------------------------------------------------------')
        sys.stdout.flush()
        curr_case = env.get_case()
        r, traj_len = game24_run(curr_case, env, BASE_PROMPT, llm, oracle_stats=oracle_stats)
        results.append(r)
        all_traj_lengths.append(traj_len)
        if r:
            perfect_traj_lengths.append(traj_len)
        if (i + 1) % 1 == 0:
            sr = sum(results) / len(results)
            print(f"id:{i + 1}, success rate:{sr}")
            print('---------------------------------------------------------------------------')
            sys.stdout.flush()

    print(f"Success rate:{sum(results) / len(results)}")

    # ---- trajectory-length distribution + histograms (perfect = winning) ---- #
    N = len(results)
    succ = (sum(results) / N) if N else 0.0
    _print_dist("ALL puzzles", all_traj_lengths)
    _print_dist("PERFECT (solved)", perfect_traj_lengths)
    print("\nGenerating histograms ...")
    _plot_dist("ALL", all_traj_lengths, "traj_len_hist_reflact_all.png", "#4C72B0", succ)
    _plot_dist("PERFECT", perfect_traj_lengths, "traj_len_hist_reflact_perfect.png", "#55A868", succ)

    # ---- aggregate ground-truth (oracle) report over the whole run ---- #
    rates = [s['solvability_preserved_rate'] for s in oracle_stats
             if s['solvability_preserved_rate'] is not None]
    decided = sum(s['decidable_moves'] for s in oracle_stats)
    correct = sum(s['correct_moves'] for s in oracle_stats)
    print("==== oracle report ====")
    print(f"solved (final answer == 24):        {sum(s['solved'] for s in oracle_stats)}/{len(oracle_stats)}")
    print(f"solvability-preserving moves:       {correct}/{decided}"
          f"  ({(correct / decided if decided else 0):.3f})")
    print(f"mean per-episode preserved rate:    {(sum(rates) / len(rates) if rates else 0):.3f}")
    print(f"illegal moves:                      {sum(s['illegal'] for s in oracle_stats)}")
    print(f"wrong-arithmetic moves:             {sum(s['wrong_arithmetic'] for s in oracle_stats)}")
    print(f"dead-end (24-killing) moves:        {sum(s['dead_end'] for s in oracle_stats)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100, help="The number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Sample seed")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="model name served by vLLM")
    parser.add_argument("--base_url", type=str, default=VLLM_BASE_URL, help="vLLM OpenAI-compatible base url")
    parser.add_argument("--mode", type=str, default='reflact', help="reflact / react / act")
    args = parser.parse_args()

    main(args)
