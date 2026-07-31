import os
import sys
import json
import re
import argparse
import collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openai import OpenAI
from typing import Any, List, Dict, Tuple
from game24 import Game24Task
from g24_oracle import TrajectoryScorer


_HERE = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(_HERE, 'sample_test_tasks.csv')
FOLDER = os.path.join(_HERE, 'prompts') + os.sep
ACTIONLY_FILE = 'game24_base_actionly.txt'
REACT_FILE = 'game24_base_react.txt'

MODEL_NAME = "Qwen/Qwen3-8B"
VLLM_BASE_URL = "http://10.5.30.29:8001/v1"


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

        if len(action) > 1:
            action = action[0].lower() + action[1:]

        return action.strip()


def _remove_duplicate_think_prefix(text):
    """Strip a leading 'think:'/'Thought:'/'Plan:' the model may re-emit
    when we already primed the prompt with 'Act k> think:'."""
    text = text.strip()
    text = re.sub(r"^think\s*:\s*", "", text, flags=re.I).strip()
    text = re.sub(r"^Thought\s*:\s*", "", text, flags=re.I).strip()
    text = re.sub(r"^Plan\s*:\s*", "", text, flags=re.I).strip()
    return text


def _print_dist(label, values):
    """Print a count/% distribution table (same style as reflact/bfs)."""
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
    """Save a histogram of trajectory lengths with count/% labels."""
    if not values:
        print(f"  [plot] no data for {label}, skipping {out_path}")
        return
    dist = collections.Counter(values)
    xs = sorted(dist.keys())
    counts = [dist[x] for x in xs]
    total = len(values)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(xs, counts, width=0.8, edgecolor="black", color=color)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{c}\n({c/total*100:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Game of 24 — {label}  ({total} puzzles)\n"
                 f"success rate: {succ*100:.1f}%")
    ax.set_xlabel("Trajectory length (steps)")
    ax.set_ylabel("Number of puzzles")
    ax.set_xticks(xs)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved -> {out_path}")


def game24_run(task_input, env, prompt, llm, to_print=True, oracle_stats=None,
               force_think=True, max_steps=11):
    ob = '\n# Here is the task:\nInput: ' + task_input
    init_prompt = prompt + ob
    prompt = ''
    scorer = TrajectoryScorer(task_input)   # ground-truth per-state labels
    if to_print:
        print(ob)
        sys.stdout.flush()
    reward = 0
    observation = ''
    step_id = 1
    while step_id <= max_steps:
        # ----------------------------------------------------------------
        # 1) Forced visible think step: prime with 'Act k> think:' so the
        #    model produces a visible ReAct thought instead of jumping
        #    straight to an arithmetic action.
        # ----------------------------------------------------------------
        if force_think:
            thought = llm._generate(
                init_prompt + prompt + f'\nAct {step_id}> think:',
                stop='\n',
            )
            thought = _remove_duplicate_think_prefix(thought)
            if not thought:
                thought = ("I will choose a valid operation that keeps the "
                           "puzzle solvable.")
            action = f'think: {thought}'
            lab = scorer.step(action)
            response = env.step(action)
            reward, observation = response['r'], response['ob']
            if to_print:
                print(f'Act {step_id}> {action}    [oracle: {lab["label"]}]\nObs {step_id}> {observation}')
                sys.stdout.flush()
            prompt += f'\nAct {step_id}> {action}\nObs {step_id}> {observation}'
            if reward or 'Exceeded' in observation:
                break
            step_id += 1
            if step_id > max_steps:
                break

        # ----------------------------------------------------------------
        # 2) Arithmetic / final-answer action.
        # ----------------------------------------------------------------
        action = llm._generate(init_prompt + prompt + f'\nAct {step_id}>', stop='\n')
        if not action:
            action = "think: I need to try another valid operation."
        lab = scorer.step(action)           # label this proposal vs. true state
        response = env.step(action)
        reward, observation = response['r'], response['ob']
        if to_print:
            print(f'Act {step_id}> {action}    [oracle: {lab["label"]}]\nObs {step_id}> {observation}')
            sys.stdout.flush()
        prompt += f'\nAct {step_id}> {action}\nObs {step_id}> {observation}'
        if reward or 'Exceeded' in observation:
            break
        step_id += 1

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
    return reward, min(step_id, max_steps)


def main(args):
    env = Game24Task(args.data_file, args.seed, args.num_samples)
    if args.mode == 'react':
        with open(os.path.join(FOLDER, REACT_FILE), 'r') as f:
            BASE_PROMPT = f.read()
            # print("\nBase Prompt is: ", BASE_PROMPT)
        # Reinforce exact visible formatting so Qwen emits 'think: ...' lines
        # instead of XML-style <think> blocks.
        BASE_PROMPT = BASE_PROMPT.rstrip() + (
            "\n\nImportant formatting rule for this run:\n"
            "Use visible ReAct thoughts exactly as 'think: ...'. "
            "Do not use XML tags such as <think>. "
            "When asked for an arithmetic action, output only one line.\n"
        )
    elif args.mode == 'act':
        with open(os.path.join(FOLDER, ACTIONLY_FILE), 'r') as f:
            BASE_PROMPT = f.read()
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Mode must be 'react' or 'act'")

    force_think = (args.mode == 'react')

    llm = VLLM_llm(args.model_name, args.base_url)
    results = []
    oracle_stats = []
    traj_lens = []                     # steps used per episode (all puzzles)
    perfect_lens = []                  # steps used per solved episode

    for i in range(args.num_samples):
        print('---------------------------------------------------------------------------')
        sys.stdout.flush()
        curr_case = env.get_case()
        r, tlen = game24_run(curr_case, env, BASE_PROMPT, llm, oracle_stats=oracle_stats,
                             force_think=force_think)
        results.append(r)
        traj_lens.append(tlen)
        if r:
            perfect_lens.append(tlen)
        if (i + 1) % 1 == 0:
            sr = sum(results) / len(results)
            print(f"id:{i + 1}, success rate:{sr}")
            print('---------------------------------------------------------------------------')
            sys.stdout.flush()

    succ = sum(results) / len(results)
    print(f"Success rate:{succ}")

    # ---- trajectory-length report (same style as reflact/bfs) ---- #
    _print_dist("ALL puzzles", traj_lens)
    _print_dist("PERFECT (solved)", perfect_lens)
    print("\nGenerating histograms ...")
    _plot_dist("trajectory length (all)", traj_lens,
               "traj_len_hist_react_all.png", "#4C72B0", succ)
    _plot_dist("trajectory length (solved)", perfect_lens,
               "traj_len_hist_react_perfect.png", "#55A868", succ)

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
    parser.add_argument("--mode", type=str, default='react', help="act / react")
    parser.add_argument("--data_file", type=str, default=file_path,
                        help="CSV with a 'Puzzles' column; puzzles are played in row order")
    args = parser.parse_args()
    
    main(args)
