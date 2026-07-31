import os
import sys
import json
import argparse
import collections
from game24 import Game24Task
from openai import OpenAI
from typing import Any, List, Dict, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    ax.set_title(f"Game of 24 Reflexion — trajectory length ({label})  "
                 f"({total} puzzles)\nsuccess rate: {succ*100:.1f}%")
    ax.set_xlabel("Trajectory length (actions taken)")
    ax.set_ylabel("Number of puzzles")
    ax.set_xticks(xs)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved -> {out_path}")


# Fixed 200-task set saved from sample.txt so the same exact puzzles play every run.
_HERE = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(_HERE, 'sample_test_tasks.csv')
FOLDER = os.path.join(_HERE, 'prompts') + os.sep
ACTIONLY_FILE = 'game24_base_actionly.txt'
REACT_FILE = 'game24_base_react.txt'

MODEL_NAME = "Qwen/Qwen3-8B"
VLLM_BASE_URL = "http://10.5.18.73:8002/v1"

with open(os.path.join(_HERE, "prompts", "game24_base_react.txt"), 'r') as f:
    FEW_SHOT_EXAMPLES = f.read()


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
        generate_text = response.choices[0].text
        # stop is applied server-side, but keep the split as a safety net
        if stop and stop in generate_text:
            generate_text = generate_text.split(stop)[0]

        action = extract_substring(generate_text)
        if len(action) > 1:
            action = action[0].lower() + action[1:]

        return action
        
        
class EnvironmentHistory:
    def __init__(self, base_query: str, start_info, memory: List[str], history: List[Dict[str, str]] = []) -> None:
        self._cur_query: str = f'{_get_base_query(base_query, start_info, memory)}'
        self._history: List[Dict[str, str]] = history
        self._last_action: str = ''
        self._is_exhausted: bool = False

    def add(self, label: str, value: str) -> None:
        assert label in ['action', 'observation', 'human_edit']
        self._history += [{
            'label': label,
            'value': value,
        }]
        if label == 'action':
            if value == self._last_action:
                self._is_exhausted = True
            else:
                self._last_action = value

    def check_is_exhausted(self) -> bool:
        return self._is_exhausted

    def reset(self) -> None:
        self._history = []

    def __str__(self) -> str:
        s: str = self._cur_query + '\n'
        for i, item in enumerate(self._history):
            if item['label'] == 'action':
                s += f'Act {i // 2 + 1}> {item["value"]}'
            elif item['label'] == 'observation':
                s += f'Obs {i // 2 + 1}> {item["value"]}'
            s += '\n'
        return s


def extract_substring(s: str) -> str:
    start_pos = s.find('>')
    if start_pos > 2:
        start_pos = -1
    return s[start_pos + 1:].strip()


def _get_scenario(s: str) -> str:
    """Parses the relevant scenario from the experience log."""
    return s.split("Here is the task:")[-1].strip()


def _generate_reflection_query(log_str: str, memory: List[str]) -> str:
    """Allows the Agent to reflect upon a past experience."""
    scenario: str = _get_scenario(log_str)
    query: str = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan after "Plan". Here are two examples:

{FEW_SHOT_EXAMPLES}

{scenario}"""

    if len(memory) > 0:
        query += '\n\nPlans from past attempts:\n'
        for i, m in enumerate(memory):
            query += f'Trial #{i}: {m}\n'

    query += '\n\nNew plan>'
    return query


def update_memory(trial_log_path: str, env_configs: List[Dict[str, Any]], model) -> List[Dict[str, Any]]:
    """Updates the given env_config with the appropriate reflections."""
    with open(trial_log_path, 'r', encoding='utf-8') as f:
        full_log: str = f.read()
        
    env_logs: List[str] = full_log.split('#####\n\n#####')
    assert len(env_logs) == len(env_configs), print(f'bad: {len(env_logs)}, {len(env_configs)}')
    for i, env in enumerate(env_configs):
        # if unsolved, get reflection and update env config
        if not env['is_success'] and not env['skip']:
            if len(env['memory']) > 3:
                memory: List[str] = env['memory'][-3:]
            else:
                memory: List[str] = env['memory']
            reflection_query: str = _generate_reflection_query(env_logs[i], memory)
            reflection: str = model._generate(reflection_query, '\n').strip()
            env_configs[i]['memory'] += [reflection]
                
    return env_configs


def _get_base_query(base_query: str, start_info: str, memory: List[str]) -> str:
    query = base_query

    # add memory if it exists
    if len(memory) > 0:
        query += '\n\nYour memory for the task below:'
        for i, m in enumerate(memory):
            query += f'\nTrial {i}:\n{m.strip()}'
    query += f"\nHere is the task:\n{start_info}"
    return query


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob


def game24_run(env, base_prompt, memory: List[str], model, to_print=True, ob=''):
    if len(memory) > 3:
        env_history = EnvironmentHistory(base_prompt, ob, memory[-3:], [])
    else:
        env_history = EnvironmentHistory(base_prompt, ob, memory, [])
    env_history.reset()
    if to_print:
        print(ob)
        sys.stdout.flush()
    cur_step = 0
    while cur_step < 12:
        action = model._generate(str(env_history) + f"Act {cur_step + 1}>", '\n').strip()
        env_history.add("action", action)

        response = env.step(action)
        reward, observation = response['r'], response['ob']

        env_history.add("observation", observation)
        if to_print:
            print(f'Act {cur_step + 1}> {action}\nObs {cur_step + 1}> {observation}')
            sys.stdout.flush()
        # reward = 0 / 1
        if reward:
            return env_history, True
        elif 'Exceeded' in observation:
            return env_history, False
        cur_step += 1
    return env_history, False


def run_trial(
        seed: int,
        num_samples: int,
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        mode: str,
        env_configs: List[Dict[str, Any]],
        use_memory: bool,
        model,
    ) -> List[Dict[str, Any]]:
    env = Game24Task(file_path, seed, num_samples)

    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)

    for z, env_config in enumerate(env_configs):
        curr_case = env.get_case()
        ob = '\n# Here is the task:\nInput: ' + curr_case

        if env_config["is_success"]:
            num_successes += 1

            # log to world log
            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a', encoding='utf-8') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            print(f'[Trial #{trial_idx}] Episode {z + 1}/{num_envs} completed, '
                  f'success rate: {num_successes}/{z + 1} = {num_successes/(z + 1):.3f}')
            sys.stdout.flush()
            continue

        if mode == 'react':
            with open(os.path.join(FOLDER, REACT_FILE), 'r') as f:
                BASE_PROMPT = f.read()
        elif mode == 'act':
            with open(os.path.join(FOLDER, ACTIONLY_FILE), 'r') as f:
                BASE_PROMPT = f.read()
        final_env_history, is_success = game24_run(env, BASE_PROMPT, env_config["memory"] if use_memory else [], model=model, to_print=True, ob=ob)

        # trajectory length = number of actions taken this attempt
        traj_len = sum(1 for it in final_env_history._history if it['label'] == 'action')
        env_configs[z].setdefault('traj_lens', []).append(traj_len)

        # update env config
        if is_success:
            status_str: str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
            env_configs[z]['is_success'] = True
            env_configs[z]['win_len'] = traj_len
            num_successes += 1
            num_additional_successes += 1
        else:
            status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'

        # log to world log
        with open(world_log_path, 'a') as f:
            f.write(status_str + '\n')

        # log env results to trial log
        with open(trial_log_path, 'a', encoding='utf-8') as wf:
            wf.write(f'\n#####\n\nEnvironment #{z}:\n{str(final_env_history)}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')

        print(f'[Trial #{trial_idx}] Episode {z + 1}/{num_envs} completed, '
              f'success rate: {num_successes}/{z + 1} = {num_successes/(z + 1):.3f}')
        sys.stdout.flush()

    # log trial results to trial and world logs
    log_str: str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 3)}
-----"""
    print(log_str)
    with open(trial_log_path, 'a', encoding='utf-8') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    return env_configs


def main(args) -> None:
    if args.is_resume:
        if not os.path.exists(args.resume_dir):
            raise ValueError(f"Resume directory `{args.resume_dir}` does not exist")
        logging_dir = args.resume_dir

        # load environment configs
        env_config_path: str = os.path.join(args.resume_dir, f'env_results_trial_{args.start_trial_num - 1}.json')
        if not os.path.exists(env_config_path):
            raise ValueError(f"Environment config file `{env_config_path}` does not exist")
        with open(env_config_path, 'r') as rf:
            env_configs: List[Dict[str, Any]] = json.load(rf)
    else:
        # Create the run directory
        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)
        logging_dir = args.run_name

        # initialize environment configs
        env_configs: List[Dict[str, Any]] = []
        for i in range(args.num_envs):
            env_configs += [{
                'name': f'env_{i}',
                'memory': [],
                'is_success': False,
                'skip': False,
                'traj_lens': [],     # trajectory length of each attempt
                'win_len': None,     # trajectory length when solved
            }]
    
    world_log_path: str = os.path.join(logging_dir, 'world.log')

    # print start status to user
    if args.is_resume:
        print(f"""
    -----
    Resuming run with the following parameters:
    Run name: {logging_dir}
    Number of trials: {args.num_trials}
    Number of environments: {args.num_envs}
    Use memory: {args.use_memory}
    Resume trial number: {args.start_trial_num}

    Sending all logs to `{args.run_name}`
    -----
    """)
    else:
        print(f"""
    -----
    Starting run with the following parameters:
    Run name: {logging_dir}
    Number of trials: {args.num_trials}
    Number of environments: {args.num_envs}
    Use memory: {args.use_memory}

    Sending all logs to `{args.run_name}`
    -----
    """)

    # run trials
    trial_idx = args.start_trial_num
    llm = VLLM_llm(args.model_name, args.base_url)
    while trial_idx < args.num_trials:
        with open(world_log_path, 'a') as wf:
            wf.write(f'\n\n***** Start Trial #{trial_idx} *****\n\n')

        # set paths to log files
        trial_log_path: str = os.path.join(args.run_name, f'trial_{trial_idx}.log')
        trial_env_configs_log_path: str = os.path.join(args.run_name, f'env_results_trial_{trial_idx}.json')
        if os.path.exists(trial_log_path):
            open(trial_log_path, 'w').close()
        if os.path.exists(trial_env_configs_log_path):
            open(trial_env_configs_log_path, 'w').close()

        # run trial
        run_trial(args.seed, args.num_envs, trial_log_path, world_log_path, trial_idx, args.mode, env_configs, args.use_memory, llm)

        # update memory if needed
        if args.use_memory:
            env_configs: List[Dict[str, Any]] = update_memory(trial_log_path, env_configs, llm)

        # log env configs for trial
        with open(trial_env_configs_log_path, 'w') as wf:
            json.dump(env_configs, wf, indent=4)

        # log world for trial
        with open(world_log_path, 'a') as wf:
            wf.write(f'\n\n***** End Trial #{trial_idx} *****\n\n')

        trial_idx += 1

    # ── FINAL RESULTS across all trials ─────────────────────────────────── #
    total = len(env_configs)
    n_solved = sum(1 for e in env_configs if e['is_success'])
    succ = (n_solved / total) if total else 0.0
    # winning-trajectory length for solved puzzles
    perfect_traj_lengths = [e['win_len'] for e in env_configs if e.get('win_len') is not None]
    # per-puzzle trajectory length: the winning attempt if solved, else last attempt
    all_traj_lengths = [
        (e['win_len'] if e.get('win_len') is not None
         else (e['traj_lens'][-1] if e.get('traj_lens') else 0))
        for e in env_configs
    ]
    avg_all = (sum(all_traj_lengths) / len(all_traj_lengths)) if all_traj_lengths else 0.0
    avg_prf = (sum(perfect_traj_lengths) / len(perfect_traj_lengths)) if perfect_traj_lengths else None

    print("\n" + "=" * 72)
    print(f"  FINAL RESULTS  (after {args.num_trials} trial(s))")
    print(f"  Puzzles run             : {total}  (num_envs={args.num_envs}, seed={args.seed})")
    print(f"  Success rate (solved)   : {n_solved}/{total} = {succ:.3f}")
    print(f"  Avg traj len (all)      : {avg_all:.2f}")
    if avg_prf is not None:
        print(f"  Avg traj len (perfect)  : {avg_prf:.2f}")
    _print_dist("ALL puzzles", all_traj_lengths)
    _print_dist("PERFECT (solved)", perfect_traj_lengths)
    print("=" * 72)

    print("\nGenerating histograms ...")
    _plot_dist("ALL", all_traj_lengths,
               os.path.join(logging_dir, "traj_len_hist_all_reflexion.png"), "#4C72B0", succ)
    _plot_dist("PERFECT", perfect_traj_lengths,
               os.path.join(logging_dir, "traj_len_hist_perfect_reflexion.png"), "#55A868", succ)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Sample seed")
    parser.add_argument("--num_trials", type=int, default=3, help="The number of trials to run")
    parser.add_argument("--num_envs", type=int, default=100, help="The number of environments per trial")
    parser.add_argument("--run_name", type=str, default="reflexion_run", help="The name of the run")
    parser.add_argument("--use_memory", action='store_true', help="Allow the Agent to use memory")
    parser.add_argument("--is_resume", action='store_true', help="To resume run")
    parser.add_argument("--resume_dir", type=str, help="If resume, the logging directory", default="")
    parser.add_argument("--start_trial_num", type=int, help="If resume, the start trial num", default=0)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="model name served by vLLM")
    parser.add_argument("--base_url", type=str, default=VLLM_BASE_URL, help="vLLM OpenAI-compatible base url")
    parser.add_argument("--mode", type=str, default='react', help="act / react")
    args = parser.parse_args()

    assert args.num_trials > 0, "Number of trials should be positive"
    assert args.num_envs > 0, "Number of environments should be positive"

    main(args)
    