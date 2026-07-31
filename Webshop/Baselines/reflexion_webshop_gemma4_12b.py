"""Reflexion (Shinn et al., NeurIPS 2023) on WebShop, layered on the canonical
ReAct WebShop notebook (https://github.com/ysymyth/ReAct/blob/master/WebShop.ipynb).

Identical to reflexion_webshop_gpt4.1.py except the inference model: the Azure
OpenAI gpt-4.1 chat endpoint is replaced by the local vLLM Gemma-4-12B-it
completions endpoint (same backend as react_webshop_qwen3_8b.py). Since it is a
text-completion endpoint, the chat system message and action-extraction regex
are dropped — the prompt ending in "Action:" is continued literally. The
Qwen3-specific "/no_think" prefix is not needed for Qwen2.5.

Pipeline mirrors `noahshinn/reflexion/webshop_runs/`:
  - Outer loop: per env (one WebShop session `fixed_<i>`), up to --num-trials
    trials. After each FAILED trial, generate a verbal "reflection" / new plan,
    append it to that env's memory (capped at the last 3), and retry. On success
    (reward == 1.0) skip the remaining trials for that env.
  - Inner trial: one ReAct rollout — the notebook's webshop_run, using the
    one-shot `think[]` prompt (prompt1) with a "Your memory for the task below:"
    block (last 3 reflections) appended to the few-shot, so it precedes the live
    task instruction (mirrors Reflexion's EnvironmentHistory).
  - Evaluator (heuristic): success iff the episode finishes with reward == 1.0
    (a "perfect" purchase). A trial FAILS on: a Buy Now that scores < 1.0, the
    per-trial step cap, OR 3 consecutive identical executed actions (Reflexion's
    exhaustion heuristic).
  - Reflection: a separate LLM call with the failed trajectory + past reflections
    + the 2 few-shot examples in reflexion_webshop_few_shot_examples.txt; returns
    a "New plan: ..." string.

Examples:
  # Single-env smoke (3 trials of fixed_0):
  python reflexion_webshop_gemma4_12b.py --num-episodes 1 --num-trials 3 --max-steps 15

  # 200-env eval, 3 outer trials each, with JSONL + resume:
  python reflexion_webshop_gemma4_12b.py --num-episodes 200 --num-trials 3 \\
      --results reflexion_webshop_gemma4_12b_results.jsonl --resume
"""

import os
import re
import sys
import json
import time
import re
import argparse
from collections import Counter
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from openai import OpenAI

# ─── config ──────────────────────────────────────────────────────────────────
WEBSHOP_URL = "http://10.5.30.30:3000/"                       # live local server
MODEL_NAME = "google/gemma-4-12B-it"
client     = OpenAI(base_url="http://10.5.18.73:8003/v1", api_key="EMPTY")  # vLLM
MAX_STEPS   = 15
MAX_TOKENS  = 256
TEMPERATURE = 0.0

# How many past reflections to include in the prompt (Reflexion's Ω).
MAX_MEMORY = 3
# Exhaustion: end the trial after this many consecutive identical executed actions.
MAX_CONSECUTIVE_REPEATS = 3
# Cap the trajectory length when serialising it into the reflection prompt.
REFLECTION_TRAJ_LAST_N = 40

RESULTS_LOG = "reflexion_webshop_gemma4_12b_results.txt"
PLOT_PREFIX = "reflexion_webshop_gemma4_12b_traj_hist"

# Reflection few-shot examples (WebShop-specific) — shared with the parent dir.
_HERE = os.path.dirname(os.path.abspath(__file__))
REFLEXION_FEW_SHOT_PATH = os.path.join(
    _HERE, "prompts", "reflexion_webshop_few_shot_examples.txt")
with open(REFLEXION_FEW_SHOT_PATH) as f:
    REFLEXION_FEW_SHOT = f.read()


# Gemma-4-12B-it is instruction-tuned: on the raw /completions endpoint it does
# NOT continue "Action:" literally — it degenerates (corrupted tokens, loops
# search<->back and never buys). So use the chat endpoint with a system message
# that pins the output to exactly one action, then strip any "Action:"/"Action N:"
# prefix the model still adds. (Same fix web_rollback_gemma4_12b.py already applies.)
_ACTION_PREFIX_RE = re.compile(r'^\s*Action\s*\d*\s*:\s*', re.IGNORECASE)


def llm(prompt, stop=["\n"]):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content":
                "You are acting in the WebShop environment. Reply with EXACTLY ONE action "
                "and nothing else. Valid actions: search[...], click[...], think[...]. "
                "Do not write 'Action:', explanations, or any prose — output only the single "
                "action, e.g. search[red running shoes] or click[Buy Now]."},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop,
    )
    text = (response.choices[0].message.content or "").strip()
    text = _ACTION_PREFIX_RE.sub("", text)     # drop a leading "Action:"/"Action 3:"
    text = text.split("\n", 1)[0].strip()      # keep only the first action line
    return text


# ─── WebShop env (verbatim from the notebook / reflact_webshop.py) ───────────
ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}


def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )


def webshop_text(session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
    # Percent-encode LLM-controlled URL parts so values containing '#', '&', '?',
    # spaces, etc. aren't truncated at the URL fragment delimiter and crash the
    # server's literal_eval. (Ported from the BFS method.)
    qs   = quote(query_string, safe='')
    opts = quote(str(options),  safe='')
    if page_type == 'init':
      url = (
          f'{WEBSHOP_URL}/{session}'
      )
    if page_type == 'search':
      url = (
          f'{WEBSHOP_URL}/search_results/{session}/'
          f'{qs}/{page_num}'
      )
    elif page_type == 'item':
      url = (
          f'{WEBSHOP_URL}/item_page/{session}/'
          f'{asin}/{qs}/{page_num}/{opts}'
      )
    elif page_type == 'item_sub':
      url = (
          f'{WEBSHOP_URL}/item_sub_page/{session}/'
          f'{asin}/{qs}/{page_num}/{subpage}/{opts}'
      )
    elif page_type == 'end':
      url = (
          f'{WEBSHOP_URL}/done/{session}/'
          f'{asin}/{opts}'
      )
    html = requests.get(url).text
    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.findAll(text=True)
    visible_texts = list(filter(tag_visible, texts))
    if False:
        return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
    else:
        observation = ''
        option_type = ''
        options = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0
        for t in visible_texts:
            if t == '\n': continue
            if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue
            if t.parent.name == 'button':  # button
                processed_t = f'\n[{t}] '
            elif t.parent.name == 'label':  # options
                if f"'{t}'" in url:
                    processed_t = f'[[{t}]]'
                else:
                    processed_t = f'[{t}]'
                options[str(t)] = option_type
            elif t.parent.get('class') == ["product-link"]: # product asins
                processed_t = f'\n[{t}] '
                if prod_cnt >= 3:
                  processed_t = ''
                prod_cnt += 1
                asins.append(str(t))
                just_prod = 0
            else: # regular, unclickable text
                processed_t =  '\n' + str(t) + ' '
                if cnt < 2 and page_type != 'init': processed_t = ''
                if just_prod <= 2 and prod_cnt >= 4: processed_t = ''
                option_type = str(t)
                cnt += 1
            just_prod += 1
            observation += processed_t
        info = {}
        if options:
          info['option_types'] = options
        if asins:
          info['asins'] = asins
        if 'Your score (min 0.0, max 1.0)' in visible_texts:
          idx = visible_texts.index('Your score (min 0.0, max 1.0)')
          info['reward'] = float(visible_texts[idx + 1])
          observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
        return clean_str(observation), info


class webshopEnv:
  def __init__(self):
    self.sessions = {}

  def step(self, session, action):
    done = False
    observation_ = None
    if action == 'reset':
      self.sessions[session] = {'session': session, 'page_type': 'init'}
    elif action.startswith('think['):              # ReAct reasoning, no-op
      observation = 'OK.'
    elif action.startswith('search['):
      assert self.sessions[session]['page_type'] == 'init'
      query = action[7:-1]
      self.sessions[session] = {'session': session, 'page_type': 'search',
                                'query_string': query, 'page_num': 1}
    elif action.startswith('click['):
      button = action[6:-1]
      if button == 'Buy Now':
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'end'
        done = True
      elif button == 'Back to Search':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        self.sessions[session] = {'session': session, 'page_type': 'init'}
      elif button == 'Next >':
        assert False # ad hoc page limitation
        assert self.sessions[session]['page_type'] == 'search'
        self.sessions[session]['page_num'] += 1
      elif button == '< Prev':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        if self.sessions[session]['page_type'] == 'search':
          assert False
          self.sessions[session]['page_num'] -= 1
        elif self.sessions[session]['page_type'] == 'item_sub':
          self.sessions[session]['page_type'] = 'item'
        elif self.sessions[session]['page_type'] == 'item':
          self.sessions[session]['page_type'] = 'search'
          self.sessions[session]['options'] = {}
      elif button in ACTION_TO_TEMPLATE:
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'item_sub'
        self.sessions[session]['subpage'] = button
      else:
        if self.sessions[session]['page_type'] == 'search':
          assert button in self.sessions[session].get('asins', [])  # must be asins
          self.sessions[session]['page_type'] = 'item'
          self.sessions[session]['asin'] = button
        elif self.sessions[session]['page_type'] == 'item':
          assert 'option_types' in self.sessions[session]
          opt_types = self.sessions[session]['option_types']
          # Case-insensitive match: the LLM often copies the casing from the
          # product title while the page label is lowercase. (Ported from BFS.)
          if button not in opt_types:
            ci_map = {k.lower(): k for k in opt_types}
            if button.lower() in ci_map:
              button = ci_map[button.lower()]
            else:
              assert False, (button, opt_types)
          option_type = opt_types[button]
          if not 'options' in self.sessions[session]:
            self.sessions[session]['options'] = {}
          self.sessions[session]['options'][option_type] = button
          observation_ = f'You have clicked {button}.'
    else:
      assert False
    observation, info = webshop_text(**self.sessions[session])
    if observation_:
      observation = observation_
    self.sessions[session].update(info)
    reward = info.get('reward', 0.0)
    return observation, reward, done


env = webshopEnv()


# ─── ReAct one-shot prompt (the notebook's prompt1, think[]) ─────────────────
prompt1 = """Webshop
Instruction:
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
[Search]

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B078GWRC1J]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
$10.99
[B078GTKVXY]
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
$10.99
[B08KBVJ4XN]
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack)
$15.95

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]
Observation: OK.

Action: click[B078GWRC1J]
Observation:
[Back to Search]
[< Prev]
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
Price: $10.99
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus.

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).

Action: click[Buy Now]

Note: search[...] is only valid from the initial search page. If you are on a search-results or item page and no item matches the instruction, first do click[Back to Search], then search again with a revised query.
"""


# ─── memory-augmented prompt (Reflexion's EnvironmentHistory) ────────────────
def build_init_prompt(memory):
    """Append the last-3-reflection memory block to the few-shot prompt, so it
    precedes the live task instruction in the rollout."""
    if not memory:
        return prompt1
    mem_block = "\nYour memory for the task below:\n"
    for i, m in enumerate(memory):
        mem_block += f"Trial {i}:\n{m.strip()}\n"
    return prompt1 + mem_block


# ─── rollout loop (notebook's webshop_run; + transcript + exhaustion) ────────
def webshop_run(idx, memory, to_print=True):
  """One ReAct rollout for session `idx` with `memory` prepended.
  Returns (reward, n_real, done, transcript, exhausted, instruction)."""
  init_prompt = build_init_prompt(memory)
  action = 'reset'
  prompt = ''
  n_real = 0                                   # trajectory length = search[]/click[] only
  forced_next = None                           # action queued after a forced click[Back to Search]
  transcript = []
  instruction = ''
  reward = 0.0
  done = False
  last_exec = None
  consec_same = 0
  exhausted = False

  for i in range(MAX_STEPS):
    invalid = False
    try:
      res = env.step(idx, action)
      observation = res[0]
    except AssertionError:
      observation = 'Invalid action!'
      invalid = True
      res = (observation, 0.0, False)

    if action.startswith('think'):
      observation = 'OK.'

    real_action = (not invalid) and (action.startswith('search[') or action.startswith('click['))
    if real_action:
      n_real += 1

    # First env observation is the task instruction page.
    if i == 0:
      instruction = observation.strip()

    if action != 'reset':
      transcript.append({"step": i, "action": action, "observation": observation})

    if to_print:
      print(f'Action: {action}\nObservation: {observation}\n')
      sys.stdout.flush()
    if i:
      prompt += f' {action}\nObservation: {observation}\n\nAction:'
    else:
      prompt += f'{observation}\n\nAction:'

    reward, done = res[1], res[2]
    if done:
      return reward, n_real, done, transcript, exhausted, instruction

    # Exhaustion heuristic: 3 consecutive identical executed (real) actions.
    if real_action:
      if action == last_exec:
        consec_same += 1
      else:
        consec_same = 1
        last_exec = action
      if consec_same >= MAX_CONSECUTIVE_REPEATS:
        exhausted = True
        if to_print:
          print(f'[exhaustion] action {action!r} repeated {consec_same}x; ending trial.')
          sys.stdout.flush()
        return reward, n_real, done, transcript, exhausted, instruction

    if forced_next is not None:
      action = forced_next                     # execute the deferred search now (page is 'init')
      forced_next = None
    else:
      action = llm(init_prompt + prompt[-(6400-len(init_prompt)):], stop=['\n']).lstrip(' ')
      # Auto-redirect: search[] is only valid from the init page.
      if (action.startswith('search[')
          and env.sessions.get(idx, {}).get('page_type', 'init') != 'init'):
        if to_print:
          print('  [auto-redirect] search[] proposed off the init page '
                '-> inserting click[Back to Search] first')
          sys.stdout.flush()
        forced_next = action
        action = 'click[Back to Search]'

  return reward, n_real, done, transcript, exhausted, instruction


# ─── reflection generation (Reflexion's self-reflection LLM call) ────────────
def _serialize_trajectory(instruction, transcript, status="FAIL"):
    """Serialise a (failed) trial's trajectory for the reflection prompt, in the
    same Action/Observation format as the few-shot examples."""
    body = transcript[-REFLECTION_TRAJ_LAST_N:]
    lines = ["Webshop", "Instruction:", instruction]
    if len(transcript) > REFLECTION_TRAJ_LAST_N:
        lines.append(f"(... earlier {len(transcript) - REFLECTION_TRAJ_LAST_N} steps truncated ...)")
    for entry in body:
        lines.append(f"Action: {entry['action']}")
        lines.append(f"Observation: {entry['observation']}")
    lines.append(f"STATUS: {status}")
    return "\n".join(lines)


def generate_reflection(instruction, transcript, memory):
    """Reflexion-style reflection query: failed trajectory + past reflections
    + 2 few-shot examples; the LLM returns the 'New plan' text."""
    scenario = _serialize_trajectory(instruction, transcript, status="FAIL")
    query = (
        "You will be given the history of a past experience in which you were placed in a "
        "WebShop environment and given an instruction to buy an item. You were unsuccessful "
        "(you did not earn a perfect score of 1.0). Do not summarize the environment, but "
        "rather think about the strategy and path you took to attempt to complete the task. "
        "Devise a concise, new plan of action that accounts for your mistake with reference "
        "to specific actions you should have taken (e.g. choosing the cheapest in-budget item, "
        "selecting every required option before Buy Now, or revising the search query). You "
        "will need this later when solving the same task. Give your plan after \"New plan\". "
        "Here are two examples:\n\n"
        f"{REFLEXION_FEW_SHOT}\n\n"
        f"{scenario}"
    )
    if memory:
        query += "\n\nPlans from past attempts:"
        for i, m in enumerate(memory):
            query += f"\nTrial #{i}: {m}"
    query += "\n\nNew plan:"

    # Chat endpoint (Gemma-it degenerates on raw /completions — see llm() above).
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": query}],
        n=1,
        temperature=0.1,
        max_tokens=256,
    )
    text = (resp.choices[0].message.content or "").strip()
    # Stop at obvious continuation markers if the model rambles past its plan.
    for stop in ("\nWebshop", "\nInstruction:", "\nSTATUS:", "\nTrial #", "\n\nNew plan"):
        cut = text.find(stop)
        if cut >= 0:
            text = text[:cut]
    return text.strip()


# ─── resume support ──────────────────────────────────────────────────────────
def _load_resume(results_path):
    """Read existing JSONL records to restore per-env state.
    Returns {idx: {'is_success': bool, 'memory': [reflection, ...], 'best_reward': float}}."""
    state = {}
    if not (results_path and os.path.exists(results_path)):
        return state
    with open(results_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            key = rec.get("idx")
            entry = state.setdefault(
                key, {"is_success": False, "memory": [], "best_reward": 0.0, "repr_len": 0})
            if rec.get("won"):
                entry["is_success"] = True
            r = rec.get("reward", 0.0) or 0.0
            if r >= entry["best_reward"]:
                entry["repr_len"] = rec.get("n_real", 0) or 0
            entry["best_reward"] = max(entry["best_reward"], r)
            if rec.get("reflection"):
                entry["memory"].append(rec["reflection"])
    for v in state.values():
        if len(v["memory"]) > MAX_MEMORY:
            v["memory"] = v["memory"][-MAX_MEMORY:]
    return state


# ─── outer Reflexion loop ────────────────────────────────────────────────────
def run_reflexion(env_configs, num_trials, results_path, to_print=True):
    """Outer Reflexion loop. Writes one JSONL record per (env, trial)."""
    out = None
    if results_path:
        os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
        out = open(results_path, "a")

    n_envs = len(env_configs)
    t_start = time.time()

    for trial_idx in range(num_trials):
        already = sum(1 for c in env_configs if c["is_success"])
        print("\n" + "=" * 80)
        print(f"TRIAL #{trial_idx}  (already succeeded: {already}/{n_envs})")
        print("=" * 80)
        sys.stdout.flush()

        for z, cfg in enumerate(env_configs):
            if cfg["is_success"]:
                continue
            print(f"\n[Env {z + 1}/{n_envs}] Trial {trial_idx}  session={cfg['session']}")
            sys.stdout.flush()

            t0 = time.time()
            try:
                reward, n_real, done, transcript, exhausted, instruction = webshop_run(
                    cfg["session"], cfg["memory"][-MAX_MEMORY:], to_print=to_print,
                )
                err = None
            except AssertionError as ex:
                reward, n_real, done = 0.0, 0, False
                transcript, exhausted, instruction = [], False, ""
                err = f"AssertionError: {ex}"
            except Exception as ex:
                reward, n_real, done = 0.0, 0, False
                transcript, exhausted, instruction = [], False, ""
                err = f"{type(ex).__name__}: {ex}"
            dt = time.time() - t0

            won = (reward == 1.0)
            # Track the trajectory length of the best-scoring trial (ties -> latest),
            # used as this env's representative length for the ALL-games distribution.
            if reward >= cfg["best_reward"]:
                cfg["repr_len"] = n_real
            cfg["best_reward"] = max(cfg["best_reward"], reward)

            reflection = None
            if not won and err is None and transcript:
                if to_print:
                    print("\n[Reflexion] Generating reflection...")
                    sys.stdout.flush()
                try:
                    reflection = generate_reflection(
                        instruction, transcript, cfg["memory"][-MAX_MEMORY:]
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
            elif reflection:
                cfg["memory"].append(reflection)
                cfg["memory"] = cfg["memory"][-MAX_MEMORY:]

            rec = {
                "trial_idx":   trial_idx,
                "idx":         cfg["idx"],
                "session":     cfg["session"],
                "instruction": instruction,
                "reward":      reward,
                "won":         bool(won),
                "exhausted":   bool(exhausted),
                "n_real":      n_real,
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
            print(f"  reward={reward}  won={won}  exhausted={exhausted}  "
                  f"steps={n_real}  ({dt:.1f}s)  |  overall {n_succ}/{n_envs} "
                  f"({n_succ / n_envs * 100:.1f}%)")
            sys.stdout.flush()

        if all(c["is_success"] for c in env_configs):
            print("\n[Reflexion] All envs solved; stopping trials early.")
            break

    if out:
        out.close()

    total_dt = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"DONE  ({total_dt / 60:.1f} min)")
    print("=" * 80)

    # Per-env final outcome: best reward across trials, and the trajectory length
    # of that best-scoring trial. Distributions are reported for ALL envs and for
    # PERFECT (reward==1.0) envs separately (mirrors reflact_webshop.py).
    rewards = [c["best_reward"] for c in env_configs]
    lengths = [c["repr_len"] for c in env_configs]
    report_trajectory_lengths(rewards, lengths, num_trials)


# ─── reporting / histograms (mirrors reflact_webshop.py) ─────────────────────
def plot_hist(label, lengths, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts_by = Counter(lengths)
    steps  = sorted(counts_by)
    counts = [counts_by[s] for s in steps]
    total  = sum(counts)
    color  = "#4C72B0" if "ALL" in label else "#55A868"
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(steps, counts, width=0.8, edgecolor="black", color=color)
    for b, c in zip(bars, counts):
        pct = 100.0 * c / total if total else 0.0
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(counts) * 0.01,
                f"{c}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Trajectory length (real search/click actions)")
    ax.set_ylabel("Number of episodes")
    ax.set_title(f"WebShop Reflexion — trajectory length [{label}]  ({total} episodes)")
    ax.set_xticks(steps)
    ax.set_ylim(0, max(counts) * 1.18 if counts else 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def report_trajectory_lengths(rewards, lengths, num_trials):
    """Trajectory-length distribution for ALL envs and for PERFECT(=1.0) envs
    separately, after the final trial (mirrors reflact_webshop.py). Each env's
    length is the trajectory length of its best-scoring trial."""
    n = len(rewards)
    perfect_lengths = [l for l, r in zip(lengths, rewards) if r == 1.0]
    avg_all = sum(lengths) / n if n else 0.0
    avg_prf = sum(perfect_lengths) / len(perfect_lengths) if perfect_lengths else 0.0
    sr      = len(perfect_lengths) / n if n else 0.0

    out = []
    def emit(s):
        print(s); out.append(s)

    emit("")
    emit(f"=== SUMMARY: {n} envs | {num_trials} trials/env | "
         f"avg_best_reward={sum(rewards)/n if n else 0:.4f} | success_rate={sr:.4f} ===")
    emit(f"Perfect=1.0 : {sr:.4f}")
    emit(f"Avg traj len (all) : {avg_all:.2f}")
    emit(f"Avg traj len (perfect) : {avg_prf:.2f}")

    blocks = [("ALL", lengths)]
    if perfect_lengths:
        blocks.append(("PERFECT=1.0", perfect_lengths))
    for label, data in blocks:
        total = len(data)
        emit("")
        emit("=" * 60)
        emit(f"TRAJECTORY LENGTH DISTRIBUTION ({label})")
        emit(f'{"Steps":>6}  {"Count":>6}   Pct')
        cby = Counter(data)
        for steps in sorted(cby):
            c = cby[steps]
            pct = 100.0 * c / total if total else 0.0
            emit(f"{steps:>6}  {c:>6}   {pct:5.1f}%")
        emit("=" * 60)

    with open(RESULTS_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"[distribution written to {RESULTS_LOG}]")

    try:
        plot_hist("ALL", lengths, f"{PLOT_PREFIX}_all.png")
        if perfect_lengths:
            plot_hist("PERFECT=1.0", perfect_lengths, f"{PLOT_PREFIX}_perfect.png")
        else:
            print("[no perfect (score=1.0) episodes — skipping perfect histogram]")
    except Exception as e:
        print(f"[plotting skipped: {e}]")


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Reflexion on WebShop (multi-trial ReAct with verbal self-reflection).")
    p.add_argument("--num-episodes", type=int, default=200, help="Number of WebShop envs (fixed_<start..start+n>).")
    p.add_argument("--start",        type=int, default=0,  help="Starting env index (fixed_<start>).")
    p.add_argument("--num-trials",   type=int, default=3,  help="Reflexion: outer trials per env.")
    p.add_argument("--max-steps",    type=int, default=MAX_STEPS, help="Per-trial step cap.")
    p.add_argument("--results",      type=str, default=None, help="JSONL output path (optional).")
    p.add_argument("--resume",       action="store_true",
                   help="Restore per-env memory + is_success from an existing --results JSONL.")
    return p.parse_args()


def main():
    print("WebShop Reflexion (multi-trial ReAct + verbal self-reflection, vLLM Gemma-4-12B-it)")
    args = parse_args()

    global MAX_STEPS
    MAX_STEPS = args.max_steps

    env_configs = [
        {
            "idx":         i,
            "session":     f"fixed_{i}",
            "memory":      [],
            "is_success":  False,
            "best_reward": 0.0,
            "repr_len":    0,      # traj length of the best-scoring trial (for ALL-games dist)
        }
        for i in range(args.start, args.start + args.num_episodes)
    ]

    if args.resume:
        state = _load_resume(args.results)
        if state:
            for cfg in env_configs:
                if cfg["idx"] in state:
                    cfg["is_success"]  = state[cfg["idx"]]["is_success"]
                    cfg["memory"]      = state[cfg["idx"]]["memory"][-MAX_MEMORY:]
                    cfg["best_reward"] = state[cfg["idx"]]["best_reward"]
                    cfg["repr_len"]    = state[cfg["idx"]]["repr_len"]
            n_done = sum(1 for c in env_configs if c["is_success"])
            n_mem  = sum(1 for c in env_configs if c["memory"])
            print(f"[Resume] restored: {n_done} already-solved envs + {n_mem} envs with memory.")

    print(f"Envs: {len(env_configs)}  |  Trials per env: {args.num_trials}  |  step_cap={MAX_STEPS}")
    if args.results:
        print(f"Results appended to: {args.results}")
    else:
        print("Results: (no JSONL output — --results not set)")

    run_reflexion(env_configs, args.num_trials, args.results, to_print=True)


if __name__ == "__main__":
    main()
