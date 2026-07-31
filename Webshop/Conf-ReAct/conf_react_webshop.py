"""
BFS Conformal Planning for WebShop — CSV-instruction variant of bfs_2.py.

What's different from bfs_2.py:
  1. Loads (id, Original_instruction) pairs from
     WebShop_ScoreFunc_Train_Data.csv.
  2. Saves them to webshop_csv_instructions.json on disk.
  3. Iterates BFS over CSV ids (instead of range(n_episodes)) and at episode
     start, prints both the CSV-recorded instruction and the WebShop env's
     actual instruction for that session id. Tags each episode [MATCH] /
     [MISMATCH] so you can see whether CSV ids align with env sessions.
  4. By default, mismatched episodes are SKIPPED. Use --include_mismatched
     to run them anyway.

Everything else (BFS, prompt, predictor, env helpers) is identical to bfs_2.py.
"""

import os
import re
import copy
import collections
import math
import ast
import csv
import json
import sys
import argparse
import torch
import pickle
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from bs4.element import Comment
from openai import OpenAI                              # For vllm use
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote

from score_model_webshop import (
    ScoreFunctionWebShop, BertEmbeddingCache,
    SOFTMAX_BINS, BERT_DIM, _logprobs_to_softmax_bins,
)
from conformal_predictor_webshop import ConformalPredictorWebShop

# ── Paths ──────────────────────────────────────────────────────────────────────
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_HERE           = os.path.dirname(os.path.abspath(__file__))
DATA_PATH       = os.path.join(_HERE, "WebShop_ScoreFunc_Train_Data.csv")
MODEL_PATH      = os.path.join(_HERE, "trained_models_webshop", "best_score_model_webshop.pt")
BERT_CACHE_PATH = os.path.join(_HERE, "trained_models_webshop", "bert_cache_webshop.pkl")
WEBSHOP_URL     = "http://10.5.30.30:3000/"
INSTRUCTIONS_DUMP = "webshop_csv_instructions.json"

MODEL_NAME = "google/gemma-4-12B-it"
client = OpenAI(base_url="http://10.5.18.73:8003/v1", api_key="EMPTY") 
# MODEL_NAME = "Qwen/Qwen3-8B"                                                # For vllm use
# client = OpenAI(base_url="http://10.5.30.30:8001/v1", api_key="EMPTY")      # For vllm use

ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features':    'features_page.html',
    'Reviews':     'review_page.html',
    'Attributes':  'attributes_page.html',
}


FEW_SHOT_PROMPT = """You are an agent in a WebShop environment.
Your goal is to follow the user's instruction and complete the task.
You MUST output ONLY ONE action per step.

Allowed actions:
- search[query]
- think[reasoning]
- click[item_id]
- click[option]
- click[Buy Now]
- click[Back to Search]

Below are 2 example episodes of how you should act:


### Example 1
WebShop
Instruction:
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
[Search]

Action: search[3 ounce bright citrus deodorant sensitive skin price under 50$]
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

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less than 50 dollars. I can check B078GWRC1J first.]
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

Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy, I will select the given options.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus.

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).

Action: click[Buy Now]
Observation: Your score (min 0.0, max 1.0): 1.0


### Example 2
WebShop
Instruction:
i am looking for a square area rug that is grey and ivory and measures 2 feet by 17 feet, and price lower than 160.00 dollars
[Search]

Action: search[square area rug grey ivory 2x17 feet under 160 dollars]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B01LX64ELS]
SAFAVIEH Amsterdam Collection AMS108K Moroccan Boho Non-Shedding Living Room Bedroom Dining Home Office Area Rug, 6'7" x 9'2", Ivory / Multi
$103.59

Action: think[Only B01LX64ELS is under the budget, it might satisfy our requirements. I will check this item first.]
Observation: OK.

Action: click[B01LX64ELS]
Observation:
[Back to Search]
[< Prev]
size [3' square][4' square][5'1" square][6'7" square][8' square][9' square][10' square]
color [ivory | grey][ivory | multi][dark grey | ivory]
SAFAVIEH Amsterdam Collection AMS108K
Price: $103.59
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Action: think[This item is under budget and offers square sizes and an ivory grey color option. I will select a square size and the ivory | grey color.]
Observation: OK.

Action: click[3' square]
Observation: You have clicked 3' square.

Action: click[ivory | grey]
Observation: You have clicked ivory | grey.

Action: click[Buy Now]
Observation: Your score (min 0.0, max 1.0): 1.0

Now follow the same style for the new task.
If no item matches the instruction, you may click[Back to Search] and search with a revised query.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# LLM HELPERS — identical to bfs_2.py
# ═══════════════════════════════════════════════════════════════════════════════

def logprobs_to_softmax_bins(token_logprobs: List[float], num_bins: int = SOFTMAX_BINS) -> List[int]:
    bins = [0] * num_bins
    for lp in token_logprobs:
        idx = min(int(math.exp(lp) * num_bins), num_bins - 1)
        bins[idx] += 1
    return bins


def _extract_action(text: str) -> Optional[str]:
    """
    Extract the first valid WebShop action from an LLM response.
    Valid prefixes: search[, click[, think[
    Returns the full action string (including closing ]) or None.
    Mirrors bfs_2.py: think[] is a stand-alone action, NOT a sidecar.
    """
    text = text.strip()
    for prefix in ('search[', 'click[', 'think['):
        if prefix in text:
            start = text.index(prefix)
            sub = text[start:]
            if ']' in sub:
                return sub[:sub.index(']') + 1]
    return None


def sample_candidates_with_logprobs(
    llm_prompt:  str,
    n:           int   = 10,
    max_tokens:  int   = 150,
    temperature: float = 0.7,
) -> Tuple[List[str], List[str], List[List[float]]]:
    """
    Sample n LLM responses AND capture their token log-probs in a single call.

    Returns (mirrors bfs_2.py):
        think_actions  — all think[] responses (for majority-vote, no logprobs needed)
        real_actions   — deduplicated search[]/click[] actions
        logprobs_list  — token log-prob list per real action, positionally aligned
                         with real_actions; ready for ConformalPredictorWebShop
    """
    response = client.completions.create(
        model       = MODEL_NAME,
        prompt      = llm_prompt,
        n           = n,
        max_tokens  = max_tokens,
        temperature = temperature,
        echo        = False,
        logprobs    = 1,
    )

    think_actions: List[str]         = []
    real_actions:  List[str]         = []
    logprobs_list: List[List[float]] = []
    seen_real: set = set()

    all_parsed = []
    for choice in response.choices:
        action = _extract_action(choice.text)
        if action is None:
            continue
        lps = []
        if choice.logprobs and choice.logprobs.token_logprobs:
            lps = [lp for lp in choice.logprobs.token_logprobs if lp is not None]
        all_parsed.append((action, lps))

    print(f"  [LLM sampled actions] {[a for a, _ in all_parsed]}")

    for action, lps in all_parsed:
        if action.startswith('think['):
            think_actions.append(action)
        elif action not in seen_real:
            seen_real.add(action)
            real_actions.append(action)
            logprobs_list.append(lps)

    return think_actions, real_actions, logprobs_list


# ═══════════════════════════════════════════════════════════════════════════════
# ENV HELPERS — identical to bfs_2.py
# ═══════════════════════════════════════════════════════════════════════════════

def clean_str(p: str) -> str:
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element) -> bool:
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return element.parent.name not in ignore and not isinstance(element, Comment)


def webshop_text(
    session, page_type, query_string='', page_num=1,
    asin='', options={}, subpage='', **kwargs,
) -> Tuple[str, dict]:
    # Percent-encode LLM-controlled URL parts so values containing '#', '&',
    # '?', spaces, etc. (e.g. option label "natural black #1b") don't get
    # truncated by the URL fragment delimiter and crash Flask's literal_eval.
    qs   = quote(query_string, safe='')
    opts = quote(str(options),  safe='')
    if page_type == 'init':
        url = f'{WEBSHOP_URL}/{session}'
    if page_type == 'search':
        url = f'{WEBSHOP_URL}/search_results/{session}/{qs}/{page_num}'
    elif page_type == 'item':
        url = f'{WEBSHOP_URL}/item_page/{session}/{asin}/{qs}/{page_num}/{opts}'
    elif page_type == 'item_sub':
        url = (f'{WEBSHOP_URL}/item_sub_page/{session}/'
               f'{asin}/{qs}/{page_num}/{subpage}/{opts}')
    elif page_type == 'end':
        url = f'{WEBSHOP_URL}/done/{session}/{asin}/{opts}'

    html     = requests.get(url).text
    html_obj = BeautifulSoup(html, 'html.parser')
    texts    = html_obj.findAll(text=True)
    visible_texts = list(filter(tag_visible, texts))

    observation = ''
    option_type = ''
    options     = {}
    asins       = []
    cnt         = 0
    prod_cnt    = 0
    just_prod   = 0

    for t in visible_texts:
        if t == '\n':
            continue
        if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '':
            continue
        if t.parent.name == 'button':
            processed_t = f'\n[{t}] '
        elif t.parent.name == 'label':
            processed_t = f'[[{t}]]' if f"'{t}'" in url else f'[{t}]'
            options[str(t)] = option_type
        elif t.parent.get('class') == ['product-link']:
            processed_t = f'\n[{t}] '
            if prod_cnt >= 3:
                processed_t = ''
            prod_cnt += 1
            asins.append(str(t))
            just_prod = 0
        else:
            processed_t = '\n' + str(t) + ' '
            if cnt < 2 and page_type != 'init':
                processed_t = ''
            if just_prod <= 2 and prod_cnt >= 4:
                processed_t = ''
            option_type = str(t)
            cnt += 1
        just_prod   += 1
        observation += processed_t

    info = {}
    if options:
        info['option_types'] = options
    if asins:
        info['asins'] = asins
    if 'Your score (min 0.0, max 1.0)' in visible_texts:
        idx = visible_texts.index('Your score (min 0.0, max 1.0)')
        info['reward'] = float(visible_texts[idx + 1])
        observation    = 'Your score (min 0.0, max 1.0): ' + str(visible_texts[idx + 1])

    return clean_str(observation), info


class WebShopEnv:
    def __init__(self):
        self.sessions: Dict[int, dict] = {}

    def step(self, session: int, action: str) -> Tuple[str, float, bool]:
        done         = False
        observation_ = None

        if action == 'reset':
            self.sessions[session] = {'session': session, 'page_type': 'init'}
        elif action.startswith('think['):
            observation_ = 'OK.'
        elif action.startswith('search['):
            assert self.sessions[session]['page_type'] == 'init'
            query = action[7:-1]
            self.sessions[session] = {
                'session': session, 'page_type': 'search',
                'query_string': query, 'page_num': 1,
            }
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
                assert self.sessions[session]['page_type'] == 'search'
                self.sessions[session]['page_num'] += 1
            elif button == '< Prev':
                assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
                if self.sessions[session]['page_type'] == 'search':
                    assert False
                elif self.sessions[session]['page_type'] == 'item_sub':
                    self.sessions[session]['page_type'] = 'item'
                elif self.sessions[session]['page_type'] == 'item':
                    self.sessions[session]['page_type'] = 'search'
                    self.sessions[session]['options']   = {}
            elif button in ACTION_TO_TEMPLATE:
                assert self.sessions[session]['page_type'] == 'item'
                self.sessions[session]['page_type'] = 'item_sub'
                self.sessions[session]['subpage']   = button
            else:
                if self.sessions[session]['page_type'] == 'search':
                    assert button in self.sessions[session].get('asins', [])
                    self.sessions[session]['page_type'] = 'item'
                    self.sessions[session]['asin']      = button
                elif self.sessions[session]['page_type'] == 'item':
                    assert 'option_types' in self.sessions[session]
                    opt_types = self.sessions[session]['option_types']
                    # Case-insensitive match: the LLM often copies the casing
                    # from the product title (e.g. "Natural Black #1B") while
                    # the option label on the page is lowercase ("natural
                    # black #1b"). Normalize the LLM's button to the page's
                    # canonical casing so the downstream URL is well-formed.
                    if button not in opt_types:
                        ci_map = {k.lower(): k for k in opt_types}
                        if button.lower() in ci_map:
                            button = ci_map[button.lower()]
                        else:
                            assert False, (button, opt_types)
                    option_type = opt_types[button]
                    if 'options' not in self.sessions[session]:
                        self.sessions[session]['options'] = {}
                    self.sessions[session]['options'][option_type] = button
                    observation_ = f'You have clicked {button}.'
        else:
            assert False, f"Unknown action: {action!r}"

        observation, info = webshop_text(**self.sessions[session])
        if observation_:
            observation = observation_
        self.sessions[session].update(info)
        reward = info.get('reward', 0.0)
        return observation, reward, done


def get_instruction(obs: str) -> str:
    try:
        return obs.split("Instruction:")[1].split("[Search]")[0].strip()
    except IndexError:
        return obs.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING DATA HELPERS — identical to bfs_2.py
# ═══════════════════════════════════════════════════════════════════════════════

def load_webshop_data(csv_path: str) -> List[Dict]:
    data    = []
    skipped = 0

    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            raw_adm = row['Admissiable_action'].strip()
            if not raw_adm or raw_adm == '[]':
                skipped += 1
                continue
            try:
                admissible = ast.literal_eval(raw_adm)
                logits     = ast.literal_eval(row['Logits_from_LLM'])
                prev_raw   = ast.literal_eval(row['Sequence_of_previous_action'])
            except (SyntaxError, ValueError):
                skipped += 1
                continue

            instruction = row['Original_instruction'].strip()
            optimal     = row['optimal_action'].strip()

            if not instruction or not admissible or not optimal:
                skipped += 1
                continue

            if optimal not in admissible:
                admissible.append(optimal)
                logits.append([])

            prev_actions = [a for a in prev_raw if a != 'reset']
            data.append({
                'instruction':        instruction,
                'prev_actions':       prev_actions,
                'admissible_actions': admissible,
                'logits_from_llm':    logits,
                'optimal_action':     optimal,
            })

    print(f"  Loaded {len(data)} valid rows ({skipped} skipped)")
    return data


def build_records(data: List[Dict], bert_cache: BertEmbeddingCache) -> List[Dict]:
    records = []

    for item in data:
        instruction     = item['instruction']
        prev_actions    = item['prev_actions']
        admissible      = item['admissible_actions']
        logits_from_llm = item['logits_from_llm']
        optimal         = item['optimal_action']

        instr_emb = bert_cache.get(instruction)
        prev_pool = (
            bert_cache.get_batch(prev_actions).mean(dim=0)
            if prev_actions else torch.zeros(BERT_DIM)
        )
        state_raw = torch.cat([instr_emb, prev_pool])

        action_embs = bert_cache.get_batch(admissible)
        bins = [_logprobs_to_softmax_bins(lp) for lp in logits_from_llm]

        existing_bins = [b for b, lp in zip(bins, logits_from_llm) if lp]
        mean_bin = (
            [sum(b[i] for b in existing_bins) / len(existing_bins)
             for i in range(SOFTMAX_BINS)]
            if existing_bins else [0.0] * SOFTMAX_BINS
        )
        bins = [mean_bin if not lp else b for b, lp in zip(bins, logits_from_llm)]
        softmax_bins = torch.tensor(bins, dtype=torch.float32)

        optimal_idx = admissible.index(optimal)
        records.append({
            'state_raw':    state_raw,
            'action_embs':  action_embs,
            'softmax_bins': softmax_bins,
            'optimal_idx':  optimal_idx,
        })

    print(f"  Built {len(records)} records")
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# CSV INSTRUCTION LOADING (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def load_csv_instructions(csv_path: str) -> List[Tuple[int, str]]:
    """
    Return [(csv_id, instruction), ...] — one per unique csv id, in id order.
    """
    seen: Dict[int, str] = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            sid_str = r['id'].strip()
            inst    = r['Original_instruction'].strip()
            if not sid_str or not inst:
                continue
            try:
                sid = int(float(sid_str))
            except ValueError:
                continue
            seen.setdefault(sid, inst)
    pairs = sorted(seen.items())
    print(f"  Loaded {len(pairs)} unique (csv_id, instruction) pairs from CSV")
    return pairs


_PRICE_CLAUSE_RE = re.compile(
    r",?\s*(?:and\s+)?price\s+(?:lower|less)\s+than\s+\$?\d+(?:\.\d+)?\s*dollars?\.?\s*$",
    re.IGNORECASE,
)


def strip_price_clause(s: str) -> str:
    """
    Drop the trailing 'price lower than X dollars' clause from a WebShop
    instruction so two instructions that only differ in price compare equal.
    """
    s = _PRICE_CLAUSE_RE.sub("", s).strip().rstrip(",").strip()
    return s.lower()


def save_instructions_json(pairs: List[Tuple[int, str]], path: str) -> None:
    with open(path, 'w') as f:
        json.dump(
            [{'csv_id': sid, 'instruction': inst} for sid, inst in pairs],
            f, indent=2,
        )
    print(f"  Saved CSV instructions → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# BFS — identical to bfs_2.py
# ═══════════════════════════════════════════════════════════════════════════════

BFSNode = collections.namedtuple(
    'BFSNode', ['session_dict', 'previous_actions', 'llm_prompt', 'instruction']
)


def run_bfs_episode(
    session_id,                                      # str like 'fixed_0' (matches sc.py convention)
    env:        WebShopEnv,
    predictor:  ConformalPredictorWebShop,
    args,
    expected_instruction: Optional[str] = None,
) -> Tuple[float, List[str], str, bool]:
    """
    Returns: (best_reward, best_traj, env_instruction, was_executed)
    was_executed = False if we skipped the episode due to mismatch.
    """
    obs, _, _ = env.step(session_id, 'reset')
    instruction = get_instruction(obs)

    if expected_instruction is not None:
        exact_match    = (instruction == expected_instruction)
        no_price_match = (
            strip_price_clause(instruction) ==
            strip_price_clause(expected_instruction)
        )
        if exact_match:
            tag = "[MATCH-EXACT]"
        elif no_price_match:
            tag = "[MATCH-IGNORING-PRICE]"
        else:
            tag = "[MISMATCH]"
        print(f"  CSV instruction : {expected_instruction}")
        print(f"  Env instruction : {instruction}")
        print(f"  {tag}")
        if not (exact_match or no_price_match) and not args.include_mismatched:
            print("  → SKIPPED (use --include_mismatched to run anyway)")
            return 0.0, [], instruction, False
    else:
        print(f"  Instruction: {instruction}")

    init_llm_prompt = FEW_SHOT_PROMPT + obs + '\n\nAction:'

    queue          = collections.deque([BFSNode(
        session_dict     = copy.deepcopy(env.sessions[session_id]),
        previous_actions = [],
        llm_prompt       = init_llm_prompt,
        instruction      = instruction,
    )])
    visited        = set()
    nodes_explored = 0
    best_reward    = 0.0
    best_traj: List[str] = []
    completed_trajectories: List[Tuple[float, List[str]]] = []

    while queue:
        node = queue.popleft()
        prev_actions = node.previous_actions
        llm_prompt   = node.llm_prompt
        instr        = node.instruction

        if len(prev_actions) >= args.max_depth:
            continue
        if nodes_explored >= args.node_budget:
            print(f"\n[BFS] Node budget ({args.node_budget}) exhausted.")
            break
        traj_key = tuple(prev_actions)
        if traj_key in visited:
            continue
        visited.add(traj_key)
        nodes_explored += 1

        env.sessions[session_id] = copy.deepcopy(node.session_dict)

        print(f"\n[BFS Node {nodes_explored}]  depth={len(prev_actions)}")
        print(f"  Traj     : {prev_actions if prev_actions else '(start)'}")

        think_actions, real_actions, logprobs_list = sample_candidates_with_logprobs(
            llm_prompt,
            n           = args.n_samples,
            temperature = args.temperature,
        )
        print(f"  Think ({len(think_actions)}): {think_actions}")
        print(f"  Real  ({len(real_actions)}):  {real_actions}")

        # ── Page-aware candidate filter ───────────────────────────────────────
        # The env only accepts search[] from page_type == 'init', and only
        # click[] from non-init pages. Filtering wrong-type candidates here
        # prevents wasted conformal scoring + guaranteed INVALIDs at env.step.
        # To search again from a results/item page, the agent must first
        # click[Back to Search] (which resets page_type to 'init').
        current_page = node.session_dict.get('page_type', 'init')
        if current_page == 'init':
            # On home page: drop click[] candidates (only search[] is legal here)
            bad_prefix, reason = 'click[', 'home page'
        else:
            # Elsewhere: drop search[] candidates (only click[] is legal here)
            bad_prefix, reason = 'search[', f'page_type={current_page!r}'

        kept_actions, kept_lps, dropped = [], [], []
        for a, lps in zip(real_actions, logprobs_list):
            if a.startswith(bad_prefix):
                dropped.append(a)
            else:
                kept_actions.append(a)
                kept_lps.append(lps)
        if dropped:
            print(f"  [filter] {reason} → dropping {len(dropped)} "
                  f"{bad_prefix}...] candidate(s): {dropped}")
        real_actions, logprobs_list = kept_actions, kept_lps

        # ── Think-only: majority vote, execute, push single child (no branching) ─
        if think_actions and not real_actions:
            counts       = collections.Counter(think_actions)
            chosen_think = counts.most_common(1)[0][0]
            print(f"  → THINK only → {chosen_think}")

            env.sessions[session_id] = copy.deepcopy(node.session_dict)
            try:
                _, _, _ = env.step(session_id, chosen_think)
            except AssertionError:
                print(f"  THINK {chosen_think!r} → invalid, skip")
                continue

            # think[] doesn't count as a real action; prev_actions unchanged
            new_prompt = llm_prompt + f' {chosen_think}\nObservation: OK.\n\nAction:'
            queue.appendleft(BFSNode(   # appendleft → depth-first continuation
                session_dict     = copy.deepcopy(env.sessions[session_id]),
                previous_actions = list(prev_actions),
                llm_prompt       = new_prompt,
                instruction      = instr,
            ))
            continue

        if not real_actions:
            print("  [skip] No parseable actions.")
            continue

        candidates = real_actions

        pred_set, scores, n_sel, _ = predictor.get_prediction_set(
            instruction        = instr,
            previous_actions   = prev_actions,
            admissible_actions = candidates,
            logits_from_llm    = logprobs_list,
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
            tag = " [IN SET]" if a in pred_set else ""
            print(f"    {scores[a]:.4f}  {a}{tag}")

        for action in pred_set:
            env.sessions[session_id] = copy.deepcopy(node.session_dict)

            try:
                obs, reward, done = env.step(session_id, action)
                print("\nAction taken: ", action)
                print(f"\nObservation from Webshop: ", obs)
            except AssertionError:
                print(f"    {action!r} → INVALID (skip)")
                continue

            new_traj   = prev_actions + [action]
            new_prompt = llm_prompt + f' {action}\nObservation: {obs}\n\nAction:'

            print(f"    → {action!r}   reward={reward:.4f}   done={done}")

            if done and reward == 1.0:
                print(f"\n{'='*60}")
                print(f"[BFS] PERFECT REWARD 1.0 in {len(new_traj)} step(s)!")
                print(f"  Winning trajectory: {new_traj}")
                print(f"{'='*60}")
                return 1.0, new_traj, instruction, True

            if done:
                if reward > best_reward:
                    best_reward = reward
                    best_traj   = new_traj
                completed_trajectories.append((reward, new_traj))
                print(f"    ↳ partial {reward:.4f}  "
                      f"(best_so_far={best_reward:.4f}) — BFS continues")
                continue

            queue.append(BFSNode(
                session_dict     = copy.deepcopy(env.sessions[session_id]),
                previous_actions = new_traj,
                llm_prompt       = new_prompt,
                instruction      = instr,
            ))

        if scores and pred_set:
            best_action = pred_set[0]
            try:
                predictor.commit_step(
                    executed_action = best_action,
                    metadata        = {
                        'instruction':    instr,
                        'prev_actions':   prev_actions,
                        'optimal_action': best_action,
                    },
                )
            except (AttributeError, AssertionError):
                pass

    if completed_trajectories:
        best_reward, best_traj = max(completed_trajectories, key=lambda x: x[0])
        print(f"  [completed_trajectories] {len(completed_trajectories)} terminal "
              f"trajectories found, best reward = {best_reward:.4f}")

    print(f"\n{'='*60}")
    if best_reward == 1.0:
        print(f"  RESULT : SUCCESS  ({len(best_traj)} steps)")
        print(f"  Trajectory : {best_traj}")
    else:
        print(f"  RESULT : BEST PARTIAL  reward={best_reward:.4f}"
              f"  (node budget exhausted / queue empty)")
    print(f"{'='*60}")
    return best_reward, best_traj, instruction, True


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BFS Conformal Planning — WebShop (CSV instructions)")
    parser.add_argument("--alpha",        type=float, default=0.3)
    parser.add_argument("--k",            type=int,   default=50)
    parser.add_argument("--n_components", type=int,   default=10)
    parser.add_argument("--n_samples",    type=int,   default=10)
    parser.add_argument("--temperature",  type=float, default=0.7)
    parser.add_argument("--max_depth",    type=int,   default=15)
    parser.add_argument("--node_budget",  type=int,   default=100)
    parser.add_argument("--n_episodes",   type=int,   default=200,
                        help="Cap on number of CSV ids to run (truncates the pair list)")
    parser.add_argument("--include_mismatched", action="store_true",
                        help="Run episodes even when env instruction != CSV instruction")
    args = parser.parse_args()

    print("=" * 72)
    print("  BFS Conformal Planning — WebShop (CSV-instruction variant)")
    print("=" * 72)
    print(f"  Device              : {DEVICE}")
    print(f"  α                   : {args.alpha}  (target coverage ≥ {(1-args.alpha)*100:.0f}%)")
    print(f"  k                   : {args.k}")
    print(f"  n_components        : {args.n_components}")
    print(f"  n_samples           : {args.n_samples}")
    print(f"  temperature         : {args.temperature}")
    print(f"  max_depth           : {args.max_depth}")
    print(f"  node_budget         : {args.node_budget}")
    print(f"  n_episodes (cap)    : {args.n_episodes}")
    print(f"  include_mismatched  : {args.include_mismatched}")

    print(f"\nLoading BERT cache from {BERT_CACHE_PATH} ...")
    bert_cache = BertEmbeddingCache(device=DEVICE)
    bert_cache.load(BERT_CACHE_PATH)

    print(f"Loading score model from {MODEL_PATH} ...")
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    model = ScoreFunctionWebShop(d_proj=ckpt['d_proj'], hidden=ckpt['hidden']).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"\nBuilding calibration pool from ALL training data ...")
    print(f"  Source: {DATA_PATH}")
    training_data = load_webshop_data(DATA_PATH)
    records       = build_records(training_data, bert_cache)
    predictor     = ConformalPredictorWebShop(model, bert_cache, alpha=args.alpha)
    predictor.calibrate_from_records(records, DEVICE)
    print(f"  Pool size: {len(predictor.cal_scores)}")

    print(f"\nLoading CSV instructions ...")
    csv_pairs = load_csv_instructions(DATA_PATH)
    save_instructions_json(csv_pairs, INSTRUCTIONS_DUMP)
    csv_pairs = csv_pairs[: args.n_episodes]
    print(f"  Will run {len(csv_pairs)} episodes (capped by --n_episodes)")

    env = WebShopEnv()

    rewards:              List[float] = []
    traj_list:            List[List]  = []
    all_traj_lengths:     List[int]   = []
    perfect_traj_lengths: List[int]   = []
    n_skipped_mismatch                = 0
    n_match_exact                     = 0
    n_match_no_price                  = 0

    for i, (csv_id, csv_instruction) in enumerate(csv_pairs):
        # Match sc.py convention: prefix the integer id with 'fixed_' so the
        # WebShop server URL becomes WEBSHOP_URL/fixed_<id> — same goal pool
        # the CSV was built against.
        session_id_str = f'fixed_{csv_id}'
        print(f"\n{'#'*72}\n  EPISODE {i+1} / {len(csv_pairs)}  "
              f"(csv_id={csv_id}, session={session_id_str})\n{'#'*72}")

        try:
            reward, traj, env_instruction, executed = run_bfs_episode(
                session_id           = session_id_str,
                env                  = env,
                predictor            = predictor,
                args                 = args,
                expected_instruction = csv_instruction,
            )
        except Exception as e:
            print(f"  Episode {session_id_str} crashed: {e}")
            reward, traj, env_instruction, executed = 0.0, [], "", False

        if not executed:
            n_skipped_mismatch += 1
            continue
        if env_instruction == csv_instruction:
            n_match_exact += 1
        elif strip_price_clause(env_instruction) == strip_price_clause(csv_instruction):
            n_match_no_price += 1

        rewards.append(reward)
        traj_list.append(traj)
        all_traj_lengths.append(len(traj))
        if reward == 1.0:
            perfect_traj_lengths.append(len(traj))

        n = len(rewards)
        print(f"\n{'='*72}")
        print(f"  EPISODE {i+1} SUMMARY  (csv_id={csv_id}, session={session_id_str})")
        print(f"  Reward       : {reward:.4f}")
        print(f"  Traj length  : {len(traj)}")
        print(f"  Trajectory   : {traj}")
        print(f"  Avg reward   : {sum(rewards)/n:.3f}")
        print(f"  Perfect=1.0  : {sum(1 for x in rewards if x == 1.0)/n:.3f}")
        print(f"  Avg traj len (all)     : {sum(all_traj_lengths)/n:.2f}")
        if perfect_traj_lengths:
            print(f"  Avg traj len (perfect) : {sum(perfect_traj_lengths)/len(perfect_traj_lengths):.2f}")
        print(f"  Match running-tally    : exact={n_match_exact}  "
              f"ignoring-price={n_match_no_price}  skipped(mismatch)={n_skipped_mismatch}")
        print(f"{'='*72}")

        # break

    if not rewards:
        print("\nNo episodes were executed (all mismatched and --include_mismatched not set).")
        return

    print(f"\n{'='*72}")
    print(f"  FINAL RESULTS  ({len(rewards)} episodes executed, "
          f"{n_skipped_mismatch} skipped due to instruction mismatch)")
    print(f"  Match exact             : {n_match_exact}")
    print(f"  Match ignoring price    : {n_match_no_price}")
    print(f"  Avg reward              : {sum(rewards)/len(rewards):.3f}")
    print(f"  Perfect=1.0             : {sum(1 for x in rewards if x == 1.0)/len(rewards):.3f}")
    print(f"  Avg traj len (all)      : {sum(all_traj_lengths)/len(all_traj_lengths):.2f}")
    if perfect_traj_lengths:
        print(f"  Avg traj len (perfect)  : {sum(perfect_traj_lengths)/len(perfect_traj_lengths):.2f}")

    perfect_pct = sum(1 for x in rewards if x == 1.0) / len(rewards)
    avg_all_len = sum(all_traj_lengths) / len(all_traj_lengths)
    avg_prf_len = (sum(perfect_traj_lengths) / len(perfect_traj_lengths)
                   if perfect_traj_lengths else None)

    def _print_dist(label: str, lengths: List[int]) -> None:
        if not lengths:
            print(f"\n  TRAJECTORY LENGTH DISTRIBUTION ({label})")
            print(f"  (no episodes)")
            return
        dist    = collections.Counter(lengths)
        max_len = max(lengths)
        total   = len(lengths)
        print(f"\n  TRAJECTORY LENGTH DISTRIBUTION ({label})")
        print(f"  {'Steps':>6}  {'Count':>6}  {'%':>7}")
        for steps in range(0, max_len + 1):
            count = dist.get(steps, 0)
            if count == 0:
                continue
            pct = count / total * 100
            print(f"  {steps:>6}  {count:>6}  {pct:>6.1f}%")

    def _plot_dist(label: str, lengths: List[int], out_path: str, color: str) -> None:
        if not lengths:
            print(f"  [plot] no episodes for {label}, skipping {out_path}")
            return
        dist    = collections.Counter(lengths)
        steps   = sorted(dist.keys())
        counts  = [dist[s] for s in steps]
        total   = len(lengths)
        pcts    = [c / total * 100 for c in counts]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(steps, counts, width=0.8, edgecolor="black", color=color)
        for bar, c, p in zip(bars, counts, pcts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.01,
                    f"{c}\n({p:.1f}%)", ha="center", va="bottom", fontsize=9)

        pretty = "all episodes" if label == "ALL" else f"{label} episodes"
        title  = f"WebShop trajectory length distribution — {pretty}  ({total} episodes)"
        sub    = [f"Perfect=1.0: {perfect_pct*100:.1f}%"]
        if label == "ALL":
            sub.append(f"avg len (all): {avg_all_len:.2f}")
        elif "PERFECT" in label and avg_prf_len is not None:
            sub.append(f"avg len (perfect): {avg_prf_len:.2f}")
        title += "\n" + "  |  ".join(sub)
        ax.set_title(title)
        ax.set_xlabel("Trajectory length (steps)")
        ax.set_ylabel("Number of episodes")
        ax.set_xticks(steps)
        ax.set_ylim(0, max(counts) * 1.18)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  [plot] saved → {out_path}")

    _print_dist("ALL", all_traj_lengths)
    _print_dist("PERFECT=1.0", perfect_traj_lengths)
    print(f"{'='*72}")

    print("\nGenerating histograms ...")
    # Filenames carry the model tag and α so concurrent runs at different α
    # don't overwrite each other (they were previously hardcoded to qwen_3_8b_0.3).
    _plot_dist("ALL",         all_traj_lengths,
               f"traj_len_hist_all_gemma4_12b_{args.alpha}.png",     "#4C72B0")
    _plot_dist("PERFECT=1.0", perfect_traj_lengths,
               f"traj_len_hist_perfect_gemma4_12b_{args.alpha}.png", "#55A868")


if __name__ == "__main__":
    main()
