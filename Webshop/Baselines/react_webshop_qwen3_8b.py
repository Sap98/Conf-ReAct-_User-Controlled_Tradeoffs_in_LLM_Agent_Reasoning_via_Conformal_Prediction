# ReAct baseline on WebShop (from https://github.com/ysymyth/ReAct/blob/master/WebShop.ipynb)
# Only change from the original notebook: llm() now calls a local vLLM Qwen3-8B
# (text completions) instead of text-davinci-002.
#
# This mirrors react_webshop_gpt4.1.py but swaps the Azure gpt-4.1 chat endpoint
# for the vLLM /completions endpoint (same backend as reflexion_webshop.py). Since
# this is a text-completion endpoint, the ReAct prompt ending in "Action:" is
# continued literally by the model, so no system message or action extraction is
# needed (unlike the chat-model gpt-4.1 variant).

import sys
import json
from collections import Counter
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from openai import OpenAI

# ---------------------------------------------------------------------------
# LLM (local vLLM Qwen3-8B, OpenAI-compatible /completions)
# ---------------------------------------------------------------------------
# MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_NAME = "Qwen/Qwen2.5-3B"
client     = OpenAI(base_url="http://10.5.18.73:8002/v1", api_key="EMPTY")  # vLLM


def llm(prompt, stop=["\n"]):
    # "/no_think" keeps Qwen3 from emitting <think>...</think> reasoning blocks,
    # which would otherwise leak into the rollout as invalid actions.
    response = client.completions.create(
        model=MODEL_NAME,
        prompt="/no_think\n" + prompt,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop,
    )
    return response.choices[0].text


# ---------------------------------------------------------------------------
# WebShop environment
# ---------------------------------------------------------------------------
WEBSHOP_URL = "http://10.5.30.30:3000"  # live local server
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
    # Percent-encode LLM-controlled URL parts so values containing '/', '#', '&', '?',
    # spaces, etc. (e.g. option "natural black / brown") aren't truncated at the URL
    # path/fragment delimiter and crash the server's literal_eval.
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
    # print(url)
    html = requests.get(url).text
    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.findAll(text=True)
    visible_texts = list(filter(tag_visible, texts))
    # visible_texts = [str(text).strip().strip('\\n') for text in visible_texts]
    # if page_type == 'end': import pdb; pdb.set_trace()
    if False:
        # For `simple` mode, return just [SEP] separators
        return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
    else:
        # Otherwise, return an observation with tags mapped to specific, unique separators
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
            # if t.startswith('Instruction:') and page_type != 'init': continue
            # print(t.parent.name, t)
            if t.parent.name == 'button':  # button
                processed_t = f'\n[{t}] '
            elif t.parent.name == 'label':  # options
                if f"'{t}'" in url:
                    processed_t = f'[[{t}]]'
                    # observation = f'You have clicked {t}.\n' + observation
                else:
                    processed_t = f'[{t}]'
                options[str(t)] = option_type
                # options[option_type] = options.get(option_type, []) + [str(t)]
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
    elif action.startswith('think['):
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
          # product title (e.g. "natural black #1B") while the page label is
          # lowercase ("natural black #1b"). (Ported from the Reflexion/BFS port.)
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

# ---------------------------------------------------------------------------
# ReAct prompts
# ---------------------------------------------------------------------------
# trivial search & item, choose option
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
"""

# trivial search & item, choose option
prompt1_actonly = """Webshop
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

Action: click[bright citrus]
Observation: You have clicked bright citrus.

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).

Action: click[Buy Now]
"""

# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------
def webshop_run(idx, prompt, to_print=True):
  action = 'reset'
  init_prompt = prompt
  prompt = ''
  n_real = 0                                   # trajectory length = search[]/click[] only
  forced_next = None                           # action queued after a forced click[Back to Search]
  for i in range(15):
    invalid = False
    try:
      res = env.step(idx, action)
      observation = res[0]
    except AssertionError:
      observation = 'Invalid action!'
      invalid = True

    if action.startswith('think'):
      observation = 'OK.'

    if (not invalid) and (action.startswith('search[') or action.startswith('click[')):
      n_real += 1

    if to_print:
      print(f'Action: {action}\nObservation: {observation}\n')
      sys.stdout.flush()
    if i:
      prompt += f' {action}\nObservation: {observation}\n\nAction:'
    else:
      prompt += f'{observation}\n\nAction:'

    if res[2]:
      return res[1], n_real

    if forced_next is not None:
      action = forced_next                     # execute the deferred search now (page is 'init')
      forced_next = None
    else:
      action = llm(init_prompt + prompt[-(6400-len(init_prompt)):], stop=['\n']).lstrip(' ')
      # Corner case: search[] is only valid from the search page (page_type 'init').
      # If the model proposes search[] while on a results/item page, first revert to
      # the search page via click[Back to Search] and defer the search to the next step.
      if (action.startswith('search[')
          and env.sessions.get(idx, {}).get('page_type', 'init') != 'init'):
        if to_print:
          print('  [auto-redirect] search[] proposed off the search page '
                '-> inserting click[Back to Search] first')
          sys.stdout.flush()
        forced_next = action
        action = 'click[Back to Search]'

  return 0, n_real


# ─── trajectory-length distribution: ALL vs PERFECT(=1.0) ─────────────────────
RESULTS_LOG = "react_webshop_qwen3_8b_traj_dist.txt"
PLOT_PREFIX = "react_webshop_qwen3_8b_traj_hist"


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
    ax.set_title(f"WebShop ReAct — trajectory length [{label}]  ({total} episodes)")
    ax.set_xticks(steps)
    ax.set_ylim(0, max(counts) * 1.18 if counts else 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def report_trajectory_lengths(rewards, lengths):
    n = len(rewards)
    perfect_lengths = [l for l, r in zip(lengths, rewards) if r == 1]
    avg_all = sum(lengths) / n if n else 0.0
    avg_prf = sum(perfect_lengths) / len(perfect_lengths) if perfect_lengths else 0.0
    sr      = len(perfect_lengths) / n if n else 0.0

    out = []
    def emit(s):
        print(s); out.append(s)

    emit("")
    emit(f"=== SUMMARY: {n} episodes | avg_reward={sum(rewards)/n if n else 0:.4f} | "
         f"success_rate={sr:.4f} ===")
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


def run_episodes(prompt, n=50):
  rs = []
  lens = []
  cnt = 0
  for i in range(n):
    print('-----------------')
    print(i)
    try:
      r, tlen = webshop_run(f'fixed_{i}', prompt, to_print=True)
    except AssertionError:
      r, tlen = 0, 0
      cnt += 1
    rs.append(r)
    lens.append(tlen)
    if (i+1) % 1 == 0:
      r, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / len(rs), cnt / len(rs)
      print(i+1, r, sr, fr)
      print('-------------')
  r, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / n, cnt / n
  print(r, sr, fr)
  report_trajectory_lengths(rs, lens)
  return rs


if __name__ == '__main__':
  res1 = run_episodes(prompt1, 200)
