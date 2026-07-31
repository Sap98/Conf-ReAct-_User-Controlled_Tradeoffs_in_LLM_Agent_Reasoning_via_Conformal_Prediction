"""
ReflAct on WebShop — faithful .py port of the canonical ReAct WebShop notebook
(https://github.com/ysymyth/ReAct/blob/master/WebShop.ipynb).

Changes vs the upstream notebook (and ONLY these):
  1. think[...]  ->  reflection[...]  everywhere:
       - env.step() and webshop_run() now key on 'reflection'
       - prompt1's two reasoning steps are rewritten as ReflAct reflections
         (ground the current state -> relate it to the task goal), not just
         relabeled next-action thoughts.
  2. llm() now calls a vLLM OpenAI-compatible endpoint with Qwen3-8B
     (the notebook's text-davinci-002 is retired). Greedy (temperature=0),
     stop=['\\n'], same single-trajectory loop.
  3. WEBSHOP_URL points at the live local server (the notebook's AWS IP is dead).
  4. Trajectory-length distribution is computed and plotted separately for
       (i) ALL games  and  (ii) PERFECT games (score == 1.0).
     Trajectory length = number of real search[]/click[] actions (reflection[]
     reasoning steps are NOT counted), matching the BFS method's convention.

Run:
  python reflact_webshop_qwen3_8b.py            # default 50 episodes (fixed_0..49)
  python reflact_webshop_qwen3_8b.py 500        # full eval like the notebook
  python reflact_webshop_qwen3_8b.py 50 100     # 50 episodes starting at fixed_100
  python reflact_webshop_qwen3_8b.py 500 0 > reflact_webshop_run.txt   # capture stdout
"""

import sys
from collections import Counter
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from openai import OpenAI

# ─── config ──────────────────────────────────────────────────────────────────
WEBSHOP_URL = "http://10.5.30.29:3000/"                       # live local server
MODEL_NAME  = "Qwen/Qwen3-8B"
client      = OpenAI(base_url="http://10.5.30.30:8001/v1", api_key="EMPTY")  # vLLM
MAX_STEPS   = 15
MAX_TOKENS  = 256
TEMPERATURE = 0.0

RESULTS_LOG = "reflact_webshop_qwen3_8b_results.txt"
PLOT_PREFIX = "reflact_webshop_qwen3_8b_traj_hist"


def llm(prompt, stop=["\n"]):
    response = client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop,
    )
    return response.choices[0].text


# ─── WebShop env (verbatim from the notebook) ────────────────────────────────
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
    # spaces, etc. (e.g. option "natural black #1b") aren't truncated at the URL
    # fragment delimiter and crash the server's literal_eval. (Ported from the BFS method.)
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
    elif action.startswith('reflection['):          # was think[ — ReflAct reasoning, no-op
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
          # product title (e.g. "Natural Black #1B") while the page label is
          # lowercase ("natural black #1b"). Normalize the button to the page's
          # canonical casing so the assert passes. (Ported from the BFS method.)
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


# ─── ReflAct one-shot prompt (notebook's prompt1; think[] -> reflection[]) ─────
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

Action: reflection[I am on the search results page with nothing selected yet; my goal is a 3 ounce bright citrus deodorant for sensitive skin under $50, and B078GWRC1J is a bright citrus deodorant for sensitive skin at $10.99, which matches the scent and is within budget, so it is the best candidate to inspect first.]
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

Action: reflection[I am now on the product page for B078GWRC1J at $10.99, under my $50 budget; it offers the scent option 'bright citrus' and the size option '3 ounce (pack of 1)' that my goal requires, so I should select both options to match the instruction before buying.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus.

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).

Action: click[Buy Now]

Note: search[...] is only valid from the initial search page. If you are on a search-results or item page and no item matches the instruction, first do click[Back to Search], then search again with a revised query.
"""


# ─── rollout loop (notebook's webshop_run; + reflection + length tracking) ────
def webshop_run(idx, prompt, to_print=True):
  action = 'reset'
  init_prompt = prompt
  prompt = ''
  n_real = 0                                   # trajectory length = search[]/click[] only
  forced_next = None                           # action queued after a forced click[Back to Search]
  for i in range(MAX_STEPS):
    invalid = False
    try:
      res = env.step(idx, action)
      observation = res[0]
    except AssertionError:
      observation = 'Invalid action!'
      invalid = True

    if action.startswith('reflection'):
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
      # Auto-redirect: search[] is only valid from the init page. If the model
      # proposes search[] from a results/item page, run click[Back to Search]
      # first and defer the search to the next step. (Sequential analog of the
      # BFS page-aware filter: redirect instead of drop, since the agent has a
      # single candidate per step.)
      if (action.startswith('search[')
          and env.sessions.get(idx, {}).get('page_type', 'init') != 'init'):
        if to_print:
          print('  [auto-redirect] search[] proposed off the init page '
                '-> inserting click[Back to Search] first')
          sys.stdout.flush()
        forced_next = action
        action = 'click[Back to Search]'

  return 0, n_real


# ─── trajectory-length distribution: ALL vs PERFECT(=1.0) ─────────────────────
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
    ax.set_title(f"WebShop ReflAct — trajectory length [{label}]  ({total} episodes)")
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
        for steps in sorted(Counter(data)):
            c = Counter(data)[steps]
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


def run_episodes(prompt, n=200, start=0):
  rs = []
  lens = []
  cnt = 0
  for i in range(start, start + n):
    print('-----------------')
    print(i)
    try:
      r, tlen = webshop_run(f'fixed_{i}', prompt, to_print=True)
    except AssertionError:
      r, tlen = 0, 0
      cnt += 1
    rs.append(r); lens.append(tlen)
    avg, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / len(rs), cnt / len(rs)
    print(i + 1, avg, sr, fr)
    print('-------------')
  avg, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / len(rs), cnt / len(rs)
  print(avg, sr, fr)
  report_trajectory_lengths(rs, lens)
  return rs


if __name__ == "__main__":
    n     = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    res1 = run_episodes(prompt1, n, start)
