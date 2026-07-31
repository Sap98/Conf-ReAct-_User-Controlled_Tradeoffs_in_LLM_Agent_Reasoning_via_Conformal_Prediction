================================================================================
WebShop — Baseline Reproduction Guide
================================================================================

Layout
    README.txt      this file
    Baselines/      all baseline runner scripts
      prompts/      few-shot / prompt assets loaded by the scripts
      website.py    WebShop environment client (imported by GA-Rollback only)

All baselines are text agents that drive a WebShop server over HTTP. Nothing
runs until that server is up (Step 1) and the scripts point at it (Step 2).


================================================================================
1. SET UP THE WEBSHOP SERVER
================================================================================

  # 1. get the environment
  git clone https://github.com/princeton-nlp/webshop.git
  cd webshop

  # 2. install deps + download product data
  #    -d all   = full 1.18M-product index (used for the reported numbers)
  #    -d small = 1k-product subset (fast smoke test)
  #    requires conda: pulls faiss-cpu and openjdk=11
  ./setup.sh -d all

  # 3. launch (Flask, binds 0.0.0.0:3000)
  ./run_dev.sh

  # 4. sanity check — this should render a task page
  curl http://localhost:3000/fixed_0

Episodes are addressed as sessions  fixed_0 ... fixed_N.
The reported evaluation uses 200 episodes: fixed_0 .. fixed_199.

Indexing the full dataset on first launch takes several minutes. Keep the
server running in its own terminal for the whole evaluation.


================================================================================
2. POINT THE SCRIPTS AT YOUR ENDPOINTS
================================================================================

All models are open-weight and served locally with vLLM — no hosted/commercial
API is used anywhere. Every script has a config block near the top; two things
to set:

(a) WebShop server
    WEBSHOP_URL = "http://<host>:3000"
    Currently set to our lab hosts (10.5.30.29 / 10.5.30.30) — change these.

    !! GA-Rollback exception: web_rollback_*.py ALSO hardcodes WEBSHOP_URL a
       second time inside the __main__ block (~line 588-613), and that copy is
       the one used. Its --port flag is inherited from upstream and is IGNORED.
       Edit the __main__ line.

(b) Model endpoint (Qwen2.5-3B, Qwen3-8B, Gemma3-12B) — served via vLLM:
       vllm serve Qwen/Qwen3-8B --port 8001
    then set in the script:
       client = OpenAI(base_url="http://<host>:8001/v1", api_key="EMPTY")

Python packages:  requests  beautifulsoup4  openai  matplotlib
web_rollback_*.py additionally imports torch + transformers at module level
(inherited from upstream GA-Rollback; only needed for --model_source open, but
imported unconditionally).


================================================================================
3. RUN THE BASELINES
================================================================================

    cd Baselines

Prompts and website.py are resolved relative to each script's own location, so
any working directory works; result files are written to the current directory.

Below, <model> is one of:  qwen25_3b | qwen3_8b | gemma4_12b
(see the coverage table in Step 6 for which combinations exist).


--- ReAct ----------------------------------------------------------------------

    python react_webshop_qwen3_8b.py
    python react_webshop_gemma4_12b.py

No CLI arguments. The episode count is set in the __main__ block
(run_episodes(prompt1, 200)); edit it to change the number of episodes.

NOTE: react_webshop_qwen3_8b.py serves BOTH Qwen models. It currently has
      MODEL_NAME = "Qwen/Qwen2.5-3B"  (line 26, with the Qwen3-8B line
      commented out just above). Switch MODEL_NAME to pick the model.


--- ReflAct --------------------------------------------------------------------

    python reflact_webshop_qwen25_3b.py  200 0
    python reflact_webshop_qwen3_8b.py   200 0
    python reflact_webshop_gemma4_12b.py 200 0

Positional arguments:  <n_episodes> <start_index>
Defaults if omitted:   50 episodes starting at fixed_0.


--- Reflexion ------------------------------------------------------------------

    python reflexion_webshop_qwen25_3b.py  --num-episodes 200 --num-trials 3
    python reflexion_webshop_qwen3_8b.py   --num-episodes 200 --num-trials 3
    python reflexion_webshop_gemma4_12b.py --num-episodes 200 --num-trials 3

Options:
    --num-episodes N   number of WebShop envs           (default 200)
    --start N          starting env index, fixed_<N>    (default 0)
    --num-trials N     Reflexion outer trials per env   (default 3)
    --max-steps N      per-trial step cap               (default 15)
    --results PATH     write per-episode JSONL
    --resume           restore memory + successes from an existing --results file

Long runs: pass --results and --resume so an interrupted run continues instead
of restarting.


--- GA-Rollback ----------------------------------------------------------------

    python web_rollback_qwen25_3b.py  --sample_num 200 --max_roll_num 6
    python web_rollback_qwen3_8b.py   --sample_num 200 --max_roll_num 6
    python web_rollback_gemma4_12b.py --sample_num 200 --max_roll_num 6

Options:
    --sample_num N        number of test instances       (default 200)
    --max_roll_num N      maximum rollbacks per episode  (default 6)
    --model_source        close | open                   (default close)
                          "close" routes through the OpenAI-compatible client,
                          which here is the local vLLM server from Step 2(b).
    --out_record_path P   rollback trace JSON            (default rollback_records_<model>.json)
    --port                IGNORED — see the warning in Step 2(a)


================================================================================
4. OUTPUTS
================================================================================

Written to the current directory, namespaced by method and model so the
variants never overwrite each other:

    <method>_webshop_<model>_results.txt     score + trajectory-length summary
    <method>_webshop_<model>_traj_dist.txt      (ReAct / GA-Rollback name)
    <method>_webshop_<model>_traj_hist_all.png      traj-length hist, all episodes
    <method>_webshop_<model>_traj_hist_perfect.png  traj-length hist, score == 1.0
    rollback_records_<model>.json            GA-Rollback rollback traces

Trajectory length = number of executed search[]/click[] actions; think[] and
reflection[] reasoning steps are NOT counted.

Full transcripts are captured by redirecting stdout:

    python react_webshop_qwen3_8b.py > react_webshop_qwen_3.txt


================================================================================
5. LLM-CALL ACCOUNTING
================================================================================

    count_llm_calls_winning.py
    count_llm_calls_winning_allmodels.py

These compute LLM calls per winning episode by parsing the stdout transcripts
from Step 4. They do not call any model. Edit the path constants at the top of
each file to point at your saved .txt logs before running.

They report the three evaluated models (Qwen2.5-3B, Qwen3-8B, Gemma-4-12B).
Note: the path constant named GPT refers to the source directory name only —
no gpt model is invoked.


================================================================================
6. BASELINE x MODEL COVERAGE
================================================================================

    Method        Qwen2.5-3B   Qwen3-8B   Gemma3-12B
    ------------------------------------------------
    ReAct          (shared)      yes         yes
    ReflAct          yes         yes         yes
    Reflexion        yes         yes         yes
    GA-Rollback      yes         yes         yes

(shared) = react_webshop_qwen3_8b.py covers Qwen2.5-3B via the MODEL_NAME
           switch described in Step 3.
