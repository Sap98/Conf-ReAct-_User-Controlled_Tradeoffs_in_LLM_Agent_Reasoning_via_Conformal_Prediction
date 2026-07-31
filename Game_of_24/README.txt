================================================================================
Game of 24 — Baseline Reproduction Guide
================================================================================

Unlike WebShop, Game of 24 needs no server: it's a pure-Python task (Game24Task
in game24.py) — the agent gets 4 numbers and must reach exactly 24 using
+ - * /. All baselines drive an open-weight model over a local vLLM
OpenAI-compatible endpoint. No hosted/commercial API is used anywhere.

Layout
    README.txt      this file
    Baselines/      baseline runner scripts (this section)
      prompts/      few-shot / prompt assets
      game24.py     shared task environment (Game24Task, IntraHistory)
      g24_oracle.py ground-truth per-step oracle (solvability-preserving
                    move checker) — used by ReAct/ReflAct to label every
                    proposed action, independent of the LLM's own guess
      sample_test_tasks.csv   the 200-puzzle evaluation set all baselines use


================================================================================
1. SETUP
================================================================================

  vllm serve Qwen/Qwen3-8B --port 8002       # or Qwen2.5-3B / gemma-4-12B-it

Python packages: torch, transformers, openai, sympy, matplotlib, pandas, numpy

All 5 scripts below default to vLLM endpoints on lab hosts
(10.5.18.73:8002/8003) — override with --base_url / --model_name (or, for
GA-Rollback, edit the MODEL_NAME/VLLM_BASE_URL constants at the top of the
file) to point at your own server.


================================================================================
2. RUN THE BASELINES
================================================================================

    cd Baselines

All prompt/task-set paths are resolved relative to each script's own
location, so any working directory works.

--- ReAct -----------------------------------------------------------------------

    python g24_react.py --model_name Qwen/Qwen3-8B --base_url http://<host>:8002/v1

Options:
    --num_samples N   samples per LLM call                (default 100)
    --seed N          sampling seed                        (default 42)
    --mode            act | react                          (default react)
    --data_file PATH  task CSV                              (default sample_test_tasks.csv)
    --model_name / --base_url   vLLM endpoint

Every proposed action is also scored against g24_oracle.py's ground-truth
solvability check (independent of the LLM), so the log records both what the
model did and whether it was actually correct.


--- ReflAct ---------------------------------------------------------------------

    python g24_reflact.py --model_name Qwen/Qwen3-8B --base_url http://<host>:8002/v1

Options: same as ReAct, plus
    --mode            reflact | react | act                 (default reflact)


--- Reflexion -------------------------------------------------------------------

    python g24_reflexion.py --model_name Qwen/Qwen3-8B --base_url http://<host>:8002/v1

Options:
    --num_trials N       outer Reflexion trials             (default 3)
    --num_envs N         puzzles per trial                  (default 100)
    --run_name DIR        logging directory                  (default "reflexion_run")
    --use_memory          allow the agent to use reflection memory
    --is_resume            resume from --resume_dir / --start_trial_num
    --model_name / --base_url   vLLM endpoint

Writes per-trial logs and `env_results_trial_<i>.json` under --run_name.


--- GA-Rollback -----------------------------------------------------------------

    python g24_rollback_qwen3_8b.py   --base_url http://<host>:8002/v1
    python g24_rollback_gemma4_12b.py --base_url http://<host>:8003/v1

No Qwen2.5-3B GA-Rollback run exists in the source project — only Qwen3-8B and
Gemma-4-12B were evaluated with this method.

Options:
    --num_samples N        samples per LLM call              (default 100)
    --max_roll_num N       max rollbacks per episode          (default 6)
    --model_source          open | close                       (default open;
                            "open" = local vLLM via --base_url, matching every
                            other script here — NOT a call to an external API)
    --mode                  act | react                         (default act)
    --prob_threshold F      min mean per-token prob to trust a rollback analysis


================================================================================
3. COVERAGE
================================================================================

    Method        Qwen2.5-3B   Qwen3-8B   Gemma-4-12B
    ------------------------------------------------
    ReAct            yes         yes          yes       (--model_name)
    ReflAct           yes         yes          yes       (--model_name)
    Reflexion         yes         yes          yes       (--model_name)
    GA-Rollback        -          yes          yes

ReAct/ReflAct/Reflexion are single scripts that take the model as a CLI
argument, so all three model results were produced by the same file — no
separate per-model copies exist (verified: the per-model directory copies in
the source project are byte-identical). GA-Rollback has separate files per
model because the open-weight local-inference branch differs; no Qwen2.5-3B
variant was ever run.

See Conf-ReAct/README.txt for the conformal-guided search method (Conf-ReAct)
itself, which is this project's main contribution and is evaluated against
these baselines.
