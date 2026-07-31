================================================================================
ScienceWorld — Baseline Reproduction Guide
================================================================================

Like Game of 24 and unlike WebShop, ScienceWorld needs no server: the
`scienceworld` package (allenai/ScienceWorld) is a local, pip-installed,
Java-backed text-game simulator (ScienceWorldEnv). All baselines drive an
open-weight model over a local vLLM OpenAI-compatible endpoint. No
hosted/commercial API is used anywhere.

Layout
    README.txt      this file
    Baselines/      baseline runner scripts (this section)
      examples/     the ReAct / ReflAct prompt templates (Kim et al., ReflAct,
                    EMNLP 2025, Appendix K.2)
      mpo_data/     the 211 (task, variation) test-set indices + per-task step
                    caps, from the MPO/ReflAct paper's test split


================================================================================
1. SETUP
================================================================================

    pip install scienceworld            # or: git clone the repo, pip install .
    vllm serve Qwen/Qwen3-8B --port 8001   # or Qwen2.5-3B / gemma-4-12B-it

`scienceworld` needs a JVM (the simulator itself is Java/Scala under the
hood) — a JDK on PATH is required; see the ScienceWorld repo's README if
`pip install scienceworld` doesn't pull one in for you.

All 3 scripts default to vLLM endpoints on lab hosts (10.5.30.29/30:8001) —
override with --base_url / --model_name.


================================================================================
2. RUN THE BASELINES
================================================================================

    cd Baselines

Prompt templates and mpo_data/ resolve relative to each script's own
location, so any working directory works.

Every script has two run modes:
  - Single task: --task-num (0-29) + --num-episodes, loops variations of ONE
    ScienceWorld task.
  - --mpo-testset: iterates the full 211 (task, variation) pairs from
    mpo_data/test_indices.json, using mpo_data/max_steps.json's per-task step
    caps — this is the mode that produced the reported numbers.

--- ReAct -----------------------------------------------------------------------

    python react_sciworld.py --mpo-testset --results results_react_211.jsonl \
        --model_name Qwen/Qwen3-8B --base_url http://<host>:8001/v1

Options:
    --task-num N / --var-num N / --num-episodes N   single-task mode (default
                                                     task 13, 134 episodes)
    --env-step-limit N   ScienceWorld internal step limit   (default 100)
    --max-steps N        LLM (ReAct) step cap per episode    (default 50;
                          ignored in --mpo-testset, which uses mpo_data's
                          per-task caps instead)
    --simplifications-preset   env simplification preset      (default "paper",
                                matching the assumptions baked into the
                                ReflAct paper's prompt)
    --resume              resume an interrupted --mpo-testset run from
                          --results' existing JSONL
    --model_name / --base_url   vLLM endpoint


--- ReflAct ---------------------------------------------------------------------

    python reflact_sciworld.py --mpo-testset --results results_reflact_211.jsonl \
        --model_name Qwen/Qwen3-8B --base_url http://<host>:8001/v1

Same options as ReAct. Uses examples/scienceworld_reflact_prompt.py
(Reflection / Action / Observation format) instead of the plain ReAct prompt.


--- Reflexion -------------------------------------------------------------------

    python reflexion_sciworld.py --mpo-testset --results results_reflexion_211.jsonl \
        --model_name Qwen/Qwen3-8B --base_url http://<host>:8001/v1

Imports react_sciworld.py directly (its ReAct rollout is the inner trial), so
it must stay in the same directory. Additional option:
    --num-trials N   outer Reflexion trials per env   (default 10; the paper
                     uses ~12)

Reflection few-shot examples: reflexion_few_shot_examples.txt (shipped).


================================================================================
3. COVERAGE
================================================================================

    Method        Qwen2.5-3B   Qwen3-8B   Gemma-4-12B
    ------------------------------------------------
    ReAct            yes         yes          yes       (--model_name)
    ReflAct           yes         yes          yes       (--model_name)
    Reflexion         yes         yes          yes       (--model_name)

All three baselines are single scripts that take the model as a CLI argument
— no per-model file copies exist.

GA-Rollback and THREAD were NOT ported to ScienceWorld in the source project
(no test_scienceworld/ dir exists in the GA-Rollback repo, and a
thread_sciworld.py exists but hardcodes Qwen3-8B only with no CLI override —
omitted here for the same reason THREAD was excluded from the WebShop
package).

See Conf-ReAct/README.txt for the conformal-guided search method (Conf-ReAct)
itself, evaluated against these baselines.
