"""
Full-MPO-211 BFS conformal play — RAM-light + shardable for parallelism.

Unlike smoke_sweep_sciworld.py (which loads the 36-59 GB BERT cache and warm-
starts the calibration pool from training data), this:
  * loads the ALREADY-SAVED calibration pool (calibrated_models/conformal_scores.pkl)
    via predictor.load() — no need to rebuild it from data, and
  * uses a FRESH, on-demand BertEmbeddingCache (BertEmbeddingCache.get() computes
    embeddings lazily on a miss), so each worker needs only ~0.5-1 GB instead of
    tens of GB.
Both together make it cheap to run MANY workers concurrently.

Sharding: worker `--shard-id` of `--num-shards` handles the MPO entries whose
global index ≡ shard-id (mod num-shards). Each worker writes its own JSONL;
merge afterwards (or just concatenate — one line per episode, disjoint sets).

One process = one alpha, one shard. Example (6 workers = 3 alphas x 2 shards):
    for a in 0.1 0.2 0.3; do for s in 0 1; do
      python bfs_conformal_211.py --alpha $a --shard-id $s --num-shards 2 \
        --base_url http://10.5.30.30:8001/v1 --model_name Qwen/Qwen3-8B \
        --results results/qwen_211/qwen_a${a}_shard${s}.jsonl &
    done; done
"""
import os
import sys
import json
import time
import argparse

import torch
from openai import OpenAI
from scienceworld import ScienceWorldEnv

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import bfs_conformal_sciworld_game24_style as m   # run_bfs_episode + helpers
from score_model import ScoreFunction, BertEmbeddingCache
from conformal_predictor import ConformalPredictor
from react_sciworld import PRESETS, load_mpo_testset, aggregate

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = os.path.join(_HERE, "trained_models", "best_score_model.pt")
POOL_PATH   = os.path.join(_HERE, "calibrated_models", "conformal_scores.pkl")


def parse_args():
    p = argparse.ArgumentParser(description="MPO-211 BFS conformal (RAM-light, shardable)")
    p.add_argument("--alpha",        type=float, required=True)
    p.add_argument("--shard-id",     type=int,   default=0)
    p.add_argument("--num-shards",   type=int,   default=1)
    # conformal / BFS (defaults match the final leaner sweep)
    p.add_argument("--method",       choices=["dual_lp", "quantile"], default="dual_lp")
    p.add_argument("--k",            type=int,   default=50)
    p.add_argument("--n_components", type=int,   default=10)
    p.add_argument("--n_samples",    type=int,   default=10)
    p.add_argument("--temperature",  type=float, default=0.7)
    p.add_argument("--score_batch_size", type=int, default=1)
    p.add_argument("--max_nodes",    type=int,   default=40)
    p.add_argument("--max_depth",    type=int,   default=50)
    p.add_argument("--dfs",          action="store_true", default=True)
    p.add_argument("--oracle_candidates", action="store_true", default=False)
    # env / io
    p.add_argument("--env-step-limit", type=int, default=200)
    p.add_argument("--simplifications-preset", choices=list(PRESETS), default="paper")
    p.add_argument("--results",      type=str,   required=True)
    p.add_argument("--resume",       action="store_true", default=True)
    p.add_argument("--model",        type=str,   default=MODEL_PATH)
    p.add_argument("--pool",         type=str,   default=POOL_PATH)
    p.add_argument("--base_url",     type=str,   default="http://10.5.30.30:8001/v1")
    p.add_argument("--model_name",   type=str,   default="Qwen/Qwen3-8B")
    p.add_argument("--verbose",      action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Wire the LLM client/model into the imported play module.
    m.MODEL_NAME = args.model_name
    m.client = OpenAI(base_url=args.base_url, api_key="EMPTY",
                      timeout=120.0, max_retries=5)

    tag = f"a{args.alpha}/shard{args.shard_id}"
    print(f"[{tag}] endpoint={args.base_url} model={args.model_name}", flush=True)
    try:
        served = [x.id for x in m.client.models.list().data]
        print(f"[{tag}] LLM OK: {served}", flush=True)
    except Exception as ex:
        sys.exit(f"[{tag}] LLM unreachable: {ex}")

    # Score model + on-demand BERT cache (RAM-light) + saved calibration pool.
    ckpt  = torch.load(args.model, map_location=DEVICE)
    model = ScoreFunction(d_proj=ckpt['d_proj'], hidden=ckpt['hidden']).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    bert_cache = BertEmbeddingCache(device=DEVICE)          # empty → computes on demand
    predictor  = ConformalPredictor(model, bert_cache, alpha=args.alpha)
    predictor.load(args.pool)          # restores cal_scores/state_raws/metadata
    predictor.alpha = args.alpha       # pool is alpha-independent; set live alpha
    print(f"[{tag}] pool={len(predictor.cal_scores)}  alpha={args.alpha}", flush=True)

    # Env + sharded MPO entries.
    env = ScienceWorldEnv("", envStepLimit=args.env_step_limit)
    simpl_str = PRESETS[args.simplifications_preset]
    entries = load_mpo_testset(env)
    my = [(i, e) for i, e in enumerate(entries)
          if i % args.num_shards == args.shard_id]
    print(f"[{tag}] {len(my)} of {len(entries)} MPO entries", flush=True)

    # Resume: skip (sw_name, var) already in results.
    seen = set()
    if args.resume and os.path.exists(args.results):
        for line in open(args.results):
            try:
                r = json.loads(line); seen.add((r['sw_name'], int(r['var'])))
            except Exception:
                pass
    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    out_f = open(args.results, 'a')

    scores, wons = [], []
    for i, e in my:
        sw_name, var_idx = e['sw_name'], e['var']
        if (sw_name, var_idx) in seen:
            continue
        cap = e['max_steps'] or args.max_depth
        max_depth = min(cap, args.max_depth)
        env.load(sw_name, var_idx, simpl_str, generateGoldPath=True)
        env.reset()
        task_desc = env.get_task_description().strip()
        gold = m.get_gold_sequence(env)

        t0 = time.time()
        (solved, score, traj, nodes, n_states, n_cov, win_len,
         n_cond_states, n_cond_cov) = m.run_bfs_episode(
            env, predictor, sw_name, var_idx, simpl_str, task_desc, gold,
            args, max_depth=max_depth, max_nodes=args.max_nodes)
        dt = time.time() - t0

        scores.append(score); wons.append(solved)
        print(f"[{tag}] [{i}] {e['mpo_name']} var={var_idx} solved={solved} "
              f"score={score} nodes={nodes} ({dt:.0f}s)  "
              f"running SR={sum(wons)}/{len(wons)} avg={sum(scores)/len(scores):.1f}",
              flush=True)
        out_f.write(json.dumps({
            'idx': i, 'mpo_name': e['mpo_name'], 'sw_name': sw_name, 'var': var_idx,
            'alpha': args.alpha, 'score': int(score), 'won': bool(solved),
            'trajectory': traj, 'n_nodes': nodes, 'n_states': n_states,
            'n_covered': n_cov, 'n_cond_states': n_cond_states,
            'n_cond_covered': n_cond_cov, 'win_len': win_len, 'wall_time': dt,
        }) + "\n")
        out_f.flush()

    out_f.close()
    n = len(scores)
    if n:
        agg = aggregate(scores, wons)
        print(f"[{tag}] DONE  n={n}  avg={agg['avg_reward']:.2f}  "
              f"SR={agg['success_rate']*100:.1f}%", flush=True)
    else:
        print(f"[{tag}] DONE  (all {len(my)} entries already in results)", flush=True)


if __name__ == "__main__":
    main()
