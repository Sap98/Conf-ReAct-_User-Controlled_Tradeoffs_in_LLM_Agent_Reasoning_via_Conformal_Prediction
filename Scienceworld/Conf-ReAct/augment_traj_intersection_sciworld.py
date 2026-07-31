"""
Stage 2b — Trajectory-intersection augmentation for ScienceWorld training data.

Generates off-gold-anchor states with reliable labels by:
  1. For each MPO (task, variation) pair, replaying gold once and snapshotting
     a (room, frozenset(inventory_items)) signature at every gold waypoint.
  2. For each gold step t, take one non-gold perturbation X from LLM-sampled
     candidates, then greedy-bushwhack up to `--bushwhack_depth` steps. If the
     env state-signature lands on a gold waypoint with index > t, the rollout
     recovered to gold. Record an off-gold training state:
         prev_actions   = gold[:t] + [X]
         optimal_action = a_1   (the first bushwhack action that started recovery)
     and generate the same Stage-2 features (candidate sampling at the off-gold
     state + echo-scored token logprobs).
  3. Output a pickle in the SAME schema as data_generation_sciworld.py, so
     Stage 3 (add_optimal_and_bins_sciworld.py) can re-process it directly.

Usage:
    python augment_traj_intersection_sciworld.py \
        --out training_data_aug_mpo_sciworld.pkl

    python augment_traj_intersection_sciworld.py --start 0 --end 10 --verbose
"""

import os
import re
import sys
import json
import time
import pickle
import argparse

_HERE  = os.path.dirname(os.path.abspath(__file__))   # .../conformal_prediction
_REACT = os.path.dirname(_HERE)                        # .../react
_REPO  = os.path.dirname(_REACT)                       # .../ScienceWorld
sys.path.insert(0, _REACT)
sys.path.insert(0, os.path.join(_REPO, "examples"))

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
from scienceworld import ScienceWorldEnv

from react_sciworld import (
    parse_thought_action, normalize_obs, PRESETS,
    mpo_to_sciworld_name, load_mpo_testset,
)
from scienceworld_react_prompt import SCIENCEWORLD_REACT_PROMPT
from data_generation_sciworld import (
    sample_candidate_actions, get_action_logprobs,
    INITIAL_LOCATION, _MOVE_PREFIXES, update_location,
)

# Reuse the same vLLM client as Stage 2 (already imported lazily by helpers).
MODEL_NAME = "Qwen/Qwen3-8B"
client = OpenAI(
    base_url="http://10.5.30.29:8001/v1",
    api_key="EMPTY",
    timeout=120.0,        # tolerate slow generations
    max_retries=5,        # retry transient APIConnectionError / 5xx
)


_TRANSIENT_EXC = (APIConnectionError, APITimeoutError, RateLimitError)

def _retry(fn, *args, _label="call", _tries=4, _backoff=2.0, **kwargs):
    """Call fn(*args, **kwargs); retry on transient connection errors with
    exponential backoff. Returns fn's result, or raises after _tries failures."""
    last = None
    for i in range(_tries):
        try:
            return fn(*args, **kwargs)
        except _TRANSIENT_EXC as ex:
            last = ex
            wait = _backoff ** i
            print(f"\n    [{_label}] transient {type(ex).__name__} (try {i+1}/{_tries}); sleeping {wait:.1f}s",
                  flush=True)
            time.sleep(wait)
    raise last


# ═══════════════════════════════════════════════════════════════════════════════
# STATE-SIGNATURE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_ROOM_RE = re.compile(r"called the (.+?)\.")


def get_room(env) -> str:
    """Pull a stable room identifier from env.look()."""
    look = env.look()
    m = _ROOM_RE.search(look)
    if m:
        return m.group(1).strip().lower()
    return look.split("\n", 1)[0].strip().lower()


def get_inventory_items(env) -> frozenset:
    """Pull a stable, order-independent set of inventory items."""
    inv = env.inventory()
    if not inv or "empty" in inv.lower():
        return frozenset()
    items = []
    for line in inv.split("\n"):
        line = line.strip().lstrip("-* \t")
        low = line.lower()
        if not line:
            continue
        if "inventory" in low or line.endswith(":"):
            continue
        items.append(line.lower())
    return frozenset(items)


def state_signature(env) -> tuple:
    """Cheap, deterministic state fingerprint used for gold-waypoint matching."""
    return (get_room(env), get_inventory_items(env))


# ═══════════════════════════════════════════════════════════════════════════════
# PER-EPISODE AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def replay_gold_and_snapshot(env, gold):
    """Replay gold from a fresh reset (env must already be `load`ed), snapshotting
    (signature -> earliest waypoint index t*) at every step. Returns
    (wp_index, obs_per_step) where:
        wp_index     : dict[signature] -> int  (the smallest t* matching it)
        obs_per_step : list[str]  (normalized observation after each gold action)
    """
    env.reset()
    wp_index = {}
    sig0 = state_signature(env)
    wp_index.setdefault(sig0, 0)

    obs_per_step = []
    for t, g in enumerate(gold, start=1):
        obs, _r, done, info = env.step(g)
        obs_per_step.append(normalize_obs(obs))
        sig = state_signature(env)
        wp_index.setdefault(sig, t)
        if done:
            break
    return wp_index, obs_per_step


def replay_to_anchor(env, gold_prefix, obs_per_step):
    """env.reset() + replay gold_prefix. Caller must have env.load()-ed the
    (task, var) already; we don't re-load every anchor (it's ~1-2 s and there
    are 20+ anchors per episode). Returns (running, prev_actions, location)
    where `running` uses an empty placeholder Thought per step."""
    env.reset()
    running = ""
    prev_actions = []
    location = INITIAL_LOCATION
    for t, a in enumerate(gold_prefix):
        obs, _r, done, info = env.step(a)
        obs_norm = obs_per_step[t] if t < len(obs_per_step) else normalize_obs(obs)
        running += f"Thought: \nAction: {a}\nObservation: {obs_norm}\n"
        prev_actions.append(a)
        location = update_location(location, a)
        if done:
            break
    return running, prev_actions, location


def bushwhack(env, base_prompt, depth, temperature=0.0, max_tokens=200):
    """Greedy LLM-driven rollout from the current env state. Yields
    (a_d, obs_norm, signature, done) after each step, up to `depth` times,
    stopping early on done."""
    cur = base_prompt
    for d in range(depth):
        resp = _retry(
            client.completions.create, _label=f"bushwhack-d{d}",
            model=MODEL_NAME, prompt=cur + "Thought:",
            n=1, max_tokens=max_tokens, temperature=temperature,
        )
        th, a = parse_thought_action(resp.choices[0].text)
        if not a:
            return
        a = a.strip()
        obs, _r, done, info = env.step(a)
        obs_norm = normalize_obs(obs)
        sig = state_signature(env)
        yield d, a, obs_norm, sig, done, info
        cur += f"Thought: {th}\nAction: {a}\nObservation: {obs_norm}\n"
        if done:
            return


def augment_episode(env, entry, simpl, args, save_cb=None):
    """Generate off-gold-anchor training states for one MPO entry.
    Returns (records, stats) where records is a list of state dicts.
    If save_cb is provided, it's called with (key, records_so_far) after every
    new record (incremental persistence)."""
    sw_name = entry["sw_name"]
    var     = entry["var"]

    # ── Gold path + task description ────────────────────────────────────────
    # ONE env.load per episode. All subsequent state restoration uses env.reset().
    env.load(sw_name, var, simpl, generateGoldPath=True)
    env.reset()
    task_desc = env.get_task_description().strip()
    gold = [g.strip() for g in env.get_gold_action_sequence()]
    if not gold or (len(gold) == 1 and str(gold[0]).startswith("ERROR")):
        return [], {"skip": f"no gold path ({gold})"}

    # ── Snapshot gold waypoint signatures ────────────────────────────────────
    wp_index, obs_per_step = replay_gold_and_snapshot(env, gold)

    # The ReAct prompt that prefaces every state's running buffer.
    base = "/no_think\n" + SCIENCEWORLD_REACT_PROMPT + task_desc + "\n"

    records  = []
    stats    = {"anchors": 0, "no_perturb": 0, "no_recovery": 0, "hits": 0}

    # ── For each anchor t, take one perturbation and try to recover ──────────
    print(f"  gold path length: {len(gold)}  (scanning anchors)")
    sys.stdout.flush()
    stats["transient_anchor"] = 0
    for t in range(len(gold)):
        gold_action = gold[t]

        try:
            # Restore env to gold[:t] and rebuild running prompt.
            running, prev_actions, location = replay_to_anchor(
                env, gold[:t], obs_per_step,
            )
            running = base + running
            stats["anchors"] += 1

            print(f"  t={t:2d}/{len(gold)-1}  sampling candidates...",
                  end="", flush=True)
            # Sample candidates at the anchor; pick first non-gold as perturbation.
            sampling_prompt = running + "Thought:"
            cand_actions, action_think = _retry(
                sample_candidate_actions, _label=f"cand-t{t}",
                sampling_prompt=sampling_prompt, n=args.n_samples,
                temperature=args.temperature,
            )
            print(f'\nThese are the candidate actions: {type(cand_actions)}')
            non_gold = [c for c in cand_actions if c != gold_action]
            if not non_gold:
                stats["no_perturb"] += 1
                print(" no_perturb", flush=True)
                continue
            X = non_gold[0]
            print(f" X={X!r}  bushwhacking...", end="", flush=True)
            th_X = action_think.get(X, "")

            # Take perturbation X.
            obs_X, _r, done_X, info_X = env.step(X)
            obs_X_norm = normalize_obs(obs_X)
            sig_after_X = state_signature(env)

            # Off-gold prompt running buffer for the bushwhack.
            off_gold_running = running + (
                f"Thought: {th_X}\nAction: {X}\nObservation: {obs_X_norm}\n"
            )

            # Edge case: X itself landed on a gold waypoint with t* > t.
            # Then X is effectively a different-but-valid action; not a useful
            # off-gold record (the agent didn't actually deviate). Skip.
            if (not done_X) and sig_after_X in wp_index and wp_index[sig_after_X] > t:
                stats["no_recovery"] += 1
                print(" X-on-gold (skip)", flush=True)
                continue
            if done_X:
                stats["no_recovery"] += 1
                print(" X-done (skip)", flush=True)
                continue

            # Bushwhack for up to D steps; first action that lands on a gold
            # waypoint > t is a valid recovery.
            recovery_action = None
            sig_d = None
            for d, a_d, obs_d, sig_d, done_d, info_d in bushwhack(
                env, off_gold_running, depth=args.bushwhack_depth,
                temperature=args.bushwhack_temperature, max_tokens=args.max_tokens,
            ):
                if d == 0:
                    recovery_action = a_d
                if sig_d in wp_index and wp_index[sig_d] > t:
                    break
                else:
                    recovery_action = None
                    if done_d:
                        break

            if recovery_action is None:
                stats["no_recovery"] += 1
                print(" no_recovery", flush=True)
                continue
            print(" HIT", flush=True)

            # ── Recovery confirmed. Generate features for the off-gold anchor. ──
            replay_to_anchor(env, gold[:t], obs_per_step)
            env.step(X)  # back to off-gold anchor

            feat_cands, feat_think = _retry(
                sample_candidate_actions, _label=f"featcand-t{t}",
                sampling_prompt=off_gold_running + "Thought:",
                n=args.n_samples, temperature=args.temperature,
            )
            feat_cands = list(dict.fromkeys(feat_cands + [recovery_action]))
            logprobs = _retry(
                get_action_logprobs, _label=f"logp-t{t}",
                scoring_prompt=off_gold_running + "Action:", actions=feat_cands,
                batch_size=args.score_batch_size,
            )

            record_B = {
                'task_name':          task_desc,
                'prev_actions':       prev_actions + [X],
                'location':           update_location(location, X),
                'admissible_actions': list(feat_cands),
                'log_probs_of_admissible_actions':
                    {a: logprobs.get(a, []) for a in feat_cands},
                'optimal_action':     recovery_action,
                'record_type':        'B',  # off-gold state, label = recovery a_1
            }
            records.append(record_B)
            stats["hits"] += 1

            # ── State-A alt-optimal record: at gold[:t], X is also a valid label. ──
            # The bushwhack proved that gold[:t] + X + ... rejoins gold within
            # `bushwhack_depth` steps, so X is an alt-optimal at State A alongside
            # gold[t] (the gold-pickle record's optimal_action at the same state).
            state_a_cands, _ = _retry(
                sample_candidate_actions, _label=f"a-cand-t{t}",
                sampling_prompt=running + "Thought:",
                n=args.n_samples, temperature=args.temperature,
            )
            state_a_cands = list(dict.fromkeys(state_a_cands + [X]))
            state_a_logprobs = _retry(
                get_action_logprobs, _label=f"a-logp-t{t}",
                scoring_prompt=running + "Action:", actions=state_a_cands,
                batch_size=args.score_batch_size,
            )
            record_A = {
                'task_name':          task_desc,
                'prev_actions':       list(prev_actions),
                'location':           location,
                'admissible_actions': list(state_a_cands),
                'log_probs_of_admissible_actions':
                    {a: state_a_logprobs.get(a, []) for a in state_a_cands},
                'optimal_action':     X,
                'record_type':        'A',  # gold anchor, label = alt-optimal X
            }
            records.append(record_A)
            stats.setdefault("a_records", 0)
            stats["a_records"] += 1

            print(f"    t={t:2d}  X={X!r}  → recovery={recovery_action!r}  "
                  f"(t*={wp_index[sig_d]})  [hit #{stats['hits']}, +A-record]")
            sys.stdout.flush()

            if save_cb is not None:
                save_cb(f"{sw_name}__var{var}__mpoaug", list(records))

        except _TRANSIENT_EXC as ex:
            stats["transient_anchor"] += 1
            print(f"\n    t={t:2d}  transient LLM error after retries: "
                  f"{type(ex).__name__}: {ex} — skipping this anchor",
                  flush=True)
            continue

    return records, stats


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="MPO trajectory-intersection augmentation")
    p.add_argument("--out", type=str,
                   default=os.path.join(_HERE, "training_data_aug_mpo_sciworld.pkl"))
    p.add_argument("--simplifications-preset", choices=list(PRESETS), default="paper")
    p.add_argument("--env-step-limit", type=int, default=2000,
                   help="Set high so reset+replay+perturb+bushwhack never trips it.")
    p.add_argument("--n-samples", type=int, default=10,
                   help="LLM samples for candidate set at each anchor.")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Temperature for the candidate-sampling step.")
    p.add_argument("--bushwhack-depth", type=int, default=5,
                   help="Max bushwhack depth in search of a gold waypoint.")
    p.add_argument("--bushwhack-temperature", type=float, default=0.0,
                   help="Greedy by default; raise for stochastic recovery.")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--score-batch-size", type=int, default=1)
    p.add_argument("--start", type=int, default=0, help="MPO entry index start.")
    p.add_argument("--end",   type=int, default=-1, help="MPO entry index end (-1 = all).")
    p.add_argument("--save-every", type=int, default=1,
                   help="Pickle to disk after every N episodes.")
    p.add_argument("--resume", action="store_true",
                   help="Skip MPO entries already in the output pickle.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args  = parse_args()
    simpl = PRESETS[args.simplifications_preset]

    print("=" * 72)
    print("  Stage 2b — Trajectory-intersection augmentation (ScienceWorld / MPO)")
    print("=" * 72)
    print(f"  simplifications : {simpl or '(none)'}")
    print(f"  n_samples       : {args.n_samples}")
    print(f"  bushwhack depth : {args.bushwhack_depth}")
    print(f"  out             : {args.out}")

    # Preflight LLM endpoint
    try:
        served = [m.id for m in client.models.list().data]
        print(f"  LLM endpoint    : OK  (serving {served})")
    except Exception as ex:
        sys.exit(f"\nLLM endpoint unreachable: {type(ex).__name__}: {ex}")

    env = ScienceWorldEnv("", envStepLimit=args.env_step_limit)
    entries = load_mpo_testset(env)

    end = len(entries) if args.end < 0 else min(args.end, len(entries))
    entries = entries[args.start:end]
    print(f"  episodes        : {len(entries)}  (range {args.start}:{end})")

    # Resume / load existing partial output
    training_data = {}
    if args.resume and os.path.exists(args.out):
        with open(args.out, 'rb') as f:
            training_data = pickle.load(f)
        print(f"  resumed         : {len(training_data)} episodes already in {args.out}")

    grand_stats = {"anchors": 0, "no_perturb": 0, "no_recovery": 0, "hits": 0,
                   "ok_ep": 0, "skip_ep": 0, "crash_ep": 0, "states": 0}

    for idx, entry in enumerate(entries, start=1):
        key = f"{entry['sw_name']}__var{entry['var']}__mpoaug"
        if args.resume and key in training_data:
            print(f"[{idx}/{len(entries)}] {entry['mpo_name']} var={entry['var']}  (cached)")
            continue

        t0 = time.time()
        print(f"\n[{idx}/{len(entries)}] {entry['mpo_name']}  var={entry['var']}  "
              f"sw_name={entry['sw_name']}")
        sys.stdout.flush()

        def _save_cb(ep_key, ep_records):
            training_data[ep_key] = ep_records
            with open(args.out, 'wb') as f:
                pickle.dump(training_data, f)

        try:
            records, stats = augment_episode(env, entry, simpl, args, save_cb=_save_cb)
        except Exception as ex:
            print(f"  CRASH: {type(ex).__name__}: {ex}")
            grand_stats["crash_ep"] += 1
            continue

        if not records:
            note = stats.get("skip") or (
                f"no_recovery on {stats.get('anchors',0)} anchors")
            print(f"  SKIP: {note}")
            grand_stats["skip_ep"] += 1
        else:
            training_data[key] = records
            grand_stats["ok_ep"]  += 1
            grand_stats["states"] += len(records)

        for k in ("anchors", "no_perturb", "no_recovery", "hits"):
            grand_stats[k] += stats.get(k, 0)

        dt = time.time() - t0
        anchors  = stats.get("anchors", 0)
        hits     = stats.get("hits", 0)
        hit_rate = (hits / anchors * 100.0) if anchors else 0.0
        print(f"  → anchors={anchors}  hits={hits}  ({hit_rate:.1f}% hit rate)  {dt:.1f}s")

        if idx % args.save_every == 0:
            with open(args.out, 'wb') as f:
                pickle.dump(training_data, f)

    # Final save
    with open(args.out, 'wb') as f:
        pickle.dump(training_data, f)

    print("\n" + "=" * 72)
    print(f"  DONE — ok={grand_stats['ok_ep']} skip={grand_stats['skip_ep']} "
          f"crash={grand_stats['crash_ep']}")
    print(f"  anchors={grand_stats['anchors']} no_perturb={grand_stats['no_perturb']} "
          f"no_recovery={grand_stats['no_recovery']} hits={grand_stats['hits']}")
    print(f"  recorded states: {grand_stats['states']}  in {args.out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
