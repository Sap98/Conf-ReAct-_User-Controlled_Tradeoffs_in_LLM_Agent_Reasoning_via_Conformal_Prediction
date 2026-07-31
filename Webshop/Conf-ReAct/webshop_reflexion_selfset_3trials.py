"""
Reflexion self-sized-set baseline (trials 1/2/3) — WebShop, list-based k=3.

Reuses the log-reconstruction in webshop_reflexion_selfset.py to rebuild each
state's running prompt, then (like game-24) asks the LLM for AT MOST K next
actions — self-sized. An [ANALYSIS MODE] instruction suspends the few-shot's
one-action rule so click states also yield a set. Reflection between trials is
self-refinement (no oracle). covered = CSV-optimal action (fuzzy-matched) is in
the returned set.

Per state, per trial: set_size (# unique valid actions), covered (0/1).

Usage:
  python webshop_reflexion_selfset_3trials.py \
     --logs bfs_webshop_qwen_3_8b_0.1.txt bfs_webshop_qwen_3_8b_0.2.txt bfs_webshop_qwen_3_8b_0.3.txt \
     --base_url http://10.5.30.29:8001/v1 --model_name Qwen/Qwen3-8B \
     --out_csv webshop_reflexion_k3_3trials_qwen3_8b.csv
"""
import os, sys, re, argparse, statistics as st
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import webshop_reflexion_selfset as W
from fraction_of_states_vs_prediction_set_size import CSV_PATH, load_csv_lookup, in_action_set

ACT_RE = re.compile(r'(search\[[^\]]*\]|click\[[^\]]*\])')   # think[] not a real move


def _complete(client, model, prompt, temperature, seed, max_tokens=100, chat=False):
    if chat:
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature, top_p=1, seed=seed)
        return r.choices[0].message.content or ""
    r = client.completions.create(model=model, prompt=prompt, max_tokens=max_tokens,
                                  temperature=temperature, top_p=1, seed=seed)
    return r.choices[0].text


def _parse_actions(text, k):
    acts = []
    for m in ACT_RE.finditer(text):
        a = m.group(1)
        if a not in acts:
            acts.append(a)
        if len(acts) >= k:
            break
    return acts


def _gen_prompt(base, k, prev_set=None, reflection=None):
    p = base.rstrip()
    if p.endswith("Action:"):
        p = p[:-len("Action:")].rstrip()
    if prev_set is None:
        instr = (f"[ANALYSIS MODE] Ignore the one-action-per-step rule for THIS answer only. "
                 f"Given the current page above, list up to {k} DIFFERENT concrete next actions "
                 f"you might take (use REAL ids/options/queries visible above; no placeholders, "
                 f"no reasoning). As few as 1 if you are confident. Output only the actions, "
                 f"one per line:\n1. ")
    else:
        instr = (f"[ANALYSIS MODE] Your previous candidate actions were: {prev_set}\n"
                 f"Reflection: {reflection}\n"
                 f"Give your REVISED set of up to {k} concrete next actions (drop bad ones, add a "
                 f"better one if missed; use REAL ids/options/queries from the page; no reasoning). "
                 f"Output only the actions, one per line:\n1. ")
    return p + "\n\n" + instr


def _reflect(client, model, base, prev_set, temperature, seed, chat=False):
    p = base.rstrip()
    if p.endswith("Action:"):
        p = p[:-len("Action:")].rstrip()
    rp = (p + f"\n\n[ANALYSIS MODE] You proposed these next actions: {prev_set}\n"
          f"Briefly reflect (1 sentence): which look wrong for the instruction, and is there a "
          f"better concrete action from the page you missed? Reflection:")
    if not prev_set:
        return ""
    lines = _complete(client, model, rp, temperature, seed, max_tokens=80, chat=chat).strip().splitlines()
    return lines[0] if lines else ""


def run_state(client, model, base, optimal, k, jaccard, temperature, seed, chat=False):
    results = []
    prev = None
    for trial in range(3):
        if trial == 0:
            prompt = _gen_prompt(base, k)
        else:
            refl = _reflect(client, model, base, prev, temperature, seed, chat)
            prompt = _gen_prompt(base, k, prev_set=prev, reflection=refl)
        raw = _complete(client, model, prompt, temperature, seed, chat=chat)
        acts = _parse_actions(raw if chat else ("1. " + raw), k)
        covered = 1 if in_action_set(optimal, acts, jaccard) else 0
        results.append((len(acts), covered))
        prev = acts if acts else prev
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--base_url", default="http://10.5.30.29:8001/v1")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--jaccard", type=float, default=0.8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_states", type=int, default=0)
    ap.add_argument("--out_csv", default="webshop_reflexion_k3_3trials_qwen3_8b.csv")
    ap.add_argument("--workers", type=int, default=16, help="concurrent states")
    ap.add_argument("--chat", action="store_true", help="chat.completions (instruct models e.g. gemma-it)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    csv_lookup, _ = load_csv_lookup(args.csv)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=120.0, max_retries=3)

    # build unique reconstructed states across all logs (dedupe by instr_norm+traj)
    combined = {}     # key -> (base_prompt, optimal)
    for lg in args.logs:
        states, obs_map, inst_map = W.parse_log(lg)
        for s in states:
            key = (s["instr_norm"], s["traj"])
            if key in combined or key not in csv_lookup:
                continue
            base = W.build_prompt(s["episode"], s["traj"], inst_map.get(s["episode"], ""), obs_map)
            if base is None:
                continue
            combined[key] = (base, csv_lookup[key][0])
    items = list(combined.items())
    if args.max_states > 0:
        items = items[:args.max_states]
    print(f"unique reconstructed CSV-matched states: {len(items)}  "
          f"(model={args.model_name}, k={args.k}, T={args.temperature}, seed={args.seed})")

    res_by_i = [None] * len(items)
    done = [0]; fail = [0]; lock = Lock()

    def work(i):
        key, (base, optimal) = items[i]
        try:
            r = run_state(client, args.model_name, base, optimal, args.k,
                          args.jaccard, args.temperature, args.seed, args.chat)
        except Exception as e:
            r = None
            with lock:
                fail[0] += 1
                if fail[0] <= 5:
                    print(f"  state {i} failed: {type(e).__name__}", flush=True)
        with lock:
            res_by_i[i] = r
            done[0] += 1
            if done[0] % 20 == 0:
                print(f"  ... {done[0]}/{len(items)} done (fail={fail[0]})", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i) for i in range(len(items))]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception:
                pass

    rows = []; cov = [0, 0, 0]
    for i, (key, _bo) in enumerate(items):
        res = res_by_i[i]
        if res is None:
            continue
        for t in range(3):
            cov[t] += res[t][1]
        rows.append((key, res))

    n = len(rows)
    print(f"\n=== WebShop reflexion self-set (k={args.k}) trials 1/2/3 ({args.model_name}) ===")
    print(f"  states: {n}  seed={args.seed}  T={args.temperature}")
    for t in range(3):
        sizes = [r[1][t][0] for r in rows]
        print(f"  Trial {t+1}: mean_size={st.mean(sizes):.2f}  "
              f"empirical_coverage={100*cov[t]/n:.1f}%  size_hist={dict(sorted(Counter(sizes).items()))}")
    with open(args.out_csv, "w") as f:
        f.write("instr,traj,size_t1,cov_t1,size_t2,cov_t2,size_t3,cov_t3\n")
        for (instr, traj), res in rows:
            f.write(f'"{instr}","{traj}",{res[0][0]},{res[0][1]},{res[1][0]},{res[1][1]},{res[2][0]},{res[2][1]}\n')
    print(f"\nSaved -> {args.out_csv}")


if __name__ == "__main__":
    main()
