"""Reflexion self-sized-set baseline — ScienceWorld (game24 analog).

For each held-out gold state (the SAME seed-42 test' split the offline conformal
eval / dump_setsize_coverage_sciworld.py use), ask the LLM for a SELF-SIZED set
of at most K next actions chosen from that state's admissible actions, across 3
trials with a reflection step in between. covered = a gold (optimal) action is in
the returned set. Writes one CSV per trial:

    reflexion_selfset_trial{1,2,3}_<stem>.csv   (cols: state_id,set_size,covered)

so plot_setsize_distribution_coverage_sciworld.py picks them up as the
'reflexion T1/T2/T3' bars.

Run once per model (each hits its own endpoint):
  python reflexion_selfset_sciworld.py --data training_data_merged_sciworld.pkl \
     --base_url http://10.5.30.29:8001/v1 --model_name Qwen/Qwen3-8B  --stem qwen3
  python reflexion_selfset_sciworld.py --data ../conformal_prediction_qwen25_3b/training_data_2_sciworld.pkl \
     --base_url http://10.5.30.29:8003/v1 --model_name Qwen/Qwen2.5-3B --stem qwen25
  python reflexion_selfset_sciworld.py --data ../conformal_prediction_gemma/training_data_2_sciworld.pkl \
     --base_url http://10.5.18.73:8003/v1 --model_name google/gemma-4-12B-it --stem gemma
"""
import os, csv, random, argparse, pickle, statistics as st
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from openai import OpenAI

SEED, VAL_SPLIT, CAL_FRAC = 42, 0.2, 0.5


def build_meta_records(training_data):
    """Same order + skip logic as train_score.build_records, meta only (no BERT)."""
    recs = []
    for _game_file, states in training_data.items():
        for state in states:
            admissible = state.get('admissible_actions', [])
            if 'optimal_actions' in state:
                optimals = list(state['optimal_actions'])
            else:
                opt = state.get('optimal_action', '')
                optimals = [opt] if opt else []
            optimal_idxs = [admissible.index(o) for o in optimals if o in admissible]
            if not admissible or not optimal_idxs:
                continue
            recs.append({
                'task_name': state['task_name'], 'location': state['location'],
                'prev_actions': list(state.get('prev_actions', [])),
                'admissible_actions': list(admissible),
                'optimal_actions': [admissible[i] for i in sorted(set(optimal_idxs))],
            })
    return recs


def get_test_records(training_data, max_test=0):
    recs = build_meta_records(training_data)
    random.seed(SEED)
    idx = list(range(len(recs))); random.shuffle(idx)
    val = [recs[i] for i in idx[:int(len(recs) * VAL_SPLIT)]]
    random.seed(SEED + 1)
    vidx = list(range(len(val))); random.shuffle(vidx)
    n_cal = int(len(val) * CAL_FRAC)
    test = [val[i] for i in vidx[n_cal:]]
    return test[:max_test] if max_test > 0 else test


def _state_block(rec):
    adm = "\n".join(f"  - {a}" for a in rec['admissible_actions'])
    prev = ", ".join(rec['prev_actions']) if rec['prev_actions'] else "none"
    return (f"You are an agent solving a ScienceWorld task.\n"
            f"Task: {rec['task_name']}\n"
            f"Current location: {rec['location']}\n"
            f"Actions taken so far: {prev}\n\n"
            f"Admissible next actions:\n{adm}\n\n")


def gen_prompt(rec, k, prev_set=None, reflection=None):
    if prev_set is None:
        instr = (f"From the admissible actions listed above, output the AT MOST {k} actions "
                 f"most likely to be the correct next step. Copy each EXACTLY as written "
                 f"above. Use as few as 1 if you are confident. No explanations. "
                 f"Output only the actions, one per line:\n1. ")
    else:
        instr = (f"Your previous candidate actions were: {prev_set}\nReflection: {reflection}\n\n"
                 f"Give your REVISED set of at most {k} actions chosen from the admissible list "
                 f"above (exact copies, drop wrong ones, add a better one if missed). "
                 f"Output only the actions, one per line:\n1. ")
    return _state_block(rec) + instr


def reflect_prompt(rec, prev_set):
    return (_state_block(rec) +
            f"You proposed these next actions: {prev_set}\n"
            f"In ONE sentence, reflect: which look wrong for this task/state, and is there a "
            f"better admissible action you missed? Reflection:")


def complete(client, model, prompt, nothink, temperature, seed, chat, max_tokens=120):
    if chat:
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature, top_p=1, seed=seed)
        return r.choices[0].message.content or ""
    r = client.completions.create(model=model, prompt=(nothink + prompt),
                                  max_tokens=max_tokens, temperature=temperature,
                                  top_p=1, seed=seed)
    return r.choices[0].text


def parse_set(text, admissible, k):
    """Return up to k admissible actions mentioned in text (exact-line, else substring)."""
    adm_lower = {a.lower(): a for a in admissible}
    out = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("0123456789.)-* ").strip().rstrip(".").strip()
        if not line:
            continue
        hit = adm_lower.get(line.lower())
        if hit is None:
            cands = [a for a in admissible if a.lower() == line.lower()
                     or a.lower() in line.lower() or line.lower() in a.lower()]
            hit = min(cands, key=len) if cands else None
        if hit and hit not in out:
            out.append(hit)
        if len(out) >= k:
            break
    return out


def run_state(client, model, rec, k, nothink, temperature, seed, chat):
    golds = set(rec['optimal_actions']); adm = rec['admissible_actions']
    results, prev = [], None
    for trial in range(3):
        if trial == 0:
            prompt = gen_prompt(rec, k)
        else:
            refl = complete(client, model, reflect_prompt(rec, prev), nothink,
                            temperature, seed, chat, max_tokens=80).strip().splitlines()
            refl = refl[0] if refl else ""
            prompt = gen_prompt(rec, k, prev_set=prev, reflection=refl)
        acts = parse_set("1. " + complete(client, model, prompt, nothink,
                                           temperature, seed, chat), adm, k)
        covered = 1 if (golds & set(acts)) else 0
        results.append((len(acts), covered))
        prev = acts if acts else prev
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--stem", required=True, help="qwen3 / qwen25 / gemma (output file suffix)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max_test", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--workers", type=int, default=16, help="concurrent states (vLLM batches)")
    ap.add_argument("--chat", action="store_true", help="use chat.completions (instruct models e.g. gemma-it)")
    args = ap.parse_args()

    with open(args.data, "rb") as f:
        training_data = pickle.load(f)
    test = get_test_records(training_data, args.max_test)
    nothink = "/no_think\n" if "qwen" in args.model_name.lower() else ""
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=90.0, max_retries=2)
    print(f"[{args.stem}] model={args.model_name}  states={len(test)}  k={args.k}  "
          f"workers={args.workers}  chat={args.chat}", flush=True)

    results = [None] * len(test)
    done = [0]; fail = [0]; lock = Lock()

    def work(i):
        try:
            res = run_state(client, args.model_name, test[i], args.k, nothink,
                            args.temperature, args.seed, args.chat)
        except Exception as e:               # a hung/failed state must not kill the run
            res = None
            with lock:
                fail[0] += 1
                if fail[0] <= 10:
                    print(f"  [{args.stem}] state {i} failed: {type(e).__name__}", flush=True)
        with lock:
            results[i] = res
            done[0] += 1
            if done[0] % 50 == 0:
                print(f"  [{args.stem}] {done[0]}/{len(test)} done (fail={fail[0]})", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i) for i in range(len(test))]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception:
                pass

    rows = [(i, results[i]) for i in range(len(test)) if results[i] is not None]
    cov = [sum(r[1][t][1] for r in rows) for t in range(3)]
    n = len(rows)
    if n == 0:
        print(f"[{args.stem}] ERROR: all states failed (endpoint down?) — nothing written", flush=True)
        return
    for t in range(3):
        sizes = [r[1][t][0] for r in rows]
        with open(os.path.join(args.out_dir, f"reflexion_selfset_trial{t+1}_{args.stem}.csv"),
                  "w", newline="") as f:
            w = csv.writer(f); w.writerow(["state_id", "set_size", "covered"])
            for sid, res in rows:
                w.writerow([sid, res[t][0], res[t][1]])
        print(f"[{args.stem}] Trial {t+1}: mean_size={st.mean(sizes):.2f}  "
              f"coverage={100*cov[t]/max(n,1):.1f}%  size_hist={dict(sorted(Counter(sizes).items()))}",
              flush=True)
    print(f"[{args.stem}] done ({n} states)", flush=True)


if __name__ == "__main__":
    main()
