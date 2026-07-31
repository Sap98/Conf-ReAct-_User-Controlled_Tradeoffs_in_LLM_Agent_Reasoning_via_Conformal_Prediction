"""
Reflexion-style SELF-SIZED prediction-set baseline — trials 1/2/3, Game of 24.

Per state, three trials of a self-sized move set:
  Trial 1 : LLM lists the move(s) it would seriously try (1..K)         [no reflection]
  Reflect : LLM critiques its OWN previous set (which look bad? any missed?)  — NO oracle
  Trial 2 : LLM revises the set given its previous set + reflection
  Reflect : again on trial-2 set
  Trial 3 : LLM revises again

The reflection uses only the model's own reasoning (never the ground-truth correct
set), so it is a clean self-refinement baseline. Per trial we record
    set_size = # unique valid moves,  covered = a correct move (oracle) is in the set
→ EMPIRICAL coverage per trial, comparable to the conformal method.

Temperature 0.7 (set size varies); fixed --seed for reproducibility.

Usage:
  python reflexion_selfset_3trials_game24.py \
     --logs compare_logs_100_qwen3_8b/m5_alpha01.txt ... \
     --base_url http://10.5.30.29:8001/v1 --out_csv reflexion_selfset_3trials_qwen3_8b.csv
"""
import os, sys, re, argparse, statistics as st
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g24_oracle import label_action   # noqa: E402

REACT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "game24_base_react.txt")
LOC_RE = re.compile(r"^\s*Location\s*:\s*(.+?)\s*$")
MOVE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")


def parse_states(log_path):
    states = {}
    for line in open(log_path):
        m = LOC_RE.match(line)
        if m:
            loc = m.group(1).strip()
            if len(loc.split()) >= 2:
                states[loc] = states.get(loc, 0) + 1
    return states


def _complete(client, model, prompt, temperature, seed, max_tokens=200, chat=False):
    if chat:
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature, top_p=1, seed=seed)
        return r.choices[0].message.content or ""
    r = client.completions.create(model=model, prompt=prompt, max_tokens=max_tokens,
                                  temperature=temperature, top_p=1, seed=seed)
    return r.choices[0].text


def _parse_moves(text, k):
    moves = []
    for m in MOVE_RE.finditer(text):
        mv = f"{m.group(1)} {m.group(2)} {m.group(3)} = {m.group(4)}"
        if mv not in moves:
            moves.append(mv)
        if len(moves) >= k:
            break
    return moves


def _validate(moves, loc):
    """Keep valid intermediate moves; return (valid_moves, covered)."""
    valid, covered = [], False
    for mv in moves:
        try:
            lab = label_action(loc.split(), mv)
        except Exception:
            continue
        if lab.get("kind") != "intermediate" or lab.get("label") not in ("correct", "dead_end"):
            continue
        valid.append(mv)
        if lab.get("label") == "correct":
            covered = True
    return valid, covered


def _gen_prompt(base, loc, k, prev_set=None, reflection=None):
    head = f"{base}\n# Current position. Remaining numbers: {loc}\n"
    if prev_set is None:
        body = (f"Think about which next move is best. Then output ONLY the move(s) you would "
                f"seriously try — typically just 1, at most {k} if you truly cannot decide. "
                f"Be selective: do NOT list moves that are clearly unhelpful. ")
    else:
        body = (f"Your previous candidate moves were: {prev_set}\n"
                f"Your reflection: {reflection}\n"
                f"Now give your REVISED final set of the move(s) you would seriously try "
                f"(as few as 1, at most {k}). Drop bad ones, add any good one you missed. ")
    return head + body + "Format 'a op b = c', one per line.\nMoves:\n1."


def _reflect_prompt(base, loc, prev_set):
    return (f"{base}\n# Current position. Remaining numbers: {loc}\n"
            f"You previously considered these next moves: {prev_set}\n"
            f"Briefly reflect (1-2 sentences): which of these look clearly bad and should be "
            f"dropped, and is there a promising move you may have missed? Reflection:")


def run_state(client, model, base, loc, k, temperature, seed, chat=False):
    """Return list of (set_size, covered) for trials 1,2,3."""
    results = []
    prev_valid = None
    for trial in range(3):
        if trial == 0:
            prompt = _gen_prompt(base, loc, k)
        else:
            refl = _complete(client, model, _reflect_prompt(base, loc, prev_valid),
                             temperature, seed, max_tokens=120, chat=chat).strip().splitlines()
            refl = refl[0] if refl else ""
            prompt = _gen_prompt(base, loc, k, prev_set=prev_valid, reflection=refl)
        # completion mode primes the prompt with "Moves:\n1." so re-attach it; chat
        # mode returns the full list itself.
        raw = _complete(client, model, prompt, temperature, seed, chat=chat)
        txt = raw if chat else ("1." + raw)
        valid, covered = _validate(_parse_moves(txt, k), loc)
        results.append((len(valid), int(covered)))
        prev_valid = valid if valid else prev_valid   # reflect on last non-empty set
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--base_url", default="http://10.5.30.29:8001/v1")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_states", type=int, default=0)
    ap.add_argument("--out_csv", default="reflexion_selfset_3trials_qwen3_8b.csv")
    ap.add_argument("--workers", type=int, default=16, help="concurrent states")
    ap.add_argument("--chat", action="store_true", help="chat.completions (instruct models e.g. gemma-it)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    base = Path(REACT_FILE).read_text()
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=120.0, max_retries=3)

    states = {}
    for lg in args.logs:
        for loc, c in parse_states(lg).items():
            states[loc] = states.get(loc, 0) + c
    locs = list(states.keys())
    if args.max_states > 0:
        locs = locs[:args.max_states]
    print(f"{len(states)} unique states; querying {len(locs)}  "
          f"(model={args.model_name}, k_cap={args.k}, T={args.temperature}, seed={args.seed}, "
          f"workers={args.workers}, chat={args.chat})")

    res_by_i = [None] * len(locs)
    done = [0]; fail = [0]; lock = Lock()

    def work(i):
        try:
            r = run_state(client, args.model_name, base, locs[i], args.k,
                          args.temperature, args.seed, args.chat)
        except Exception as e:
            r = None
            with lock:
                fail[0] += 1
                if fail[0] <= 5:
                    print(f"  state {i} failed: {type(e).__name__}", flush=True)
        with lock:
            res_by_i[i] = r
            done[0] += 1
            if done[0] % 50 == 0:
                print(f"  ... {done[0]}/{len(locs)} done (fail={fail[0]})", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i) for i in range(len(locs))]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception:
                pass

    rows = []   # (loc, s1,c1, s2,c2, s3,c3)
    cov = [0, 0, 0]
    for i, loc in enumerate(locs):
        res = res_by_i[i]
        if res is None:
            continue
        for t in range(3):
            cov[t] += res[t][1]
        rows.append((loc, *[x for pair in res for x in pair]))

    n = len(rows)
    print(f"\n=== SELF-SIZED SET baseline, trials 1/2/3 ({args.model_name}) ===")
    print(f"  states: {n}   seed={args.seed}  T={args.temperature}")
    for t in range(3):
        sizes = [r[1 + 2 * t] for r in rows]
        print(f"  Trial {t+1}: mean_size={st.mean(sizes):.2f}  "
              f"empirical_coverage={100*cov[t]/n:.1f}%  "
              f"size_hist={dict(sorted(Counter(sizes).items()))}")
    with open(args.out_csv, "w") as f:
        f.write("location,size_t1,cov_t1,size_t2,cov_t2,size_t3,cov_t3\n")
        for r in rows:
            f.write(f'"{r[0]}",{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]}\n')
    print(f"\nSaved -> {args.out_csv}")


if __name__ == "__main__":
    main()
