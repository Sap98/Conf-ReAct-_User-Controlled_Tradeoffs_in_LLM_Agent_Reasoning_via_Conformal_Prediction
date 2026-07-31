"""
Count LLM inference calls per method — Game of 24, from the output logs.
Methodology matches baselines_gpt_4.1/count_llm_calls_winning.py (WebShop):
BFS conformal = ONE completions request per [BFS Node] (the sampling that draws
n_samples generations).  The echo-scoring / logprob calls that feed the trained
score model are NOT counted (they are the "score model" side, not an agent LLM
decision).  Game-24 additionally issues one natural_think generation per node,
reported separately so you can include it or not.

Baselines (task-solving agents), per model dir compare_logs_100_<model>/:
  base_react / base_reflact / base_rollback : 1 LLM call per action step
      = each 'Act <n>>' / 'Act <n>:' line.
  base_reflexion : cumulative over its 3 trials (all trials' action steps)
      — "calls needed to (eventually) solve".

BFS conformal (m5_alpha0{1,2,3}.txt), per node:
  sample : 1 request  (draws n_samples generations)      <- matches WebShop "req"
  think  : 1 request  (natural_think, 1 generation)      <- game-24 only
  score  : echo-scoring, EXCLUDED (feeds the score model)
Reported: nodes, sample_req(=nodes), agent_req(=think+sample=2*nodes),
          generations(=nodes*(1+n_samples)).

Usage:  python count_llm_calls_game24.py
"""
import re
import argparse
from pathlib import Path

ACT_RE   = re.compile(r"^\s*Act\s+\d+\s*[>:]")            # one agent step = 1 gen call
NODE_RE  = re.compile(r"^\[BFS Node")
NSAMP_RE = re.compile(r"^\s*n_samples\s*:\s*(\d+)")


def count_baseline(path):
    if not Path(path).exists():
        return None
    return sum(1 for line in open(path, errors="ignore") if ACT_RE.match(line))


def count_bfs(path):
    if not Path(path).exists():
        return None
    nodes = 0
    n_samples = None
    for line in open(path, errors="ignore"):
        if n_samples is None:
            m = NSAMP_RE.match(line)
            if m:
                n_samples = int(m.group(1))
        if NODE_RE.match(line):
            nodes += 1
    ns = n_samples or 10
    return dict(nodes=nodes, n_samples=ns,
                sample_req=nodes,               # WebShop-style: 1 request/node
                agent_req=2 * nodes,            # + natural_think request/node
                generations=nodes * (1 + ns))   # think(1) + sample(n_samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--models", nargs="+", default=["qwen25_3b", "qwen3_8b", "gemma4_12b"])
    ap.add_argument("--alphas", nargs="+", default=["01", "02", "03"])
    args = ap.parse_args()
    root = Path(args.root)

    NICE = {"qwen25_3b": "Qwen2.5-3B", "qwen3_8b": "Qwen3-8B", "gemma4_12b": "Gemma-4-12B"}
    BASELINES = [("ReAct", "base_react.txt"), ("ReflAct", "base_reflact.txt"),
                 ("Reflexion", "base_reflexion.txt"), ("Rollback", "base_rollback.txt")]

    for m in args.models:
        d = root / f"compare_logs_100_{m}"
        print(f"\n{'='*74}\n{NICE.get(m, m)}   (100 test puzzles)\n{'='*74}")
        if not d.exists():
            print(f"  dir not found: {d}"); continue

        print("  BASELINES  (1 LLM call per action step; Reflexion cumulative over 3 trials):")
        print(f"    {'method':<12}{'LLM calls':>12}")
        for name, fn in BASELINES:
            c = count_baseline(d / fn)
            print(f"    {name:<12}{('—' if c is None else str(c)):>12}")

        print("\n  BFS CONFORMAL  (echo-scoring / score-model calls EXCLUDED):")
        print(f"    {'α':>4}{'nodes':>7}{'sample_req':>12}{'+think_req':>12}{'generations':>13}")
        tot = dict(nodes=0, sample_req=0, agent_req=0, generations=0)
        for a in args.alphas:
            r = count_bfs(d / f"m5_alpha{a}.txt")
            if r is None:
                print(f"    0.{a[-1]}   (log missing)"); continue
            print(f"    0.{a[-1]:<3}{r['nodes']:>7}{r['sample_req']:>12}"
                  f"{r['agent_req']:>12}{r['generations']:>13}")
            for k in tot:
                tot[k] += r[k]
        print(f"    {'SUM':>4}{tot['nodes']:>7}{tot['sample_req']:>12}"
              f"{tot['agent_req']:>12}{tot['generations']:>13}")
        print("    (sample_req = 1 request/node like WebShop; +think_req adds the "
              "per-node natural_think; generations = nodes*(1+n_samples))")


if __name__ == "__main__":
    main()
