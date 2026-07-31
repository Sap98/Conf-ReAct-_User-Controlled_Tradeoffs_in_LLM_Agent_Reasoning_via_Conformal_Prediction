"""
Reflexion self-sized-set baseline (trials 1/2/3) — WebShop, via LOG RECONSTRUCTION.

WebShop states are page observations, not just numbers. We rebuild each state's
running prompt straight from the conformal log:
  * FEW_SHOT_PROMPT (same as the runner)
  * initial page  = "WebShop\\nInstruction:\\n{instr}\\n[Search]"
  * for a node with Traj=[a1..ak], we look up each prefix's observation from an
    obs-map built by scanning "Action taken: a / Observation from Webshop: o"
    pairs (an action a taken under a node whose Traj=P yields the observation for
    trajectory P+[a]).
Then we ask the LLM (fresh call) for a SELF-SIZED set of next actions (at most K),
reflect, and repeat — like the game-24 baseline. covered = the CSV-optimal action
(fuzzy-matched) is in the set.

Stage 1 (this file, --dump): reconstruct + print a prompt to verify it looks sane
BEFORE spending LLM calls.
"""
import re
import ast
import argparse
from pathlib import Path

from fraction_of_states_vs_prediction_set_size import (
    CSV_PATH, load_csv_lookup, strip_price_clause,
    EPISODE_RE, ENV_INSTR_RE, CSV_INSTR_RE, INSTR_RE, NODE_HDR_RE, TRAJ_RE,
    _safe_literal_list,
)

# few-shot text lives in the runner; import it so we match exactly
# don't exec the whole runner (it connects to env/LLM); just read FEW_SHOT_PROMPT text
_src = (Path(__file__).resolve().parent / "conf_react_webshop.py").read_text()
_m = re.search(r'FEW_SHOT_PROMPT\s*=\s*"""(.*?)"""', _src, re.DOTALL)
FEW_SHOT_PROMPT = _m.group(1) if _m else ""

ACTION_TAKEN_RE = re.compile(r"^\s*Action taken:\s*(.+?)\s*$")
OBS_HDR_RE      = re.compile(r"^\s*Observation from Webshop:\s*(.*)$")
ARROW_RE        = re.compile(r"^\s*→")


def parse_log(path):
    """Return (states, obs_map, inst_map).
    states  : list of dicts {episode, instr_norm, traj(tuple)}
    obs_map : {(episode, traj_tuple): observation_text}
    inst_map: {episode: raw_instruction}
    """
    lines = Path(path).read_text(errors="ignore").splitlines()
    states = []
    obs_map = {}
    inst_map = {}
    episode = -1
    cur_instr_norm = None
    cur_traj = None                      # traj of the node currently being expanded
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if EPISODE_RE.match(line):
            episode = int(EPISODE_RE.match(line).group(1))
            cur_instr_norm = None; cur_traj = None
            i += 1; continue
        em = ENV_INSTR_RE.match(line) or INSTR_RE.match(line) or CSV_INSTR_RE.match(line)
        if em:
            raw = em.group(1).strip()
            if episode not in inst_map or ENV_INSTR_RE.match(line):
                inst_map[episode] = raw
            cur_instr_norm = strip_price_clause(raw)
            i += 1; continue
        if NODE_HDR_RE.match(line):
            # find this node's Traj
            j = i + 1; traj = None
            while j < n and not NODE_HDR_RE.match(lines[j]) and not EPISODE_RE.match(lines[j]):
                tm = TRAJ_RE.match(lines[j])
                if tm:
                    parsed = _safe_literal_list(tm.group(1))
                    if parsed is not None:
                        traj = tuple(parsed)
                    break
                j += 1
            cur_traj = traj if traj is not None else tuple()
            if cur_instr_norm is not None and traj is not None:
                states.append({"episode": episode, "instr_norm": cur_instr_norm, "traj": traj})
            i += 1; continue
        # capture "Action taken: a" + following observation → obs for cur_traj + (a,)
        am = ACTION_TAKEN_RE.match(line)
        if am and cur_traj is not None:
            action = am.group(1).strip()
            # find the Observation block
            k = i + 1
            while k < n and not OBS_HDR_RE.match(lines[k]) and not ACTION_TAKEN_RE.match(lines[k]) \
                    and not NODE_HDR_RE.match(lines[k]):
                k += 1
            if k < n and OBS_HDR_RE.match(lines[k]):
                obs_lines = [OBS_HDR_RE.match(lines[k]).group(1)]
                k += 1
                while k < n and not ARROW_RE.match(lines[k]) and not ACTION_TAKEN_RE.match(lines[k]) \
                        and not NODE_HDR_RE.match(lines[k]) and not EPISODE_RE.match(lines[k]):
                    obs_lines.append(lines[k])
                    k += 1
                obs_txt = "\n".join(obs_lines).strip("\n")
                obs_map[(episode, tuple(cur_traj) + (action,))] = obs_txt
            i = k; continue
        i += 1
    return states, obs_map, inst_map


def initial_obs(instr_raw):
    return f"WebShop\nInstruction:\n{instr_raw}\n[Search]"


def build_prompt(episode, traj, instr_raw, obs_map):
    """Reconstruct the running WebShop prompt up to (but not including) the next action.
    Returns None if any intermediate observation is missing."""
    obs0 = initial_obs(instr_raw)
    prompt = FEW_SHOT_PROMPT + "\n\n### Now complete this episode:\n" + obs0 + "\n\nAction:"
    for t in range(len(traj)):
        action = traj[t]
        prefix = tuple(traj[:t + 1])
        obs = obs_map.get((episode, prefix))
        if obs is None:
            return None
        prompt += f" {action}\nObservation: {obs}\n\nAction:"
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(Path(__file__).resolve().parent / "logs" / "bfs_webshop_qwen_3_8b_0.1.txt"))
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--dump", type=int, default=2, help="print this many reconstructed prompts")
    args = ap.parse_args()

    csv_lookup, _ = load_csv_lookup(args.csv)
    states, obs_map, inst_map = parse_log(args.log)
    print(f"parsed {len(states)} node-states  |  {len(obs_map)} observations  |  {len(inst_map)} episodes")

    # dedupe by (instr_norm, traj) and keep those matched in CSV (optimal known)
    seen = set(); usable = []
    for s in states:
        key = (s["instr_norm"], s["traj"])
        if key in seen or key not in csv_lookup:
            continue
        seen.add(key); usable.append(s)
    print(f"unique CSV-matched states: {len(usable)}")

    shown = 0
    for s in usable:
        instr_raw = inst_map.get(s["episode"], "")
        prompt = build_prompt(s["episode"], s["traj"], instr_raw, obs_map)
        if prompt is None:
            continue
        optimal = csv_lookup[(s["instr_norm"], s["traj"])][0]
        print("\n" + "=" * 78)
        print(f"episode={s['episode']}  traj={s['traj']}  OPTIMAL={optimal!r}")
        print("-" * 78)
        print(prompt[-1200:])          # tail of the prompt (current state)
        shown += 1
        if shown >= args.dump:
            break
    # coverage of reconstructability
    ok = sum(1 for s in usable
             if build_prompt(s["episode"], s["traj"], inst_map.get(s["episode"], ""), obs_map) is not None)
    print(f"\n\nReconstructable prompts: {ok}/{len(usable)}")


if __name__ == "__main__":
    main()
