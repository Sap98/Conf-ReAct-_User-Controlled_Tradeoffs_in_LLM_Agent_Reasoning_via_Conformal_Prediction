"""
Ground-truth oracle for Game of 24 (state = multiset of numbers "left").

This gives you a PER-STATE label for any agent-proposed action (ReAct / Reflexion
/ rollback / ToT), instead of only the final-answer check that the environments do.

Core idea
---------
A state is a multiset of rationals (e.g. {3, 3, 4}). An action consumes two of
them with one of {+,-,*,/} and produces one new number. `solvable(state)` answers
"can this multiset still reach exactly 24?" via exhaustive DFS + memoization.

An action is CORRECT (== OPTIMAL) iff the state it leads to is still solvable.
Every Game-of-24 solution uses exactly n-1 ops, so there is no "shorter" line to
prefer -- the optimal policy is simply: never move into an unsolvable state.

Action format understood (matches game24.py's `step`):
    intermediate :  "1 + 2 = 3"          (a op b = c)
    answer       :  "answer: (1+2+3)*4 = 24"
    think        :  "think: ..."          (no state change)

Public API
----------
    solvable(state_tuple)                  -> bool
    correct_actions(numbers)               -> list of solvability-preserving moves
    label_action(numbers, action)          -> dict label for one action
    TrajectoryScorer(puzzle)               -> .step(action) per step, .summary()
    score_trajectory(puzzle, actions)      -> per-step labels + summary
    optimal_action_supervision(numbers)    -> {action_str: 'correct'|'dead_end'}
"""
import re
from fractions import Fraction as F
from functools import lru_cache
from itertools import combinations

import sympy

TARGET = F(24)

# number token: optional sign, integer, optional /denominator  (e.g. 3, -2, 8/3)
_NUM = r'-?\d+(?:/\d+)?'
_INTERMEDIATE_RE = re.compile(
    rf'^\(*\s*({_NUM})\s*([+\-*/])\s*({_NUM})\s*\)*$'
)


# --------------------------------------------------------------------------- #
# Core solvability oracle
# --------------------------------------------------------------------------- #
def _ops(a: F, b: F):
    """All (result, symbol) reachable from ordered pair (a, b)."""
    yield a + b, '+'
    yield a * b, '*'
    yield a - b, '-'
    yield b - a, '-'
    if b != 0:
        yield a / b, '/'
    if a != 0:
        yield b / a, '/'


@lru_cache(maxsize=None)
def solvable(state: tuple) -> bool:
    """state: a sorted tuple of Fractions. True iff 24 is still reachable."""
    if len(state) == 1:
        return state[0] == TARGET
    nums = list(state)
    for i, j in combinations(range(len(nums)), 2):
        rest = nums[:i] + nums[i + 1:j] + nums[j + 1:]
        for r, _ in _ops(nums[i], nums[j]):
            if solvable(tuple(sorted(rest + [r]))):
                return True
    return False


def to_state(numbers) -> tuple:
    """Coerce a string ("3 3 4"), or an iterable of ints/strs/Fractions,
    into a sorted Fraction tuple."""
    if isinstance(numbers, str):
        numbers = numbers.split()
    out = []
    for n in numbers:
        out.append(n if isinstance(n, F) else F(str(n)))
    return tuple(sorted(out))


def fmt(x: F) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


# --------------------------------------------------------------------------- #
# Enumerate correct / all actions from a state
# --------------------------------------------------------------------------- #
def _binops(x: F, y: F):
    """Yield (result, 'lhs op rhs = result') with CORRECT operand ordering."""
    yield x + y, f"{fmt(x)} + {fmt(y)} = {fmt(x + y)}"
    yield x * y, f"{fmt(x)} * {fmt(y)} = {fmt(x * y)}"
    yield x - y, f"{fmt(x)} - {fmt(y)} = {fmt(x - y)}"
    yield y - x, f"{fmt(y)} - {fmt(x)} = {fmt(y - x)}"
    if y != 0:
        yield x / y, f"{fmt(x)} / {fmt(y)} = {fmt(x / y)}"
    if x != 0:
        yield y / x, f"{fmt(y)} / {fmt(x)} = {fmt(y / x)}"


def _child_moves(state: tuple):
    """Yield (child_state, action_str) for every distinct legal action."""
    nums = list(state)
    seen = set()
    for i, j in combinations(range(len(nums)), 2):
        rest = nums[:i] + nums[i + 1:j] + nums[j + 1:]
        for r, action in _binops(nums[i], nums[j]):
            child = tuple(sorted(rest + [r]))
            key = (action, child)
            if key in seen:
                continue
            seen.add(key)
            yield child, action


def correct_actions(numbers):
    """List of action strings whose resulting state keeps 24 reachable."""
    state = to_state(numbers)
    return [a for child, a in _child_moves(state) if solvable(child)]


def optimal_action_supervision(numbers):
    """
    Full supervision signal for a state: every legal action mapped to
    'correct' (solvability-preserving) or 'dead_end'. Useful for building
    step-level training targets.
    """
    state = to_state(numbers)
    return {a: ('correct' if solvable(child) else 'dead_end')
            for child, a in _child_moves(state)}


# --------------------------------------------------------------------------- #
# Label a single proposed action against a known state
# --------------------------------------------------------------------------- #
def _try_remove(state_list, value):
    """Remove one occurrence of `value` from list; return new list or None."""
    for k, v in enumerate(state_list):
        if v == value:
            return state_list[:k] + state_list[k + 1:]
    return None


def label_action(numbers, action: str, reference=None) -> dict:
    """
    Classify one agent action against the current state `numbers`.
    `reference` (the original puzzle numbers) is used to validate `answer:`
    expressions, mirroring the env's check against the original 4 numbers;
    defaults to `numbers` when not given.

    Returns a dict with:
        kind          : 'intermediate' | 'answer' | 'think'
        label         : 'correct' | 'dead_end' | 'illegal' | 'wrong_arithmetic'
                        | 'win' | 'lose' | 'neutral'
        parent_solvable : was 24 reachable BEFORE this action?
        child          : resulting state (list of str) for intermediate moves
        child_solvable : is 24 reachable AFTER this action?
        detail         : human-readable note
    """
    state = list(to_state(numbers))
    parent_solvable = solvable(tuple(state))
    a = action.strip()
    low = a.lower()

    # -- think: no state change ------------------------------------------- #
    if low.startswith('think'):
        return dict(kind='think', label='neutral',
                    parent_solvable=parent_solvable, child=None,
                    child_solvable=parent_solvable, detail='reasoning step')

    # -- answer: terminal check (mirrors the env's sympy check) ----------- #
    if low.startswith('answer'):
        expr = a.split(':', 1)[1] if ':' in a else a
        expr = expr.split('=')[0]
        used = re.findall(r'\d+', expr)
        ref = reference if reference is not None else numbers
        have = re.findall(r'\d+', ref) if isinstance(ref, str) \
            else [str(x) for x in ref]
        if sorted(used) != sorted(have):
            return dict(kind='answer', label='lose',
                        parent_solvable=parent_solvable, child=None,
                        child_solvable=False,
                        detail='answer numbers do not match the state')
        try:
            ok = int(sympy.simplify(expr) == 24)
        except Exception as e:                       # noqa: BLE001
            return dict(kind='answer', label='lose',
                        parent_solvable=parent_solvable, child=None,
                        child_solvable=False, detail=f'unparseable: {e}')
        return dict(kind='answer', label='win' if ok else 'lose',
                    parent_solvable=parent_solvable, child=None,
                    child_solvable=bool(ok),
                    detail='equals 24' if ok else 'does not equal 24')

    # -- intermediate move:  a op b = c ----------------------------------- #
    m = _INTERMEDIATE_RE.match(a.split('=')[0].strip()) if '=' in a else None
    if m is None:
        return dict(kind='intermediate', label='illegal',
                    parent_solvable=parent_solvable, child=None,
                    child_solvable=False, detail='cannot parse "a op b = c"')

    op1, sym, op2 = F(m.group(1)), m.group(2), F(m.group(3))
    rhs = a.split('=', 1)[1].strip()
    try:
        claimed = F(re.match(rf'\(*\s*({_NUM})', rhs).group(1))
    except Exception:                                # noqa: BLE001
        return dict(kind='intermediate', label='illegal',
                    parent_solvable=parent_solvable, child=None,
                    child_solvable=False, detail='cannot parse result')

    # operands must actually be present in the current state
    after = _try_remove(state, op1)
    after = _try_remove(after, op2) if after is not None else None
    if after is None:
        return dict(kind='intermediate', label='illegal',
                    parent_solvable=parent_solvable, child=None,
                    child_solvable=False,
                    detail=f'operands {fmt(op1)},{fmt(op2)} not both in state')

    true = {'+': op1 + op2, '-': op1 - op2, '*': op1 * op2,
            '/': (op1 / op2 if op2 != 0 else None)}[sym]
    if true is None or true != claimed:
        return dict(kind='intermediate', label='wrong_arithmetic',
                    parent_solvable=parent_solvable, child=None,
                    child_solvable=False,
                    detail=f'{fmt(op1)} {sym} {fmt(op2)} != {fmt(claimed)}')

    child = tuple(sorted(after + [true]))
    child_ok = solvable(child)
    return dict(kind='intermediate',
                label='correct' if child_ok else 'dead_end',
                parent_solvable=parent_solvable,
                child=[fmt(x) for x in child], child_solvable=child_ok,
                detail='keeps 24 reachable' if child_ok else 'kills 24')


# --------------------------------------------------------------------------- #
# Trajectory-level scoring
# --------------------------------------------------------------------------- #
class TrajectoryScorer:
    """
    Feed it one action at a time; it maintains the true rational state and
    labels each step. Advances the state only on legal, arithmetically-valid
    intermediate moves (think/illegal/wrong leave the state unchanged).
    """
    def __init__(self, puzzle):
        nums = re.findall(r'\d+', puzzle) if isinstance(puzzle, str) else puzzle
        self.original = [str(n) for n in nums]   # for validating `answer:`
        self.state = list(to_state(nums))
        self.labels = []

    def step(self, action: str) -> dict:
        lab = label_action(self.state, action, reference=self.original)
        self.labels.append(lab)
        if lab['kind'] == 'intermediate' and lab['label'] in ('correct', 'dead_end'):
            self.state = list(to_state(lab['child']))
        return lab

    def summary(self) -> dict:
        moves = [l for l in self.labels if l['kind'] == 'intermediate']
        decided = [l for l in moves if l['parent_solvable']]  # had a right answer
        correct = [l for l in decided if l['label'] == 'correct']
        win = any(l['kind'] == 'answer' and l['label'] == 'win' for l in self.labels)
        return dict(
            total_actions=len(self.labels),
            moves=len(moves),
            illegal=sum(l['label'] == 'illegal' for l in moves),
            wrong_arithmetic=sum(l['label'] == 'wrong_arithmetic' for l in moves),
            dead_end=sum(l['label'] == 'dead_end' for l in moves),
            correct_moves=len(correct),
            decidable_moves=len(decided),
            # of moves made from a still-solvable state, how many kept it solvable
            solvability_preserved_rate=(len(correct) / len(decided)) if decided else None,
            solved=win,
        )


def score_trajectory(puzzle, actions):
    """Convenience: label a full list of actions, return (labels, summary)."""
    sc = TrajectoryScorer(puzzle)
    labels = [sc.step(a) for a in actions]
    return labels, sc.summary()


# --------------------------------------------------------------------------- #
# Demo / self-test
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    print("# correct first moves per state")
    for p in ["1 2 3 4", "3 3 8 8", "4 4 6 8", "1 1 1 1"]:
        acts = correct_actions(p)
        print(f"  {p:10s} solvable={solvable(to_state(p))} "
              f"#correct={len(acts)}")

    print("\n# labeling a mixed trajectory on '1 1 4 6' (a solvable puzzle)")
    traj = [
        "think: I'll try to build 24",
        "6 * 4 = 24",       # correct (keeps 24: state -> 1 1 24)
        "1 + 1 = 2",        # dead_end (24 -> ... can't recover with 1 1 24? check)
        "1 + 1 = 3",        # wrong_arithmetic
        "9 * 9 = 81",       # illegal (no 9s)
        "answer: 6 * 4 * 1 * 1 = 24",
    ]
    labels, summ = score_trajectory("1 1 4 6", traj)
    for a, l in zip(traj, labels):
        print(f"  [{l['label']:16s}] {a}   -> {l['detail']}")
    print("  summary:", summ)
