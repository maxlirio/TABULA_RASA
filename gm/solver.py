"""A SOLVER the brain can call as a tool — the old TABULA_RASA idea (grow your own understanding
by searching for the rule that explains a puzzle), now a backend the language brain invokes rather
than the brain itself.

Given a sequence it SEARCHES a space of candidate rules (constant, arithmetic, geometric,
quadratic, fibonacci-like, multiply-then-add, two interleaved patterns, recursively) and returns
the next value + the rule it discovered. It also GROWS: every pattern it solves is cached and
reused, so its library of understood patterns expands with use.

  CALL: solve 2 4 8 16    ->  RESULT: 32  (rule: multiply by 2 each step)
"""
import json
import os
import re

_MEM_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "solver_mem.json")
try:
    _CACHE = json.load(open(_MEM_PATH))
except (OSError, ValueError):
    _CACHE = {}


def _save_cache():
    try:
        json.dump(_CACHE, open(_MEM_PATH, "w"))
    except OSError:
        pass


def _num(x):
    return int(x) if abs(x - round(x)) < 1e-9 else round(x, 4)


def _arith(s):
    d = s[1] - s[0]
    if all(abs((s[i + 1] - s[i]) - d) < 1e-9 for i in range(len(s) - 1)):
        return _num(s[-1] + d), f"add {_num(d)} each step"


def _geom(s):
    if any(v == 0 for v in s[:-1]):
        return None
    r = s[1] / s[0]
    if all(abs(s[i + 1] - s[i] * r) < 1e-9 for i in range(len(s) - 1)):
        return _num(s[-1] * r), f"multiply by {_num(r)} each step"


def _quad(s):
    d1 = [s[i + 1] - s[i] for i in range(len(s) - 1)]
    if len(d1) < 2:
        return None
    d2 = d1[1] - d1[0]
    if all(abs((d1[i + 1] - d1[i]) - d2) < 1e-9 for i in range(len(d1) - 1)):
        return _num(s[-1] + d1[-1] + d2), f"the gaps grow by {_num(d2)} each time"


def _fib(s):
    if len(s) < 3:
        return None
    if all(abs(s[i] - (s[i - 1] + s[i - 2])) < 1e-9 for i in range(2, len(s))):
        return _num(s[-1] + s[-2]), "each number is the sum of the previous two"


def _mul_add(s):
    if len(s) < 3:
        return None
    for k in range(-3, 7):
        for c in range(-12, 13):
            if (k, c) in ((1, 0),):
                continue
            if all(abs(s[i + 1] - (s[i] * k + c)) < 1e-9 for i in range(len(s) - 1)):
                return _num(s[-1] * k + c), f"multiply by {k} then add {c}"


def _interleaved(s):
    if len(s) < 4:
        return None
    a, b = _rule(s[0::2]), _rule(s[1::2])
    if a and b:
        nxt = a[0] if len(s) % 2 == 0 else b[0]
        return nxt, f"two patterns taking turns ({a[1]}; {b[1]})"


def _rule(s):
    """Search the rule space in order of simplicity; return (next, description) or None."""
    s = list(s)
    if len(s) < 2:
        return None
    for hyp in (_arith, _geom, _quad, _fib, _mul_add, _interleaved):
        try:
            r = hyp(s)
        except Exception:
            r = None
        if r:
            return r
    return None


def solve(query):
    """Solve a number-sequence puzzle. Returns 'NEXT  (rule: ...)' or a can't-solve note."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(query))
    if len(nums) < 3:
        return "give me at least 3 numbers and i'll find the pattern"
    key = ",".join(nums)
    if key in _CACHE:                                  # already understood -> reuse (grown library)
        return _CACHE[key]
    seq = [float(n) for n in nums]
    r = _rule(seq)
    if not r:
        return "i can't find a pattern in that one"
    out = f"{r[0]}  (rule: {r[1]})"
    _CACHE[key] = out                                  # learn it, so the library grows
    _save_cache()
    return out
