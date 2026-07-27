#!/usr/bin/env python3
"""DOES THE BRAIN DESIGN ITS OWN TENDRIL? — model-structure discovery, tested against ground truth.

Four worlds, four different TRUE model shapes. The brain is told none of them; it must discover the
right shape from interaction data:

  1. SIMPLE   effect is a constant vector      -> should pick the plain form (and NOT over-complicate).
  2. EGOCENTRIC effect rotates with heading    -> should DISCOVER it must condition on the heading
                                                  feature. (This is the fix I hard-coded 3x this
                                                  session; here the brain must find it itself.)
  3. WALLED   constant, but blocked at walls   -> should COMPOSE gating on top.
  4. RANDOM   no predictable structure         -> should HONESTLY report no form predicts.

    python3 suit/experiment_discover.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from suit.discover import Discoverer, Const, Cond   # noqa: E402

NACT = 4
_MOVES = [(1, 0), (0, 1), (-1, 0), (0, -1)]          # the true per-action body-frame steps


def _rot(dx, dy, yaw):
    for _ in range(yaw % 4):
        dx, dy = -dy, dx
    return dx, dy


def gen(kind, n=4000, seed=0, W=7, H=7):
    """Generate (o, a, o') transitions for a world of the given TRUE shape. o = (x, y, yaw)."""
    rng = np.random.default_rng(seed)
    walls = {(int(rng.integers(W)), int(rng.integers(H))) for _ in range((W * H) // 6)}
    O, A, O2 = [], [], []
    x, y, yaw = 3, 3, 0
    for _ in range(n):
        a = int(rng.integers(NACT))
        nx, ny, nyaw = x, y, yaw
        if kind == "simple":
            dx, dy = _MOVES[a]                        # fixed world-frame step, no heading dependence
            nx, ny = x + dx, y + dy
        elif kind == "egocentric":
            dx, dy = _rot(*_MOVES[a], yaw)            # step rotates with heading
            if a in (0, 1):
                nx, ny = x + dx, y + dy
            else:
                nyaw = (yaw + (1 if a == 2 else -1)) % 4   # two actions turn instead of move
        elif kind == "walled":
            dx, dy = _MOVES[a]
            tx, ty = x + dx, y + dy
            if (tx, ty) not in walls and 0 <= tx < W and 0 <= ty < H:
                nx, ny = tx, ty                       # blocked by walls
        elif kind == "random":
            nx, ny = int(rng.integers(W)), int(rng.integers(H))
        O.append([x, y, yaw]); A.append(a); O2.append([nx, ny, nyaw])
        x, y, yaw = int(nx) % W, int(ny) % H, int(nyaw)
    return np.array(O, float), np.array(A), np.array(O2, float)


def run(kind, expect):
    O, A, O2 = gen(kind, seed=1)
    form, rep = Discoverer().discover(O, A, O2, NACT, seed=1)
    print(f"\n{'='*74}\n{kind.upper()}  (true shape: {expect})\n{'='*74}")
    print("  candidates it tried (form, held-out error, complexity):")
    for nm, e, c in rep["candidates"]:
        mark = " <- chosen" if nm == rep["chosen"] else ""
        print(f"     {nm:26s} err {e:6.3f}  cx {c:4d}{mark}")
    print(f"\n  DESIGNED: {rep['chosen']}")
    print(f"  meaning : {rep['why']}")
    print(f"  predicts held-out: {rep['predicts']} (err {rep['chosen_error']})   "
          f"invented new structure: {rep['invented_structure']}")
    return rep


r1 = run("simple", "constant effect")
r2 = run("egocentric", "effect depends on heading")
r3 = run("walled", "constant + blocked by walls")
r4 = run("random", "no predictable structure")

print(f"\n{'='*74}\nVERDICT\n{'='*74}")
ok1 = r1["chosen"] == "constant"
ok2 = r2["chosen"].startswith("depends-on-feature[2]")     # feature 2 = yaw/heading, DISCOVERED
ok3 = "gated" in r3["chosen"]
ok4 = not r4["predicts"]
print(f"  simple   -> plain form, not over-built     : {'PASS' if ok1 else 'FAIL'} ({r1['chosen']})")
print(f"  egocentric -> DISCOVERED heading-conditioning: {'PASS' if ok2 else 'FAIL'} ({r2['chosen']})")
print(f"  walled   -> COMPOSED gating                 : {'PASS' if ok3 else 'FAIL'} ({r3['chosen']})")
print(f"  random   -> HONESTLY admits no model        : {'PASS' if ok4 else 'FAIL'} "
      f"(predicts={r4['predicts']})")
print(f"\n  the brain designed the right tendril shape from data in {sum([ok1,ok2,ok3,ok4])}/4 worlds "
      f"-- including the heading-conditioning that was hand-coded before.")
