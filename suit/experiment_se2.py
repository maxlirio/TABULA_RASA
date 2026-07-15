#!/usr/bin/env python3
"""THE ROBOT TEST — does the tendril method survive an EGOCENTRIC body on a LiDAR map?

  0. DOES THE GRIDWORLD ASSUMPTION SHATTER?  a constant-effect model averages 'forward' across
     headings to ~zero and calls the legs a dud. It must fail here, or the suit proves nothing.
  1. DOES IT LEARN THE BODY?    per-heading prediction error must drop on held-out transitions.
  2. DID IT UNDERSTAND ITSELF?  untold, it must name each command walk/turn/dud, recover the
     body-frame step of each walk, and match the two synonyms.
  3. DOES IT NAVIGATE?          reach novel goal cells over the LiDAR map, from random start poses.
  4. PORTABLE?                  rewire the commands AND remap the room; it must relearn, no code change.

    python3 suit/experiment_se2.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from suit.world_se2 import RobotSuit, _rot                 # noqa: E402
from suit.tendrils_se2 import RobotTendrils, babble_robot  # noqa: E402


def rule(m):
    print("\n" + "=" * 80 + f"\n{m}\n" + "=" * 80)


def heldout_error(world, tend, n=400, seed=7):
    rng = np.random.default_rng(seed)
    errs, obs = [], world.reset()
    for _ in range(n):
        a = int(rng.integers(tend.n))
        pred = tend.predict(obs, a)
        obs = world.step(a)
        errs.append(float(np.abs(obs - pred).sum()))
    return float(np.mean(errs))


def grow(seed, steps=8000):
    w = RobotSuit(seed=seed)
    t = RobotTendrils(w.handles)
    babble_robot(w, t, steps=steps, rng=np.random.default_rng(seed))
    return w, t


# ============================================================ 0. the gridworld assumption shatters
rule("0. DOES THE CONSTANT-EFFECT (GRIDWORLD) MODEL SHATTER ON AN EGOCENTRIC BODY?")
world, tend = grow(seed=1)
# a naive model = one mean delta per command, ignoring heading
naive = np.zeros((tend.n, 3))
for a in range(tend.n):
    ds = [(*dp, dy) for y in range(tend.n_yaw) for (dp, dy) in tend.samples[a][y]]
    if ds:
        naive[a] = np.mean(ds, axis=0)
truth = world._truth()
walkers = [a for a in range(tend.n) if truth[a][0] == "move"]
print("  a constant model's idea of each WALK command (averaging over headings):")
for a in walkers:
    print(f"    {world.handles[a]}: mean world-delta {tuple(round(v, 2) for v in naive[a])}"
          f"   |move| = {abs(naive[a][0]) + abs(naive[a][1]):.2f}")
collapsed = sum(abs(naive[a][0]) + abs(naive[a][1]) < 0.5 for a in walkers)
print(f"\n  => {collapsed}/{len(walkers)} of the robot's WALK commands average to ~0 and look like duds.")
print("     The constant-effect model that ACED the gridworld cannot see a body that turns.")


# ============================================================ 1-3. the egocentric brain
rule("1-3. THE EGOCENTRIC BRAIN  (conditions each command's effect on heading)")
before = heldout_error(RobotSuit(seed=1), RobotTendrils(RobotSuit(seed=1).handles))
after = heldout_error(world, tend)
print(f"  per-heading prediction error  BEFORE: {before:5.2f}   AFTER: {after:5.2f}")

names = {"move": "walk", "turn": "turn", "dud": "dud"}
cls = tend.classify()
ok = 0
print("\n  what it worked out about each command (it was told only the NAMES):")
for a in range(tend.n):
    kind, body = cls[world.handles[a]]
    tk, tv = truth[a]
    # ground-truth body step for a walk, for scoring only
    tbody = (tv[0], tv[1]) if tk == "move" else None
    good = (kind.startswith("walk") and tk == "move" and body == tbody) \
        or (kind.startswith("turn") and tk == "turn") \
        or (kind == "dud" and tk == "dud")
    ok += good
    extra = f" body-step {body}" if body else ""
    print(f"    {'ok' if good else 'XX'} {world.handles[a]}: brain says '{kind}'{extra}"
          f"   (truly {names[tk]}{' ' + str(tbody) if tbody else ''})")
print(f"\n  commands understood: {ok}/{tend.n}")

# navigate to novel goals over the LiDAR map
nav = world.nav_grid()
rng = np.random.default_rng(5)
tried = solved = 0
for _ in range(200):
    sx, sy = int(rng.integers(world.w)), int(rng.integers(world.h))
    gx, gy = int(rng.integers(world.w)), int(rng.integers(world.h))
    if not nav[sx, sy] or not nav[gx, gy] or (sx, sy) == (gx, gy):
        continue
    tried += 1
    world.reset(pose=(sx, sy, int(rng.integers(4))))
    p = tend.plan(world.observe(), (gx, gy), nav)
    if not p:
        continue
    for a in p:
        world.step(a)
    solved += tuple(int(v) for v in world.pose[:2]) == (gx, gy)
print(f"  novel goals reached over the mapped room: {solved}/{tried}  ({solved / max(tried, 1):.0%})")

T1 = after < 0.1
T2 = ok == tend.n
T3 = solved / max(tried, 1) >= 0.9


# ============================================================ 4. portable across body + room
rule("4. PORTABLE?  (rewire the commands AND remap the room; same brain code)")
print(f"  {'suit':>6}{'err after':>12}{'understood':>13}{'goals':>16}")
allok = True
for s in (1, 2, 3, 4, 5):
    w2, t2 = grow(seed=s)
    aft = heldout_error(w2, t2)
    tr2 = w2._truth()
    c2 = t2.classify()
    good = 0
    for a in range(t2.n):
        kind, body = c2[w2.handles[a]]
        tk, tv = tr2[a]
        tbody = (tv[0], tv[1]) if tk == "move" else None
        good += (kind.startswith("walk") and tk == "move" and body == tbody) \
            or (kind.startswith("turn") and tk == "turn") or (kind == "dud" and tk == "dud")
    nav2 = w2.nav_grid()
    rng = np.random.default_rng(90 + s)
    tr = sv = 0
    for _ in range(120):
        sx, sy = int(rng.integers(w2.w)), int(rng.integers(w2.h))
        gx, gy = int(rng.integers(w2.w)), int(rng.integers(w2.h))
        if not nav2[sx, sy] or not nav2[gx, gy] or (sx, sy) == (gx, gy):
            continue
        tr += 1
        w2.reset(pose=(sx, sy, int(rng.integers(4))))
        p = t2.plan(w2.observe(), (gx, gy), nav2)
        if p:
            for a in p:
                w2.step(a)
            sv += tuple(int(v) for v in w2.pose[:2]) == (gx, gy)
    allok = allok and good == t2.n and sv / max(tr, 1) >= 0.9
    print(f"  {s:>6}{aft:>12.3f}{f'{good}/{t2.n}':>13}{f'{sv}/{tr} ({sv/max(tr,1):.0%})':>16}")

rule(f"RESULT   shatters-naive={'PASS' if collapsed else 'FAIL'}   learn={'PASS' if T1 else 'FAIL'}   "
     f"understood-self={'PASS' if T2 else 'FAIL'}   navigate={'PASS' if T3 else 'FAIL'}   "
     f"portable={'PASS' if allok else 'FAIL'}")
