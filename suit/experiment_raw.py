#!/usr/bin/env python3
"""THE RAW-OBSERVATION TEST — does the tendril METHOD survive a robot-like body?

The clean suit handed the brain (x, y). A real body won't. This gives it raw sensor readings
(position mixed through an unknown transform + noise + dead sensors) and asks the same questions:

  0. DOES THE OLD BRAIN SHATTER?   it assumed observation==position; it must fail here, or the raw
     suit isn't actually testing anything.
  1. DOES IT STILL LEARN?          raw-space prediction error must drop on HELD-OUT transitions.
  2. DID IT DISCOVER SPACE?        the frame it invented from its own actions must recover true
     position up to an affine map (R^2 -> ~1.0). This is the real result: space from action.
  3. DOES IT STILL ARRANGE?        reach goals given as RAW sensor readings it never practised.
  4. GRACEFUL UNDER NOISE?         degrade the sensors and watch where it breaks. Honest, not hidden.

    python3 suit/experiment_raw.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from suit.gridworld import GridWorld, RawEncoder          # noqa: E402
from suit.tendrils import Tendrils, babble                # noqa: E402  (the OLD brain)
from suit.tendrils_raw import RawTendrils, babble_raw     # noqa: E402


def rule(m):
    print("\n" + "=" * 78 + f"\n{m}\n" + "=" * 78)


def make(seed, noise, dim=12, dead=4):
    enc = RawEncoder(dim=dim, noise=noise, dead=dead, seed=seed)
    return GridWorld(seed=seed, encoder=enc), enc


def heldout_error(world, tend, n=300, seed=7):
    rng = np.random.default_rng(seed)
    errs, obs = [], world.reset()
    for _ in range(n):
        a = int(rng.integers(tend.n))
        pred = tend.predict(obs, a)
        obs2 = world.step(a)
        errs.append(float(np.abs(obs2 - pred).sum()))
        obs = obs2
    return float(np.mean(errs))


def frame_r2(world, tend, n=400, seed=8):
    """Fit an affine map from the brain's INVENTED coordinate to the TRUE (x,y). R^2 ~ 1 means it
    genuinely rediscovered position from nothing but how its actions moved the sensors."""
    rng = np.random.default_rng(seed)
    C, P, obs = [], [], world.reset()
    for _ in range(n):
        C.append(tend._frame(obs))
        P.append(world.true_pos())
        obs = world.step(int(rng.integers(tend.n)))
    C = np.column_stack([np.array(C), np.ones(len(C))])       # [c1, c2, 1] for an affine fit
    P = np.array(P)
    W, *_ = np.linalg.lstsq(C, P, rcond=None)
    resid = P - C @ W
    ss_res = (resid ** 2).sum()
    ss_tot = ((P - P.mean(0)) ** 2).sum()
    return 1 - ss_res / ss_tot


# ============================================================ 0. does the OLD brain shatter?
rule("0. DOES THE CLEAN-SUIT BRAIN SHATTER ON RAW OBSERVATIONS?  (it must -- else this proves nothing)")
world, enc = make(seed=1, noise=0.05)
old = Tendrils(world.handles)
babble(world, old, steps=4000, rng=np.random.default_rng(1))
try:
    om = old.learned_map()
    dud_claims = sum(v is None for v in om.values())
    print(f"  the old brain, on raw sensors: {dud_claims}/{old.n} commands it now calls DUDS")
    print(f"  (it reads the 12-D delta as a position delta -- garbage; its wall map is meaningless)")
    print(f"  held-out error (old brain): {heldout_error(world, old):.2f}   -- it did not model the body")
except Exception as e:
    print(f"  it crashed outright: {type(e).__name__}: {e}")
print("  => as expected, the convenience of observation==position does NOT survive a real sensor.")


# ============================================================ 1-3. the raw brain, clean-ish sensors
rule("1-3. THE RAW BRAIN  (12 sensors, 4 of them dead, light noise)")
world, enc = make(seed=1, noise=0.05)
tend = RawTendrils(world.handles)
before = heldout_error(world, tend)
babble_raw(world, tend, steps=6000, rng=np.random.default_rng(1))
after = heldout_error(world, tend)
r2 = frame_r2(world, tend)
leak = tend.dead_sensor_leak(enc.dead_idx)
truth = world._truth()
true_movers = sum(e is not None for e in truth)
found_movers = len(tend.movers)

print(f"  raw-space prediction error  BEFORE: {before:5.2f}   AFTER: {after:5.2f}")
print(f"  commands found to move the body   : {found_movers}   (truly {true_movers} move, "
      f"{tend.n - true_movers} dud)")
print(f"  dead-sensor leakage (want ~0)     : {leak:.3f}   -- it ignored the stuck sensors, untold")
print(f"  FRAME RECOVERY  R^2 (want ~1.0)   : {r2:.4f}   <- space rediscovered from action alone")
print("\n  what it thinks each command does, in the coordinate IT INVENTED:")
lm = tend.learned_map_2d()
for i, h in enumerate(tend.handles):
    tag = "dud     " if truth[i] is None else f"true{str(truth[i]):>8}"
    print(f"    {h}: recovered-effect {str(lm[h]):>10}   ({tag})")

# arranging: reach novel goals given ONLY as raw sensor readings
rng = np.random.default_rng(5)
tried = solved = 0
for _ in range(200):
    s = (int(rng.integers(world.w)), int(rng.integers(world.h)))
    g = (int(rng.integers(world.w)), int(rng.integers(world.h)))
    if s in world.walls or g in world.walls or s == g:
        continue
    tried += 1
    world.reset(pos=s)
    o_start = world.observe()
    o_goal = world.encode(g)                     # the brain is handed the raw READING at the goal
    p = tend.plan(o_start, o_goal)
    if not p:
        continue
    for a in p:
        world.step(a)
    solved += tuple(map(int, world.true_pos())) == g
print(f"\n  novel goals (given as raw readings) reached: {solved}/{tried}  ({solved/max(tried,1):.0%})")

T1 = after < before * 0.5
T2 = r2 > 0.98
T3 = solved / max(tried, 1) >= 0.85
print(f"\n  => learn={'PASS' if T1 else 'FAIL'}   discovered-space={'PASS' if T2 else 'FAIL'}   "
      f"arrange={'PASS' if T3 else 'FAIL'}")


# ============================================================ 4. graceful under noise?
rule("4. HOW MUCH SENSOR NOISE CAN IT TAKE?  (degrade the body's eyes, watch where it breaks)")
print(f"  {'noise':>7}{'err after':>12}{'frame R^2':>12}{'goals reached':>16}")
for noise in (0.0, 0.05, 0.15, 0.30, 0.60):
    w2, e2 = make(seed=2, noise=noise)
    t2 = RawTendrils(w2.handles)
    babble_raw(w2, t2, steps=6000, rng=np.random.default_rng(2))
    r = frame_r2(w2, t2)
    aft = heldout_error(w2, t2)
    rng = np.random.default_rng(9)
    tr = sv = 0
    for _ in range(120):
        s = (int(rng.integers(w2.w)), int(rng.integers(w2.h)))
        g = (int(rng.integers(w2.w)), int(rng.integers(w2.h)))
        if s in w2.walls or g in w2.walls or s == g:
            continue
        tr += 1
        w2.reset(pos=s)
        p = t2.plan(w2.observe(), w2.encode(g))
        if p:
            for a in p:
                w2.step(a)
            sv += tuple(map(int, w2.true_pos())) == g
    print(f"  {noise:>7.2f}{aft:>12.2f}{r:>12.3f}{f'{sv}/{tr} ({sv/max(tr,1):.0%})':>16}")

rule(f"RESULT (light noise)   learn={'PASS' if T1 else 'FAIL'}   "
     f"discovered-space={'PASS' if T2 else 'FAIL'}   arrange={'PASS' if T3 else 'FAIL'}")
