#!/usr/bin/env python3
"""THE FEASIBILITY EXPERIMENT — the three tests from BRAIN_DESIGN.md §6, run for real.

    python3 suit/experiment.py

  1. DOES IT LEARN THE SUIT?    prediction error must DROP, on HELD-OUT transitions.
  2. DOES IT ARRANGE OR MEMORISE?  it must reach goals it NEVER practised, needing action
     sequences never taken during babbling. This project has been fooled by a memorised lookup
     once already (reward design); assume it will be again unless the test forbids it.
  3. IS IT PORTABLE?  swap the suit -- re-randomise which handle does what -- and it must relearn
     with NO code change and NO knowledge carried over. This is the actual thesis.
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from suit.gridworld import GridWorld                    # noqa: E402
from suit.tendrils import Tendrils, babble, plan, execute  # noqa: E402


def rule(msg):
    print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78)


def grow(seed, steps=4000, flip_at=None, quiet=False):
    w = GridWorld(seed=seed, flip_at=flip_at)
    t = Tendrils(w.handles)
    early = []
    obs = w.observe()                                    # HELD-OUT error: measure BEFORE learning
    rng = np.random.default_rng(seed + 99)
    for _ in range(200):
        a = int(rng.integers(t.n))
        o2 = w.step(a)
        early.append(float(np.abs(o2 - t.predict(obs, a)).sum()))
        obs = o2
    w.reset()
    babble(w, t, steps=steps, rng=np.random.default_rng(seed))
    return w, t, float(np.mean(early))


def heldout_error(world, tend, n=300, seed=7):
    """Error on FRESH transitions the brain never trained on. Training error can be flattered by
    the curiosity policy revisiting what it already knows; this cannot."""
    rng = np.random.default_rng(seed)
    errs = []
    obs = world.reset()
    for _ in range(n):
        a = int(rng.integers(tend.n))
        pred = tend.predict(obs, a)
        obs2 = world.step(a)
        errs.append(float(np.abs(obs2 - pred).sum()))
        obs = obs2
    return float(np.mean(errs))


def score_map(world, tend):
    """How much of the suit did it actually figure out? Compared to ground truth -- TEST ONLY."""
    truth, got, ok = world._truth(), tend.learned_map(), 0
    for i, h in enumerate(tend.handles):
        want = truth[i]
        want = None if want is None else (want[0], want[1])
        if got[h] == want:
            ok += 1
    return ok, len(tend.handles)


# ============================================================ TEST 1 — does it learn the suit?
rule("TEST 1 — DOES IT LEARN THE SUIT?  (prediction error must drop, on held-out transitions)")
world, tend, err_before = grow(seed=1)
err_after = heldout_error(world, tend)
ok, n = score_map(world, tend)
print(f"  held-out prediction error BEFORE babbling : {err_before:.3f}")
print(f"  held-out prediction error AFTER  babbling : {err_after:.3f}")
print(f"  handles correctly understood              : {ok}/{n}")
print("\n  what the brain thinks its body is (it was told ONLY the names):")
truth = world._truth()
for i, h in enumerate(tend.handles):
    g = tend.learned_map()[h]
    t_ = truth[i]
    t_ = None if t_ is None else (t_[0], t_[1])
    mark = "ok " if g == t_ else "XX "
    print(f"    {mark}{h}: discovered {str(g):>9}   (truly {str(t_):>9}, probed {tend.tries[i]:>4}x)")
T1 = err_after < err_before * 0.5 and ok >= n - 1
print(f"\n  => {'PASS' if T1 else 'FAIL'}: it {'grew tendrils' if T1 else 'did NOT learn the suit'}")


# ==================================================== TEST 2 — does it arrange, or memorise?
rule("TEST 2 — DOES IT ARRANGE, OR MEMORISE?  (reach goals it NEVER practised)")
rng = np.random.default_rng(5)
tried = solved = 0
fails = []
for _ in range(200):
    start = (int(rng.integers(world.w)), int(rng.integers(world.h)))
    goal = (int(rng.integers(world.w)), int(rng.integers(world.h)))
    if start in world.walls or goal in world.walls or start == goal:
        continue
    tried += 1
    world.reset(pos=start)
    # NO ground-truth walls, NO ground-truth bounds. The brain plans with the body it BELIEVES it
    # has. (The first version of this passed `walls=world.walls` -- handing it the answer.)
    p = plan(tend, start, goal)
    if p is None:
        fails.append((start, goal, "no plan"))
        continue
    if execute(world, p, goal):
        solved += 1
    else:
        fails.append((start, goal, "plan wrong"))
print(f"  novel start->goal pairs attempted : {tried}")
print(f"  reached by planning over the LEARNED model : {solved}  ({solved/max(tried,1):.0%})")
if fails[:3]:
    print(f"  example failures: {fails[:3]}")
print("  (goals are random; the babble never practised these routes -- a lookup table cannot do this)")
T2 = solved / max(tried, 1) >= 0.90
print(f"\n  => {'PASS' if T2 else 'FAIL'}: it {'ARRANGES' if T2 else 'does NOT arrange'}")


# ============================================================== TEST 3 — is it portable?
rule("TEST 3 — IS IT PORTABLE?  (swap the suit: rewire every handle, change nothing else)")
print("  same brain code, same babble budget, a suit whose handles mean something ELSE.\n")
rows = []
for s in (1, 2, 3, 4, 5):
    w2, t2, before = grow(seed=s)
    after = heldout_error(w2, t2)
    o, nn = score_map(w2, t2)
    rows.append((s, before, after, o, nn))
    print(f"    suit seed {s}: held-out error {before:5.2f} -> {after:5.2f}   handles understood {o}/{nn}")
worst = min(r[3] / r[4] for r in rows)
T3 = worst >= 0.875
print(f"\n  => {'PASS' if T3 else 'FAIL'}: the brain {'grew into every suit' if T3 else 'is not portable'}"
      f" (worst {worst:.0%} of handles understood)")


# ============================================ BONUS — the rule flip: does it NOTICE it is wrong?
rule("BONUS — THE RULE FLIP  (the suit rewires itself mid-life; does the brain notice and adapt?)")
w3 = GridWorld(seed=11, flip_at=2500)
t3 = Tendrils(w3.handles)
babble(w3, t3, steps=2500, rng=np.random.default_rng(11))
pre = t3.recent_error(200)
babble(w3, t3, steps=200, rng=np.random.default_rng(12))       # first 200 steps AFTER the flip
spike = t3.recent_error(200)
babble(w3, t3, steps=4000, rng=np.random.default_rng(13))      # keep living
post = t3.recent_error(200)
o, nn = score_map(w3, t3)
print(f"  prediction error before the flip      : {pre:.3f}")
print(f"  ... immediately after the rewire      : {spike:.3f}   <- SURPRISE (the world lied)")
print(f"  ... after living with the new body    : {post:.3f}")
print(f"  handles re-understood in the new wiring: {o}/{nn}")
T4 = spike > pre * 1.5 and post < spike * 0.6
print(f"\n  => {'PASS' if T4 else 'FAIL'}: it {'noticed and re-grew' if T4 else 'did NOT adapt'}")

rule(f"RESULT   learn={'PASS' if T1 else 'FAIL'}   arrange={'PASS' if T2 else 'FAIL'}   "
     f"portable={'PASS' if T3 else 'FAIL'}   adapt={'PASS' if T4 else 'FAIL'}")
