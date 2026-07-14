#!/usr/bin/env python3
"""WATCH THE BRAIN GROW INTO ITS SUIT.

Two grids, side by side: the world as it REALLY is, and the world AS THE BRAIN BELIEVES IT.
The right-hand grid starts blank -- the brain has never been anywhere and knows nothing -- and
fills in as it pokes around. You are watching understanding accumulate.

    python3 suit/watch.py                 # babble -> plan -> execute -> rule-flip
    python3 suit/watch.py --fast          # less dawdling
    python3 suit/watch.py --frames 40     # non-interactive: dump N frames and exit

Legend
    @  the body            #  a wall (truth)          ?  never visited (the brain has no idea)
    *  the goal            X  a wall the brain BUMPED INTO and remembers
    .  open ground         ~  ground the brain has stood on and knows is open
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from suit.gridworld import GridWorld              # noqa: E402
from suit.tendrils import Tendrils, plan          # noqa: E402

DIRNAME = {(-1, 0): "west ", (1, 0): "east ", (0, -1): "north", (0, 1): "south"}
CLEAR, HOME = "\033[2J", "\033[H"


def render(world, tend, note="", goal=None, err=None, step=0, total=0):
    W, H = world.w, world.h
    pos = tuple(map(int, world.pos))
    truth, belief = [], []
    for y in range(H):
        rt, rb = [], []
        for x in range(W):
            c = (x, y)
            # ---- the world as it IS
            if c == pos:
                rt.append("@")
            elif goal and c == goal:
                rt.append("*")
            elif c in world.walls:
                rt.append("#")
            else:
                rt.append(".")
            # ---- the world AS THE BRAIN BELIEVES IT
            if c == pos:
                rb.append("@")
            elif goal and c == goal:
                rb.append("*")
            elif c in tend.blocked:
                rb.append("X")          # it has been stopped here -- it KNOWS
            elif c in tend.free:
                rb.append("~")          # it has stood here -- it KNOWS
            else:
                rb.append("?")          # never been -- total ignorance
        truth.append(" ".join(rt))
        belief.append(" ".join(rb))

    out = [f"{'THE SUIT (truth)':<{W*2+4}}{'WHAT THE BRAIN BELIEVES'}"]
    for a, b in zip(truth, belief):
        out.append(f"  {a}    {b}")
    out.append("")
    out.append("  handle   the brain's belief        times probed")
    for i, h in enumerate(tend.handles):
        e = tend.intent(i)
        if e == (0, 0):
            word = "does nothing" if tend.tries[i] > 8 else "no idea yet"
        else:
            word = f"moves me {DIRNAME.get(e, str(e))}"
        bar = "#" * min(int(tend.tries[i]) // 12, 22)
        out.append(f"  {h:<8} {word:<24} {bar} {int(tend.tries[i])}")
    out.append("")
    cells = {(x, y) for x in range(W) for y in range(H)}
    known = len((tend.free | tend.blocked) & cells)      # a cell is known if stood-in OR bumped-into
    out.append(f"  cells it has figured out: {known}/{W*H}"
               + (f"     prediction error (surprise): {err:.3f}" if err is not None else ""))
    if total:
        out.append(f"  step {step}/{total}")
    out.append("")
    out.append(f"  >> {note}")
    return "\n".join(out)


def show(txt, pause, live=True):
    if live:
        sys.stdout.write(HOME + txt + "\n")
        sys.stdout.flush()
        time.sleep(pause)
    else:
        print(txt + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--frames", type=int, default=0, help="non-interactive: print N frames, no ANSI")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    live = args.frames == 0
    pause = 0.02 if args.fast else 0.06
    shown = [0]

    world = GridWorld(seed=args.seed)
    tend = Tendrils(world.handles)
    rng = np.random.default_rng(args.seed)
    if live:
        sys.stdout.write(CLEAR)

    def frame(note, goal=None, err=None, step=0, total=0, every=6):
        shown[0] += 1
        if not live and shown[0] > args.frames:
            raise SystemExit
        if live and shown[0] % every:
            return
        show(render(world, tend, note, goal, err, step, total), pause, live)

    # ---------------------------------------------------------------- 1. BABBLE
    BAB = 1400
    obs = world.observe()
    for t in range(BAB):
        a = int(np.argmin(tend.tries + rng.random(tend.n) * 3.0)) if rng.random() < 0.7 \
            else int(rng.integers(tend.n))
        obs2 = world.step(a)
        err = tend.observe(obs, a, obs2)
        obs = obs2
        frame(f"BABBLING — poking {world.handles[a]} to see what it does "
              f"(it was told the NAME and nothing else)", err=tend.recent_error(60),
              step=t + 1, total=BAB)
        if t % 400 == 399:
            obs = world.reset()
    show(render(world, tend, "BABBLING DONE — it has never been told a single thing about this body.",
                err=tend.recent_error(60)), 0.8 if live else 0, live)

    # ---------------------------------------------------------------- 2. PLAN + EXECUTE
    start = tuple(map(int, world.pos))
    free = sorted(tend.free - {start})
    goal = free[len(free) // 2] if free else start
    p = plan(tend, start, goal)
    if not p:
        show(render(world, tend, "no plan found", goal=goal), 1.0, live)
        return
    names = " -> ".join(world.handles[a] for a in p)
    show(render(world, tend, f"PLANNED a route to * using ONLY what it discovered:\n     {names}",
                goal=goal), 1.4 if live else 0, live)
    for i, a in enumerate(p):
        world.step(a)
        frame(f"EXECUTING — {world.handles[a]} ({DIRNAME.get(tend.intent(a), '?')})   step {i+1}/{len(p)}",
              goal=goal, every=1)
        if live:
            time.sleep(0.18)
    hit = tuple(map(int, world.pos)) == goal
    show(render(world, tend,
                "REACHED THE GOAL. It was never taught this route — it searched the map it built itself."
                if hit else "missed — it routed through a wall it has never bumped into",
                goal=goal), 1.6 if live else 0, live)

    # ---------------------------------------------------------------- 3. THE RULE FLIP
    show(render(world, tend, "NOW I REWIRE ITS BODY. Every handle will mean something else.\n"
                             "     It is not told. Watch the surprise."), 1.8 if live else 0, live)
    world._wiring = list(world._orig_wiring)
    rng2 = np.random.default_rng(args.seed + 77)
    rng2.shuffle(world._wiring)
    obs = world.observe()
    FLIP = 2600                       # long enough for belief-decay to actually finish re-growing
    for t in range(FLIP):
        a = int(rng2.integers(tend.n))
        obs2 = world.step(a)
        tend.observe(obs, a, obs2)
        obs = obs2
        e = tend.recent_error(40)
        note = ("SURPRISE! its body no longer does what it believed — error SPIKING"
                if e > 0.25 else "re-grown: it has learned the new body and is calm again")
        frame(note, err=e, step=t + 1, total=FLIP, every=10)
        if t % 300 == 299:
            obs = world.reset()
    show(render(world, tend, "IT RE-GREW. Same brain, no code changed, a body that lied to it.",
                err=tend.recent_error(40)), 0, live)


if __name__ == "__main__":
    main()
