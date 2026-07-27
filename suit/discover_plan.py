#!/usr/bin/env python3
"""LEARN the buttons, then PLAN a sequence of presses to reach a goal / go around a thing.

discover.py works out WHAT each button does. This searches that learned model for a SEQUENCE of
presses that gets from here to a goal without crossing an obstacle -- the plan. So the brain both
figures out its remote AND works out which buttons to press, in what order, to do a job.

    python3 suit/discover_plan.py
"""
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from suit.discover import Discoverer   # noqa: E402

# the robot's THREE buttons (the brain is NOT told these meanings — it learns them):
FWD, LEFT, RIGHT = 0, 1, 2
NAME = {FWD: "forward", LEFT: "turn-left", RIGHT: "turn-right"}
_DIR = [(1, 0), (0, 1), (-1, 0), (0, -1)]           # facing 0=+x, 1=+y, 2=-x, 3=-y


def _truth(x, y, yaw, a):
    if a == FWD:
        dx, dy = _DIR[yaw]; return x + dx, y + dy, yaw       # forward is EGOCENTRIC (rotates with facing)
    return x, y, (yaw + 1) % 4 if a == LEFT else (yaw - 1) % 4


def babble(n=3000, seed=0):
    """Press buttons at random, write down (before, button, after) — the notebook."""
    rng = np.random.default_rng(seed)
    O, A, O2 = [], [], []
    x, y, yaw = 3, 3, 0
    for _ in range(n):
        a = int(rng.integers(3))
        O.append([x, y, yaw])
        nx, ny, nyaw = _truth(x, y, yaw, a)
        A.append(a); O2.append([nx, ny, nyaw])
        x, y, yaw = int(nx) % 7, int(ny) % 7, int(nyaw)
    return np.array(O, float), np.array(A), np.array(O2, float)


def plan(form, start, goal, obstacles, bounds, max_len=60):
    """Search the LEARNED model for a button sequence from start pose to goal, around obstacles."""
    W, H = bounds
    s = (int(start[0]), int(start[1]), int(start[2]))
    goal = (int(goal[0]), int(goal[1]))
    seen, q = {s}, deque([(s, [])])
    while q:
        (x, y, yaw), path = q.popleft()
        if (x, y) == goal:
            return path
        if len(path) >= max_len:
            continue
        for a in range(3):
            o2 = form.predict(np.array([x, y, yaw], float), a)   # ask the learned model what happens
            nx, ny, nyaw = int(round(o2[0])), int(round(o2[1])), int(round(o2[2])) % 4
            if (nx, ny) != (x, y):                               # a move (not a turn): check the map
                if not (0 <= nx < W and 0 <= ny < H) or (nx, ny) in obstacles:
                    continue
            st = (nx, ny, nyaw)
            if st in seen:
                continue
            seen.add(st); q.append((st, path + [a]))
    return None


def _render(bounds, start, goal, obstacles, path):
    """Draw the plan as a little map so you can see it go around the object."""
    W, H = bounds
    x, y, yaw = start
    visited = {(x, y)}
    for a in path:
        x, y, yaw = _truth(x, y, yaw, a)
        visited.add((x, y))
    rows = []
    for j in range(H - 1, -1, -1):
        row = []
        for i in range(W):
            if (i, j) == (start[0], start[1]):
                row.append("S")
            elif (i, j) == tuple(goal):
                row.append("G")
            elif (i, j) in obstacles:
                row.append("#")
            elif (i, j) in visited:
                row.append("·")
            else:
                row.append(" ")
        rows.append(" ".join(row))
    return "\n".join("   " + r for r in rows)


if __name__ == "__main__":
    # 1) learn the buttons from the notebook
    O, A, O2 = babble()
    form, rep = Discoverer().discover(O, A, O2, 3)
    print("STEP 1 — learn the remote")
    print(f"  the brain worked out: {rep['why']}  (sure? {rep['predicts']})\n")

    # 2) a job: get from S to G, but an OBJECT (a wall of blocks) is in the way
    bounds = (9, 9)
    start = (1, 4, 0)                 # bottom-left-ish, facing +x (toward the goal)
    goal = (7, 4)
    obstacles = {(4, 2), (4, 3), (4, 4), (4, 5), (4, 6)}   # the object blocking the straight path

    print("STEP 2 — plan the presses to go around the object")
    seq = plan(form, start, goal, obstacles, bounds)
    if seq is None:
        print("  no path found")
    else:
        print("  the plan (press these in order):")
        print("    " + " → ".join(NAME[a] for a in seq))
        print(f"\n  that's {seq.count(FWD)} forwards and {seq.count(LEFT)+seq.count(RIGHT)} turns, "
              f"{len(seq)} presses total.\n")
        print(_render(bounds, start, goal, obstacles, seq))
        # 3) prove it: follow the plan in the real world, does it reach G without hitting the object?
        x, y, yaw = start
        hit = False
        for a in seq:
            x, y, yaw = _truth(x, y, yaw, a)
            if (x, y) in obstacles:
                hit = True
        print(f"\n  followed the plan for real -> ended at ({x},{y}); goal is {goal}; "
              f"hit the object: {hit}   => {'REACHED IT, went around' if (x, y) == goal and not hit else 'FAILED'}")
