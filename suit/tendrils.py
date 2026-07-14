#!/usr/bin/env python3
"""TENDRILS — the brain growing into the suit.

It is told the command HANDLES and nothing else. It pokes them, watches what happens, and learns
what each one does:  f(observation, action) -> next_observation.

The learning signal is SURPRISE. No reward, no labels, no human-authored action model. The brain
predicts, the world disagrees, and the disagreement is the gradient. (BRAIN_DESIGN.md §4.)

THE CENTRAL LESSON, learned by getting it wrong first:

    You cannot learn what a command DOES by averaging what HAPPENED, because the world sometimes
    prevents it. A wall makes a real command look like a no-op. Averaging the outcomes drags the
    handle's effect toward zero and the brain files a working command as a dud.

So the model is explicitly TWO things, and keeping them apart is the whole trick:

    INTENT     what the command tries to do          -> the MODE of the non-zero deltas, not the mean
    AFFORDANCE where the world will actually allow it -> a blocked-cell map, DISCOVERED by being stopped

A move that produced no motion is not evidence the command is dead. It is evidence that the cell
ahead is blocked. Same observation, opposite conclusion -- and telling them apart is what makes
the difference between a brain that understands its body and one that averages it into mush.

The ARRANGER then plans over the LEARNED model -- never the real world, and never a wall map handed
to it by a human. If the tendrils are wrong, the plan is wrong. That is the honesty we want.
"""
from collections import Counter, deque

import numpy as np


class Tendrils:
    """A learned model of the body: what each opaque handle intends, and where it is permitted.

    Deliberately NOT a neural net. The honest model of "what does cmd_3 do" is "it tries to move me
    by (dx, dy), and here is how sure I am". An MLP would hide, behind a loss curve, whether the
    brain actually knows its suit -- and this project has been fooled by a pretty loss curve before.
    """

    def __init__(self, handles, dud_tol=0.30):
        self.handles = list(handles)
        self.n = len(handles)
        self.deltas = [Counter() for _ in range(self.n)]   # every outcome ever seen, per handle
        self.tries = np.zeros(self.n, dtype=np.int64)
        self.dud_tol = dud_tol
        self.blocked = set()        # cells it has been STOPPED by -- walls and edges alike
        self.free = set()           # cells it has actually stood in
        self.errors = []

    # ---------------------------------------------------------------- what it believes
    def intent(self, a):
        """What handle `a` TRIES to do: the most common NON-ZERO outcome. A command that is often
        blocked still has a clear intent -- the blocked samples simply do not get a vote here."""
        nz = [(d, c) for d, c in self.deltas[a].items() if d != (0, 0)]
        if not nz:
            return (0, 0)
        (dx, dy), best = max(nz, key=lambda kv: kv[1])
        zero = self.deltas[a].get((0, 0), 0)
        # a genuine DUD never moves us. If we have plenty of samples and almost all are no-ops,
        # believe it is dead; otherwise the no-ops are the world blocking a real command.
        if best / max(best + zero, 1) < self.dud_tol:
            return (0, 0)
        return (dx, dy)

    def predict(self, obs, a):
        """Where the brain thinks it will end up -- intent, GATED by what it has learned is blocked."""
        dx, dy = self.intent(a)
        if (dx, dy) == (0, 0):
            return obs.copy()
        tgt = (int(obs[0]) + dx, int(obs[1]) + dy)
        if tgt in self.blocked:                 # it has been stopped here before; expect to be again
            return obs.copy()
        return obs + np.array([dx, dy], dtype=obs.dtype)

    def observe(self, obs, a, obs2):
        """SURPRISE = |actual - predicted|. Then update BOTH beliefs: what the command wanted, and
        whether the world let it."""
        err = float(np.abs(obs2 - self.predict(obs, a)).sum())
        self.errors.append(err)

        d = (int(obs2[0] - obs[0]), int(obs2[1] - obs[1]))
        self.deltas[a][d] += 1
        self.tries[a] += 1
        self.free.add((int(obs2[0]), int(obs2[1])))

        if d == (0, 0):
            dx, dy = self.intent(a)             # we did not move. WHY?
            if (dx, dy) != (0, 0):              # the command has a real intent -> the world stopped us
                self.blocked.add((int(obs[0]) + dx, int(obs[1]) + dy))
        else:
            # we DID move here, so this cell is not blocked after all (a stale belief, e.g. a rule
            # flip changed which handle does what and we mis-blamed a cell). Retract it.
            self.blocked.discard((int(obs2[0]), int(obs2[1])))
        return err

    # ---------------------------------------------------------------- its own account of its body
    def learned_map(self):
        """handle -> discovered effect, or None for a dud. The brain's own words for its suit."""
        out = {}
        for i, h in enumerate(self.handles):
            e = self.intent(i)
            out[h] = None if e == (0, 0) else e
        return out

    def bounds(self):
        """The extent of the world, DISCOVERED by standing in it -- not handed over."""
        if not self.free:
            return (0, 0, 0, 0)
        xs = [p[0] for p in self.free]
        ys = [p[1] for p in self.free]
        return (min(xs), max(xs), min(ys), max(ys))

    def recent_error(self, k=200):
        return float(np.mean(self.errors[-k:])) if self.errors else float("nan")


def babble(world, tend, steps=4000, rng=None, curiosity=True):
    """Poke the levers. CURIOSITY prefers the least-probed handle, which is what makes this an
    insect probing a control room rather than a random walk."""
    rng = rng or np.random.default_rng(0)
    obs = world.observe()
    for t in range(steps):
        if curiosity and rng.random() < 0.7:
            a = int(np.argmin(tend.tries + rng.random(tend.n) * 3.0))
        else:
            a = int(rng.integers(tend.n))
        obs2 = world.step(a)
        tend.observe(obs, a, obs2)
        obs = obs2
        if t % 400 == 399:                     # unstick from corners so effects stay observable
            obs = world.reset()
    return tend


# ---------------------------------------------------------------------------- the ARRANGER
def plan(tend, start, goal, max_len=40):
    """Breadth-first search over the LEARNED model ONLY.

    No ground-truth walls, no ground-truth bounds. The brain plans with the body it believes it
    has: the handles it discovered an intent for, and the cells it discovered it cannot enter.
    If it never learned that cmd_5 goes east, it cannot use cmd_5 to go east -- and it shouldn't.
    """
    x0, x1, y0, y1 = tend.bounds()
    start, goal = tuple(map(int, start)), tuple(map(int, goal))
    moves = [(a, tend.intent(a)) for a in range(tend.n)]
    moves = [(a, e) for a, e in moves if e != (0, 0)]      # duds are useless for planning
    seen, q = {start}, deque([(start, [])])
    while q:
        pos, path = q.popleft()
        if pos == goal:
            return path
        if len(path) >= max_len:
            continue
        for a, (dx, dy) in moves:
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in seen or nxt in tend.blocked:
                continue
            if not (x0 <= nxt[0] <= x1 and y0 <= nxt[1] <= y1):
                continue
            seen.add(nxt)
            q.append((nxt, path + [a]))
    return None


def execute(world, actions, goal):
    """Run the plan in the REAL suit. Did the body actually end up where the brain intended?"""
    for a in actions:
        world.step(a)
    return tuple(map(int, world.observe())) == tuple(map(int, goal))
