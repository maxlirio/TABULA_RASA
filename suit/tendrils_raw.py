#!/usr/bin/env python3
"""RAW TENDRILS — growing into a body whose observations are a robot's, not a clean (x, y).

The clean-suit Tendrils cheated in a way that only shows up now: it assumed the observation WAS the
position. `obs2 - obs` was the true movement; `int(obs[0]), int(obs[1])` were grid cells. Feed it a
raw sensor vector (position mixed through an unknown transform, blurred by noise, padded with dead
sensors) and every one of those assumptions is false.

So this brain assumes NOTHING about what the numbers mean. It recovers the idea of "where am I"
from the only thing it controls -- its own actions:

  1. LEARN THE EFFECT of each command as a constant vector in raw-sensor space. A linear encoding
     turns a fixed move into a fixed sensor-delta, so `d_a` is still constant per command -- it is
     just no longer the true (dx, dy). Robust-average it, so NOISE cancels and a DEAD sensor (which
     never responds to any command) shows ~0 in every d_a and is thereby ignored, untold.

  2. DISCOVER SPACE FROM ACTION. All those effect-vectors lie in a 2-D plane (the body only moves in
     two dimensions, whatever the sensor count). The top two principal directions of the effects ARE
     that plane. Project sensor readings onto it and the brain has invented its own 2-D coordinate --
     grounded in what its commands DO, not in being told which numbers are coordinates.

  3. PLAN in that self-made frame, exactly as before: search the learned effects to reach a goal,
     avoiding the cells where a command was found to do nothing (walls, in recovered coordinates).

If this works, the tendril METHOD -- not the clean-grid convenience -- is what survives to the robot.
"""
from collections import deque

import numpy as np


class RawTendrils:
    def __init__(self, handles, dead_tol=0.5, refit_every=200):
        self.handles = list(handles)
        self.n = len(handles)
        self.samples = [[] for _ in range(self.n)]     # (obs, obs') per command, raw
        self.tries = np.zeros(self.n, dtype=np.int64)
        self.errors = []
        self.dead_tol = dead_tol                        # a delta smaller than this = "did not move"
        self.refit_every = refit_every
        # everything below is DISCOVERED, not given:
        self.o0 = None          # a reference reading, so the recovered frame has an origin
        self.d = None           # d[a] = command a's effect in raw-sensor space
        self.B = None           # the recovered 2-D plane (dim x 2), position lives here
        self.rd = None          # rd[a] = command a's effect in RECOVERED coordinates
        self.movers = []        # commands that actually move the body
        self.blocked = set()    # recovered-frame cells where a mover was stopped (walls/edges)
        self.step = 1.0         # spacing of the recovered lattice, for cell keys + tolerance
        self._bbox = None

    # ------------------------------------------------------------------ the recovered frame
    def _frame(self, o):
        """Raw reading -> the brain's self-invented 2-D coordinate."""
        return self.B.T @ (o - self.o0)

    def _key(self, c):
        return tuple(np.round(c / (self.step * 0.5)).astype(int))

    # ------------------------------------------------------------------ predict / learn
    def predict(self, o, a):
        if self.d is None or a not in self.movers:
            return o.copy()
        if self._key(self._frame(o) + self.rd[a]) in self.blocked:   # expect to be stopped here
            return o.copy()
        return o + self.d[a]

    def observe(self, o, a, o2):
        err = float(np.abs(o2 - self.predict(o, a)).sum())
        self.errors.append(err)
        if self.o0 is None:
            self.o0 = o.copy()
        self.samples[a].append((o.copy(), o2.copy()))
        self.tries[a] += 1
        if sum(self.tries) % self.refit_every == 0:
            self.fit()
        return err

    def fit(self):
        """Re-estimate every command's effect, then re-derive the coordinate frame from them."""
        dim = len(self.o0)
        d = np.zeros((self.n, dim))
        for a in range(self.n):
            if not self.samples[a]:
                continue
            arr = np.array([o2 - o for o, o2 in self.samples[a]])
            mags = np.linalg.norm(arr, axis=1)
            if mags.max() < self.dead_tol:              # never moved -> a dud command
                continue
            moved = mags > mags.max() * 0.5             # ignore the samples where a wall stopped us
            d[a] = arr[moved].mean(axis=0)              # robust effect; NOISE averages away
        self.d = d
        self.movers = [a for a in range(self.n) if np.linalg.norm(d[a]) > self.dead_tol]
        if len(self.movers) < 2:
            return
        # the 2-D plane the body moves in = top-2 principal directions of the effect-vectors
        M = np.array([d[a] for a in self.movers])
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        self.B = Vt[:2].T
        self.rd = d @ self.B
        self.step = min(np.linalg.norm(self.rd[a]) for a in self.movers)
        # rebuild the wall map IN RECOVERED COORDINATES, and the explored bounding box
        self.blocked, frames = set(), []
        for a in self.movers:
            for o, o2 in self.samples[a]:
                frames.append(self._frame(o))
                if np.linalg.norm(o2 - o) < self.dead_tol:          # a mover that did not move
                    self.blocked.add(self._key(self._frame(o) + self.rd[a]))
        F = np.array(frames)
        self._bbox = (F.min(0) - self.step, F.max(0) + self.step)

    # ------------------------------------------------------------------ its own account of its body
    def learned_map_2d(self):
        """command -> its effect in the recovered frame (rounded), or None for a dud."""
        out = {}
        for i, h in enumerate(self.handles):
            if self.rd is None or i not in self.movers:
                out[h] = None
            else:
                out[h] = tuple(int(v) for v in np.round(self.rd[i] / self.step))
        return out

    def dead_sensor_leak(self, dead_idx):
        """How much signal the DEAD sensors wrongly picked up. Should be ~0 -- the brain ignored
        them without being told they were dead."""
        if self.d is None or not dead_idx:
            return 0.0
        idx = list(dead_idx)
        return float(np.abs(self.d[self.movers][:, idx]).max()) if self.movers else 0.0

    def recent_error(self, k=200):
        return float(np.mean(self.errors[-k:])) if self.errors else float("nan")

    # ------------------------------------------------------------------ the arranger
    def plan(self, o_start, o_goal, max_len=80):
        if self.d is None or self.B is None:
            return None
        cs, cg = self._frame(o_start), self._frame(o_goal)
        moves = [(a, self.rd[a]) for a in self.movers]
        tol = self.step * 0.4
        lo, hi = self._bbox
        seen, q = {self._key(cs)}, deque([(cs, [])])
        while q:
            c, path = q.popleft()
            if np.linalg.norm(c - cg) < tol:
                return path
            if len(path) >= max_len:
                continue
            for a, rda in moves:
                nc = c + rda
                k = self._key(nc)
                if k in seen or k in self.blocked:
                    continue
                if np.any(nc < lo) or np.any(nc > hi):       # don't wander past what it has explored
                    continue
                seen.add(k)
                q.append((nc, path + [a]))
        return None


def babble_raw(world, tend, steps=6000, rng=None):
    """Poke the levers, curiosity-first, in a body whose readings mean nothing yet."""
    rng = rng or np.random.default_rng(0)
    obs = world.observe()
    for t in range(steps):
        a = int(np.argmin(tend.tries + rng.random(tend.n) * 3.0)) if rng.random() < 0.7 \
            else int(rng.integers(tend.n))
        obs2 = world.step(a)
        tend.observe(obs, a, obs2)
        obs = obs2
        if t % 400 == 399:
            obs = world.reset()
    tend.fit()
    return tend
