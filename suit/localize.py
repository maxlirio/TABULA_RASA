#!/usr/bin/env python3
"""MONTE-CARLO LOCALISATION — the brain figuring out WHERE it is when a single look can't tell it.

The suits so far were Markov: the observation named the pose. A real LiDAR scan doesn't -- every
open patch of floor looks the same, so the robot is genuinely uncertain and must reason over TIME.

The answer is the one real robots use: keep a cloud of hypotheses (particles) about where you are,
move them with your MOTION model, weight them by how well each one's expected view matches what you
actually see, and resample. The cloud collapses onto the truth as movement rules out the imposters.

The nice part for this project: the motion model the particles are pushed through is the one the
brain GREW by babbling (tendrils_se2.eff) -- not a formula handed to it. It localises itself with
the body it taught itself.
"""
from collections import Counter

import numpy as np

from suit.world_se2 import _YAW_TO_DIR


def predicted_scan(nav, x, y):
    """What the 4-neighbour occupancy pattern SHOULD look like at (x, y), read off the known map."""
    W, H = nav.shape
    return tuple(0 if (0 <= x + dx < W and 0 <= y + dy < H and nav[x + dx, y + dy]) else 1
                 for dx, dy in _YAW_TO_DIR)


class Localizer:
    def __init__(self, nav, tend, n=800, seed=0):
        self.nav, self.tend, self.n = nav, tend, n
        self.use_heading = True                                       # IMU compass; off = symmetry demo
        self.rng = np.random.default_rng(seed)
        self.free = [(x, y) for x in range(nav.shape[0]) for y in range(nav.shape[1]) if nav[x, y]]
        idx = self.rng.integers(0, len(self.free), n)                 # a UNIFORM prior: kidnapped
        self.P = np.array([[*self.free[i], self.rng.integers(0, 4)] for i in idx], dtype=int)

    def motion(self, a):
        """Push every hypothesis through the LEARNED egocentric effect, colliding with the map."""
        eff = self.tend.eff
        W, H = self.nav.shape
        for i in range(self.n):
            x, y, yaw = self.P[i]
            dx, dy, dyaw = eff[a][yaw % 4]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H and self.nav[nx, ny]):
                nx, ny = x, y                                         # blocked: this hypothesis stays
            self.P[i] = (nx, ny, (yaw + dyaw) % 4)

    def sense(self, scan):
        bits, yaw = scan
        w = np.empty(self.n)
        for i in range(self.n):
            x, y, py = self.P[i]
            match = sum(a == b for a, b in zip(predicted_scan(self.nav, x, y), bits))
            head = (5.0 if py % 4 == yaw else 0.05) if self.use_heading else 1.0
            w[i] = np.exp(match) * head                              # heading (IMU) helps disambiguate
        w /= w.sum()
        self.P = self.P[self._resample(w)]

    def _resample(self, w):
        pos = (self.rng.random() + np.arange(self.n)) / self.n         # low-variance resampling
        c = np.cumsum(w)
        idx = np.zeros(self.n, dtype=int)
        i = 0
        for j, p in enumerate(pos):
            while i < self.n - 1 and p > c[i]:
                i += 1
            idx[j] = i
        return idx

    def estimate(self):
        """Best guess of the cell, the confidence in it, and how many hypotheses still survive."""
        cells = Counter((int(x), int(y)) for x, y, _ in self.P)
        (cx, cy), cnt = cells.most_common(1)[0]
        return (cx, cy), cnt / self.n, len(cells)

    def belief_grid(self):
        g = np.zeros(self.nav.shape)
        for x, y, _ in self.P:
            g[x, y] += 1
        return g / self.n
