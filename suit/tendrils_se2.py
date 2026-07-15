#!/usr/bin/env python3
"""ROBOT TENDRILS — growing into a body that FACES a direction.

The gridworld brain learned one effect per command: a constant delta. That is exactly wrong for a
robot. 'forward' does something different depending on which way you are pointing, so its effect is
constant only in the BODY frame, and the turns rotate that frame. Learn one constant delta and you
average 'forward' across all four headings into nothing -- and call your own legs a dud.

So this brain conditions each command's effect on the heading component of the pose (SLAM labels
which number is orientation). It learns a small table:  effect[command][yaw] = (dx, dy, dyaw). From
that table it can tell, untold, which commands walk (and recover the body-frame step they take),
which turn, which do nothing, and which are two names for one command -- then plan over the LiDAR map.
"""
from collections import Counter, deque

import numpy as np

from suit.world_se2 import _rot                       # to UN-rotate a world delta back to body frame


class RobotTendrils:
    def __init__(self, handles, n_yaw=4):
        self.handles = list(handles)
        self.n = len(handles)
        self.n_yaw = n_yaw
        self.samples = [[[] for _ in range(n_yaw)] for _ in range(self.n)]  # [(dpos,dyaw)] per (a,yaw)
        self.tries = np.zeros(self.n, dtype=np.int64)
        self.errors = []
        self.eff = None                                # eff[a][yaw] = (dx, dy, dyaw)
        self.nav = None                                # the LiDAR occupancy map, if the robot has it

    def set_map(self, nav_free):
        """Hand the brain the floor map its LiDAR built. Its world model is now action-effects PLUS
        geometry: it predicts a walk into a mapped obstacle will go nowhere, instead of being
        surprised by every wall. (The command SEMANTICS are still its own to discover -- only the
        room shape is given, which is exactly what LiDAR is for.)"""
        self.nav = np.asarray(nav_free, dtype=bool)

    def _yaw(self, o):
        return int(round(float(o[2]))) % self.n_yaw

    # ---------------------------------------------------------------- predict / learn
    def predict(self, o, a):
        if self.eff is None:
            return o.copy()
        dx, dy, dyaw = self.eff[a][self._yaw(o)]
        nx, ny = o[0] + dx, o[1] + dy
        if (dx, dy) != (0, 0) and self.nav is not None:            # a walk -- will the map allow it?
            ix, iy = int(round(float(nx))), int(round(float(ny)))
            if not (0 <= ix < self.nav.shape[0] and 0 <= iy < self.nav.shape[1]) or not self.nav[ix, iy]:
                return o.copy()                                    # blocked: the body stays put
        return np.array([nx, ny, (o[2] + dyaw) % self.n_yaw], dtype=o.dtype)

    def observe(self, o, a, o2):
        err = float(np.abs(o2 - self.predict(o, a)).sum())
        self.errors.append(err)
        y = self._yaw(o)
        dpos = (int(round(float(o2[0] - o[0]))), int(round(float(o2[1] - o[1]))))
        dyaw = int(round(float(o2[2] - o[2]))) % self.n_yaw
        self.samples[a][y].append((dpos, dyaw))
        self.tries[a] += 1
        if int(sum(self.tries)) % 300 == 0:
            self.fit()
        return err

    def fit(self):
        eff = [[(0, 0, 0)] * self.n_yaw for _ in range(self.n)]
        for a in range(self.n):
            for y in range(self.n_yaw):
                s = self.samples[a][y]
                if not s:
                    continue
                cnt = Counter(s)
                (dpos, dyaw), _ = cnt.most_common(1)[0]
                # a blocked walk looks like the identity outcome. If identity won but there is a
                # clear real effect, prefer it -- the wall must not convince us the command is dead.
                if (dpos, dyaw) == ((0, 0), 0):
                    nonid = [(k, c) for k, c in cnt.items() if k != ((0, 0), 0)]
                    if nonid:
                        (k, c) = max(nonid, key=lambda kv: kv[1])
                        if c / (c + cnt[((0, 0), 0)]) > 0.30:
                            dpos, dyaw = k
                eff[a][y] = (dpos[0], dpos[1], dyaw)
        self.eff = eff

    # ---------------------------------------------------------------- its own account of its body
    def classify(self):
        """Name each command WITHOUT being told: walk / turn / dud, and the body-frame step a walk
        takes (recovered by un-rotating its per-yaw world deltas -- if they agree, it is egocentric)."""
        out = {}
        for a in range(self.n):
            h = self.handles[a]
            if self.eff is None:
                out[h] = ("?", None)
                continue
            dyaws = {self.eff[a][y][2] for y in range(self.n_yaw)}
            movemag = max(abs(self.eff[a][y][0]) + abs(self.eff[a][y][1]) for y in range(self.n_yaw))
            turns = sorted(d for d in dyaws if d)
            if movemag == 0 and turns:
                out[h] = (f"turn {turns[0]:+d}", None)
            elif movemag > 0 and not turns:
                body = {_rot(self.eff[a][y][0], self.eff[a][y][1], -y) for y in range(self.n_yaw)
                        if self.eff[a][y][:2] != (0, 0)}
                if len(body) == 1:                     # same body-frame step at every heading
                    out[h] = ("walk (egocentric)", next(iter(body)))
                else:
                    out[h] = ("walk (inconsistent)", None)
            elif movemag == 0 and not turns:
                out[h] = ("dud", None)
            else:
                out[h] = ("mixed", None)
        return out

    def recent_error(self, k=200):
        return float(np.mean(self.errors[-k:])) if self.errors else float("nan")

    # ---------------------------------------------------------------- the arranger
    def plan(self, start, goal_xy, nav_free, max_len=120):
        """Search (x, y, yaw) using the learned per-heading effects, colliding against the LiDAR
        map (nav_free) -- the robot SEES obstacles instead of bumping them."""
        if self.eff is None:
            return None
        W, H = nav_free.shape
        s = (int(start[0]), int(start[1]), self._yaw(start))
        goal_xy = (int(goal_xy[0]), int(goal_xy[1]))
        seen, q = {s}, deque([(s, [])])
        while q:
            (x, y, yaw), path = q.popleft()
            if (x, y) == goal_xy:
                return path
            if len(path) >= max_len:
                continue
            for a in range(self.n):
                dx, dy, dyaw = self.eff[a][yaw]
                nx, ny, nyaw = x + dx, y + dy, (yaw + dyaw) % self.n_yaw
                if not (0 <= nx < W and 0 <= ny < H) or not nav_free[nx, ny]:
                    continue
                st = (nx, ny, nyaw)
                if st in seen:
                    continue
                seen.add(st)
                q.append((st, path + [a]))
        return None


def babble_robot(world, tend, steps=8000, rng=None):
    rng = rng or np.random.default_rng(0)
    tend.set_map(world.nav_grid())                     # the robot carries the map its LiDAR built
    obs = world.observe()
    for t in range(steps):
        a = int(np.argmin(tend.tries + rng.random(tend.n) * 3.0)) if rng.random() < 0.7 \
            else int(rng.integers(tend.n))
        obs2 = world.step(a)
        tend.observe(obs, a, obs2)
        obs = obs2
        if t % 300 == 299:
            obs = world.reset()
    tend.fit()
    return tend
