#!/usr/bin/env python3
"""THE ROBOT'S SUIT — a legged body navigating a room its LiDAR has mapped in 3D.

The user chose LiDAR over a camera on purpose: LiDAR + SLAM hand the brain clean GEOMETRY -- its
pose (x, y, heading) and a 3-D occupancy map of the room -- so there is no vision model to build and
no pixels to interpret. Perception is solved upstream. The brain's job is not to see; it is to learn
what its opaque commands DO to that pose, and arrange them over the mapped obstacles.

THE NEW HARD THING vs the gridworld: the commands are EGOCENTRIC. 'forward' moves along the current
heading, so its effect in world coordinates ROTATES with yaw -- it is NOT a constant vector. A brain
that averages 'forward' across headings averages it to ZERO and files a working command as a dud.
Discovering that some commands act in a frame that OTHER commands (the turns) rotate is exactly the
step from "buttons on a grid" to "a body that faces a direction". That is what this suit demands.

Config space is SE(2): (x, y, yaw), yaw in {0,1,2,3}. The room is a real 3-D voxel map; the floor
where the robot's body can stand is its projection (no obstacle in the robot's height column).
"""
import numpy as np

# world heading for each yaw. THE BRAIN NEVER SEES THIS -- it only sees the integer yaw in its pose.
_YAW_TO_DIR = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def _rot(bx, by, yaw):
    """Rotate a BODY-frame step (bx, by) into world coordinates by yaw*90 degrees."""
    for _ in range(yaw % 4):
        bx, by = -by, bx
    return bx, by


class RobotSuit:
    LANDMARK_NAMES = ["stairs", "door", "table", "window", "charger"]

    def __init__(self, w=9, h=9, depth=3, n_handles=8, seed=0, obstacles=True, landmarks=False):
        self.w, self.h, self.depth, self.seed = w, h, depth, seed
        self.rng = np.random.default_rng(seed)
        self.landmarks = {}          # name -> cell (from LiDAR + a label; e.g. "stairs", "door")
        self.stairs_height = 3       # how many step-ups the stairs take
        self.level = 0               # which floor the robot is on (climbing stairs raises it)
        self._want_landmarks = landmarks

        # --- opaque command wiring: each handle is secretly one of these primitives ---
        prim = [("move", (1, 0, 0)),    # forward     (+1 in body-x)
                ("move", (-1, 0, 0)),   # back
                ("move", (0, 1, 0)),    # strafe-left (+1 in body-y)
                ("move", (0, -1, 0)),   # strafe-right
                ("turn", (0, 0, 1)),    # turn one way
                ("turn", (0, 0, -1))]   # turn the other
        wiring = list(prim)
        while len(wiring) < n_handles:
            wiring.append(("move", (1, 0, 0)) if len(wiring) == 6 else ("dud", (0, 0, 0)))  # synonym+dud
        self.rng.shuffle(wiring)
        self._wiring = wiring
        self._orig_wiring = list(wiring)
        self.handles = [f"cmd_{i}" for i in range(n_handles)]

        # --- the 3-D occupancy map the LiDAR built ---
        self.occ = np.zeros((w, h, depth), dtype=bool)
        if obstacles:
            self.occ[0, :, :] = self.occ[-1, :, :] = True          # room walls
            self.occ[:, 0, :] = self.occ[:, -1, :] = True
            for _ in range((w * h) // 7):
                x, y = int(self.rng.integers(1, w - 1)), int(self.rng.integers(1, h - 1))
                ht = int(self.rng.integers(1, depth + 1))          # obstacle of some height
                self.occ[x, y, :ht] = True
        # where the robot's body can stand: floor cells with a clear height column
        self.free = np.array([[not self.occ[x, y, :].any() for y in range(h)] for x in range(w)])
        if self._want_landmarks:
            cells = [(x, y) for x in range(w) for y in range(h) if self.free[x, y]]
            self.rng.shuffle(cells)
            for nm in self.LANDMARK_NAMES:
                if cells:
                    self.landmarks[nm] = cells.pop()
        self.reset()

    # ---------------------------------------------------------------- the socket
    def nav_grid(self):
        """Floor-projection of the 3-D map: cells the body can occupy. This is what LiDAR gives the
        brain -- it SEES the obstacles rather than discovering them by collision."""
        return self.free.copy()

    def reset(self, pose=None):
        if pose is None:
            while True:
                x, y = int(self.rng.integers(self.w)), int(self.rng.integers(self.h))
                if self.free[x, y]:
                    break
            pose = (x, y, int(self.rng.integers(4)))
        self.pose = tuple(int(v) for v in pose)
        return self.observe()

    def observe(self):
        return np.array(self.pose, dtype=np.float32)               # (x, y, yaw) from SLAM

    def _free_at(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h and bool(self.free[x, y])

    def scan(self):
        """A LOCAL LiDAR reading: occupancy of the 4 neighbouring cells + heading. It does NOT
        reveal (x, y) -- every open-floor cell returns the same all-clear pattern -- so a single
        reading cannot locate the robot. This is the perceptual-aliasing (non-Markov) case: the
        robot must MOVE and integrate readings to know where it is."""
        x, y, yaw = self.pose
        bits = tuple(0 if self._free_at(x + dx, y + dy) else 1 for dx, dy in _YAW_TO_DIR)
        return bits, int(yaw)

    def step(self, a):
        kind, vec = self._wiring[a]
        x, y, yaw = self.pose
        if kind == "turn":
            yaw = (yaw + vec[2]) % 4
        elif kind == "move":
            dx, dy = _rot(vec[0], vec[1], yaw)                      # egocentric -> world
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h and self.free[nx, ny]:
                x, y = nx, ny                                      # blocked move = no-op
        self.pose = (x, y, yaw)
        return self.observe()

    def _truth(self):
        return list(self._wiring)
