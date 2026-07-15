#!/usr/bin/env python3
"""THE SUIT — a body the brain wakes up inside, knowing nothing about it.

The contract is deliberately hostile to cheating (see BRAIN_DESIGN.md §3):

  * commands are OPAQUE HANDLES. `cmd_0`, `cmd_1`, ... The brain is told the LIST and nothing
    else. No names like "move_north" for it to pattern-match on -- that would be handing it the
    understanding we are claiming it discovers.
  * the mapping handle -> effect is RANDOMISED PER SUIT (the seed). So a brain cannot carry the
    meaning of `cmd_3` from one suit to the next; it can only carry the METHOD for finding out.
    This is what makes the suit-swap test (BRAIN_DESIGN.md §6.3) real rather than decorative.
  * some handles are DUDS (do nothing) and some are SYNONYMS (two handles, one effect). A brain
    that truly probes will find this out. A brain that assumes "n handles = n distinct useful
    actions" will not.
  * there is a rule-flip: past `flip_at` steps the action mapping is PERMUTED, so a brain that
    stopped learning is wrong afterwards and must notice its own prediction error rise.

The world itself is trivial on purpose. The hard part of this project is not the environment; it
is proving the brain grew into it rather than memorised it.
"""
import numpy as np

# the real semantics, which the BRAIN NEVER SEES
_EFFECTS = [(-1, 0), (1, 0), (0, -1), (0, 1)]        # north, south, west, east
NOOP = None


class RawEncoder:
    """A ROBOT'S EYES instead of a clean (x, y).

    A real body never hands its brain "you are at cell (3, 4)". It hands it a pile of sensor
    numbers -- joint angles, IMU, range-finders -- in which position is BURIED: mixed together
    through some unknown transform, blurred by sensor noise, and padded with readings that carry
    no positional information at all (a sensor pointed at a blank wall). The brain is NOT told the
    dimension means anything, nor which numbers are position, nor even that position is in there.

    This encoder is exactly that: true (x, y) -> a `dim`-vector = an unknown linear MIXING of the
    two coordinates, plus gaussian NOISE, plus `dead` dimensions that never respond to movement.
    If the tendrils only worked because observation==state, they will shatter here. That is the
    point -- it is the same wall the robot will present.
    """

    def __init__(self, dim=12, noise=0.05, dead=4, seed=0):
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(dim, 2))                 # position lives on a 2-D plane tilted in dim-D
        self.dead_idx = set()
        if dead:
            for i in rng.choice(dim, size=min(dead, dim), replace=False):
                A[i] = 0.0                            # a stuck sensor: no positional signal at all
                self.dead_idx.add(int(i))
        self.A, self.b, self.noise, self.dim = A, rng.normal(size=dim), noise, dim
        self._rng = rng

    def __call__(self, x, y):
        base = self.A @ np.array([x, y], dtype=float) + self.b
        return (base + self._rng.normal(size=self.dim) * self.noise).astype(np.float32)


class GridWorld:
    """A W x H grid with walls. Observation = (agent_x, agent_y). Actions = opaque handles."""

    def __init__(self, w=7, h=7, n_handles=8, seed=0, flip_at=None, walls=True, encoder=None):
        self.w, self.h, self.seed = w, h, seed
        self.rng = np.random.default_rng(seed)
        self.flip_at, self.steps = flip_at, 0
        self.encoder = encoder                        # None -> clean (x,y); RawEncoder -> robot eyes

        # --- the secret wiring: handle -> effect. Randomised, with duds and synonyms. ---
        eff = [_EFFECTS[i % 4] for i in range(n_handles)]      # guarantees all 4 directions exist
        if n_handles >= 6:
            eff[-1] = NOOP                                     # a dud: does nothing
            eff[-2] = eff[0]                                   # a synonym: same effect as handle 0
        self.rng.shuffle(eff)
        self._wiring = list(eff)
        self._orig_wiring = list(eff)

        self.handles = [f"cmd_{i}" for i in range(n_handles)]  # ALL the brain is given
        self.walls = set()
        if walls:
            for _ in range((w * h) // 6):
                self.walls.add((int(self.rng.integers(w)), int(self.rng.integers(h))))
        self.reset()

    # ---------------------------------------------------------------- the socket
    def reset(self, pos=None):
        if pos is None:
            while True:
                pos = (int(self.rng.integers(self.w)), int(self.rng.integers(self.h)))
                if pos not in self.walls:
                    break
        self.pos = pos
        return self.observe()

    def observe(self):
        if self.encoder is not None:
            return self.encoder(*self.pos)            # raw sensor reading, position buried inside
        return np.array(self.pos, dtype=np.float32)

    def encode(self, pos):
        """The raw reading the body WOULD emit at `pos` (for setting a goal), without moving there."""
        return self.encoder(*pos) if self.encoder is not None else np.array(pos, dtype=np.float32)

    def true_pos(self):
        return np.array(self.pos, dtype=np.float32)   # ground truth -- scoring only, never the brain

    def step(self, handle_idx):
        """Take a command by INDEX into self.handles. Returns the new observation."""
        self.steps += 1
        if self.flip_at and self.steps == self.flip_at:        # the rule flip
            self._wiring = list(self._orig_wiring)
            self.rng.shuffle(self._wiring)

        eff = self._wiring[handle_idx]
        if eff is not None:
            nx, ny = self.pos[0] + eff[0], self.pos[1] + eff[1]
            if 0 <= nx < self.w and 0 <= ny < self.h and (nx, ny) not in self.walls:
                self.pos = (nx, ny)                            # a blocked move is a NO-OP, silently
        return self.observe()

    # ---------------------------------------------------------------- ground truth (TESTS ONLY)
    def _truth(self):
        """The wiring. For scoring the brain's discovered map -- never for training it."""
        return list(self._wiring)
