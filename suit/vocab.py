#!/usr/bin/env python3
"""THE CONSTRAINED VOCABULARY — the only words the mouth is allowed to say.

The idea (the user's, and it is the right one): don't let the language model emit free text it can
confabulate. Restrict its output to `CALL: <verb> <target>` where verb and target come from a CLOSED
set the grounded brain understands. Then "please make the robot go up the stairs" can only come out
as `CALL: climb stairs` -- it CANNOT come out as "solve this robot go down the stairs", which is
exactly what the untrained model does today.

Division of labour (this is the important part):
  - the MOUTH (LM) turns language into a CALL. That is all. It does NOT count or spatial-reason --
    we proved it says "north two" for "north three". Its job is understanding, not planning.
  - the INTERPRETER grounds the target to a spot on the LiDAR map and asks the ARRANGER to plan.
    The spatial reasoning -- "the stairs are ahead, so forward, forward, then step up" -- is the
    PLANNER's, produced from real geometry, where it is reliable.

`parse()` here is a GROUNDED STAND-IN for the LM (keyword rules), so the whole pipeline runs today.
prep_robotcmd.py generates the data that trains the real LM to emit these same CALLs -- the same way
tool-use CALLs were trained. Swap the trained model in behind the same CALL grammar and nothing else
changes.
"""
import sys

from suit.commander import _corners, _free_cells, _nearest_free

VERBS = ["goto", "climb", "face", "stop"]
LANDMARKS = ["stairs", "door", "table", "window", "charger"]
SPATIAL = ["far_corner", "near_corner", "middle", "other_side", "start",
           "north", "south", "east", "west"]
TARGETS = LANDMARKS + SPATIAL                                    # the closed target vocabulary


def parse(text):
    """Grounded stand-in for the constrained-output LM: English -> (verb, target) in the closed
    vocabulary, or (None, None) if it means nothing the brain can act on."""
    t = " " + text.lower().strip() + " "
    if "stair" in t and any(w in t for w in ("up", "climb", "ascend", "top")):
        return "climb", "stairs"
    for nm in LANDMARKS:
        if nm in t:
            return "goto", nm
    if "corner" in t:
        return "goto", "far_corner" if any(w in t for w in ("far", "other", "opposite")) else "near_corner"
    if any(w in t for w in ("middle", "center", "centre")):
        return "goto", "middle"
    if any(w in t for w in ("other side", "across", "far side", "far end")):
        return "goto", "other_side"
    if "start" in t or "home" in t or ("back" in t and " to " in t):
        return "goto", "start"
    for d, canon in (("north", "north"), ("south", "south"), ("east", "east"), ("west", "west"),
                     ("up", "north"), ("down", "south"), ("left", "west"), ("right", "east")):
        if f" {d} " in t or f" {d}." in t:
            return "goto", canon
    return None, None


# short primitive tokens for the plan the arranger produces (the {forward}{step_up} chain)
_PRIM = {(1, 0): "forward", (-1, 0): "back", (0, 1): "strafe_left", (0, -1): "strafe_right"}


class Interpreter:
    """Turns a CALL into grounded action: resolves the target on the map, plans with the brain's
    learned effects, and expands to a primitive chain. This is the whole brain speaking one language."""

    def __init__(self, commander):
        self.cmd = commander
        self.world = commander.world
        self.tend = commander.tend
        self.nav = commander.nav

    def _prim_name(self, a):
        kind, step = self.tend.classify()[self.world.handles[a]]
        if kind.startswith("walk"):
            return _PRIM.get(step, "walk")
        if kind.startswith("turn"):
            return "turn_left" if "+1" in kind else "turn_right"
        return "noop"

    def resolve(self, target):
        if target in self.world.landmarks:
            return self.world.landmarks[target], target
        p = (int(self.world.pose[0]), int(self.world.pose[1]))
        W, H = self.nav.shape
        cs = _corners(self.nav)
        if target == "far_corner":
            k = max(cs, key=lambda k: abs(cs[k][0] - p[0]) + abs(cs[k][1] - p[1]))
            return _nearest_free(self.nav, cs[k]), f"the far corner ({k})"
        if target == "near_corner":
            k = min(cs, key=lambda k: abs(cs[k][0] - p[0]) + abs(cs[k][1] - p[1]))
            return _nearest_free(self.nav, cs[k]), f"the nearest corner ({k})"
        if target == "middle":
            return _nearest_free(self.nav, (W // 2, H // 2)), "the middle"
        if target == "other_side":
            cells = _free_cells(self.nav)
            return max(cells, key=lambda c: abs(c[0] - p[0]) + abs(c[1] - p[1])), "the far side"
        if target == "start":
            return self.cmd.start, "the start"
        dirs = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
        if target in dirs:
            dx, dy = dirs[target]
            return _nearest_free(self.nav, (p[0] + dx * W, p[1] + dy * H)), f"the {target} wall"
        return None, target

    def run(self, text):
        verb, target = parse(text)
        call = f"CALL: {verb} {target}" if verb else None
        if verb is None:
            return {"call": None, "prims": None, "ok": False,
                    "say": "(no CALL -- outside my vocabulary)"}
        goal, phrase = self.resolve(target)
        if goal is None:
            return {"call": call, "prims": None, "ok": False, "say": f"{call}  -> can't find {target}"}
        plan = self.tend.plan(self.world.observe(), goal, self.nav)
        if plan is None:
            return {"call": call, "prims": None, "ok": False,
                    "say": f"{call}  -> no route to {phrase}"}
        prims = [self._prim_name(a) for a in plan]
        for a in plan:                                          # walk it
            self.world.step(a)
        reached = tuple(int(v) for v in self.world.pose[:2]) == tuple(goal)
        if verb == "climb" and target == "stairs":             # the skill the planner tops it off with
            prims += ["step_up"] * self.world.stairs_height
            self.world.level += 1
        chain = " ".join("{" + p + "}" for p in prims)
        return {"call": call, "prims": prims, "chain": chain, "ok": reached, "goal": goal,
                "say": f"{call}   ->   {chain}"}
