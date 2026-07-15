#!/usr/bin/env python3
"""THE COMMANDER — all four parts of the brain, running together.

    you (English)  ->  MOUTH: turn the words into a goal
                   ->  ARRANGER: plan a route over the LiDAR map
                   ->  SUIT: walk the body there
                   ->  MOUTH: say what it did, in the terms IT discovered its commands mean

The mouth's language job is deliberately GROUNDED, not left to the language model. We tested the
285M LM on navigation goals and it deflected ("take care!", "see you later!") or fell into the
reward template -- the same failure reward design has. "Where is the far corner" is a fact about the
MAP, not about English, so it is resolved from geometry, the same way the reward tool and the solver
are grounded. The LM remains the conversational voice; the spatial reference is code. That division
-- language understanding by the net, precise grounding by code -- is this project's recurring answer.

The important honesty: the commander never uses the world's ground truth. It plans with the brain's
DISCOVERED command effects and narrates in the names the brain worked out ("walk forward", "turn
left") -- so if the tendrils are wrong, the walk is wrong, out loud.
"""
import re

import numpy as np

# grid convention: x = column (0 = west/left .. W-1 = east/right); y = row (0 = south/bottom .. top)
_DIRS = {"north": (0, 1), "up": (0, 1), "top": (0, 1), "forward": (0, 1),
         "south": (0, -1), "down": (0, -1), "bottom": (0, -1), "back": (0, -1),
         "east": (1, 0), "right": (1, 0), "west": (-1, 0), "left": (-1, 0)}
_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10, "a": 1, "an": 1}


def _free_cells(nav):
    return [(x, y) for x in range(nav.shape[0]) for y in range(nav.shape[1]) if nav[x, y]]


def _nearest_free(nav, cell):
    """Snap a desired point to the closest cell the body can actually stand on."""
    if nav[int(np.clip(cell[0], 0, nav.shape[0] - 1)), int(np.clip(cell[1], 0, nav.shape[1] - 1))]:
        pass
    cells = _free_cells(nav)
    return min(cells, key=lambda c: abs(c[0] - cell[0]) + abs(c[1] - cell[1])) if cells else None


def _corners(nav):
    W, H = nav.shape
    return {"southwest": (0, 0), "southeast": (W - 1, 0), "northwest": (0, H - 1),
            "northeast": (W - 1, H - 1)}


def resolve_goal(text, nav, pose, start=None, here=None):
    """English -> a goal cell on the map (grounded in geometry). Returns (cell, spoken_phrase) or
    (None, reason)."""
    t = " " + text.lower().strip() + " "
    W, H = nav.shape
    p = (int(pose[0]), int(pose[1]))
    said = None
    goal = None

    def dirs_in(s):
        return [v for k, v in _DIRS.items() if f" {k} " in s or f" {k}-" in s or f"-{k} " in s]

    if "corner" in t:
        cs = _corners(nav)
        if any(w in t for w in ("far", "farthest", "opposite", "other")):
            name = max(cs, key=lambda k: abs(cs[k][0] - p[0]) + abs(cs[k][1] - p[1]))
            goal, said = cs[name], f"the far corner ({name})"
        elif any(w in t for w in ("near", "nearest", "closest")):
            name = min(cs, key=lambda k: abs(cs[k][0] - p[0]) + abs(cs[k][1] - p[1]))
            goal, said = cs[name], f"the nearest corner ({name})"
        else:                                              # a named corner from its two directions
            names = {"north": "north", "up": "north", "top": "north", "south": "south",
                     "down": "south", "bottom": "south", "east": "east", "right": "east",
                     "west": "west", "left": "west"}
            ns = {names[k] for k in _DIRS if f" {k} " in t or f"{k}-" in t or f"-{k}" in t
                  and k in names}
            key = ("north" if "north" in ns else "south" if "south" in ns else "") + \
                  ("west" if "west" in ns else "east" if "east" in ns else "")
            if key in cs:
                goal, said = cs[key], f"the {key} corner"
            else:                                          # bare "the corner" -> nearest, sensibly
                name = min(cs, key=lambda k: abs(cs[k][0] - p[0]) + abs(cs[k][1] - p[1]))
                goal, said = cs[name], f"a corner ({name})"

    elif any(w in t for w in ("middle", "center", "centre")):
        goal, said = (W // 2, H // 2), "the middle of the room"

    elif ("start" in t or "home" in t or ("back" in t and "to" in t)) and start is not None:
        goal, said = (int(start[0]), int(start[1])), "the start point"

    elif ("here" in t or " me " in t or "come" in t) and here is not None:
        goal, said = (int(here[0]), int(here[1])), "your position"

    elif any(w in t for w in ("other side", "far side", "across", "other end", "far end", "farthest")):
        cells = _free_cells(nav)
        goal = max(cells, key=lambda c: abs(c[0] - p[0]) + abs(c[1] - p[1])) if cells else None
        said = "the far side of the room"

    else:                                                  # a relative move: "go north (three)"
        ds = dirs_in(t)
        if ds:
            dx, dy = ds[0]
            m = re.search(r"\b(\d+)\b", t)
            n = int(m.group(1)) if m else next((v for w, v in _WORDNUM.items()
                                                if f" {w} " in t), None)
            if n is None:                                  # "go north" with no count -> to the wall
                gx, gy = p[0] + dx * W, p[1] + dy * H
                said = f"as far {_dir_name(dx, dy)} as the room allows"
            else:
                gx, gy = p[0] + dx * n, p[1] + dy * n
                said = f"{n} {'cell' if n == 1 else 'cells'} {_dir_name(dx, dy)}"
            goal = (gx, gy)

    if goal is None:
        return None, "I couldn't tell where you want me to go."
    snapped = _nearest_free(nav, goal)
    if snapped is None:
        return None, "there's nowhere I can stand near there."
    if snapped != tuple(goal):
        said += f" (nearest open floor to it)"
    return snapped, said


def _dir_name(dx, dy):
    return {(0, 1): "north", (0, -1): "south", (1, 0): "east", (-1, 0): "west"}.get((dx, dy), "over")


class Commander:
    """Wraps a suit + its grown brain and turns English into walking."""

    def __init__(self, world, tend):
        self.world, self.tend = world, tend
        self.nav = world.nav_grid()
        self.start = tuple(int(v) for v in world.pose[:2])
        self._names = self._name_commands()

    def _name_commands(self):
        """Friendly names for the brain's DISCOVERED commands (not the suit's secret wiring)."""
        body = {(1, 0): "walk forward", (-1, 0): "walk back",
                (0, 1): "strafe left", (0, -1): "strafe right"}
        out = {}
        for a, (kind, step) in self.tend.classify().items():
            idx = self.world.handles.index(a)
            if kind.startswith("walk"):
                out[idx] = body.get(step, f"walk {step}")
            elif kind.startswith("turn"):
                out[idx] = "turn left" if "+1" in kind else "turn right"
            else:
                out[idx] = "(does nothing)"
        return out

    def _narrate(self, plan):
        if not plan:
            return "no moves"
        runs, cur, k = [], plan[0], 1
        for a in plan[1:]:
            if a == cur:
                k += 1
            else:
                runs.append((cur, k))
                cur, k = a, 1
        runs.append((cur, k))
        return ", then ".join(f"{self._names[a]}" + (f" x{k}" if k > 1 else "") for a, k in runs)

    def command(self, text, here=None):
        pose = self.world.observe()
        goal, said = resolve_goal(text, self.nav, pose, start=self.start, here=here)
        if goal is None:
            return {"ok": False, "say": said, "goal": None, "plan": None, "path": None}
        plan = self.tend.plan(pose, goal, self.nav)
        if plan is None:
            return {"ok": False, "say": f"I understand '{said}', but I can't find a route there.",
                    "goal": goal, "plan": None, "path": None}
        path = [tuple(int(v) for v in self.world.pose[:2])]
        for a in plan:
            self.world.step(a)
            path.append(tuple(int(v) for v in self.world.pose[:2]))
        reached = tuple(int(v) for v in self.world.pose[:2]) == tuple(goal)
        plan_say = self._narrate(plan)
        say = (f"Heading to {said}. My plan: {plan_say}. "
               + ("Done -- I'm there." if reached else "...I ended up somewhere else."))
        return {"ok": reached, "say": say, "goal": goal, "plan": plan, "path": path,
                "plan_text": plan_say}

    # ---------------------------------------------------------------- a look at the room
    def render(self, goal=None, path=None):
        W, H = self.nav.shape
        pset = set(path or [])
        face = {0: ">", 1: "^", 2: "<", 3: "v"}[int(self.world.pose[2])]
        rp = (int(self.world.pose[0]), int(self.world.pose[1]))
        out = []
        for y in range(H - 1, -1, -1):
            row = []
            for x in range(W):
                c = (x, y)
                if c == rp:
                    row.append(face)
                elif goal and c == tuple(goal):
                    row.append("*")
                elif not self.nav[x, y]:
                    row.append("#")
                elif c in pset:
                    row.append(":")
                else:
                    row.append(".")
            out.append(" ".join(row))
        return "\n".join(out)
