#!/usr/bin/env python3
"""ROUTE ADAPTER — turns the brain's intent into commands the REAL robot already understands.

The Dummy 13 (ROBOT_EXPERIMENT/mujoco/robot_bridge.py) does NOT want motor primitives. Its supervisor
exposes SKILLS: navigate to a metric point (`go to <x> <y>`), `cross` a gap, `stand`, `recover`, and
egocentric nudges (`go forward|back|left|right [d]`). The low-level walking/balancing is the
supervisor's job. So the brain's contribution is the layer ABOVE that: read the map, plan a ROUTE
around obstacles, and hand the supervisor a short list of waypoints -- which is exactly what
`suit/`'s map + planner are for, once pointed at the robot's grammar instead of grid primitives.

This is also the reconciliation (the two efforts had grown two parsers). ONE symbolic vocabulary the
LM emits:  cross | stand | recover | go forward|back|left|right | goto <landmark/place>.  Every one
of those except `goto` is robot_bridge grammar VERBATIM; `goto <place>` is the single thing this
adapter expands -- resolve the place on the map, plan a route, emit `go to <x> <y>` per waypoint.
Nothing this adapter emits is outside what robot_bridge.parse_command already accepts (the demo
validates that against the real parser).

The metric mapping (cell -> metres) is a resolution + origin, exactly the metadata a SLAM occupancy
grid carries; on the real robot those come from the map, here they are constructor args.
"""
from collections import deque

import numpy as np

from suit.commander import _corners, _free_cells, _nearest_free

# every symbolic verb except `goto` is passed straight to the robot untouched
PASSTHROUGH = ("cross", "stand", "recover")
_DIRECTIONAL = ("forward", "back", "left", "right")
_LANDMARKS = ("door", "table", "window", "charger", "stairs")


def parse_robot(text):
    """English -> ONE symbolic command in the reconciled vocabulary, or None. This is what a trained
    LM would emit; grounded here so the pipeline runs today (prep_robotcmd.py trains the LM to emit
    these exact strings)."""
    t = " " + text.lower().strip() + " "
    if ("cross" in t and ("table" in t or "gap" in t)) or t.strip() in ("cross", "get across"):
        return "cross"
    if any(w in t for w in ("get up", "recover", "stand up", "stand back up")):
        return "recover"
    if any(f" {w} " in t for w in ("stop", "stand", "hold", "stay")) and "stair" not in t:
        return "stand"
    if "start" in t or "home" in t or ("back" in t and " to " in t):
        return "goto start"
    for d, canon in (("forward", "forward"), ("backward", "back"), ("back", "back"),
                     ("left", "left"), ("right", "right")):
        if f" {d} " in t or f" {d}." in t:
            return f"go {canon}"
    if "stair" in t and any(w in t for w in ("up", "climb", "ascend", "top")):
        return "goto stairs"                         # NB: a climb skill is a robot-side TODO (see run())
    for nm in _LANDMARKS:
        if nm in t:
            return f"goto {nm}"
    if "corner" in t:
        return "goto far_corner" if any(w in t for w in ("far", "other", "opposite")) else "goto near_corner"
    if any(w in t for w in ("middle", "center", "centre")):
        return "goto middle"
    if any(w in t for w in ("other side", "across", "far side", "far end")):
        return "goto other_side"
    for d in ("north", "south", "east", "west"):
        if f" {d} " in t:
            return f"goto {d}"
    return None


class RouteAdapter:
    def __init__(self, nav_free, landmarks, resolution=0.12, origin=(0.0, 0.0), start_cell=None):
        self.nav = np.asarray(nav_free, dtype=bool)
        self.landmarks = dict(landmarks)
        self.res, (self.ox, self.oy) = resolution, origin
        self.cur = tuple(start_cell) if start_cell else (_free_cells(self.nav) or [(0, 0)])[0]

    # ------------------------------------------------------------------ geometry
    def cell_to_world(self, c):
        return (self.ox + (c[0] + 0.5) * self.res, self.oy + (c[1] + 0.5) * self.res)

    def resolve(self, place):
        if place in self.landmarks:
            return self.landmarks[place]
        W, H = self.nav.shape
        p, cs = self.cur, _corners(self.nav)
        if place == "far_corner":
            k = max(cs, key=lambda k: abs(cs[k][0] - p[0]) + abs(cs[k][1] - p[1]))
            return _nearest_free(self.nav, cs[k])
        if place == "near_corner":
            k = min(cs, key=lambda k: abs(cs[k][0] - p[0]) + abs(cs[k][1] - p[1]))
            return _nearest_free(self.nav, cs[k])
        if place == "middle":
            return _nearest_free(self.nav, (W // 2, H // 2))
        if place == "other_side":
            return max(_free_cells(self.nav), key=lambda c: abs(c[0] - p[0]) + abs(c[1] - p[1]))
        if place == "start":
            return self.cur if not hasattr(self, "_home") else self._home
        dirs = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
        if place in dirs:
            dx, dy = dirs[place]
            return _nearest_free(self.nav, (p[0] + dx * W, p[1] + dy * H))
        return None

    # ------------------------------------------------------------------ route planning
    def _bfs(self, s, g):
        W, H = self.nav.shape
        seen, q = {s}, deque([(s, [s])])
        while q:
            (x, y), path = q.popleft()
            if (x, y) == g:
                return path
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and self.nav[nx, ny] and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    q.append(((nx, ny), path + [(nx, ny)]))
        return None

    def _line_free(self, a, b):
        (x0, y0), (x1, y1) = a, b
        n = max(abs(x1 - x0), abs(y1 - y0))
        for i in range(n + 1):
            x = round(x0 + (x1 - x0) * i / n) if n else x0
            y = round(y0 + (y1 - y0) * i / n) if n else y0
            if not self.nav[x, y]:
                return False
        return True

    def _waypoints(self, path):
        """Collapse a cell-by-cell path to the few corners navigate needs (straight legs merged)."""
        wp, i = [], 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self._line_free(path[i], path[j]):
                j -= 1
            wp.append(path[j])
            i = j
        return wp

    def route(self, goal_cell):
        path = self._bfs(self.cur, goal_cell)
        if path is None:
            return None
        cmds = ["go to %.2f %.2f" % self.cell_to_world(w) for w in self._waypoints(path)]
        self.cur = goal_cell
        return cmds or ["stand"]        # already there -> a no-op the robot understands

    # ------------------------------------------------------------------ the whole translation
    def translate(self, text):
        sym = parse_robot(text)
        if sym is None:
            return {"symbolic": None, "cmds": None, "note": "outside the vocabulary"}
        head = sym.split()
        if head[0] == "goto":
            cell = self.resolve(head[1])
            if cell is None:
                return {"symbolic": sym, "cmds": None, "note": f"can't place '{head[1]}'"}
            cmds = self.route(cell)
            note = ""
            if head[1] == "stairs":
                note = "navigates to the stairs; a `climb` skill is a robot-side TODO"
            return {"symbolic": sym, "cmds": cmds, "note": note}
        return {"symbolic": sym, "cmds": [sym], "note": "robot_bridge skill verbatim"}
