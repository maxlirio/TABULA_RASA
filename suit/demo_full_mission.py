#!/usr/bin/env python3
"""THE DEEPEST CHAIN — every part at once, on the real trained model.

  1. the robot wakes up NOT knowing where it is (uniform belief), and localises itself by MOVING,
     using the motion model it GREW (non-Markov / perceptual aliasing).
  2. once it knows where it is, you give it a MULTI-STEP spoken mission.
  3. the REAL trained apollo (run #4) turns each sentence into a command.
  4. the adapter grounds it on the map and route-plans it into waypoints.
  5. every command is checked against the ROBOT'S OWN parse_command.

    python3 suit/demo_full_mission.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gm.lm import load                                       # noqa: E402
from suit.world_se2 import RobotSuit, _YAW_TO_DIR            # noqa: E402
from suit.tendrils_se2 import RobotTendrils, babble_robot    # noqa: E402
from suit.localize import Localizer                          # noqa: E402
from suit.robot_adapter import RouteAdapter                  # noqa: E402

sys.path.insert(0, os.path.expanduser("~/Developer/ROBOT_EXPERIMENT/mujoco"))
try:
    from robot_bridge import parse_command                   # the robot's REAL parser
except Exception:
    parse_command = None


def rule(m):
    print("\n" + "=" * 74 + f"\n{m}\n" + "=" * 74)


# ---- the real trained mouth
BRAIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "apollo_run4.pt")
model, coder = load(BRAIN)
model.eval()
_unk = coder.stoi.get("<unk>")
_ban = [_unk] if _unk is not None else None


def apollo(text, n=8):
    ids = coder.encode(f"USER: {text}\nBOT:")
    o = model.generate(torch.tensor([ids]), n, temp=0.1, top_k=1, ban=_ban)[0].tolist()
    return coder.decode(o[len(ids):]).split("■")[0].split("USER")[0].strip()


# ---- grow the body model, then wake up lost
world = RobotSuit(seed=7, landmarks=True)
tend = RobotTendrils(world.handles)
babble_robot(world, tend, steps=8000, rng=np.random.default_rng(7))
nav = world.nav_grid()

rule("1. KIDNAPPED — the robot wakes up not knowing where it is, and localises by moving")
true0 = (3, 5, 1)
world.reset(pose=true0)
loc = Localizer(nav, tend, n=800, seed=1)
loc.sense(world.scan())
_, _, nh0 = loc.estimate()
print(f"   truth: at cell {true0[:2]}. belief: spread over {nh0} cells (a single scan can't localise)")
fwd = next(a for a, (k, s) in ((i, tend.classify()[h]) for i, h in enumerate(world.handles))
           if k.startswith("walk") and s == (1, 0))
turn = next(a for a, (k, s) in ((i, tend.classify()[h]) for i, h in enumerate(world.handles))
            if k.startswith("turn"))
conv = None
for step in range(1, 31):
    bits, yaw = world.scan()
    a = turn if bits[yaw] else fwd
    world.step(a)
    loc.motion(a)
    loc.sense(world.scan())
    est, conf, nh = loc.estimate()
    if nh == 1 and est == tuple(int(v) for v in world.pose[:2]):
        conv = step
        break
truth_now = tuple(int(v) for v in world.pose[:2])
print(f"   after {conv} moves: belief collapsed to {est} (conf {conf:.0%}); true position {truth_now}"
      f"  -> {'CORRECT' if est == truth_now else 'WRONG'}")

# hand the recovered position to the adapter and run the mission from there
adapter = RouteAdapter(nav, world.landmarks, resolution=0.12, origin=(-0.5, 0.0), start_cell=est)
adapter._home = est

rule("2. A MULTI-STEP SPOKEN MISSION — real apollo translates, adapter routes, robot parser checks")
print("   landmarks on the map:", {k: v for k, v in world.landmarks.items()})
print()
mission = ["go to the door", "now cross the table", "then head to the charger",
           "go up the stairs", "finally go back to where you started", "nice work"]
tot = acc = 0
for text in mission:
    sym = apollo(text)
    res = adapter.expand(sym)
    print(f'   you>    "{text}"')
    print(f"   apollo> {sym!r}")
    if res["cmds"]:
        for c in res["cmds"]:
            if parse_command:
                try:
                    parse_command(c); tot += 1; acc += 1; verdict = "accepted"
                except Exception as e:
                    tot += 1; verdict = f"REJECTED ({e})"
            else:
                verdict = "(robot repo not found)"
            print(f"   robot>  {c:20s} -> {verdict}")
        if res["note"]:
            print(f"           note: {res['note']}")
        # 'execute': the supervisor navigates to the goal (abstracted -- no MuJoCo here)
        if sym.startswith("goto"):
            adapter.cur = adapter.resolve(" ".join(sym.split()[1:])) or adapter.cur
    else:
        print(f"   robot>  (no command -- {res['note']})")
    print()

rule(f"RESULT: lost -> localised in {conv} moves -> {len(mission)}-step mission -> "
     f"{acc}/{tot} commands accepted by the robot's REAL parser")
