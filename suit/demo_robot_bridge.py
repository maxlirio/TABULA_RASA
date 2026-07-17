#!/usr/bin/env python3
"""END TO END TO THE REAL ROBOT GRAMMAR.

    English  ->  symbolic command   (the mouth: reconciled vocab, robot_bridge-aligned)
             ->  RouteAdapter        (goto <place> -> `go to <x> <y>` waypoints over the map)
             ->  robot_bridge.parse_command  (the ROBOT'S OWN parser -- validated here)

Proves the brain's output is something the Dummy 13 supervisor already accepts, not a parallel
invented language. The symbolic command is produced by a grounded stand-in for the LM; prep_robotcmd.py
trains the real model to emit these same strings.

    python3 suit/demo_robot_bridge.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from suit.world_se2 import RobotSuit                        # noqa: E402
from suit.robot_adapter import RouteAdapter                 # noqa: E402

# import the ROBOT'S real parser to validate against (no MuJoCo needed -- parse_command is standalone)
ROBOT = os.path.expanduser("~/Developer/ROBOT_EXPERIMENT/mujoco")
_real_parse = None
if os.path.isdir(ROBOT):
    sys.path.insert(0, ROBOT)
    try:
        from robot_bridge import parse_command as _real_parse  # noqa: E402
    except Exception as e:
        print(f"(could not import the real robot_bridge parser: {e})")


def validate(cmd):
    """Does the ROBOT'S own parser accept this command? Returns the (skill, kwargs) or an error."""
    if _real_parse is None:
        return "no-robot-repo"
    try:
        skill, kw = _real_parse(cmd)
        return skill + (f" {kw}" if kw else "")
    except Exception as e:
        return f"REJECTED: {e}"


def main():
    world = RobotSuit(seed=7, landmarks=True)
    nav = world.nav_grid()
    # a SLAM map carries a resolution (m/cell) + origin; here the 9x9 room is ~1.1 m across
    adapter = RouteAdapter(nav, world.landmarks, resolution=0.12, origin=(-0.5, 0.0),
                           start_cell=(1, 1))
    adapter._home = (1, 1)

    print("Robot's mapped room (landmarks from LiDAR + labels):")
    for nm, cell in world.landmarks.items():
        print(f"    {nm:8s} cell {cell} = ({adapter.cell_to_world(cell)[0]:+.2f}, "
              f"{adapter.cell_to_world(cell)[1]:+.2f}) m")
    print("\nEnglish -> symbolic -> robot commands -> checked against the ROBOT'S OWN parser:\n")

    script = ["please go up the stairs", "now cross the table", "head to the charger",
              "go to the far corner", "get up", "go forward", "stand still",
              "go to the middle of the room", "what's the weather"]
    total = accepted = 0
    for text in script:
        res = adapter.translate(text)
        print("=" * 74)
        print(f'you>    "{text}"')
        print(f"mouth>  {res['symbolic'] or '(no command -- outside vocabulary)'}")
        if res["cmds"]:
            for c in res["cmds"]:
                v = validate(c)
                total += 1
                accepted += not str(v).startswith("REJECTED")
                print(f"robot>  {c:22s} -> parse_command: {v}")
            if res["note"]:
                print(f"        note: {res['note']}")
    print("=" * 74)
    if _real_parse is not None:
        print(f"\n{accepted}/{total} emitted commands accepted by the robot's REAL parse_command.")
    else:
        print("\n(robot repo not found; commands emitted but not cross-validated)")
    print("'what's the weather' produced no command -- the constrained vocab can't invent an action.")


if __name__ == "__main__":
    main()
