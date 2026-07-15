#!/usr/bin/env python3
"""ALL FOUR PARTS, END TO END. Grow a brain into a robot body it knows nothing about, then drive it
purely by English -- no code, no coordinates typed. Each command shows: what the mouth understood,
the plan the arranger built from the brain's DISCOVERED command effects, and the body walking there.

    python3 suit/demo_commander.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from suit.world_se2 import RobotSuit                        # noqa: E402
from suit.tendrils_se2 import RobotTendrils, babble_robot   # noqa: E402
from suit.commander import Commander                        # noqa: E402


def main():
    world = RobotSuit(seed=4)
    tend = RobotTendrils(world.handles)
    print("The robot wakes up knowing only that it has 8 commands named cmd_0..cmd_7.")
    print("It babbles to find out what they do...\n")
    babble_robot(world, tend, steps=8000, rng=np.random.default_rng(4))

    cls = tend.classify()
    print("What it discovered its commands are (told only the names):")
    for a in world.handles:
        kind, step = cls[a]
        print(f"    {a}: {kind}" + (f"  body-step {step}" if step else ""))
    world.reset(pose=(1, 1, 0))
    cmd = Commander(world, tend)
    print("\nNow just talk to it. It resolves the words to a spot on its LiDAR map, plans with the")
    print("effects it LEARNED, and walks there.\n")

    script = ["go to the top left corner", "now go to the far corner", "head to the middle of the room",
              "go east two", "take me to the other side", "fly to the moon", "go back to the start"]
    ok = reached = 0
    for text in script:
        res = cmd.command(text)
        reached += res["goal"] is not None
        ok += res["ok"]
        print("=" * 66)
        print(f'you> "{text}"')
        print(f"robot> {res['say']}")
        if res["goal"] is not None:
            print()
            print(cmd.render(goal=res["goal"], path=res["path"]))
        print()

    print("=" * 66)
    print(f"{reached}/{len(script)} understood, {ok} walked to the exact goal   "
          f"(1 was gibberish and correctly refused)")
    print("legend:  > ^ < v = robot (facing)   * = goal   # = obstacle/wall   : = path walked")


if __name__ == "__main__":
    main()
