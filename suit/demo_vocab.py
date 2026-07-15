#!/usr/bin/env python3
"""CONSTRAINED-VOCABULARY COMMANDING, end to end.

    English  ->  CALL: <verb> <target>   (the mouth: closed vocabulary, cannot confabulate)
             ->  grounded on the LiDAR map + planned by the arranger
             ->  a primitive chain {forward}{turn_left}{step_up}...   (the body does this)

The CALL here is produced by a GROUNDED stand-in for the LM (suit/vocab.parse) so it runs today;
prep_robotcmd.py makes the data to train the real model to emit the exact same CALLs. The point is
that the spatial reasoning -- turning "go up the stairs" into forward/turn/step-up -- is the
PLANNER's, from real geometry, not the language model's.

    python3 suit/demo_vocab.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from suit.world_se2 import RobotSuit                        # noqa: E402
from suit.tendrils_se2 import RobotTendrils, babble_robot   # noqa: E402
from suit.commander import Commander                        # noqa: E402
from suit.vocab import Interpreter                          # noqa: E402


def main():
    world = RobotSuit(seed=7, landmarks=True)
    tend = RobotTendrils(world.handles)
    babble_robot(world, tend, steps=8000, rng=np.random.default_rng(7))
    world.reset(pose=(1, 1, 0))
    interp = Interpreter(Commander(world, tend))

    print("The robot's LiDAR map is labelled with landmarks:")
    for nm, cell in world.landmarks.items():
        print(f"    {nm:8s} at {cell}")
    print("\nIts mouth may ONLY output  CALL: <verb> <target>  from a closed vocabulary.")
    print("Everything after the CALL -- grounding + the {primitive} chain -- is the planner.\n")

    script = ["please make the robot go up the stairs", "now head to the door",
              "go to the charger", "walk to the far corner", "go to the middle of the room",
              "cross the room to the other side", "what's the weather"]
    for text in script:
        world.reset(pose=(1, 1, 0))
        res = interp.run(text)
        print("=" * 70)
        print(f'you>  "{text}"')
        print(f"mouth> {res['call'] or '(nothing -- not in vocabulary)'}")
        if res.get("chain"):
            print(f"body>  {res['chain']}")
            print(f"       ({'reached the goal' if res['ok'] else 'partial'}"
                  + (f", climbed {world.stairs_height} steps to level {world.level}"
                     if res['call'] and 'climb' in res['call'] else "") + ")")
        else:
            print(f"body>  {res['say']}")
    print("=" * 70)
    print("\nThe mouth never emitted a word outside the vocabulary. 'what's the weather' produced no")
    print("CALL at all -- which is the safety of constraining the output: it cannot invent an action.")


if __name__ == "__main__":
    main()
