#!/usr/bin/env python3
"""Training data that teaches the LM to emit CONSTRAINED CALLs for the robot (suit/vocab.py).

Every example is  USER: <a natural request>  /  BOT: CALL: <verb> <target>  where verb+target are
from the closed vocabulary. Trained on this, the mouth learns to turn "please take the robot up the
stairs" into `CALL: climb stairs` instead of the confabulation it produces today -- the same way it
learned tool-use CALLs. The grounded interpreter does everything after the CALL, so the LM only ever
has to get the CALL right.

    python3 prep_robotcmd.py            # -> data/robotcmd/chat.txt

Add to t4_run.py's append list to include it in the next run.
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

# (target, [phrases that should map to it]) -- many surface forms per target, so the model learns the
# MAPPING, not a single template. The verb is fixed per target (climb for stairs, goto otherwise).
TARGETS = {
    ("climb", "stairs"): ["go up the stairs", "climb the stairs", "take the stairs up",
                          "head upstairs", "ascend the staircase", "walk up the steps",
                          "get to the top of the stairs", "go up to the next floor",
                          "climb up", "take the robot up the stairs", "up the stairs"],
    ("goto", "door"): ["go to the door", "head to the door", "walk to the doorway",
                       "approach the door", "get to the door", "move to the door"],
    ("goto", "table"): ["go to the table", "walk over to the table", "head to the table",
                        "approach the table", "get to the table"],
    ("goto", "window"): ["go to the window", "head to the window", "walk to the window",
                         "over to the window"],
    ("goto", "charger"): ["go to the charger", "go charge", "return to the charger",
                          "head to the charging dock", "go get charged", "dock at the charger"],
    ("goto", "far_corner"): ["go to the far corner", "walk to the opposite corner",
                             "head to the farthest corner", "get to the other corner"],
    ("goto", "near_corner"): ["go to the nearest corner", "head to the closest corner",
                              "walk to the corner"],
    ("goto", "middle"): ["go to the middle", "head to the center of the room", "go to the centre",
                         "stand in the middle", "move to the center"],
    ("goto", "other_side"): ["go to the other side", "cross the room", "head to the far side",
                             "get to the other end", "walk across the room"],
    ("goto", "start"): ["go back to the start", "return home", "go back home",
                        "head back to where you started", "return to base"],
    ("goto", "north"): ["go north", "head north", "move to the north wall", "go up"],
    ("goto", "south"): ["go south", "head south", "move to the south wall", "go down"],
    ("goto", "east"): ["go east", "head east", "move to the east wall", "go right"],
    ("goto", "west"): ["go west", "head west", "move to the west wall", "go left"],
}
# polite / indirect wrappers, so the model isn't thrown by "please", "can you", "I want you to", etc.
WRAP = ["{p}", "please {p}", "can you {p}", "could you {p}", "i want you to {p}", "robot, {p}",
        "now {p}", "{p} please", "i need you to {p}", "would you {p}"]
# a few NEGATIVES: chatter that is NOT a command -> the model must NOT emit a CALL (it replies plainly)
NEGATIVES = [("hello there", "hi!"), ("how are you", "i'm good, thanks!"),
             ("what's your name", "i'm the robot."), ("thanks", "you're welcome!"),
             ("tell me a story", "once, a small robot learned to walk."),
             ("what can you do", "i can go places you name, and climb the stairs.")]


def main(seed=0):
    r = random.Random(seed)
    out = []
    for (verb, target), phrases in TARGETS.items():
        for p in phrases:
            for w in WRAP:
                out.append(f"USER: {w.format(p=p)}\nBOT: CALL: {verb} {target}")
    for text, reply in NEGATIVES:
        for w in WRAP[:5]:
            out.append(f"USER: {w.format(p=text)}\nBOT: {reply}")
    r.shuffle(out)
    d = os.path.join(HERE, "data", "robotcmd")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "chat.txt"), "w") as f:
        f.write("\n\n".join(out) + "\n")
    print(f"[robotcmd] {len(out):,} command->CALL examples "
          f"({len(TARGETS)} targets, {len(WRAP)} phrasings) -> data/robotcmd/chat.txt")


if __name__ == "__main__":
    main()
