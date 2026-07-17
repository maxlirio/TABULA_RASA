#!/usr/bin/env python3
"""Training data that teaches the LM to emit the RECONCILED robot command vocabulary.

Reconciled with ROBOT_EXPERIMENT/mujoco/robot_bridge.py: every target below is either a robot_bridge
skill VERBATIM (cross / stand / recover / go forward|back|left|right) or `goto <place>`, which
suit/robot_adapter.py expands into robot_bridge's `go to <x> <y>` waypoints. So the LM is trained to
speak exactly the grammar the robot already accepts -- no parallel invented vocabulary.

Every example is  USER: <natural request>  /  BOT: <symbolic command>.  Trained on this, the mouth
turns "please take the robot up the stairs" into `goto stairs` instead of confabulating, the same
way it learned tool CALLs. The adapter + supervisor do everything after.

    python3 prep_robotcmd.py            # -> data/robotcmd/chat.txt
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

# (symbolic command, [phrasings]) -- many surface forms per target so the model learns the MAPPING.
TARGETS = {
    "cross":         ["cross the table", "cross the gap", "get across the table", "cross over",
                      "walk across the table", "go across the gap"],
    "recover":       ["get up", "recover", "stand back up", "get back on your feet", "you fell, get up"],
    "stand":         ["stand", "stop", "hold still", "stay there", "stand still", "halt", "freeze"],
    "go forward":    ["go forward", "step forward", "move forward", "go ahead", "forward a bit"],
    "go back":       ["go back", "back up", "step back", "move backward", "reverse"],
    "go left":       ["go left", "step left", "shuffle left", "move left"],
    "go right":      ["go right", "step right", "shuffle right", "move right"],
    "goto stairs":   ["go up the stairs", "climb the stairs", "head to the stairs", "go to the steps",
                      "take the stairs", "get to the stairs", "up the stairs"],
    "goto door":     ["go to the door", "head to the door", "walk to the doorway", "get to the door"],
    "goto table":    ["go to the table", "head to the table", "walk over to the table"],
    "goto window":   ["go to the window", "head to the window", "over to the window"],
    "goto charger":  ["go to the charger", "go charge", "return to the charger", "go get charged",
                      "dock at the charger"],
    "goto far corner": ["go to the far corner", "walk to the opposite corner", "head to the farthest corner"],
    "goto near corner": ["go to the nearest corner", "head to the closest corner", "go to the corner"],
    "goto middle":   ["go to the middle", "head to the center of the room", "stand in the middle"],
    "goto other side": ["go to the other side", "cross the room", "head to the far side", "other end"],
    "goto start":    ["go back to the start", "return home", "go back home", "return to base"],
    "goto north":    ["go to the north wall", "head north"],
    "goto south":    ["go to the south wall", "head south"],
    "goto east":     ["go to the east wall", "head east"],
    "goto west":     ["go to the west wall", "head west"],
}
# Compositional wrappers, so the model learns the MAPPING from thousands of UNIQUE surface forms
# rather than memorising a few hundred repeated ones (the lookup-vs-generalisation trap that sank
# reward design). prefix x phrasing x suffix multiplies genuine variety.
PREFIX = ["", "hey ", "ok ", "so ", "alright ", "please ", "can you ", "could you ",
          "i want you to ", "i need you to ", "would you ", "robot, ", "now "]
SUFFIX = ["", " please", " now", " for me", " when you get a chance", " thanks", " right now",
          " ok", " already"]
# NEGATIVES: chatter that is NOT a command -> the model replies plainly, emits no command
NEGATIVES = [("hello there", "hi!"), ("how are you", "i'm good, thanks!"),
             ("what's your name", "i'm the robot."), ("thanks", "you're welcome!"),
             ("what's up", "not much!"), ("good job", "thank you!"),
             ("what can you do", "i can walk to places you name and cross the table.")]


def main(seed=0, cap=7000):
    r = random.Random(seed)
    pairs = set()
    for target, phrases in TARGETS.items():
        for p in phrases:
            for pre in PREFIX:
                for suf in SUFFIX:
                    text = (pre + p + suf).strip()
                    text = text[0].upper() + text[1:] if r.random() < 0.15 else text  # some capitalised
                    pairs.add((text, target))
    pairs = list(pairs)
    r.shuffle(pairs)
    if len(pairs) > cap:
        pairs = pairs[:cap]
    out = [f"USER: {t}\nBOT: {c}" for t, c in pairs]
    for text, reply in NEGATIVES:                              # negatives across the same wrappers
        for pre in PREFIX[:6]:
            out.append(f"USER: {(pre + text).strip()}\nBOT: {reply}")
    r.shuffle(out)
    d = os.path.join(HERE, "data", "robotcmd")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "chat.txt"), "w") as f:
        f.write("\n\n".join(out) + "\n")
    uniq = len(set(l.split("\n")[0] for l in out))
    print(f"[robotcmd] {len(out):,} examples ({uniq:,} unique requests) across {len(TARGETS)} "
          f"robot_bridge-compatible targets -> data/robotcmd/chat.txt")


if __name__ == "__main__":
    main()
