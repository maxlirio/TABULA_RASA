#!/usr/bin/env python3
"""Three-way check of the unified brain: does it chat naturally, handle commands
(act/clarify), AND translate goals into reward specs?"""
import sys

from gm.lm import load
from gm.mind import Mind
import gm.agent as A

CKPT = sys.argv[1] if len(sys.argv) > 1 else "apollo.pt"

CHAT = ["hello", "how are you", "what is your name", "do you like me"]
CMDS = ["move the red box", "lift it", "push the door", "do the thing with the rope"]
GOALS = ["make it move forward efficiently", "teach it to walk smoothly",
         "make it move forward without falling", "make it better"]

m = Mind()
m._voices = {"apollo": load(CKPT)}
m.voice = "apollo"
for title, qs in (("CHAT", CHAT), ("COMMANDS", CMDS), ("GOALS -> REWARD", GOALS)):
    print(f"\n=== {title} ===")
    for q in qs:
        print(f"you> {q}\n bot> {A._voice_reply(m, q)}")
