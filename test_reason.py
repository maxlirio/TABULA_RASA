#!/usr/bin/env python3
"""Check APPLIED reasoning in chat: learn is-a facts, derive new conclusions (syllogism),
catch contradictions, recall derived facts — while chat/commands/goals still go to the LM."""
import os
import sys

from gm.chat import Chat
from gm.lm import load

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = sys.argv[1] if len(sys.argv) > 1 else "apollo.pt"

SCRIPT = [
    "every cat is a mammal",
    "every mammal is an animal",
    "is a cat an animal?",          # derived: cat -> mammal -> animal
    "is an animal a cat?",          # not reversible
    "what is a cat?",               # recall incl. derived
    "a cat is not an animal",       # contradiction -> reject
    "every bird is an animal",
    "is a bird a mammal?",          # no path -> no
    "hello",                        # -> LM
    "move the red box",             # -> LM
    "make it move forward efficiently",   # -> LM (reward)
]


def main():
    path = os.path.join(HERE, "_reasontest.json")
    c = Chat({"apollo": load(os.path.join(HERE, CKPT))}, "apollo", path)
    c.bot_name = c.user_name = None
    c.notes, c.facts, c.history = [], [], []
    for s in SCRIPT:
        print(f"you> {s}\n  bot: {c.reply(s)}")
    try:
        os.remove(path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
