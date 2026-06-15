#!/usr/bin/env python3
"""Check the memory layer: naming, remembering, recall, and thread context."""
import os
import sys

from gm.chat import Chat
from gm.lm import load

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = sys.argv[1] if len(sys.argv) > 1 else "apollo.pt"

SCRIPT = [
    "my name is Max",
    "what is my name?",
    "your name is Apollo",
    "what is your name?",
    "remember that the red box goes on the shelf",
    "remember that I like coffee",
    "what do you remember?",
    "move the red box",          # still a command
    "make it move forward efficiently",   # still a goal
    "forget everything",
    "what do you remember?",
]


def main():
    voices = {"apollo": load(os.path.join(HERE, CKPT))}
    path = os.path.join(HERE, "_memtest.json")
    chat = Chat(voices, "apollo", path)
    chat.bot_name = chat.user_name = None
    chat.notes, chat.history = [], []
    for p in SCRIPT:
        print(f"you> {p}\n  bot: {chat.reply(p)}")
    try:
        os.remove(path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
