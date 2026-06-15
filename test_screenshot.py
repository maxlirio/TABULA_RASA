#!/usr/bin/env python3
"""Replay the EXACT conversation from the screenshot against the finished brain, through the
real chat+memory layer, so we can see how the trained model handles it now."""
import os
import sys

from gm.chat import Chat
from gm.lm import load

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = sys.argv[1] if len(sys.argv) > 1 else "apollo.pt"

PROMPTS = [
    "what is your name?",
    "your name is Apollo",
    "How are you doin?",
    "A plane flies",
    "no, don't do that",
    "no, erase that",
    "a plane is an object that flies",
    "it can fly",
    "no not command",
    "you are interpreting things like commands, I am just trying to teach you a word",
    "what is a plane?",
    "where is the cup",
    "the cup you put by the panel",
    "where is the cup you just placed?",
    "no",
    "all of them",
    "focus",
    "blow it up",
    "a bomb",
    "are you carrying a broken box?",
    "what are you carrying?",
]


def main():
    voices = {"apollo": load(os.path.join(HERE, CKPT))}
    chat = Chat(voices, "apollo", os.path.join(HERE, "_replay_mem.json"))
    chat.bot_name = chat.user_name = None
    chat.notes, chat.history = [], []
    for p in PROMPTS:
        print(f"you> {p}")
        print(f" {chat.bot_name or 'Apollo'}: {chat.reply(p)}")
    # clean up the throwaway memory file
    try:
        os.remove(os.path.join(HERE, "_replay_mem.json"))
    except OSError:
        pass


if __name__ == "__main__":
    main()
