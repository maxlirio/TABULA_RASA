#!/usr/bin/env python3
"""Talk to the from-scratch brain. It chats, takes commands, turns goals into reward specs,
and REMEMBERS — names, things you ask it to remember, and the recent thread.

No large language model and (now) no old symbolic engine: chat goes straight to the tiny
neural voice plus a small, exact memory layer (see gm/chat.py).

  speak as apollo / speak as arthur   switch voice
  what is your name / my name is X     naming
  remember that ...                    save something
  what do you remember                 recall
  quit                                 exit
"""
import os

from gm.chat import Chat

HERE = os.path.dirname(os.path.abspath(__file__))
MEM = os.path.join(HERE, "chatmem.json")

HELLO = ("Hi - I'm a tiny from-scratch brain. Just talk to me, give me a command, or give "
         "me a goal like 'make it move forward efficiently' and I'll turn it into a reward. "
         "I'll remember what you tell me. ('speak as arthur' to switch voice, 'quit' to go.)")


def load_voices():
    from gm.lm import load
    voices = {}
    for key, ck in (("apollo", "apollo.pt"), ("arthur", "arthur.pt")):
        p = os.path.join(HERE, ck)
        if os.path.exists(p):
            voices[key] = load(p)
    return voices


def main():
    voices = load_voices()
    if not voices:
        print("No trained brain found. Train one first:  python3 train_lm.py mixed apollo.pt")
        return
    voice = "apollo" if "apollo" in voices else next(iter(voices))
    chat = Chat(voices, voice, MEM)
    print(HELLO + "\n")

    while True:
        try:
            s = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not s:
            continue
        low = s.lower()
        if low in ("quit", "exit", "bye", "goodbye"):
            print((chat.bot_name or "bot") + ": bye!")
            break
        if low.startswith("speak as ") and low.split()[-1] in voices:
            chat.voice = low.split()[-1]
            print(f"[now speaking as {chat.voice.title()}]")
            continue
        print((chat.bot_name or chat.voice.title()) + ": " + chat.reply(s))


if __name__ == "__main__":
    main()
