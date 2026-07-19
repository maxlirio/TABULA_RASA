#!/usr/bin/env python3
"""Talk to the from-scratch brain. It chats, takes commands, turns goals into reward specs,
and REMEMBERS — names, things you ask it to remember, and the recent thread.

No large language model and (now) no old symbolic engine: chat goes straight to the tiny
neural voice (Apollo) plus a small, exact memory layer (see gm/chat.py).

  what is your name / my name is X     naming
  remember that ...                    save something
  what do you remember                 recall
  quit                                 exit
"""
import os

from gm.chat import Chat

HERE = os.path.dirname(os.path.abspath(__file__))
MEM = os.path.join(HERE, "chatmem.json")

HELLO = ("Hi - I'm Apollo, a tiny from-scratch brain. Just talk to me, give me a command, or "
         "give me a goal like 'make it move forward efficiently' and I'll turn it into a "
         "reward. I'll remember what you tell me. ('quit' to go.)")


def main():
    import sys

    from gm.lm import load
    # optional path, so you can talk to a specific brain without shuffling 1GB files around:
    #     python3 talk.py apollo_280m.pt
    # default to the newest trained brain present (run #4 added robot commands, kept stories+tools).
    if len(sys.argv) > 1:
        apollo = sys.argv[1]
    else:
        apollo = next((os.path.join(HERE, f) for f in ("apollo_run4.pt", "apollo_run3.pt", "apollo.pt")
                       if os.path.exists(os.path.join(HERE, f))), os.path.join(HERE, "apollo.pt"))
    if not os.path.exists(apollo):
        print(f"No brain at {apollo}. Train one:  python3 train_lm.py mixed apollo.pt")
        return
    model, coder = load(apollo)
    chat = Chat({"apollo": (model, coder)}, "apollo", MEM)
    import time as _t
    params = sum(p.numel() for p in model.parameters())
    built = _t.strftime("%Y-%m-%d %H:%M", _t.localtime(os.path.getmtime(apollo)))
    print(f"[brain: {params/1e6:.1f}M params, vocab {len(coder.tokens) if hasattr(coder,'tokens') else len(coder.chars)}, trained {built}]")
    print(HELLO + "\n")

    while True:
        try:
            s = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not s:
            continue
        if s.lower() in ("quit", "exit", "bye", "goodbye"):
            print((chat.bot_name or "apollo") + ": bye!")
            break
        print((chat.bot_name or "Apollo") + ": " + chat.reply(s))


if __name__ == "__main__":
    main()
