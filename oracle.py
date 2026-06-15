#!/usr/bin/env python3
"""Talk to the two from-scratch minds: Apollo (trained on modern prose) and Arthur I
(trained only on classical/archaic English — Shakespeare, Milton, the King James Bible).

Give it a few words and it continues them in its own learned voice. Same brain, different
libraries — so they sound like different people. Nothing is hardcoded; it's all learned.

  speak as apollo / speak as arthur    switch voice
  both <text>                          hear both continue the same words
  apollo: <text>  /  arthur: <text>    a one-off in that voice
  quit
"""
import os

import torch

from gm.lm import load

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"apollo": ("apollo.pt", "Apollo"), "arthur": ("arthur.pt", "Arthur I")}


def go(mind, prompt, n=240, temp=0.8):
    model, coder, _ = mind
    ids = coder.encode(prompt) or [coder.stoi.get("\n", 0)]
    out = coder.decode(model.generate(torch.tensor([ids]), n, temp=temp)[0])
    return out.strip()


def main():
    minds = {}
    for key, (ck, disp) in NAMES.items():
        path = os.path.join(HERE, ck)
        if os.path.exists(path):
            model, coder = load(path)
            minds[key] = (model, coder, disp)
    if not minds:
        print("No trained models yet — run train_lm.py first.")
        return
    speaker = "arthur" if "arthur" in minds else next(iter(minds))
    print("Two minds, grown from different books: " + ", ".join(m[2] for m in minds.values()) + ".")
    print("Give me a few words and I'll go on.  ('speak as apollo', 'both <text>', 'quit')\n")

    while True:
        try:
            s = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not s:
            continue
        low = s.lower()
        if low in ("quit", "exit", "bye"):
            break
        if low.startswith("speak as ") and low.split()[-1] in minds:
            speaker = low.split()[-1]
            print(f"[now speaking as {minds[speaker][2]}]\n"); continue
        if low.startswith("both "):
            for key in minds:
                print(f"{minds[key][2]}: {go(minds[key], s[5:])}\n")
            continue
        chosen = None
        for key in minds:
            if low.startswith(key + ":"):
                chosen, s = key, s.split(":", 1)[1].strip()
                break
        mind = minds[chosen or speaker]
        print(f"{mind[2]}: {go(mind, s)}\n")


if __name__ == "__main__":
    main()
