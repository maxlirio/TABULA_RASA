#!/usr/bin/env python3
"""Teach the brain the intents it was MISSING, so it stops forcing every input into a
command or reward. Two consistent-frame families:

  - STATEMENTS (someone tells it a fact / how they feel) -> a light acknowledgement,
    NOT a command. ("a plane flies" should get "oh, neat!", not "i will paint the file".)
  - CORRECTIONS / NEGATIONS ("no", "stop", "don't do that", "that's wrong") -> handled
    as a correction, not parsed as a new command/goal.

Note (honest scope): the neural brain ACKNOWLEDGES statements; it does not store them as a
knowledge base — fact-teaching was the old symbolic engine's job and isn't part of the
language->reward goal. Output: data/misc/chat.txt.
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

NOUNS = ["plane", "dog", "bird", "car", "fish", "tree", "river", "sun", "cat", "boat",
         "horse", "cloud", "fire", "wind", "rock"]
VERBS = ["flies", "runs", "swims", "moves", "falls", "grows", "shines", "sleeps",
         "floats", "burns", "jumps", "sings"]
ADJS = ["big", "small", "fast", "slow", "blue", "red", "tall", "cold", "warm", "bright",
        "heavy", "soft", "loud", "round"]
FEELINGS = ["tired", "happy", "bored", "hungry", "okay", "good", "sad", "excited"]

ACK = ["oh, neat!", "good to know.", "interesting!", "okay, got it.",
       "thanks for telling me.", "huh, okay.", "makes sense.", "i see."]

CORRECTIONS = [
    (["no", "nope", "not that", "wrong", "that's wrong", "that is wrong",
      "no not that", "that's not right"],
     ["okay - my mistake.", "sorry, what did you mean?", "got it, i'll fix that.",
      "oops - what should it be?"]),
    (["stop", "stop it", "halt", "wait", "hold on"],
     ["okay, stopping.", "alright, i'll wait.", "okay, paused."]),
    (["don't do that", "do not do that", "no don't do that", "don't"],
     ["okay, i won't.", "alright, i'll leave it.", "understood, not doing that."]),
    (["cancel", "never mind", "nevermind", "forget it", "undo that"],
     ["okay, cancelled.", "no problem, forget it.", "alright, undone."]),
]


def frames(r):
    n, v, a, f = (r.choice(NOUNS), r.choice(VERBS), r.choice(ADJS), r.choice(FEELINGS))
    out = [
        (f"a {n} {v}", r.choice(ACK)),
        (f"the {n} is {a}", r.choice(ACK)),
        (f"{n}s are {a}", r.choice(ACK)),
        (f"i like {n}s", r.choice(["nice!", "me too!", "cool.", "good to know."])),
        (f"i am {f}", r.choice(["okay.", "thanks for sharing.", "i hear you.", "got it."])),
    ]
    for ways, replies in CORRECTIONS:
        out.append((r.choice(ways), r.choice(replies)))
    return out


def main(n=900, seed=5):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        for u, b in frames(r):
            out.append(f"USER: {u}\nBOT: {b}\n")
    r.shuffle(out)
    out_dir = os.path.join(HERE, "data", "misc")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "chat.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"[misc] {len(out):,} turns -> {path} ({os.path.getsize(path):,} bytes)")


if __name__ == "__main__":
    main()
