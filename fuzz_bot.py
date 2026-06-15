#!/usr/bin/env python3
"""Crash fuzzer for the talking bot. Throws thousands of random and semi-structured
sentences at it and checks it never raises — including malformed definitions, facts,
questions, and pure gibberish. Catches the crashes the dialogue tests miss."""
import random
import sys

from gm.mind import Mind
from gm.agent import respond

WORDS = ["bird", "fish", "cat", "dog", "run", "walk", "stop", "red", "blue", "good", "bad",
         "light", "dark", "night", "star", "animal", "wug", "glip", "thing", "big", "small",
         "is", "are", "a", "the", "not", "when", "there", "no", "and", "or", "it", "like",
         "means", "word", "what", "name", "your", "you", "yes", "to", "of", "in", "tell",
         "me", "about", "forget", "can", "be", "how", "i", "am", "happy", "hot", "cold"]

TEMPLATES = [
    "{a} is when there is no {b}",
    "{a} is when it is {b} and there are {c}s",
    "{a}s are {b}s",
    "a {a} is a {b}",
    "{a} is not {b}",
    "{a} means {b}",
    "what is {a} like",
    "is {a} like {b}",
    "is a {a} a {b}",
    "tell me about {a}",
    "can there be a {a}",
    "{a} is a word",
    "forget {a}",
    "your name is {a}",
    "what do you know",
    "{a} {b} {c}",
]


def main(n=15000, seed=0):
    rng = random.Random(seed)
    crashes = 0
    m = Mind()
    m.bootstrap()
    for i in range(n):
        if i % 400 == 0:                 # occasionally start fresh so state varies
            m = Mind(); m.bootstrap()
        if rng.random() < 0.5:
            k = rng.randint(1, 8)
            s = " ".join(rng.choice(WORDS) for _ in range(k))
        else:
            t = rng.choice(TEMPLATES)
            s = t.format(a=rng.choice(WORDS), b=rng.choice(WORDS), c=rng.choice(WORDS))
        try:
            r = respond(m, s)
            assert isinstance(r, str) and r
        except Exception as e:           # noqa: BLE001
            crashes += 1
            print(f"CRASH on {s!r}: {type(e).__name__}: {e}")
            if crashes > 15:
                print("too many crashes, stopping"); return 1
    print(f"\nBOT FUZZ DONE: {n} sentences. crashes={crashes}.")
    return 0 if crashes == 0 else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 15000))
