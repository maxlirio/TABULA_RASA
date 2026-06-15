#!/usr/bin/env python3
"""Expand the bot's real-world KNOWLEDGE using WordNet (a public lexical database, not a
pretrained model). For common nouns we pull:
  - is-a facts from hypernyms      ("a dog is a mammal", "a mammal is an animal")
  - has facts from part-meronyms   ("a car has wheels", "a tree has leaves")
into both USER/BOT teaching frames (training data) and a facts JSON that can preload the
knowledge graph so it KNOWS things out of the box. Use only when the model no longer babbles.
Output: data/knowledge/chat.txt  +  base_facts.json
"""
import json
import os

from nltk.corpus import wordnet as wn

HERE = os.path.dirname(os.path.abspath(__file__))
STOP = {"the", "of", "and", "to", "a", "an", "in", "is", "it", "for", "on", "as", "be"}


def lemma(syn):
    return syn.lemmas()[0].name().replace("_", " ").lower()


def best_noun(w):
    """Dominant noun sense, and only if that sense is actually used (count>0) — skips obscure
    homonyms like have->'haves'(person), was, will, can->container."""
    best, bc = None, 0
    for s in wn.synsets(w, pos=wn.NOUN):
        for l in s.lemmas():
            if l.name().lower() == w and l.count() > bc:
                best, bc = s, l.count()
    return best


def main(max_words=5000):
    common = [w.strip().lower() for w in open(os.path.join(HERE, "common10k.txt"))]
    frames, facts, seen = [], [], set()
    for w in common[:max_words]:
        if w in STOP or not w.isalpha() or len(w) < 3:
            continue
        syn = best_noun(w)
        if not syn:
            continue
        # is-a (one hypernym level — the immediate category)
        hyps = syn.hypernyms()
        if hyps:
            parent = lemma(hyps[0])
            if parent != w and " " not in parent and (w, "isa", parent) not in seen:
                seen.add((w, "isa", parent))
                facts.append([w, "isa", parent])
                frames.append(f"USER: is a {w} a kind of {parent}?\nBOT: yes, a {w} is a {parent}.\n")
                frames.append(f"USER: what is a {w}?\nBOT: a {w} is a {parent}.\n")
        # has (parts)
        for part in syn.part_meronyms()[:2]:
            p = lemma(part)
            if " " not in p and (w, "has", p) not in seen:
                seen.add((w, "has", p))
                facts.append([w, "has", p])
                frames.append(f"USER: does a {w} have a {p}?\nBOT: yes, a {w} has a {p}.\n")

    out_dir = os.path.join(HERE, "data", "knowledge")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chat.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(frames))
    json.dump(facts, open(os.path.join(HERE, "base_facts.json"), "w"))
    print(f"[knowledge] {len(facts):,} real-world facts -> base_facts.json + "
          f"{len(frames):,} teaching frames -> data/knowledge/chat.txt")


if __name__ == "__main__":
    main()
