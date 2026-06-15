#!/usr/bin/env python3
"""Regression suite for the talking bot — naming, small talk, learning by usage
(embeddings), and reasoning over taught ideas (definitions + contradictions)."""
from gm.mind import Mind
from gm.agent import respond


def run(setup, sentence):
    m = Mind()
    m.bootstrap()
    for s in setup:
        respond(m, s)
    return respond(m, sentence)


CASES = [
    # naming
    ([], "your name is sprout", "I'll be Sprout"),
    (["your name is sprout"], "what is your name?", "Sprout"),
    ([], "who are you?", "don't have a name yet"),
    # small talk
    ([], "hello", "Teach me"),
    ([], "thank you", "welcome"),
    ([], "what can you do?", "learn"),
    # learning by usage -> similarity (the seed primer makes these stable)
    ([], "what is bird like?", "fish"),
    ([], "what is run like?", "sit"),       # clusters with actions (sit/sleep/rest/jump)
    ([], "is a cat like a dog?", "similar"),
    # teaching ideas + reasoning (the kept engine)
    (["a robin is a bird"], "is a robin a bird?", "every robin is bird"),
    (["light is a word", "dark is when there is no light"],
     "what is dark?", "dark is when not light"),
    (["light is a word", "dark is when there is no light"],
     "can there be a dark?", "could be a dark"),
    (["light is a word", "dark is when there is no light",
      "night is when it is dark and there are stars"],
     "stars are light", "could never actually exist"),
    (["light is a word", "dark is when there is no light",
      "night is when it is dark and there are stars"],
     "stars are light", "both light and not light"),
    # entailment / derivation
    (["light is a word", "dark is when there is no light", "stars are light"],
     "are stars dark?", "never dark"),
    (["cats are mammals", "mammals are animals"],
     "cats are not animals", "both animal and not animal"),
    # tell me about (blends usage + definitions)
    (["light is a word", "dark is when there is no light"],
     "tell me about dark", "not light"),
    # a definition immediately shapes the learned-from-usage space (neuro-symbolic bridge)
    (["a robin is a bird"], "what is robin like?", "bird"),
    (["a wug is a fish"], "what is wug like?", "fish"),
    # synonyms pull words together
    (["big means large"], "is big like large?", "similar"),
    ([], "good is another word for nice", "another word for"),
    # opposites: symbolic knowledge overrides the fuzzy embedding
    (["hot is the opposite of cold"], "is hot like cold?", "opposites"),
    (["a wug is a small bird"], "what is wug like?", "bird"),    # multi-word fact
    # natural phrasings: contractions + new query forms
    ([], "what's a bird like?", "fish"),
    (["hot is the opposite of cold"], "what is the opposite of cold?", "hot"),
    ([], "do you know fish?", "I know"),
    ([], "do you know zebra?", "don't know"),
    ([], "what goes with run?", "sit"),
    # multi-word PHRASE learning: a word-pair used a lot becomes its own unit
    (["a good dog runs"] * 5 + ["a cat walks fast"] * 16,
     "what is good dog like?", "feels similar"),    # 'good dog' became its own unit
    # has / part-of relations
    (["a bird has wings"], "what does a bird have?", "wing"),
    (["a wing is part of a bird"], "what is a wing part of?", "bird"),
    (["a bird has wings", "a bird has feathers"], "tell me about bird", "has: wing"),
    # forgetting clears what you taught
    (["a bird can fly", "forget bird"], "can a bird fly?", "don't know what a bird can do"),
    (["a robin is a bird", "forget robin"], "what do you know?", "picked up"),
    # it ACTIVELY asks about a word it doesn't know, and USES what it has learned
    ([], "i have a quux", "don't know the word 'quux'"),
    (["a bird can fly", "a bird has wings"], "i saw a bird", "I know"),
    # COMMAND understanding — say something to do, it understands
    ([], "run", "I'll run"),
    ([], "please jump", "I'll jump"),
    (["you can hop"], "hop", "I'll hop"),                 # teach a new action, then obey
    (["jog means run"], "please jog", "I'll jog"),         # a synonym of an action is commandable
    ([], "please florp", "don't know how to florp"),      # unknown action -> asks to be taught
    # PART OF SPEECH asked first (it can't tell noun/verb/adjective on its own)
    ([], "i have a zebra", "what kind of word"),       # asks part of speech first
    (["florp", "a verb"], "florp", "I'll florp"),       # told it's a verb -> becomes a command
    (["florp", "an action"], "florp", "I'll florp"),       # told it's a verb -> becomes a command
    ([], "i feel glum", "describing word"),
    # 'X is a verb' tags part of speech (not 'every X is a verb'); 'X is something you Y' is a definition
    ([], "ask is a verb", "is a verb"),
    ([], "a question is something you ask", "kind of thing"),
    ([], "a cat is an animal", "every cat is animal"),     # still a real fact
    # it ACTIVELY asks its own follow-up question about what you just taught
    ([], "a cat is an animal", "what can a cat do"),
]


def embedding_checks():
    """The embedding must learn structure from usage: like words cluster."""
    m = Mind()
    m.bootstrap()
    ok = True
    pairs = [("bird", "fish", "red"), ("run", "walk", "blue"), ("dog", "cat", "yellow")]
    for a, near, far in pairs:
        cn, cf = m.embed.cosine(a, near), m.embed.cosine(a, far)
        good = cn is not None and cf is not None and cn > cf
        print(f"[{'PASS' if good else 'FAIL'}] {a}~{near} ({cn:.2f}) > {a}~{far} ({cf:.2f})")
        ok = ok and good
    return ok


def main():
    passed = 0
    for setup, q, want in CASES:
        got = run(setup, q)
        ok = want.lower() in got.lower()
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {q!r}\n        -> {got!r}"
              + ("" if ok else f"\n        WANTED ~ {want!r}"))
    emb = embedding_checks()
    print(f"\n{passed}/{len(CASES)} dialogue cases passed; embedding {'OK' if emb else 'FAILED'}.")
    return passed == len(CASES) and emb


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
