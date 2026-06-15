"""The language brain — the part that turns a MESSAGE (what it means to convey) into WORDS.

This is deliberately separate from the part that decides WHAT to say. A message is a small
structured intent, e.g. ('ask_can', {'subj':'robin','verb':'fly','known':'bird'}). realize()
strings words together to express it. Because word-choice lives here and pulls from the
learned lexicon, teaching the bot new words changes how it can say the same message — the
meaning is fixed, the wording is generated.

It's simple realization, not fluent free generation (that needs an LLM) — but the meaning
and the wording are genuinely separate, which is the whole point.
"""
from __future__ import annotations


def _pick(key, options):
    """Choose one phrasing deterministically from the content, so the bot doesn't always
    say things the exact same way, yet stays reproducible (no randomness)."""
    h = sum(ord(c) for c in str(key))
    return options[h % len(options)]


def _word_for(mind, concept):
    """Pick the word to express a concept. Today the concept IS a word, but if the user
    has taught a synonym the bot 'prefers' (most recently learned), it can use that — so
    new vocabulary actually shows up in what it says."""
    pref = getattr(mind, "prefer", {}) if mind else {}
    return pref.get(concept, concept)


def realize(mind, msg):
    """Turn a structured message into a sentence."""
    act = msg[0]
    d = msg[1] if len(msg) > 1 else {}

    if act == "ack_command":
        a = _word_for(mind, d["action"])
        tgt = d.get("target")
        extra = f" toward the {tgt}" if tgt else ""
        lead = f"I think you want me to {a} — okay, " if d.get("approx") else ""
        return _pick(a, [f"{lead}I'll {a}{extra}! (doing it)",
                         f"{lead}Sure — I'll {a}{extra}! (doing it)",
                         f"{lead}Alright, I'll {a}{extra}! (doing it)"])
    if act == "ask_can":
        v = _word_for(mind, d["verb"])
        return f" I know a {d['known']} can {v} — so can a {d['subj']} {v} too?"
    if act == "ask_has":
        return f" I know a {d['known']} has {d['part']} — does a {d['subj']} have {d['part']} too?"
    if act == "curious_do":
        return f"I've been wondering — what can a {d['subj']} do?"
    if act == "curious_have":
        return f"I've been wondering — what does a {d['subj']} have?"
    if act == "curious_word":
        return f"I've been wondering — what is a {d['word']}, really? Can you tell me about it?"
    if act == "curious_none":
        return "I'm curious about lots of things! Teach me something and I'll wonder about it."
    if act == "inferred":
        return f"I thought so! I worked out that a {d['subj']} {d['rel']} {d['val']} all by myself."
    # --- social ---
    if act == "greet":
        who = f" I'm {d['name']}." if d.get("name") else ""
        return _pick(d.get("name") or "", [
            f"Hello!{who} Teach me words and I'll learn them.",
            f"Hi there!{who} Teach me words and I'll remember them.",
            f"Hey!{who} Teach me new words and I'll learn them."])
    if act == "named":
        return f"Thank you! I'll be {d['name']} from now on. Nice to meet you!"
    if act == "my_name":
        return f"I'm {d['name']}!" if d.get("name") else \
            "I don't have a name yet — type 'your name is ...' to give me one!"
    if act == "thanks":
        return _pick(str(len(getattr(mind, "corpus", []) or [])),
                     ["You're welcome!", "Anytime — you're welcome!", "Of course, you're welcome!"])
    # --- informing / answering, with word-choice from the lexicon ---
    if act == "inform_is":
        s, o = _word_for(mind, d["subj"]), _word_for(mind, d["obj"])
        if d.get("neg"):
            return _pick(s, [f"Got it — no {s} is {o}.", f"Okay — no {s} is {o}.",
                             f"Right, no {s} is {o}."])
        return _pick(s, [f"Got it — every {s} is {o}.", f"Okay — every {s} is {o}.",
                         f"Right, every {s} is {o}."])
    if act == "similar":
        return f"'{d['word']}' feels similar to: {', '.join(d['neighbors'])}."
    if act == "can_list":
        return f"A {d['subj']} can: {', '.join(_word_for(mind, a) for a in d['acts'])}."
    if act == "has_list":
        return f"A {d['subj']} has: {', '.join(d['parts'])}."
    if act == "know_about":
        bits = ", and ".join(d["bits"])
        return _pick(d["subj"], [f"Oh, {d['subj']}! I know {bits}.",
                                 f"Ah, {d['subj']} — I know {bits}.",
                                 f"Right, {d['subj']}! I know {bits}."])
    if act == "ask_pos":
        return (f"I don't know the word '{d['word']}'. What kind of word is it — a thing (noun), "
                f"an action (verb), a describing word (adjective), or another kind (adverb, "
                f"pronoun, preposition, conjunction, interjection)?")
    if act == "cant_do":
        a = d["action"]
        return f"I don't know how to {a}. Teach me — say 'you can {a}'."
    if act == "know_word":
        w = d["word"]
        return f"Yes, I know '{w}'." if d.get("known") else \
            f"No, I don't know '{w}' yet — teach me by using it."
    if act == "opposite_of":
        return f"The opposite of {d['word']} is {d['opp']}." if d.get("opp") else \
            f"I don't know the opposite of '{d['word']}' yet."
    return "..."
