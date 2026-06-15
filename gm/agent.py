"""The talking bot. It learns words two ways and answers from both:

  • by USAGE  — every sentence you type trains the embedding model, so it picks up which
                words are alike from how you use them (no definitions needed);
  • by DEFINITION — you can teach it ideas in words ('dark is when there is no light',
                'a bird is an animal'), and it reasons over them, catching contradictions.

There is no tabletop here and no large language model — just a small neural net it grows
from what you say, plus a tiny logic engine.
"""
from __future__ import annotations

from . import abstract
from . import say
from .mind import Mind

_STOP = {"it", "that", "this", "there", "they", "them", "the", "a", "an", "you", "i", "we",
         "he", "she", "what", "who", "where", "when", "why", "how", "is", "are", "was",
         "do", "does", "did", "to", "of", "on", "in", "no", "not", "and", "or", "any",
         "be", "am", "your", "my", "me", "here", "like", "similar", "mean", "means",
         "can", "could", "would", "yes", "have", "has", "had", "get", "got", "want",
         "see", "saw", "make", "made", "let", "go", "goes", "went", "come", "came",
         "take", "took", "give", "gave", "put", "say", "said", "think", "feel", "need",
         "use", "tell", "told", "know", "about", "some", "very", "really",
         "every", "all", "each", "both", "many", "few", "most", "an"}


def tokenize(s):
    s = s.lower().strip()
    for ch in "?.!,;:\"()":
        s = s.replace(ch, " ")
    out = []
    for w in s.split():
        out.extend(_CONTRACTIONS.get(w, [w]))
    return out


_CONTRACTIONS = {
    "what's": ["what", "is"], "that's": ["that", "is"], "it's": ["it", "is"],
    "here's": ["here", "is"], "there's": ["there", "is"], "who's": ["who", "is"],
    "i'm": ["i", "am"], "you're": ["you", "are"], "we're": ["we", "are"],
    "don't": ["do", "not"], "doesn't": ["does", "not"], "isn't": ["is", "not"],
    "aren't": ["are", "not"], "can't": ["can", "not"], "won't": ["will", "not"],
    "i'll": ["i", "will"], "i've": ["i", "have"], "you've": ["you", "have"],
    "let's": ["let", "us"], "they're": ["they", "are"],
}


def _sing(w):
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


_FILLER = {"now", "then", "today", "please", "here", "there", "again", "ok", "okay"}

# all eight parts of speech, with the words you might answer with
_POS_WORDS = {
    "verb": {"action", "verb", "do", "doing", "move", "movement", "command"},
    "adjective": {"describing", "describe", "adjective", "quality", "feeling", "color", "size"},
    "adverb": {"adverb", "how", "way", "manner"},
    "pronoun": {"pronoun"},
    "preposition": {"preposition", "position", "relation"},
    "conjunction": {"conjunction", "connector", "joiner", "joining", "join"},
    "interjection": {"interjection", "exclamation", "expression"},
    "noun": {"thing", "noun", "object", "animal", "person", "place", "name", "food", "stuff"},
}
_POS_ORDER = ["verb", "adjective", "adverb", "pronoun", "preposition", "conjunction",
              "interjection", "noun"]
_POS_ALL = set().union(*_POS_WORDS.values())
# only these literally name a part of speech — 'animal'/'food' are content, not grammar
_POS_STRICT = {"noun", "verb", "adjective", "adverb", "pronoun", "preposition",
               "conjunction", "interjection", "action"}


def _pos_declare(mind, t):
    """'ask is a verb' / 'a question is a noun' -> TAG the word's part of speech, NOT make
    a categorical fact like 'every ask is a verb'."""
    tt = t[1:] if t[:1] in (["a"], ["an"], ["the"]) else t
    if len(tt) < 3 or tt[1] != "is":
        return None
    word = _sing(tt[0])
    if word in _STOP:
        return None
    rest = [w for w in tt[2:] if w not in ("a", "an", "the", "just", "kind", "of", "word")]
    if rest and all(w in _POS_STRICT for w in rest):
        cat = _classify_pos(rest)
        mind.pos[word] = cat
        if cat == "verb":
            mind.actions.add(word); mind.embed.observe(["you", "can", word])
            return f"Got it — '{word}' is a verb. Tell me to {word}!"
        return f"Got it — '{word}' is a{'n' if cat[0] in 'aeiou' else ''} {cat}."
    return None


def _classify_pos(t):
    ts = set(t)
    for cat in _POS_ORDER:
        if ts & _POS_WORDS[cat]:
            return cat
    return None


_CORRECT_LEAD = {"no", "nope", "wrong", "incorrect"}


def _correction(mind, t):
    """Ways to tell it it's using a word wrong, so it can fix it instead of just absorbing."""
    low = " ".join(t)
    # 'X is not like Y'  -> they're correcting a similarity it believes
    if "not" in t and "like" in t and t[1:3] == ["is", "not"]:
        a = _sing(t[0])
        after = [w for w in t[t.index("like") + 1:] if w not in ("a", "an", "the")]
        if after and a not in _STOP:
            b = _sing(after[-1])
            mind.embed.push(a, b)
            return f"Okay — so '{a}' isn't really like '{b}'. I'll keep that in mind."
    # 'you used X wrong' / 'X doesn't mean that' / "that's not how you use X"
    if "wrong" in t or ("not" in t and ("mean" in t or "use" in t or "right" in t)):
        cands = [_sing(w) for w in t
                 if w not in _STOP and len(w) > 2
                 and w not in ("wrong", "used", "use", "using", "mean", "means", "right",
                               "incorrect", "way", "really", "actually")]
        target = cands[-1] if cands else getattr(mind, "_last_used", None)
        if target:
            mind._asking = target
            return f"Sorry — I got '{target}' wrong. What is it like, or what does it mean? " \
                   f"Tell me and I'll fix it."
    # bare 'no' / 'that's wrong' right after it said something about a word
    if (t[0] in _CORRECT_LEAD and len(t) <= 3) or low in ("that is wrong", "that is not right"):
        w = getattr(mind, "_last_used", None)
        if w:
            mind._last_used = None
            mind._asking = w
            return f"Sorry! Let me fix '{w}'. What is it like, or what does it mean?"
        return "Sorry — which word did I get wrong?"
    return None


def _ask_unknown(mind, word):
    """When it meets a word it doesn't know, ask the PART OF SPEECH first, so it knows how
    to ask the right next question (it can't tell a noun from a verb on its own)."""
    mind._asking_pos = word
    return say.realize(mind, ("ask_pos", {"word": word}))


# ---- abstract definitions: parsing a boolean condition over word-atoms -----------------

def _abs_lit(toks, i):
    neg = False
    if toks[i:i + 2] in (["there", "is"], ["there", "are"]):
        i += 2
    elif toks[i:i + 1] in (["theres"], ["there's"], ["there"]):
        i += 1
    elif toks[i:i + 2] == ["it", "is"]:
        i += 2
    elif toks[i:i + 1] in (["its"], ["it's"]):
        i += 1
    if toks[i:i + 1] in (["no"], ["not"]):
        neg = True; i += 1
    if i < len(toks) and toks[i] not in ("and", "or"):
        w = _sing(toks[i]); i += 1
        return (("not", w) if neg else w), i
    return None


def _abs_cond(toks, i):
    def conj(k):
        lit = _abs_lit(toks, k)
        if not lit:
            return None
        items, j = [lit[0]], lit[1]
        while toks[j:j + 1] == ["and"]:
            l2 = _abs_lit(toks, j + 1)
            if not l2:
                break
            items.append(l2[0]); j = l2[1]
        return (items[0] if len(items) == 1 else ("and", items)), j

    c = conj(i)
    if not c:
        return None
    nodes, j = [c[0]], c[1]
    while toks[j:j + 1] == ["or"]:
        c2 = conj(j + 1)
        if not c2:
            break
        nodes.append(c2[0]); j = c2[1]
    return (nodes[0] if len(nodes) == 1 else ("or", nodes)), j


def _render_abs(expr):
    if isinstance(expr, str):
        return expr
    if expr[0] == "not":
        return "not " + _render_abs(expr[1])
    sep = " and " if expr[0] == "and" else " or "
    return sep.join(_render_abs(e) for e in expr[1])


def _refs_of(mind, words):
    seen, stack = set(), list(words)
    while stack:
        w = stack.pop()
        if w in seen:
            continue
        seen.add(w)
        if w in mind.defs:
            sub = set(); abstract._collect(mind.defs[w], sub); stack.extend(sub)
        for a, b in mind.implications:
            if a == w:
                stack.append(b[1] if (not isinstance(b, str) and b[0] == "not") else b)
    return seen


def _contradiction_reply(mind, bad):
    culprit = next((w for w in bad if w != "__kb__"), None)
    related = _refs_of(mind, [w for w in bad if w != "__kb__"])
    ordered = ([culprit] if culprit in mind.defs else []) + \
              [w for w in sorted(related) if w in mind.defs and w != culprit]
    facts = [f"{w} is {_render_abs(mind.defs[w])}" for w in ordered]
    for a, b in mind.implications:
        bword = b[1] if (not isinstance(b, str) and b[0] == "not") else b
        if a in related or bword in related:
            facts.append(f"no {a} is {bword}" if not isinstance(b, str) and b[0] == "not"
                         else f"every {a} is {b}")
    pivot = abstract.conflict_atom(culprit, mind.defs, mind.implications) if culprit else None
    pin = f" — a {culprit} would have to be both {pivot} and not {pivot}" if pivot else ""
    lead = (f"Wait — that doesn't add up. A {culprit} could never actually exist"
            if culprit else "Wait — that can't all be true at once")
    return f"{lead}{pin}. You told me {'; '.join(facts)}. Which did you mean?"


def _abstract(mind, sentence):
    """Definitions, facts, contradiction checks, and questions over taught ideas.
    Returns a reply, or None to let other handling take over."""
    t = tokenize(sentence)
    if len(t) < 3:
        return None

    if t[1] == "is" and t[-1] in ("word", "idea", "concept") and _sing(t[0]) not in _STOP:
        mind.atoms.add(_sing(t[0]))
        return f"OK — I'll remember the word '{_sing(t[0])}'."

    if t[1] == "is" and t[2] == "when" and _sing(t[0]) not in _STOP:
        word = _sing(t[0])
        c = _abs_cond(t, 3)
        if not c or c[1] != len(t):
            return None
        expr = c[0]
        refs = set(); abstract._collect(expr, refs)
        unknown = [r for r in sorted(refs)
                   if r not in mind.defs and r not in mind.atoms]
        if unknown:
            mind._pending_abstract = {"sentence": sentence}
            u = unknown[0]
            return f"I don't know what {u} is. What is {u}? " \
                   f"(e.g. '{u} is a word', or '{u}s are something-else')."
        old = dict(mind.defs)
        mind.defs[word] = expr
        bad = abstract.contradictions(mind.defs, mind.implications)
        if bad:
            reply = _contradiction_reply(mind, bad)
            mind.defs = old
            return reply
        return f"Got it — {word} is when {_render_abs(expr)}."

    # fact: 'a raven is a bird' / 'ravens are [not] birds'
    if ("is" in t or "are" in t) and (t.count("is") + t.count("are")) == 1:
        verb = "is" if "is" in t else "are"
        vi = t.index(verb)
        left = [x for x in t[:vi] if x not in ("a", "an", "the")]
        right = t[vi + 1:]
        neg = right[:1] == ["not"]
        if neg:
            right = right[1:]
        had_article = any(x in ("a", "an", "the") for x in t[:vi]) or right[:1] in (["a"], ["an"])
        # 'a question is something you ask' -> a noun tied to a verb, NOT 'every question is ask'
        if right[:1] in (["something"], ["anything"], ["someone"], ["somewhere"]) and len(left) == 1:
            a, vb = _sing(left[-1]), _sing(right[-1])
            if a not in _STOP:
                mind.pos[a] = "noun"
                if mind.embed.knows(vb):
                    mind.embed.pull(a, vb, 0.3)
                return f"Got it — a {a} is a kind of thing (you {vb} it)."
        right = [x for x in right if x not in ("a", "an", "the")]
        # 'a wug is a small bird' -> head noun is the last word (bird)
        if len(left) == 1 and len(right) >= 1 and (len(right) == 1 or had_article):
            a, b = _sing(left[0]), _sing(right[-1])
            cons = ("not", b) if neg else b
            if a != b and a not in _STOP and b not in _STOP \
                    and (a in mind.atoms or a in mind.defs or b in mind.atoms
                         or b in mind.defs or verb == "are" or had_article):
                mind.atoms.add(a); mind.atoms.add(b)
                old = list(mind.implications)
                mind.implications = old + [(a, cons)]
                bad = abstract.contradictions(mind.defs, mind.implications)
                if bad:
                    reply = _contradiction_reply(mind, bad)
                    mind.implications = old
                    return reply
                if not neg:                    # 'a robin is a bird' -> robin feels like bird
                    mind.embed.pull(a, b)
                    mind._last_subject = a
                    hyp = _form_hypothesis(mind, a, b)   # reason further, on its own
                    return say.realize(mind, ("inform_is", {"subj": a, "obj": b})) + hyp
                return say.realize(mind, ("inform_is", {"subj": a, "obj": b, "neg": True}))

    if t[:2] == ["what", "is"] and _sing(t[2]) in mind.defs and len(t) == 3:
        w = _sing(t[2])
        return f"{w} is when {_render_abs(mind.defs[w])}."
    if (t[:3] == ["can", "there", "be"] or t[:2] == ["is", "there"]) \
            and _sing(t[-1]) in mind.defs:
        w = _sing(t[-1])
        ok = abstract.satisfiable(w, mind.defs, mind.implications)
        return f"Yes, there could be a {w}." if ok else \
               f"No — by your definitions, a {w} can't actually exist."

    # entailment: 'is a X a Y' / 'are Xs Y'
    if t[0] in ("is", "are"):
        content = [_sing(w) for w in t[1:] if w not in ("a", "an", "the")]
        if len(content) == 2:
            x, y = content
            atoms = abstract._atoms(mind.defs, mind.implications)
            known_x = x in mind.defs or x in mind.atoms or x in atoms
            known_y = y in mind.defs or y in mind.atoms or y in atoms
            if known_x and x not in _STOP and y not in _STOP:
                if not known_y:
                    return f"I don't know what {y} is yet."
                models = abstract._models(mind.defs, mind.implications) or []
                truex = [mdl for mdl in models if abstract._eval(x, mdl, mind.defs)]
                if not truex:
                    return f"By your definitions a {x} can't exist, so I can't say."
                if all(abstract._eval(y, mdl, mind.defs) for mdl in truex):
                    return f"Yes — every {x} is {y}."
                if not any(abstract._eval(y, mdl, mind.defs) for mdl in truex):
                    return f"No — a {x} is never {y}."
                return f"Not necessarily — a {x} can be {y}, but doesn't have to be."
    return None


# ---- the main loop --------------------------------------------------------------------

def respond(mind: Mind, sentence: str) -> str:
    toks = tokenize(sentence)
    if not toks:
        return "..."
    reply = _handle(mind, sentence, toks)
    reply = _maybe_curious(mind, reply)
    mind.learn_sentence(toks)          # learn the words from how they were used
    mind.save()
    return reply


def _maybe_curious(mind, reply):
    """Once in a while, after you've taught it a few things, it volunteers something it's
    been wondering about — on its own, from a real gap in what it knows. Deterministic and
    rate-limited so it stays gentle (and never disrupts a short exchange)."""
    keys = ("every ", " can ", " has ", "part of", "is when", "now you can",
            "something i can do", "kind of thing", "another word for", "opposite of",
            "worked out that", "is a verb", "is a noun", "i'll remember")
    teaching = not reply.rstrip().endswith("?") and any(k in reply.lower() for k in keys)
    if not teaching:
        return reply
    # actively pursue understanding of the THING just taught — ask its next open gap
    subj = getattr(mind, "_last_subject", None)
    q = _followup(mind, subj)
    asked = getattr(mind, "_asked_wonders", None)
    if asked is None:
        asked = mind._asked_wonders = []
    if q and q not in asked:
        asked.append(q)
        return reply + f" Ooh — I'm curious: {q}"
    return reply


def _followup(mind, subj):
    """A real open question about a thing it just learned about — so it drives the topic."""
    if not subj:
        return ""
    if not any(o == subj for o, _ in mind.abilities):
        return f"what can a {subj} do?"
    if not any(wh == subj for _, wh in mind.parts):
        return f"what does a {subj} have?"
    if not any(a == subj for a, _ in mind.implications):
        return f"what is a {subj}?"
    return ""


def _handle(mind, sentence, t):
    # confirming/denying a thought the bot worked out itself ('so can a robin fly too?')
    hyp = getattr(mind, "_hypothesis", None)
    if hyp and t and t[0] in ("yes", "yeah", "yep", "no", "nope", "not", "maybe"):
        mind._hypothesis = None
        x, rel, val = hyp
        if t[0] in ("yes", "yeah", "yep"):
            (mind.abilities if rel == "can" else mind.parts).append(
                (x, val) if rel == "can" else (val, x))
            return say.realize(mind, ("inferred", {"subj": x, "rel": rel, "val": val}))
        if t[0] == "maybe":
            return "Okay — maybe. I'll keep wondering about it."
        return f"Oh — so a {x} does not {rel} {val}. Good to know, thanks."

    # confirming/denying a guess the bot just made ('Is glip a bit like bird?')
    g = getattr(mind, "_guessed", None)
    if g and t and t[0] in ("yes", "yeah", "yep", "right", "correct", "no", "nope", "not"):
        mind._guessed = None
        w, guess = g
        if t[0] in ("yes", "yeah", "yep", "right", "correct"):
            mind.embed.pull(w, guess, alpha=0.4)        # confirmation reinforces it
            return f"Good — so '{w}' is a bit like '{guess}'. That helps."
        return f"Okay — I'll keep learning what '{w}' really means."

    # answering the PART OF SPEECH of a word it just asked about
    ap = getattr(mind, "_asking_pos", None)
    if ap:
        mind._asking_pos = None
        cat = _classify_pos(t) or "noun"
        mind.pos[ap] = cat
        if cat == "verb":
            mind.actions.add(ap); mind.embed.observe(["you", "can", ap])
            return f"Got it — '{ap}' is a verb (an action). Tell me to {ap}!"
        if cat in ("adjective", "adverb"):
            mind._asking_adj = ap
            what = "describes things" if cat == "adjective" else "tells how something is done"
            return f"Okay, '{ap}' is an {cat} — it {what}. Is it like another {cat} you know, " \
                   f"or the opposite of one?"
        if cat == "pronoun":
            mind.embed.pull(ap, "it", 0.4)
            return f"Okay — '{ap}' is a pronoun; it stands in for a name, like 'it' or 'they'."
        if cat == "preposition":
            return f"Okay — '{ap}' is a preposition; it shows how things relate, like 'on'."
        if cat == "conjunction":
            return f"Okay — '{ap}' is a conjunction; it joins words, like 'and' or 'but'."
        if cat == "interjection":
            return f"Okay — '{ap}' is an interjection; a word you say to show feeling, like 'wow'."
        mind._asking = ap                              # noun
        return f"Okay, '{ap}' is a noun (a thing). What is a {ap} like, or what is a {ap}?"

    # answering what a describing-word is like ('it's like big' / 'the opposite of happy')
    aa = getattr(mind, "_asking_adj", None)
    if aa and ("like" in t or "opposite" in t):        # only if it looks like an answer
        mind._asking_adj = None
        opp = "opposite" in t
        words = [w for w in t if w not in _STOP and w not in ("opposite",) and len(w) > 1]
        if words:
            other = _sing(words[-1])
            if opp:
                mind.embed.push(aa, other)
                if {aa, other} not in [set(p) for p in mind.opposites]:
                    mind.opposites.append((aa, other))
                return f"Ah — so '{aa}' is the opposite of '{other}'. Got it."
            mind.embed.pull(aa, other, 0.5)
            return f"Ah — so '{aa}' is a bit like '{other}'. Got it."

    # answering a word the bot actively asked about ('I don't know zebra...')
    asking = getattr(mind, "_asking", None)
    if asking:
        mind._asking = None
        # treat as an answer only if it looks like one ('it's like a horse', 'like a horse')
        if "like" in t and (t[0] in ("it", "like", "a", "an", "the", "that") or asking in t):
            after = [w for w in t[t.index("like") + 1:] if w not in ("a", "an", "the")]
            if after:
                tgt = _target(mind, after)
                mind.embed.pull(asking, tgt)
                return f"Ah — so a {asking} is like a {_disp(tgt)}. Thanks, that helps me."
        # otherwise let the reply (a fact / 'a zebra is an animal' / etc.) teach it below

    # answering 'what is <unknown word>?' that we just asked about
    pa = getattr(mind, "_pending_abstract", None)
    if pa:
        mind._pending_abstract = None
        ans = _abstract(mind, sentence)
        if ans is not None:
            retry = _abstract(mind, pa["sentence"]) or ""
            return f"{ans} {retry}".strip()

    # --- choose which word it SAYS for a concept: 'say soar for fly' ---
    if t[:1] in (["use"], ["say"]) and len(t) >= 4 and ("for" in t or "instead" in t):
        key = "of" if "instead" in t else "for"
        if key in t:
            i = t.index(key)
            a = _sing(t[1])
            b = _sing(t[i + 1]) if i + 1 < len(t) else None
            if a and b and a not in _STOP and b not in _STOP and a != b:
                mind.prefer[b] = a
                return f"Okay — from now on I'll say '{a}' when I mean '{b}'."

    # --- corrections: tell it when it used a word wrong ---
    corr = _correction(mind, t)
    if corr is not None:
        return corr

    # --- naming ---
    if t[:3] == ["your", "name", "is"] and len(t) >= 4:
        mind.name = t[3].capitalize()
        return say.realize(mind, ("named", {"name": mind.name}))
    if (t[:3] in (["what", "is", "your"], ["what", "s", "your"]) and "name" in t) \
            or t == ["who", "are", "you"]:
        return say.realize(mind, ("my_name", {"name": mind.name}))

    # --- small talk ---
    if t[0] in ("hello", "hi", "hey", "greetings"):
        return say.realize(mind, ("greet", {"name": mind.name}))
    if t[0] in ("thanks", "thank"):
        return say.realize(mind, ("thanks", {}))
    if t[:3] == ["how", "are", "you"]:
        return "I'm learning! Teach me a word or an idea."
    # switch which learned voice it speaks in
    if (t[:2] in (["speak", "as"], ["talk", "as"], ["talk", "like"]) or t[:1] == ["be"]) \
            and getattr(mind, "_voices", None):
        want = t[-1]
        if want in mind._voices:
            mind.voice = want
            return f"[Now speaking as {want.title()}.]"
        if want in ("yourself", "normal", "plain", "you"):
            mind.voice = None
            return "[Back to my plain voice.]"
    if "curious" in t or "wondering" in t or "wonder" in t \
            or (t[:1] == ["what"] and "know" in t and ("want" in t or "like" in t)):
        return _wonder(mind)
    if t[0] in ("ok", "okay", "cool", "nice", "great"):
        return "Okay! What else can you teach me?"
    if t == ["help"] or (t[:3] == ["what", "can", "you"] and "do" in t):
        return ("Talk to me and I learn words from how you use them. Try: 'a cat is an animal', "
                "'what is cat like?', 'is a cat an animal?', 'dark is when there is no light', "
                "or teach me ideas and I'll tell you if they don't add up.")
    if t[:3] in (["what", "do", "you"], ["what", "have", "you"]) and "about" not in t \
            and ("know" in t or "learned" in t):
        return _vocab(mind)
    if t[0] == "forget" and len(t) >= 2:
        return _forget(mind, [w for w in t[1:] if w not in ("the", "word", "about")])

    # --- synonyms: 'big means large' / 'big is another word for large' ---
    syn = None
    if len(t) == 3 and t[1] == "means":
        syn = (t[0], t[2])
    elif t[1:2] == ["is"] and "another" in t and "word" in t and "for" in t:
        syn = (t[0], t[-1])
    if syn:
        a, b = _sing(syn[0]), _sing(syn[1])
        if a != b and a not in _STOP and b not in _STOP:
            mind.embed.pull(a, b, alpha=0.7)           # synonyms: strong pull together
            mind.embed.pull(b, a, alpha=0.3)
            if b in mind.actions:                      # a synonym of an action is also a command
                mind.actions.add(a)
            return f"Got it — '{a}' is another word for '{b}'."

    # --- opposites: 'hot is the opposite of cold' ---
    if t[1:2] == ["is"] and "opposite" in t and "of" in t:
        a, b = _sing(t[0]), _sing(t[-1])
        if a != b and a not in _STOP and b not in _STOP:
            mind.embed.push(a, b); mind.embed.push(b, a)
            if {a, b} not in [set(p) for p in mind.opposites]:
                mind.opposites.append((a, b))
            return f"Got it — '{a}' is the opposite of '{b}'."

    # --- teach it an action it can do: 'you can wave' ---
    ta = _teach_action(mind, t)
    if ta is not None:
        return ta

    # --- parts: 'a bird has wings' / 'a wing is part of a bird' ---
    pr = _parts_teach(mind, t)
    if pr is not None:
        return pr
    pq = _parts_query(mind, t)
    if pq is not None:
        return pq

    # --- abilities: 'a bird can fly' / 'what can a bird do' / 'can a bird fly' ---
    aq = _ability(mind, t)
    if aq is not None:
        return aq

    # --- 'do you know X' ---
    if t[:3] == ["do", "you", "know"] and len(t) >= 4:
        w = _sing(t[-1])
        return say.realize(mind, ("know_word", {"word": w, "known": mind.embed.knows(w)}))
    # --- 'what is the opposite of X' ---
    if t[:2] == ["what", "is"] and "opposite" in t and "of" in t:
        w = _sing(t[-1])
        opp = None
        for x, y in mind.opposites:
            if _sing(x) == w:
                opp = y
            elif _sing(y) == w:
                opp = x
        return say.realize(mind, ("opposite_of", {"word": w, "opp": opp}))
    # --- 'what goes with X' ---
    if t[:3] == ["what", "goes", "with"] and len(t) >= 4:
        return _like(mind, _sing(t[-1]))

    # --- similarity (learned from usage) ---
    sim = _similarity_query(mind, t)
    if sim is not None:
        return sim
    if t[:3] == ["tell", "me", "about"] and len(t) >= 4:
        rest = [w for w in t[3:] if w not in ("a", "an", "the")]
        if rest:
            return _about(mind, _sing(rest[-1]))

    # --- 'X is a verb/noun/...' -> tag part of speech (not a fact) ---
    pd = _pos_declare(mind, t)
    if pd is not None:
        return pd

    # --- reasoning over taught ideas ---
    ab = _abstract(mind, sentence)
    if ab is not None:
        return ab

    # --- a command? understand the order and (say it'll) do it ---
    cmd = _command(mind, sentence, t)
    if cmd is not None:
        return cmd

    # --- otherwise: it's a statement to learn from ---
    return _acknowledge(mind, t)


def _similarity_query(mind, t):
    # 'what is X like' / 'what is like X' / 'what is similar to X'  (X may be a phrase)
    if t[:2] == ["what", "is"] and ("like" in t or "similar" in t):
        kw = "like" if "like" in t else "similar"
        end = t.index(kw)
        before = [w for w in t[2:end] if w not in ("a", "an", "the")]
        after = [w for w in t[end + 1:] if w not in ("a", "an", "the", "to")]
        words = before or after
        return _like(mind, _target(mind, words))
    if t[:1] == ["is"] and "like" in t:              # 'is X like Y'  (X, Y may be phrases)
        i = t.index("like")
        a = _target(mind, [w for w in t[1:i] if w not in ("a", "an", "the")])
        b = _target(mind, [w for w in t[i + 1:] if w not in ("a", "an", "the")])
        if a and b:
            da, db = _disp(a), _disp(b)
            if {a, b} in [set(p) for p in mind.opposites]:
                return f"No — you told me '{da}' and '{db}' are opposites. " \
                       f"(They get used in similar ways, but they mean opposite things.)"
            c = mind.embed.cosine(a, b)
            if c is None:
                miss = da if not mind.embed.knows(a) else db
                return f"I don't know '{miss}' well enough yet — use it in a few sentences."
            if c > 0.45:
                return f"Yes, {da} and {db} feel pretty similar to me."
            if c > 0.2:
                return f"A little — {da} and {db} are somewhat alike."
            return f"Not really — {da} and {db} feel different to me."
    return None


def _disp(w):
    return w.replace("_", " ")


def _target(mind, words):
    """Pick the query target from trailing words — a known phrase if the last two form one,
    else the last content word."""
    if len(words) >= 2 and (words[-2], words[-1]) in mind.phrases:
        return words[-2] + "_" + words[-1]
    return _sing(words[-1]) if words else None


def _clean_sims(mind, w, k=5, thr=0.25):
    """Nearest words by usage, with function words and the word's own forms filtered out."""
    out = []
    for s, v in mind.embed.similar(w, k + 6, min_count=2):
        if v > thr and s not in _STOP and _sing(s) != _sing(w):
            out.append(s)
        if len(out) >= k:
            break
    return out


def _like(mind, w):
    if not w:
        return "Like what?"
    if not mind.embed.knows(w):
        return _ask_unknown(mind, _disp(w))
    sims = [_disp(s) for s in _clean_sims(mind, w, 5)]
    if not sims:
        return f"I don't have a good feel for '{_disp(w)}' yet — tell me more about it."
    mind._last_used = _sing(w)
    return say.realize(mind, ("similar", {"word": _disp(w), "neighbors": sims}))


def _about(mind, w):
    parts = []
    if mind.embed.knows(w):
        sims = _clean_sims(mind, w, 4)
        if sims:
            parts.append(f"'{w}' feels similar to {', '.join(sims)}.")
    if w in mind.defs:
        parts.append(f"You told me {w} is when {_render_abs(mind.defs[w])}.")
    facts = [f"every {w} is {b}" for a, b in mind.implications if a == w and isinstance(b, str)]
    facts += [f"no {w} is {b[1]}" for a, b in mind.implications
              if a == w and not isinstance(b, str)]
    parts += facts
    can = [a for o, a in mind.abilities if o == w]
    if can:
        parts.append(f"It can: {', '.join(can)}.")
    has = [p for p, whole in mind.parts if whole == w]
    if has:
        parts.append(f"It has: {', '.join(has)}.")
    partof = [whole for p, whole in mind.parts if p == w]
    if partof:
        parts.append(f"It's part of: {', '.join(partof)}.")
    opp = [y for x, y in mind.opposites if x == w] + [x for x, y in mind.opposites if y == w]
    if opp:
        parts.append(f"Its opposite is {', '.join(opp)}.")
    if not parts:
        return f"I don't know much about '{w}' yet — teach me by using it."
    return " ".join(parts)


def _record_part(mind, part, whole):
    mind._last_subject = whole
    if part != whole and (part, whole) not in mind.parts:
        mind.parts.append((part, whole))
        mind.embed.pull(part, whole, alpha=0.2)        # a part associates with its whole


def _parts_teach(mind, t):
    art = ("a", "an", "the", "some", "two", "many")
    # 'a bird has wings' / 'birds have feathers'
    if "has" in t or "have" in t:
        v = "has" if "has" in t else "have"
        vi = t.index(v)
        left = [w for w in t[:vi] if w not in art]
        right = [w for w in t[vi + 1:] if w not in art]
        if len(left) == 1 and right and left[0] not in _STOP and right[-1] not in _STOP:
            owner, part = _sing(left[0]), _sing(right[-1])
            _record_part(mind, part, owner)
            return f"Got it — a {owner} has {_disp(' '.join(right))}."
    # 'a wing is part of a bird'
    if "part" in t and "of" in t and t.count("is") == 1:
        ii = t.index("is")
        if t[ii + 1:ii + 3] == ["part", "of"]:
            left = [w for w in t[:ii] if w not in art]
            right = [w for w in t[ii + 3:] if w not in art]
            if left and right and left[-1] not in _STOP and right[-1] not in _STOP:
                part, whole = _sing(left[-1]), _sing(right[-1])
                _record_part(mind, part, whole)
                return f"Got it — a {part} is part of a {whole}."
    return None


def _parts_query(mind, t):
    art = ("a", "an", "the")
    # 'what does a bird have' / 'what do birds have'
    if t[:2] in (["what", "does"], ["what", "do"]) and ("have" in t or "has" in t):
        body = [w for w in t[2:] if w not in art and w not in ("have", "has")]
        if body:
            whole = _sing(body[-1])
            ps = [p for p, w in mind.parts if w == whole]
            return say.realize(mind, ("has_list", {"subj": whole, "parts": ps})) if ps else \
                f"I don't know what a {whole} has yet — tell me, like 'a {whole} has ...'."
    # 'what is part of a bird' / 'what is a wing part of'
    if "part" in t and "of" in t and t[:2] == ["what", "is"]:
        pi = t.index("part")
        head = [w for w in t[2:pi] if w not in art]      # words between 'is' and 'part'
        if not head:                                     # 'what is part of X'
            after = [w for w in t[pi + 2:] if w not in art]
            if after:
                whole = _sing(after[-1])
                ps = [p for p, w in mind.parts if w == whole]
                return f"Part of a {whole}: {', '.join(ps)}." if ps else \
                       f"I don't know what's part of a {whole} yet."
        else:                                            # 'what is X part of'
            part = _sing(head[-1])
            ws = [w for p, w in mind.parts if p == part]
            return f"A {part} is part of: {', '.join(ws)}." if ws else \
                   f"I don't know what a {part} is part of yet."
    return None


def _map_action(mind, verb):
    """Map a command word to an action it knows — exactly, or by closest meaning."""
    if verb in mind.actions:
        return verb, False
    best, bs = None, 0.0
    for a in mind.actions:
        c = mind.embed.cosine(verb, a)
        if c is not None and c > bs:
            bs, best = c, a
    if best and bs > 0.5:
        return best, True
    return None


def _teach_action(mind, t):
    """'you can wave', 'you can run and jump', 'wave is an action'."""
    if t[:2] == ["you", "can"] and len(t) >= 3:
        acts = [_sing(w) for w in t[2:] if w not in ("and", "also", "too") and len(w) > 1]
        for a in acts:
            mind.actions.add(a)
            mind.embed.observe(["you", "can", a])
        if acts:
            return f"Okay — now you can tell me to {' or '.join(acts)}!"
    if t[1:2] == ["is"] and ("action" in t or ("can" in t and "do" in t)) and _sing(t[0]) not in _STOP:
        a = _sing(t[0])
        mind.actions.add(a)
        return f"Got it — '{a}' is something I can do. Tell me to {a}!"
    return None


def _command(mind, sentence, t):
    """Understand an order: map the words to one of the actions it can do (even if you
    word it in a way it hasn't seen), or ask you to teach it. 'Doing' it = saying so."""
    if "is" in t or "are" in t or "means" in t:        # that's a statement/question
        return None
    words, marked = list(t), False
    if words[:1] == ["please"]:
        words, marked = words[1:], True
    elif words[:2] == ["can", "you"]:
        words, marked = words[2:], True
    elif words[:4] == ["i", "want", "you", "to"]:
        words, marked = words[4:], True
    elif words[:1] in (["go"], ["now"]) and len(words) > 1:
        words, marked = words[1:], True
    if not words:
        return None
    verb = _sing(words[0])
    if verb in _STOP:                                  # 'i have a ...' is not a command
        return None
    if verb in mind.actions:                           # exact action it knows
        action, approx = verb, False
    elif marked:                                       # clearly an order — allow a close match
        act = _map_action(mind, verb)
        if act:
            action, approx = act
        else:
            return say.realize(mind, ("cant_do", {"action": verb}))
    else:
        return None                                    # unmarked + not a known action -> not a command
    tgt = " ".join(w for w in words[1:]
                   if w not in _STOP and w not in mind.actions and w not in _FILLER)
    return say.realize(mind, ("ack_command",
                              {"action": action, "target": tgt or None, "approx": approx}))


def _form_hypothesis(mind, x, y):
    """Having learned 'a x is a y', reason FURTHER on its own: a y can fly / has wings, so
    maybe an x does too? Forms a hypothesis and voices it — a thought it made itself."""
    x_can = {a for o, a in mind.abilities if o == x}
    for o, a in mind.abilities:
        if o == y and a not in x_can:
            mind._hypothesis = (x, "can", a)
            return say.realize(mind, ("ask_can", {"subj": x, "verb": a, "known": y}))
    x_has = {p for p, wh in mind.parts if wh == x}
    for p, wh in mind.parts:
        if wh == y and p not in x_has:
            mind._hypothesis = (x, "has", p)
            return say.realize(mind, ("ask_has", {"subj": x, "part": p, "known": y}))
    return ""


def _wonder(mind):
    """A question the bot generates from a GAP in its own knowledge — curiosity."""
    things = [a for a, b in mind.implications if isinstance(b, str)]
    seen = set()
    for x in things:
        if x in seen:
            continue
        seen.add(x)
        if not any(o == x for o, _ in mind.abilities):
            return say.realize(mind, ("curious_do", {"subj": x}))
        if not any(wh == x for _, wh in mind.parts):
            return say.realize(mind, ("curious_have", {"subj": x}))
    if mind.embed.words:
        # a word it's heard a lot but knows nothing concrete about
        ranked = sorted(range(len(mind.embed.words)), key=lambda i: -mind.embed.counts[i])
        for i in ranked:
            w = mind.embed.words[i]
            if w in _STOP or len(w) <= 2 or "_" in w:
                continue
            if w not in mind.defs and not any(o == w for o, _ in mind.abilities) \
                    and not any(a == w for a, _ in mind.implications):
                return say.realize(mind, ("curious_word", {"word": w}))
    return say.realize(mind, ("curious_none", {}))


def _ability(mind, t):
    art = ("a", "an", "the")
    # query: 'what can a bird do'
    if t[:2] == ["what", "can"] and "do" in t:
        body = [w for w in t[2:] if w not in art and w != "do"]
        if body and _sing(body[-1]) not in _STOP:
            owner = _sing(body[-1])
            acts = [a for o, a in mind.abilities if o == owner]
            return say.realize(mind, ("can_list", {"subj": owner, "acts": acts})) if acts else \
                f"I don't know what a {owner} can do yet."
    # query: 'can a bird fly'  (but NOT 'can there be a ...' — that's a logic question)
    if t[0] == "can" and len(t) >= 3 and "there" not in t and "be" not in t:
        body = [w for w in t[1:] if w not in art]
        if len(body) >= 2 and body[0] not in _STOP:
            owner, action = _sing(body[0]), _sing(body[-1])
            if (owner, action) in mind.abilities:
                return f"Yes, a {owner} can {action}."
            if any(o == owner for o, a in mind.abilities):
                return f"Not that you've told me — I don't think a {owner} can {action}."
            return f"I don't know what a {owner} can do yet."
    # teach: 'a bird can fly' / 'a bird can fly and sing'
    if "can" in t and t[0] not in ("can", "what", "how", "why", "could", "do", "i", "you"):
        ci = t.index("can")
        left = [w for w in t[:ci] if w not in art]
        right = [_sing(w) for w in t[ci + 1:] if w not in art and w != "and"]
        right = [w for w in right if w not in _STOP]
        if len(left) == 1 and right and left[0] not in _STOP:
            owner = _sing(left[0])
            for a in right:
                if (owner, a) not in mind.abilities:
                    mind.abilities.append((owner, a))
            mind._last_subject = owner
            return f"Got it — a {owner} can {' and '.join(right)}."
    return None


def _vocab(mind):
    n = len(mind.embed.words) if mind.embed.words else 0
    bits = [f"I've picked up about {n} words from how you use them."]
    # show a couple of the groupings it has formed, from the words it's seen most
    ranked = sorted(range(len(mind.embed.words)), key=lambda i: -mind.embed.counts[i])
    shown = 0
    for i in ranked:
        w = mind.embed.words[i]
        if w in _STOP or len(w) <= 2:
            continue
        near = _clean_sims(mind, w, 3, thr=0.3)
        if near:
            bits.append(f"I think '{w}' goes with {', '.join(near)}.")
            shown += 1
        if shown >= 2:
            break
    if mind.defs:
        bits.append("Ideas you defined: " + ", ".join(sorted(mind.defs)) + ".")
    if mind.implications:
        bits.append(f"And I'm holding {len(mind.implications)} facts about how words relate.")
    return " ".join(bits)


def _forget(mind, rest):
    if not rest:
        return "Forget what?"
    word = _sing(rest[0])
    found = False
    for key in (rest[0], word):
        if key in mind.defs:
            del mind.defs[key]; found = True
        if key in mind.atoms:
            mind.atoms.discard(key); found = True
    keys = (rest[0], word)
    before = (len(mind.implications), len(mind.abilities), len(mind.parts), len(mind.opposites))
    mind.implications = [(a, b) for a, b in mind.implications
                         if a not in keys
                         and (b[1] if not isinstance(b, str) and b[0] == "not" else b) not in keys]
    mind.abilities = [(o, a) for o, a in mind.abilities if o not in keys]
    mind.parts = [(p, wh) for p, wh in mind.parts if p not in keys and wh not in keys]
    mind.opposites = [(x, y) for x, y in mind.opposites if x not in keys and y not in keys]
    found = found or before != (len(mind.implications), len(mind.abilities),
                                len(mind.parts), len(mind.opposites))
    return f"OK — I've forgotten '{rest[0]}'." if found else \
           f"I don't have anything to forget about '{rest[0]}'."


def _knowledge_about(mind, w):
    """Assemble what the bot has actually learned about a word into phrases it can SAY."""
    bits = []
    can = [a for o, a in mind.abilities if o == w]
    if can:
        bits.append(f"a {w} can {' and '.join(can[:2])}")
    has = [p for p, whole in mind.parts if whole == w]
    if has:
        bits.append(f"a {w} has {' and '.join(has[:2])}")
    isa = [b for a, b in mind.implications if a == w and isinstance(b, str)]
    if isa:
        bits.append(f"a {w} is a {isa[0]}")
    sims = _clean_sims(mind, w, 1, thr=0.35)
    if sims and not isa:
        bits.append(f"a {w} is a bit like a {_disp(sims[0])}")
    return bits


def _voice_reply(mind, text):
    """Generate a reply through the loaded LM voice (Apollo / Arthur I), seeded by what you
    said. This is a learned, from-scratch model speaking — not a template."""
    voices = getattr(mind, "_voices", None)
    key = getattr(mind, "voice", None)
    if not voices or key not in voices:
        return None
    import torch
    model, coder = voices[key]
    # The models were trained on USER:/BOT: turns, so prompt them the same way: give them
    # what you said as the USER turn and let them produce the BOT turn (its reply).
    seed = "USER: " + text.strip() + "\nBOT: "
    ids = coder.encode(seed) or [coder.stoi.get("\n", 0)]
    out = coder.decode(model.generate(torch.tensor([ids]), 200, temp=0.8)[0])
    gen = out[len(seed):] if out.startswith(seed) else out
    # the reply is everything up to the next turn marker
    for cut in ("\nUSER:", "\nBOT:", "USER:", "BOT:"):
        if cut in gen:
            gen = gen.split(cut)[0]
    gen = gen.replace("\n", " ").strip()
    return gen or None


def _acknowledge(mind, t):
    """A statement we don't parse as a command/teaching. If a learned VOICE is on, just
    converse through it. Otherwise fall back to the symbolic learner (ask about a new word,
    or use what it knows)."""
    voiced = _voice_reply(mind, " ".join(t))
    if voiced:
        return voiced
    ranked = []
    for i, w in enumerate(t):
        sw = _sing(w)
        if w not in _STOP and len(w) > 1:
            ranked.append((0 if (i > 0 and t[i - 1] in ("a", "an", "the")) else 1, sw))
    ranked.sort(key=lambda p: p[0])
    unknown = [w for _, w in ranked if not mind.embed.knows(w)]
    if unknown:                                    # ACTIVELY ask — but part of speech first
        return _ask_unknown(mind, unknown[0])
    order = [w for _, w in ranked]
    # everything's familiar: USE what I know about one of these words
    for w in order:
        bits = _knowledge_about(mind, w)
        if bits:
            mind._last_used = w
            return say.realize(mind, ("know_about", {"subj": w, "bits": bits[:2]}))
    # known but I have nothing to say yet — guess from usage and check
    for w in order:
        sims = _clean_sims(mind, w, 1, thr=0.4)
        if sims:
            mind._guessed = (w, _disp(sims[0]))
            return f"Okay. Is a {w} a bit like a {_disp(sims[0])}?"
    return "Okay — tell me more, and I'll learn."
