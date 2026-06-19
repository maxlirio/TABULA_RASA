"""A knowledge store the bot can actually USE.

Everything you tell it becomes a (subject, relation, object) triple — is-a, has, can, or a
plain property. It answers questions in BOTH directions ("a cat has fur" -> "what has fur?")
and INHERITS properties down is-a links ("a cat is a mammal", "a mammal has fur" -> "does a
cat have fur?" yes). This is where real memory and understanding live; the tiny neural net
only handles casual phrasing — it cannot store or recall facts, so we don't ask it to.
"""
import re

from gm import abstract

ARTS = ("a", "an", "the", "every", "all", "each", "some", "this", "that", "my", "your")
GRAMMAR = {"is", "are", "a", "an", "the", "not", "do", "does", "to", "of", "and", "or"}
PRON = {"they", "them", "it", "he", "she", "we", "you", "i", "both", "this", "that",
        "these", "those", "there", "here", "everyone", "someone", "something",
        "nothing", "one", "who", "what", "thing"}


def bad_subj(s):
    return (not s) or s.split()[0] in PRON or s in PRON


BAD_OBJ = {"anything", "nothing", "something", "everything", "it", "them", "that",
           "this", "one", "any", "some"}


def ok_subject(s):
    """A real, namable thing — 1-2 plain words, no commas/sentences/pronouns."""
    return (bool(s) and not bad_subj(s) and 1 <= len(s) <= 22
            and len(s.split()) <= 2 and "," not in s
            and all(c.isalpha() or c in " '" for c in s))


def ok_obj(o):
    return (bool(o) and "," not in o and len(o) <= 28 and o not in BAD_OBJ
            and not any(neg in f" {o} " for neg in (" not ", " n't ")))


def sing(w):
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def norm(phrase, sg=True):
    parts = phrase.strip().lower().rstrip("?.!,").split()
    while parts and parts[0] in ARTS:
        parts = parts[1:]
    if not parts:
        return ""
    if sg:                              # singularize the head for taxonomy matching
        parts[-1] = sing(parts[-1])
    return " ".join(parts)


def art(w):
    return ("an " if w[:1] in "aeiou" else "a ") + w


class Knowledge:
    def __init__(self):
        self.triples = []                       # list of [subj, rel, obj]; obj may be ["not", w]
        self.last = None                        # most recently taught subject (for curiosity)

    # --- persistence ----------------------------------------------------
    def dump(self):
        return self.triples

    def load(self, data):
        self.triples = []
        for t in data or []:
            if len(t) == 2:                     # legacy is-a facts [a, b] / [a, ["not", b]]
                self.triples.append([t[0], "isa", t[1]])
            else:
                self.triples.append(list(t))

    def _add(self, s, r, o):
        self.last = s
        if [s, r, o] not in self.triples:
            self.triples.append([s, r, o])

    def curiosity(self, recent=None, asked=()):
        """Make up a GENUINE question from real gaps/patterns it can't infer — never a canned
        empty slot. Two kinds, both grounded in what it actually knows:
          A) sibling transfer: you just taught 'a cat is a mammal' and it knows a dog (mammal)
             has fur but can't infer that for a cat -> 'does a cat have fur?'
          B) induction: cats and dogs (mammals) have fur, and it wasn't told mammals do
             -> 'do all mammals have fur?'
        Returns None when it has nothing it genuinely can't infer to wonder about."""
        from collections import defaultdict
        asked = set(asked)
        subs = {s for s, _, _ in self.triples}

        # A) about the thing just taught: a sibling has a trait it can't infer for this one
        if recent in subs:
            kin = self.ancestors(recent) - {recent}
            for rel in ("has", "can"):
                mine = set(self.rel_objs(recent, rel))     # includes inherited
                for s, r, o in self.triples:
                    if (r == rel and s != recent and isinstance(o, str) and o not in mine
                            and (self.ancestors(s) & kin)):
                        q = f"does a {recent} have {o}?" if rel == "has" else f"can a {recent} {o}?"
                        if q not in asked:
                            return q

        # B) induction across a category
        holders = {"has": defaultdict(set), "can": defaultdict(set)}
        for s, r, o in self.triples:
            if r in holders:
                holders[r][o].add(s)
        for r, objs in holders.items():
            for o, who in objs.items():
                if len(who) < 2:
                    continue
                for c in set.intersection(*[self.ancestors(s) for s in who]) - {o}:
                    direct_c = {oo for s2, r2, oo in self.triples if s2 == c and r2 == r}
                    if o not in direct_c:
                        q = f"do all {c}s have {o}?" if r == "has" else f"can all {c}s {o}?"
                        if q not in asked:
                            return q
        return None

    # --- graph helpers --------------------------------------------------
    def _isa_edges(self):
        return [(s, o) for s, r, o in self.triples if r == "isa" and isinstance(o, str)]

    def ancestors(self, x):
        """x and everything x is-a (transitively)."""
        seen, queue = {x}, [x]
        adj = {}
        for a, b in self._isa_edges():
            adj.setdefault(a, []).append(b)
        while queue:
            for nxt in adj.get(queue.pop(), []):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    def isa_path(self, x, y):
        if x == y:
            return [x]
        adj = {}
        for a, b in self._isa_edges():
            adj.setdefault(a, []).append(b)
        seen, queue = {x}, [(x, [x])]
        while queue:
            node, path = queue.pop(0)
            for nxt in adj.get(node, []):
                if nxt == y:
                    return path + [y]
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, path + [nxt]))
        return None

    def rel_objs(self, x, rel):
        """Everything x has/can, including what it inherits from its is-a ancestors."""
        anc = self.ancestors(x)
        return sorted({o for s, r, o in self.triples
                       if r == rel and s in anc and isinstance(o, str)})

    def subjects_with(self, rel, obj):
        """Everything that has/can <obj>, directly or by inheritance ('what has fur?')."""
        subs = {s for s, r, o in self.triples if r in ("isa", "has", "can", "is")}
        return sorted({s for s in subs if obj in self.rel_objs(s, rel)})

    def about(self, x):
        """Everything known about an entity, direct + inherited — the bundle a code
        generator would receive as constraints for 'do stuff with this X'."""
        x = norm(x)
        if not any(s == x for s, _, _ in self.triples):
            return None
        return {"entity": x,
                "is_a": self.rel_objs(x, "isa"),
                "has": self.rel_objs(x, "has"),
                "can": self.rel_objs(x, "can"),
                "is": self.rel_objs(x, "is")}

    def _contradiction(self):
        imp = [(s, tuple(o) if isinstance(o, list) else o)
               for s, r, o in self.triples if r == "isa"]
        try:
            return abstract.contradictions({}, imp)
        except Exception:
            return []

    # --- teaching -------------------------------------------------------
    def teach(self, text):
        t = re.sub(r"^remember (?:that )?", "", text.strip().rstrip(".!"), flags=re.I)
        low = t.lower()
        # accept looser phrasings: "is a kind of / type of / sort of" -> plain "is a"
        low = re.sub(r"\b(is|are)\s+(?:a |an )?(?:kind|type|sort)s?\s+of\b", r"\1 a", low)

        m = re.match(r"^(.+?) (?:is|are) not (?:a |an )?(.+)$", low)
        if m:
            s, o = norm(m.group(1)), norm(m.group(2))
            if ok_subject(s) and ok_obj(o):
                self._add(s, "isa", ["not", o])
                if self._contradiction():
                    self.triples.pop()
                    return f"That can't be right - it conflicts with what I know about {s}."
                return f"Got it - no {s} is {art(o)}."

        m = re.match(r"^(?:a |an |every |all |each |the )?(.+?) (?:has|have|with) (?:a |an |some )?(.+)$", low)
        if m:
            s, o = norm(m.group(1)), norm(m.group(2), sg=False)
            if ok_subject(s) and ok_obj(o):
                self._add(s, "has", o)
                return f"Got it - {art(s)} has {o}."

        m = re.match(r"^(?:a |an |every |all |each |the )?(.+?) can (.+)$", low)
        if m:
            s, o = norm(m.group(1)), m.group(2).strip()
            if ok_subject(s) and ok_obj(o):
                self._add(s, "can", o)
                return f"Got it - {art(s)} can {o}."

        # is-a: needs 'a/an' OR plural 'Xs are Ys' so "the sky is blue" stays a property
        m = re.match(r"^(?:every |all |each )?(.+?) (?:is|are) (?:a|an) (\w+)", low)
        if m:
            s, o = norm(m.group(1)), norm(m.group(2))
            if ok_subject(s) and ok_obj(o):
                self._add(s, "isa", o)
                if self._contradiction():
                    self.triples.pop()
                    return f"Hmm - that contradicts something I know about {s}."
                extra = self._derived_note(s, o)
                return f"Got it - every {s} is {art(o)}.{extra}"
        m = re.match(r"^(\w+)s (?:are) (\w+)s?$", low)        # cats are animals
        if m and ok_subject(sing(m.group(1))) and ok_obj(sing(m.group(2))):
            s, o = sing(m.group(1)), sing(m.group(2))
            self._add(s, "isa", o)
            extra = self._derived_note(s, o)
            return f"Got it - every {s} is {art(o)}.{extra}"

        # plain property: "the sky is blue" — only when it clearly states a fact (starts with
        # the/a, or is about something already known). Otherwise it's chat ("how are you",
        # "hey what is up") and belongs to the LM.
        m = re.match(r"^(?:the )?(.+?) (?:is|are) (\w+)$", low)
        if m and m.group(2) not in GRAMMAR:
            s, o = norm(m.group(1)), norm(m.group(2))
            known = {sub for sub, _, _ in self.triples}
            if (ok_subject(s) and ok_obj(o)
                    and (low.startswith(("the ", "a ", "an ")) or s in known)):
                self._add(s, "is", o)
                return f"Got it - {s} is {o}."
        return None

    def _derived_note(self, s, o):
        for c in self.ancestors(o) - {s, o}:
            return f" So every {s} is {art(c)}, too."
        return ""

    # --- questions ------------------------------------------------------
    def ask(self, text):
        low = text.strip().rstrip("?.!").lower()
        entities = {s for s, _, _ in self.triples}

        def known(x):
            # only answer about real, taught things — never pronouns ("you") or unknowns,
            # so casual chat ("do you have a cough?") goes to the model, not a canned template
            return bool(x) and not bad_subj(x) and x in entities

        # specific "what does X have" / "what can X do" come BEFORE the generic forms
        m = re.match(r"^what does (?:a |an |the )?(.+?) have$", low)
        if m and known(norm(m.group(1))):
            objs = self.rel_objs(norm(m.group(1)), "has")
            if objs:
                return f"{art(norm(m.group(1))).capitalize()} has " + ", ".join(objs) + "."

        m = re.match(r"^what can (?:a |an |the )?(.+?) do$", low)
        if m and known(norm(m.group(1))):
            objs = self.rel_objs(norm(m.group(1)), "can")
            if objs:
                return f"{art(norm(m.group(1))).capitalize()} can " + ", ".join(objs) + "."

        # find-queries: skip conversational/pronoun phrasings ("what can you do") so they go
        # to the model instead of becoming a literal "Nothing I know of can you do."
        _PRON = ("you", "i", "we", "they", "he", "she", "it", "u", "ya")
        m = re.match(r"^(?:what|who) has (?:a |an )?(.+)$", low)
        if m and m.group(1).strip().split()[0] not in _PRON:
            obj = norm(m.group(1), sg=False)
            subs = self.subjects_with("has", obj)
            if subs:
                return self._list_subjects(subs, f"has {obj}")
            return f"Nothing I know of has {obj}."

        m = re.match(r"^(?:what|who) can (.+)$", low)
        if (m and m.group(1).strip().split()[0] not in _PRON
                and not m.group(1).strip().endswith(" do")):   # "what can X do" -> not a find
            act = m.group(1).strip()
            subs = self.subjects_with("can", act)
            if subs:
                return self._list_subjects(subs, f"can {act}")
            return f"Nothing I know of can {act}."

        m = re.match(r"^(?:does|do) (?:a |an |the )?(.+?) have (?:a |an )?(.+)$", low)
        if m and known(norm(m.group(1))):       # only for things it actually knows
            x, o = norm(m.group(1)), norm(m.group(2), sg=False)
            if o in self.rel_objs(x, "has"):
                return f"Yes - {art(x)} has {o}." + self._why_inherited(x, "has", o)
            return f"Not that I know of - I haven't been told {art(x)} has {o}."

        m = re.match(r"^can (?:a |an |the )?(.+?) (.+)$", low)
        if m and known(norm(m.group(1))):
            x, o = norm(m.group(1)), m.group(2).strip()
            if o in self.rel_objs(x, "can"):
                return f"Yes - {art(x)} can {o}." + self._why_inherited(x, "can", o)
            # fall through if unknown so the LM can handle "can you help me"

        # similarity: "is a cat like a mammal?" -> compare what they share
        m = re.match(r"^(?:is|are) (?:a |an )?(.+?) (?:like|similar to) (?:a |an )?(.+)$", low)
        if m and known(norm(m.group(1))) and known(norm(m.group(2))):
            x, y = norm(m.group(1)), norm(m.group(2))
            sx = set(self.rel_objs(x, "has") + self.rel_objs(x, "can")
                     + [a for a in self.ancestors(x) if a != x])
            sy = set(self.rel_objs(y, "has") + self.rel_objs(y, "can")
                     + [a for a in self.ancestors(y) if a != y])
            shared = sx & sy
            if shared:
                return f"Somewhat - both have {', '.join(sorted(shared))}."
            if any(s == x for s, _, _ in self.triples) or any(s == y for s, _, _ in self.triples):
                return f"Not that I can tell - they don't share anything I know about."

        m = re.match(r"^(?:is|are) (?:a |an |every |all |each )?(.+?) (?:a |an )?(.+)$", low)
        if m:
            x, y = norm(m.group(1)), norm(m.group(2))
            if x in [s for s, _, _ in self.triples] or y in self.ancestors(x):
                path = self.isa_path(x, y)
                if path and len(path) > 1:
                    why = " and ".join(f"{art(path[i])} is {art(path[i+1])}"
                                       for i in range(len(path) - 1))
                    return f"Yes - every {x} is {art(y)}, because {why}."
                if y in self.rel_objs(x, "is"):
                    return f"Yes - {x} is {y}."
                return f"Not that I know of - nothing I've been told makes every {x} {art(y)}."

        m = re.match(r"^(?:tell me (?:more )?about|describe|what do you know about) "
                     r"(?:a |an |the )?(.+)$", low)
        if m and known(norm(m.group(1))):       # unknown -> let the model chat, no canned line
            prof = self.profile(norm(m.group(1)))
            if prof:
                return prof

        m = re.match(r"^(?:what|who) (?:is|are) (?:a |an |the )?(.+)$", low)
        if m and known(norm(m.group(1))):
            prof = self.profile(norm(m.group(1)))
            if prof:
                return prof
        return None

    def _why_inherited(self, x, rel, o):
        for t, _, oo in self.triples:
            if oo == o and [t, rel, o] in self.triples and t != x and t in self.ancestors(x):
                path = self.isa_path(x, t)
                if path:
                    chain = " and ".join(f"{art(path[i])} is {art(path[i+1])}"
                                         for i in range(len(path) - 1))
                    return f" ({chain}, and {art(t)} has {o})" if rel == "has" \
                        else f" ({chain})"
        return ""

    def _list_subjects(self, subs, tail):
        items = [art(s) for s in subs[:6]]
        subj = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
        if len(items) > 1 and tail.startswith("has "):
            tail = "have " + tail[4:]
        out = f"{subj} {tail}."
        return out[0].upper() + out[1:]

    def profile(self, x):
        if not any(s == x for s, _, _ in self.triples):
            return None
        isa = self.rel_objs(x, "isa")
        has = self.rel_objs(x, "has")
        can = self.rel_objs(x, "can")
        props = self.rel_objs(x, "is")
        sents = []
        if isa:
            sents.append(f"{art(x).capitalize()} is " + ", ".join(art(i) for i in isa))
        else:
            sents.append(art(x).capitalize())
        if props:
            sents.append("It's " + ", ".join(props))
        if has:
            sents.append("It has " + ", ".join(has))
        if can:
            sents.append("It can " + ", ".join(can))
        return ". ".join(sents) + "." if len(sents) > 1 or isa else None
