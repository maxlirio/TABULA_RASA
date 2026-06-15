"""A tiny consistency checker for ABSTRACT vocabulary — words taught only through
definitions, not grounded in the tabletop ('light', 'dark', 'night', 'star').

It lets the agent learn things like:
    light is a word
    dark is when there is no light
    night is when it is dark and there are stars
    stars are light
and NOTICE that these can't all be true at once (a night would be dark — no light —
yet contain stars, which are light), so it can ask the operator for clarification.

No LLM: a definition is a boolean relationship between word-atoms, and we brute-force
propositional satisfiability over the (small) set of primitive atoms. Each defined word
is W <-> expr(W); each 'A are B' fact is A -> B. A concept is CONTRADICTORY when no
consistent assignment of the primitive atoms can make it true.
"""
from __future__ import annotations

from itertools import product

# An expr is: a word (str), ('not', expr), ('and', [expr...]), or ('or', [expr...]).


def _eval(expr, assign, defs, depth=0):
    if depth > 60:
        return False
    if isinstance(expr, str):
        if expr in defs:
            return _eval(defs[expr], assign, defs, depth + 1)
        return assign.get(expr, False)
    tag = expr[0]
    if tag == "not":
        return not _eval(expr[1], assign, defs, depth + 1)
    if tag == "and":
        return all(_eval(e, assign, defs, depth + 1) for e in expr[1])
    if tag == "or":
        return any(_eval(e, assign, defs, depth + 1) for e in expr[1])
    return False


def _collect(expr, out):
    if isinstance(expr, str):
        out.add(expr)
    elif expr[0] == "not":
        _collect(expr[1], out)
    else:
        for e in expr[1]:
            _collect(e, out)


def _atoms(defs, implications):
    """Primitive words — those with no definition of their own."""
    words = set()
    for w, e in defs.items():
        words.add(w)
        _collect(e, words)
    for a, b in implications:            # a is a word, b may be a word or ('not', word)
        _collect(a, words)
        _collect(b, words)
    return sorted(w for w in words if w not in defs)


def _models(defs, implications):
    """All atom assignments consistent with every 'A -> B' fact (None if too large)."""
    atoms = _atoms(defs, implications)
    if len(atoms) > 18:
        return None
    out = []
    for combo in product((False, True), repeat=len(atoms)):
        assign = dict(zip(atoms, combo))
        if all(not (_eval(a, assign, defs) and not _eval(b, assign, defs))
               for a, b in implications):
            out.append(assign)
    return out


def satisfiable(word, defs, implications):
    """Could something that is a `word` exist, given all definitions and facts?"""
    models = _models(defs, implications)
    if models is None:
        return True                      # too big to be sure — don't cry wolf
    return any(_eval(word, a, defs) for a in models)


def conflict_atom(word, defs, implications):
    """If `word` is impossible, find a primitive quality it would have to both have and
    lack (the heart of the contradiction), by forcing word=True and propagating. Else None."""
    forced, work = {}, [(word, True)]

    def push_atom(w, val):
        if w in forced:
            return w if forced[w] != val else None    # pivot if forced both ways
        forced[w] = val
        if val:                                       # fire implications 'w -> ...'
            for a, b in implications:
                if a == w:
                    if isinstance(b, str):
                        work.append((b, True))
                    elif b[0] == "not":
                        work.append((b[1], False))
        return None

    steps = 0
    while work and steps < 5000:
        steps += 1
        expr, val = work.pop()
        if isinstance(expr, str):
            if expr in defs:
                work.append((defs[expr], val))
            else:
                pivot = push_atom(expr, val)
                if pivot:
                    return pivot
        elif expr[0] == "not":
            work.append((expr[1], not val))
        elif expr[0] == "and" and val:
            work.extend((e, True) for e in expr[1])
        elif expr[0] == "or" and not val:
            work.extend((e, False) for e in expr[1])
    return None


def contradictions(defs, implications):
    """Categories that cannot possibly be true (necessarily empty) — a defined word,
    or any word asserted as a category ('A are B' makes A a category). Such a word being
    impossible means the statements about it conflict."""
    models = _models(defs, implications)
    if models is None:
        return []
    if not models:
        return ["__kb__"]                # not even a single consistent world
    cats = set(defs) | {a for a, _ in implications}
    return [w for w in sorted(cats) if not any(_eval(w, m, defs) for m in models)]
