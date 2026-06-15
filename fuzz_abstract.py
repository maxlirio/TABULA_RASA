#!/usr/bin/env python3
"""Property fuzzer for the abstract consistency checker (gm/abstract.py).

Generates random small definitional knowledge bases and checks logical invariants the
reasoner must always satisfy — most importantly that conflict_atom() is SOUND: if the
propagation tracer claims a concept is contradictory (returns a pivot quality), then the
independent brute-force model search must agree the concept is genuinely impossible.
"""
import random
import sys

from gm import abstract

ATOMS = ["light", "warm", "wet", "round", "big", "loud"]


def rand_expr(rng, words, depth):
    if depth <= 0 or rng.random() < 0.4:
        return rng.choice(words)
    r = rng.random()
    if r < 0.34:
        return ("not", rand_expr(rng, words, depth - 1))
    tag = "and" if r < 0.67 else "or"
    n = rng.randint(2, 3)
    return (tag, [rand_expr(rng, words, depth - 1) for _ in range(n)])


def rand_kb(rng):
    atoms = rng.sample(ATOMS, rng.randint(2, 4))
    defs, defined = {}, list(atoms)
    for k in range(rng.randint(1, 4)):
        w = f"c{k}"
        defs[w] = rand_expr(rng, defined, rng.randint(1, 3))
        defined.append(w)
    implications = []
    for _ in range(rng.randint(0, 4)):
        a = rng.choice(defined)
        b = rng.choice(defined)
        if a != b:
            implications.append((a, ("not", b)) if rng.random() < 0.4 else (a, b))
    return defs, implications, defined


def main(n=20000, seed=0):
    rng = random.Random(seed)
    bugs = checks = 0
    for _ in range(n):
        defs, impl, defined = rand_kb(rng)
        models = abstract._models(defs, impl)
        if models is None:
            continue
        listed = set(abstract.contradictions(defs, impl))
        kb_unsat = listed == {"__kb__"}            # no consistent world at all
        for w in defined:
            # SOUNDNESS: a claimed conflict must mean the concept is really unsatisfiable
            if abstract.conflict_atom(w, defs, impl) is not None:
                checks += 1
                if abstract.satisfiable(w, defs, impl):
                    bugs += 1
                    print(f"UNSOUND: conflict_atom flagged {w} but it IS satisfiable\n"
                          f"  defs={defs}\n  impl={impl}")
            # CONSISTENCY: contradictions() agrees with satisfiable() (when a world exists)
            if w in defs and not kb_unsat:
                checks += 1
                unsat = not abstract.satisfiable(w, defs, impl)
                if unsat != (w in listed):
                    bugs += 1
                    print(f"MISMATCH: {w} satisfiable={not unsat} listed={w in listed}\n"
                          f"  defs={defs}\n  impl={impl}")
        if bugs > 20:
            print("too many bugs, stopping"); return 1
    print(f"\nABSTRACT FUZZ DONE: {n} random knowledge bases, {checks} checks. bugs={bugs}.")
    return 0 if bugs == 0 else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000))
