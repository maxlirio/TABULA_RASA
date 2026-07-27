"""SELF-DESIGNING TENDRILS — the brain invents the FORM of its world-model, not just its parameters.

A plain tendril fits ONE fixed model shape: "command a adds a constant vector". When the world doesn't
work that way, that shape can't be made to fit no matter how much data you pour in -- and this session
proved it three times, each fixed by a HUMAN redesigning the shape (heading-conditioning, frame
recovery, intent/affordance gating). This module does that redesign itself.

It holds a COMPOSABLE SPACE of model forms and, from interaction data, discovers the smallest form that
actually predicts -- including which hidden observation variable the effect depends on, and whether an
effect is gated by a precondition. Selection is by HELD-OUT prediction error (a form that only fits the
data it trained on is rejected), with an Occam rule: among forms that predict within noise of the best,
take the simplest. If NO form predicts, it says so instead of faking a fit.

Honest scope: this is model-STRUCTURE discovery over a defined, composable space (which variables gate
the dynamics, whether there's a precondition) -- not unbounded program synthesis. But it genuinely
designs the structure from data rather than being told it, which is the thing that was missing.
"""
import numpy as np


def _quantize(v, lo, w, nb):
    if w <= 0:
        return 0
    return int(min(max(np.floor((v - lo) / w), 0), nb - 1))


class Const:
    """f(o,a) = o + d_a. One constant effect per command (the plain tendril)."""
    name = "constant"

    def fit(self, O, A, O2, nact):
        self.dim = O.shape[1]
        self.d = np.zeros((nact, self.dim))
        for a in range(nact):
            D = O2[A == a] - O[A == a]
            if len(D):
                self.d[a] = np.median(D, axis=0)          # median: a blocked no-move can't sway it
        self.complexity = nact * self.dim
        return self

    def predict(self, o, a):
        return o + self.d[int(a)]

    def describe(self):
        return "each command adds a fixed vector (no state dependence)"


class Cond:
    """f(o,a) = o + d_(a, bin(o[k])). The effect DEPENDS on observation feature k -- the form the brain
    has to invent for a body that faces a direction (effect conditioned on heading)."""

    def __init__(self, k, nb):
        self.k, self.nb = k, nb
        self.name = f"depends-on-feature[{k}]"

    def fit(self, O, A, O2, nact):
        self.dim = O.shape[1]
        col = O[:, self.k]
        self.lo, hi = float(col.min()), float(col.max())
        self.w = (hi - self.lo) / self.nb if hi > self.lo else 0.0
        b = np.array([_quantize(v, self.lo, self.w, self.nb) for v in col])
        self.d = np.zeros((nact, self.nb, self.dim))
        for a in range(nact):
            for bb in range(self.nb):
                D = O2[(A == a) & (b == bb)] - O[(A == a) & (b == bb)]
                if len(D):
                    self.d[a, bb] = np.median(D, axis=0)
        self.complexity = nact * self.nb * self.dim
        return self

    def predict(self, o, a):
        return o + self.d[int(a), _quantize(o[self.k], self.lo, self.w, self.nb)]

    def describe(self):
        return f"the effect of each command depends on observation feature #{self.k}"


class Gated:
    """A base form, but a command's effect is BLOCKED at states where it was observed to do nothing
    despite the base predicting motion -- the precondition/affordance structure (walls). Composed on
    top of another form, so this is the brain building a bigger form out of smaller ones."""

    def __init__(self, base):
        self.base, self.name = base, base.name + "+gated"

    def fit(self, O, A, O2, nact):
        self.base.fit(O, A, O2, nact)
        self.dim = O.shape[1]
        self.blocked = set()
        for o, a, o2 in zip(O, A, O2):
            pred = self.base.predict(o, a)
            if np.abs(pred - o).sum() > 0.3 and np.abs(o2 - o).sum() < 0.3:   # base says move, world didn't
                self.blocked.add(self._key(pred))
        self.complexity = self.base.complexity + len(self.blocked)
        return self

    def _key(self, o):
        return tuple(np.round(o).astype(int))

    def predict(self, o, a):
        pred = self.base.predict(o, a)
        return o.copy() if self._key(pred) in self.blocked else pred

    def describe(self):
        return self.base.describe() + f", blocked by a precondition at {len(self.blocked)} states"


def _heldout_error(form, O, A, O2):
    return float(np.mean([np.abs(o2 - form.predict(o, a)).sum() for o, a, o2 in zip(O, A, O2)]))


class Discoverer:
    """Given interaction data, DESIGN the model form: search the composable space, keep the simplest
    form that predicts held-out transitions, grow the space if none do."""

    def __init__(self, tol=1.15, good=0.15):
        self.tol = tol          # a form within tol x of the best error counts as "as good" -> prefer simpler
        self.good = good        # held-out error below this = the form genuinely predicts

    def discover(self, O, A, O2, nact, seed=0):
        O, A, O2 = np.asarray(O, float), np.asarray(A, int), np.asarray(O2, float)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(O))
        cut = int(len(O) * 0.7)
        tr, ho = idx[:cut], idx[cut:]
        dim = O.shape[1]

        # candidate forms: the plain one, then "depends on feature k" for every feature worth trying
        cands = [Const()]
        for k in range(dim):
            distinct = len(np.unique(np.round(O[:, k], 2)))
            if distinct > 1:
                cands.append(Cond(k, nb=min(distinct, 6)))

        scored = []
        for f in cands:
            f.fit(O[tr], A[tr], O2[tr], nact)
            scored.append((f, _heldout_error(f, O[ho], A[ho], O2[ho])))
        scored.sort(key=lambda fe: fe[1])
        best_err = scored[0][1]

        # GROW: if even the best form leaves real error, the world needs MORE structure -> compose
        # gating. Try gating the SIMPLEST base (constant) and the best conditioned base, so Occam can
        # keep the smallest gated form that works (a wall is a precondition on a constant effect, not
        # a reason to also condition on position).
        if best_err > self.good:
            bases = [Const()]
            if not isinstance(scored[0][0], Const):
                bases.append(scored[0][0].__class__(scored[0][0].k, scored[0][0].nb))
            for base in bases:
                g = Gated(base)
                g.fit(O[tr], A[tr], O2[tr], nact)
                scored.append((g, _heldout_error(g, O[ho], A[ho], O2[ho])))
            scored.sort(key=lambda fe: fe[1])
            best_err = scored[0][1]

        # Occam: among forms within tol x of the best error, choose the SIMPLEST
        near = [fe for fe in scored if fe[1] <= best_err * self.tol + 1e-9]
        chosen, chosen_err = min(near, key=lambda fe: fe[0].complexity)

        report = {
            "chosen": chosen.name, "chosen_error": round(chosen_err, 4),
            "predicts": chosen_err < self.good,
            "why": chosen.describe(),
            "candidates": [(f.name, round(e, 4), f.complexity) for f, e in scored],
            "invented_structure": not isinstance(chosen, Const),
        }
        return chosen, report
