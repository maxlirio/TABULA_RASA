"""A tiny word-embedding model, built from scratch (numpy only — no frameworks, no
pretrained weights). This is the part that learns word meaning from USAGE rather than
from definitions: words that keep similar company end up with similar vectors.

It's plain skip-gram with negative sampling, with hand-written gradients. On the small
amount of text one person types it stays rough — but similar words really do cluster,
and that's the point: meaning the bot was never explicitly told.
"""
from __future__ import annotations

import numpy as np


class Embed:
    def __init__(self, dim=24, window=2, neg=5, lr=0.05, seed=1):
        self.dim, self.window, self.neg, self.lr = dim, window, neg, lr
        self.rng = np.random.default_rng(seed)
        self.vocab: dict[str, int] = {}     # word -> row index
        self.words: list[str] = []          # row index -> word
        self.counts: list[int] = []         # how often each word has been seen
        self.W = np.zeros((0, dim))          # "center" vectors (the meaning we read out)
        self.C = np.zeros((0, dim))          # "context" vectors (used only in training)

    # ---- vocabulary -------------------------------------------------------

    def _row(self):
        return (self.rng.random(self.dim) - 0.5) / self.dim

    def _idx(self, w, add=True):
        i = self.vocab.get(w, -1)
        if i >= 0 or not add:
            return i
        i = len(self.words)
        self.vocab[w] = i
        self.words.append(w)
        self.counts.append(0)
        self.W = np.vstack([self.W, self._row()])
        self.C = np.vstack([self.C, self._row()])
        return i

    def knows(self, w):
        return w in self.vocab

    # ---- learning ---------------------------------------------------------

    def observe(self, tokens, reps=1):
        """Learn from one sentence (a list of word tokens), online."""
        ids = [self._idx(w) for w in tokens if w]
        for w in ids:
            self.counts[w] += 1
        for _ in range(reps):
            for pos, c in enumerate(ids):
                lo, hi = max(0, pos - self.window), min(len(ids), pos + self.window + 1)
                for o in range(lo, hi):
                    if o != pos:
                        self._step(c, ids[o])

    def train(self, corpus, epochs=5, reps=1):
        """Batch-train over many sentences (each a token list)."""
        for _ in range(epochs):
            order = self.rng.permutation(len(corpus))
            for k in order:
                self.observe(corpus[int(k)], reps=reps)

    def _neg(self, avoid):
        n = len(self.words)
        for _ in range(8):
            j = int(self.rng.integers(n))
            if j != avoid:
                return j
        return j

    @staticmethod
    def _sig(x):
        x = -30.0 if x < -30.0 else (30.0 if x > 30.0 else x)   # stable, no overflow
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        z = np.exp(x)
        return z / (1.0 + z)

    def _step(self, c, o):
        """One skip-gram + negative-sampling SGD update for a (center, context) pair."""
        vc = self.W[c]
        # positive context word: push vc and uo together
        uo = self.C[o]
        g = self._sig(float(vc @ uo)) - 1.0
        grad_c = g * uo
        self.C[o] -= self.lr * g * vc
        # negative samples: push apart
        for _ in range(self.neg):
            n = self._neg(o)
            un = self.C[n]
            s = self._sig(float(vc @ un))
            grad_c += s * un
            self.C[n] -= self.lr * s * vc
        # clip the gradient so a vector can't blow up on this tiny, repetitive data
        gn = np.linalg.norm(grad_c)
        if gn > 5.0:
            grad_c *= 5.0 / gn
        self.W[c] -= self.lr * grad_c

    def pull(self, a, b, alpha=0.35):
        """Nudge word a's vector toward word b's — used when you TELL it 'a is a b',
        so a definition also shapes the learned-from-usage space. The symbolic and
        neural halves reinforce each other."""
        ia, ib = self._idx(a), self._idx(b)
        self.W[ia] += alpha * (self.W[ib] - self.W[ia])

    def push(self, a, b, alpha=0.25):
        """Nudge a away from b — for opposites ('hot is the opposite of cold')."""
        ia, ib = self._idx(a), self._idx(b)
        d = self.W[ia] - self.W[ib]
        self.W[ia] += alpha * d
        n = np.linalg.norm(self.W[ia])
        if n > 4.0:
            self.W[ia] *= 4.0 / n

    # ---- reading it back --------------------------------------------------

    def vec(self, w):
        i = self.vocab.get(w, -1)
        return self.W[i].copy() if i >= 0 else None

    def cosine(self, a, b):
        va, vb = self.vec(a), self.vec(b)
        if va is None or vb is None:
            return None
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(va @ vb / (na * nb))

    def similar(self, w, k=5, min_count=1):
        i = self.vocab.get(w, -1)
        if i < 0 or len(self.words) < 2:
            return []
        v = self.W[i]
        vn = float(np.linalg.norm(v))
        if vn < 1e-9:
            return []
        sims = []
        for j, u in enumerate(self.W):
            if j == i or self.counts[j] < min_count:
                continue
            un = float(np.linalg.norm(u))
            if un >= 1e-9:
                sims.append((self.words[j], float(u @ v) / (un * vn)))
        sims.sort(key=lambda p: -p[1])
        return sims[:k]

    # ---- persistence ------------------------------------------------------

    def to_dict(self):
        return {"dim": self.dim, "window": self.window, "neg": self.neg, "lr": self.lr,
                "words": self.words, "counts": self.counts,
                "W": self.W.tolist(), "C": self.C.tolist()}

    @classmethod
    def from_dict(cls, d):
        e = cls(dim=d["dim"], window=d.get("window", 2), neg=d.get("neg", 5),
                lr=d.get("lr", 0.05))
        e.words = list(d["words"])
        e.counts = list(d["counts"])
        e.vocab = {w: i for i, w in enumerate(e.words)}
        e.W = np.array(d["W"]) if d["W"] else np.zeros((0, e.dim))
        e.C = np.array(d["C"]) if d["C"] else np.zeros((0, e.dim))
        return e
