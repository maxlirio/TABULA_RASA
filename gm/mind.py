"""The bot's whole mind: its name, the words it has learned by USAGE (the embedding
model), the ideas it has learned by DEFINITION (the abstract reasoner's knowledge), and
the running record of what's been said to it. Everything here persists to a brain file,
so it remembers across sessions. No tabletop, no blocks — just language.
"""
from __future__ import annotations

import json

from collections import Counter

from .corpus import seed_sentences
from .embed import Embed

CORPUS_CAP = 4000          # keep the most recent N sentences for retraining

# function words that shouldn't anchor a phrase
_PHRASE_STOP = {"the", "a", "an", "is", "are", "was", "be", "can", "to", "of", "in", "on",
                "and", "or", "not", "i", "you", "it", "that", "this", "he", "she", "we",
                "they", "my", "your", "me", "no", "yes", "do", "does", "did", "with", "for",
                "at", "as", "but", "so", "if", "then", "there", "here", "what", "who", "how"}


class Mind:
    def __init__(self):
        self.name: str | None = None
        # learned by definition (symbolic, exact) — the kept reasoning engine's state
        self.defs: dict[str, object] = {}
        self.implications: list = []
        self.atoms: set = set()
        self.opposites: list = []          # pairs the user said are opposites
        self.parts: list = []              # (part, whole): 'a bird has wings' / 'wing part of bird'
        self.abilities: list = []          # (thing, action): 'a bird can fly'
        self.actions: set = set()          # things IT can be commanded to do
        self.pos: dict = {}                # word -> part of speech you told me
        self.prefer: dict = {}             # concept -> the word you want it to SAY for it
        # learned by usage (neural, fuzzy)
        self.embed = Embed()
        self.corpus: list = []
        self.phrases: set = set()          # frequent word-pairs treated as one unit
        self._brain_path = None

    def bootstrap(self):
        """Give it a thin starting vocabulary if it knows nothing yet."""
        if not self.embed.words:
            self.embed.train(seed_sentences(), epochs=60, reps=2)
        if not self.actions:               # a few things it can do out of the box
            self.actions = {"run", "walk", "stop", "jump", "sit", "sleep", "spin",
                            "dance", "wave", "wait", "go", "turn"}

    def merge(self, tokens):
        """Glue known phrase-pairs into single tokens: ['good','dog'] -> ['good_dog']."""
        out, i = [], 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) in self.phrases:
                out.append(tokens[i] + "_" + tokens[i + 1]); i += 2
            else:
                out.append(tokens[i]); i += 1
        return out

    def detect_phrases(self):
        """Notice word-pairs you keep saying together, and start treating them as units."""
        bg = Counter()
        for s in self.corpus[-CORPUS_CAP:]:
            for i in range(len(s) - 1):
                a, b = s[i], s[i + 1]
                if a not in _PHRASE_STOP and b not in _PHRASE_STOP and len(a) > 2 and len(b) > 2:
                    bg[(a, b)] += 1
        for pair, c in bg.items():
            if c >= 3 and len(self.phrases) < 200:
                self.phrases.add(pair)

    def learn_sentence(self, tokens):
        """Learn a word's meaning from how it's used — the heart of the new approach."""
        if not tokens:
            return
        self.embed.observe(self.merge(tokens), reps=4)
        self.corpus.append(list(tokens))
        if len(self.corpus) > CORPUS_CAP:
            self.corpus = self.corpus[-CORPUS_CAP:]
        self._since = getattr(self, "_since", 0) + 1
        if self._since >= 20:                  # every so often, firm up recent words/phrases
            self._since = 0
            self.consolidate()

    def consolidate(self):
        """A few extra passes over recent sentences, so words and phrases you've used a lot
        settle in. This is what makes it noticeably better the more you talk to it."""
        self.detect_phrases()
        recent = [self.merge(s) for s in self.corpus[-300:]]
        if recent:
            self.embed.train(recent, epochs=3, reps=1)

    # ---- persistence ------------------------------------------------------

    def learned(self):
        return {
            "name": self.name,
            "defs": self.defs,
            "implications": [list(p) for p in self.implications],
            "atoms": sorted(self.atoms),
            "opposites": [list(p) for p in self.opposites],
            "parts": [list(p) for p in self.parts],
            "abilities": [list(p) for p in self.abilities],
            "actions": sorted(self.actions),
            "pos": self.pos,
            "prefer": self.prefer,
            "phrases": [list(p) for p in self.phrases],
            "embed": self.embed.to_dict(),
            "corpus": self.corpus,
        }

    def teach_from(self, d):
        self.name = d.get("name")
        self.defs = d.get("defs", {})
        self.implications = [tuple(p) for p in d.get("implications", [])]
        self.atoms = set(d.get("atoms", []))
        self.opposites = [tuple(p) for p in d.get("opposites", [])]
        self.parts = [tuple(p) for p in d.get("parts", [])]
        self.abilities = [tuple(p) for p in d.get("abilities", [])]
        self.actions = set(d.get("actions", []))
        self.pos = d.get("pos", {})
        self.prefer = d.get("prefer", {})
        self.phrases = {tuple(p) for p in d.get("phrases", [])}
        if d.get("embed", {}).get("words"):
            self.embed = Embed.from_dict(d["embed"])
        self.corpus = d.get("corpus", [])

    def save(self):
        if not self._brain_path:
            return
        try:
            with open(self._brain_path, "w") as f:
                json.dump(self.learned(), f)
        except OSError:
            pass
