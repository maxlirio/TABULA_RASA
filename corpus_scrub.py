#!/usr/bin/env python3
"""Corpus scrubs shared by t4_run.py (the real run) and preflight.py (the gate).

These live in ONE module on purpose. The gate is only worth running if it assembles the SAME corpus
the run will train on, and the last time an assembly step existed as two copies they silently drifted
apart. Anything that mutates the assembled corpus belongs here, imported by both.
"""
import re

# Project Gutenberg TYPESETTING markup. It is not prose, but it rides along inside otherwise-good
# text: [Illustration] appears in 0.9% of v5's blocks and _italic_ markers in 9.1%, and the fable
# scrape carried them at 10%/22%. It is the same disease that made run #1 recite a memorized fable
# index -- the model cannot tell "junk that reliably appears in books" from "language" -- so strip it
# from every source rather than only the one we happened to catch.
_BRACKET = re.compile(r"\[[^\]\n]*\]")        # [Illustration], [Illustration: a fox], [Footnote 3]
_ITALIC = re.compile(r"_([^_\n]+)_")          # _It is unwise..._  -> keep the words, lose the markers
_ORPHAN_PUNCT = re.compile(r"\s+([.,;:!?])")  # the strips can orphan punctuation: "moral ." -> "moral."
_WS = re.compile(r"[ \t]+")


def scrub_markup(text):
    """Strip Gutenberg typesetting markup, preserving the prose (and the fable morals) inside it."""
    text = _BRACKET.sub(" ", text)
    text = _ITALIC.sub(r"\1", text)
    text = text.replace("_", " ")             # any unpaired marker left behind
    text = _ORPHAN_PUNCT.sub(r"\1", text)
    return "\n".join(_WS.sub(" ", ln).strip() for ln in text.split("\n"))


def markup_count(text):
    """How many blocks still carry markup — for the preflight gate to assert on."""
    blocks = [b for b in text.split("\n\n") if b.strip()]
    bad = sum(1 for b in blocks if _BRACKET.search(b) or "_" in b)
    return bad, len(blocks)
