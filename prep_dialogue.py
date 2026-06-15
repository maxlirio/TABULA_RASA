#!/usr/bin/env python3
"""Turn raw sources into TURN-STRUCTURED dialogue so the models learn to *answer*, not
just continue prose. Output is a single chat.txt per voice, shaped like:

    USER: <something said>
    BOT: <the reply>
    USER: ...
    BOT: ...
    <blank line between separate conversations>

Apollo learns from the Cornell movie-dialogue corpus (modern back-and-forth talk).
Arthur I learns from Shakespeare's plays, which are already speaker-by-speaker dialogue
(alternating speeches become alternating turns) — so its replies stay archaic.

Nothing is hardcoded into the bot; this only reshapes public text into a question/answer
format the same from-scratch net then learns from.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CORNELL = os.path.join(HERE, "cornell movie-dialogs corpus")


def clean(s):
    s = (s.replace("“", '"').replace("”", '"').replace("‘", "'")
         .replace("’", "'").replace("—", " - ").replace("–", "-"))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def write_chat(convos, subdir):
    """convos: list of conversations, each a list of utterance strings (in order)."""
    out_dir = os.path.join(HERE, "data", subdir)
    os.makedirs(out_dir, exist_ok=True)
    lines, kept = [], 0
    for turns in convos:
        turns = [clean(t) for t in turns]
        turns = [t for t in turns if 1 <= len(t) <= 220]
        if len(turns) < 2:
            continue
        kept += 1
        for i, t in enumerate(turns):
            lines.append(("USER: " if i % 2 == 0 else "BOT: ") + t)
        lines.append("")
    path = os.path.join(out_dir, "chat.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[{subdir}] {kept:,} conversations -> {path} "
          f"({os.path.getsize(path):,} bytes)")


def build_modern():
    lines = {}
    with open(os.path.join(CORNELL, "movie_lines.txt"), encoding="latin-1") as f:
        for ln in f:
            p = ln.split(" +++$+++ ")
            if len(p) == 5:
                lines[p[0]] = p[4].strip()
    convos = []
    with open(os.path.join(CORNELL, "movie_conversations.txt"), encoding="latin-1") as f:
        for ln in f:
            p = ln.split(" +++$+++ ")
            if len(p) != 4:
                continue
            ids = re.findall(r"L\d+", p[3])
            utt = [lines[i] for i in ids if i in lines]
            if len(utt) >= 2:
                convos.append(utt)
    write_chat(convos, "dialogue_modern")


SPEAKER = re.compile(r"^([A-Z][A-Z'’.\- ]{1,28})\.$")
STAGE = re.compile(r"^(\[|Enter |Exit|Exeunt|Re-enter|Re-?enter|Scene|SCENE|Manet)\b")


def build_classical():
    text = open(os.path.join(HERE, "data", "classical", "pg100.txt"),
               encoding="utf-8", errors="ignore").read()
    m = re.search(r"\*\*\* ?START OF.*?\*\*\*", text, re.S)   # drop Gutenberg front-matter
    if m:
        text = text[m.end():]
    m = re.search(r"\*\*\* ?END OF", text)
    if m:
        text = text[:m.start()]

    convos, scene, speech = [], [], []
    has_speaker = False

    def flush_speech():
        nonlocal has_speaker
        if speech and has_speaker:
            scene.append(" ".join(speech))
        speech.clear()
        has_speaker = False

    for raw in text.split("\n"):
        line = raw.strip()
        if re.match(r"^(ACT|SCENE|EPILOGUE|PROLOGUE|THE END|FINIS)\b", line, re.I):
            flush_speech()
            if len(scene) >= 2:
                convos.append(scene[:])
            scene.clear()
            continue
        if STAGE.match(line):                       # ignore stage directions
            continue
        if SPEAKER.match(line):                     # a new speaker -> prior speech is one turn
            flush_speech()
            has_speaker = True
            continue
        if has_speaker and line:
            speech.append(line)
    flush_speech()
    if len(scene) >= 2:
        convos.append(scene)
    write_chat(convos, "dialogue_classical")


if __name__ == "__main__":
    build_modern()
    build_classical()
