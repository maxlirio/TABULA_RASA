"""Chat with real MEMORY, built around the from-scratch LM voice.

Memory is deliberately NOT in the neural net (a tiny char-LM can't store or recall facts).
It lives in plain, reliable code, in three parts:
  - EXACT memory: your name, its name, a persona ("you are ..."), and free notes.
  - KNOWLEDGE: everything you teach it becomes (subject, relation, object) triples it can
    query in both directions and reason over with inheritance (see gm/know.py).
  - THREAD memory: recent turns are fed back to the model as context for casual chat.

The LM only handles casual phrasing it doesn't recognise as memory/knowledge.
"""
import json
import os
import re

import torch

from gm.know import Knowledge

_DEFS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "defs.json")
try:
    DEFS = json.load(open(_DEFS_PATH))
except (OSError, ValueError):
    DEFS = {}

RECALL_NAME_ME = ("what is my name", "what's my name", "whats my name",
                  "do you know my name", "who am i")
RECALL_NAME_YOU = ("what is your name", "what's your name", "whats your name",
                   "do you have a name")
RECALL_NOTES = ("what do you remember", "what did i tell you",
                "what have we talked about", "do you remember anything")
FORGET = ("forget everything", "forget it all", "clear your memory", "wipe your memory")


class Chat:
    def __init__(self, voices, voice, path):
        self.voices = voices
        self.voice = voice
        self.path = path
        self.bot_name = None
        self.user_name = None
        self.persona = None           # "you are a helpful pirate" -> role it reports/uses
        self.notes = []
        self.know = Knowledge()
        self.history = []
        self.session = []             # this run's transcript (for "what did I say" recall)
        self.asked = set()            # curiosity questions already asked (don't repeat)
        self.rules = []               # standing instructions for THIS session (fed to the model)
        self._cooldown = 0            # rate-limit proactive questions (not after every turn)
        self._load()

    # --- persistence -----------------------------------------------------
    def _load(self):
        try:
            with open(self.path) as f:
                d = json.load(f)
            self.bot_name = d.get("bot_name")
            self.user_name = d.get("user_name")
            self.persona = d.get("persona")
            self.notes = d.get("notes", [])
            self.know.load(d.get("facts", []))
            self.asked = set(d.get("asked", []))
            self.history = [tuple(p) for p in d.get("history", [])][-20:]
        except (OSError, ValueError):
            pass

    def _save(self):
        try:
            with open(self.path, "w") as f:
                json.dump({"bot_name": self.bot_name, "user_name": self.user_name,
                           "persona": self.persona, "notes": self.notes,
                           "facts": self.know.dump(), "asked": sorted(self.asked),
                           "history": self.history[-20:]}, f)
        except OSError:
            pass

    # --- main entry ------------------------------------------------------
    def reply(self, text):
        text = text.strip()
        out = self._route(text)          # routed BEFORE recording, so "what did I say" excludes this
        self.history.append(("USER", text))
        self.history.append(("BOT", out))
        self.session.append(("USER", text))
        self.session.append(("BOT", out))
        self._save()
        return out

    def _route(self, text):
        # 1) ANSWERS that must be accurate stay deterministic (this is "the logic").
        for fn in (self._recall, self._history_intent, self._define_intent, self.know.ask):
            r = fn(text)
            if r is not None:
                return r
        # 2) Learn a fact: store it silently, react naturally. It does NOT pester with a
        #    question after every fact — it only wonders out loud when explicitly asked
        #    ("what are you curious about"), and only about things it genuinely can't infer.
        taught = self.know.teach(text)
        if taught is not None:
            if "contradict" in taught.lower() or "can't be right" in taught.lower():
                return taught
            ack = self._generate(text)
            # proactively wonder — but only a GENUINE question, and not every single turn
            if self._cooldown <= 0:
                q = self.know.curiosity(recent=self.know.last, asked=self.asked)
                if q:
                    self.asked.add(q)
                    self._cooldown = 2
                    return f"{ack} {q}"
            self._cooldown -= 1
            return ack
        # 3) Other state changes (name / persona / remember / forget) run silently.
        self._apply_memory(text)
        # 4) Respond through the model — no canned acknowledgements.
        return self._generate(text)

    # --- exact retrieval (answers) --------------------------------------
    def _recall(self, text):
        t = text.lower().rstrip("?.!")
        if t in RECALL_NAME_ME:
            return (f"You're {self.user_name}." if self.user_name
                    else "I don't know your name yet - tell me 'my name is ...'.")
        if t in RECALL_NAME_YOU or t == "who are you":
            who = f"I'm {self.bot_name}." if self.bot_name else "I don't have a name yet."
            if self.persona:
                who += f" I'm acting as {self.persona}."
            return who
        if t == "what are you" and self.persona:
            return f"I'm acting as {self.persona}."
        if t in ("what are you curious about", "what do you want to know",
                 "ask me something", "ask me a question", "what do you want to learn"):
            q = self.know.curiosity(asked=self.asked)
            if q:
                self.asked.add(q)
                return q
            return "Teach me something and I'll get curious about it."
        if t in RECALL_NOTES:
            bits = []
            if self.notes:
                bits.append("you told me: " + "; ".join(self.notes))
            if self.know.triples:
                bits.append(f"I also know {len(self.know.triples)} facts you taught me")
            return ("I remember " + "; and ".join(bits) + ".") if bits \
                else "I don't have anything saved yet."
        return None

    # --- transcript recall ("what was the first thing I said this chat?") ---
    def _history_intent(self, text):
        t = text.lower().rstrip("?.!")
        me = [m for r, m in self.session if r == "USER"]
        you = [m for r, m in self.session if r == "BOT"]
        ords = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2,
                "fourth": 3, "4th": 3, "fifth": 4, "5th": 4, "last": -1}
        m = re.match(r"what (?:is|was|did) (?:the )?(\w+) thing i sa(?:id|y)", t)
        if m and m.group(1) in ords:
            i = ords[m.group(1)]
            if me and -len(me) <= i < len(me):
                return f'The {m.group(1)} thing you said was: "{me[i]}"'
            return "You haven't said that many things yet this chat."
        if re.search(r"what did i (?:just )?say|(?:what was|what's) the last thing i said", t):
            return f'You just said: "{me[-1]}"' if me else "You haven't said anything yet."
        if re.search(r"what did you (?:just )?say|(?:what was|what's) the last thing you said", t):
            return f'I just said: "{you[-1]}"' if you else "I haven't said anything yet."
        if re.search(r"how many (?:things|messages) have i (?:said|sent)", t):
            return f"You've said {len(me)} thing(s) so far this chat."
        return None

    # --- silent state changes (logic only; the reply comes from the model) ---
    def _apply_memory(self, text):
        t = text.lower().rstrip("?.!")
        # standing instruction? keep it so it's fed to the model before every future reply.
        # We only decide it's a rule worth keeping; the MODEL learned to apply it.
        if ("instead of" in t or "now means" in t or t.startswith("whenever")
                or re.match(r"replace \w+ with \w+", t) or re.match(r"use \w+ for \w+", t)):
            rule = re.sub(r"\s*(from now on|please)\.?$", "", t).strip()
            if rule and rule not in self.rules:
                self.rules.append(rule)
            return
        m = re.match(r"(?:your name is|i'?ll call you|i will call you|call yourself) (\w+)$", t)
        if m:
            self.bot_name = m.group(1).title()
            return
        # persona ONLY on explicit role phrasing — never bare "you are ..." (that's chat)
        m = re.match(r"(?:act like|act as|pretend to be|roleplay as|your role is|"
                     r"from now on,? (?:be|act like|act as)) (?:a |an )?(.+)$", t)
        if m:
            self.persona = m.group(1).strip()
            return
        m = re.match(r"(?:my name is|call me|i am called|i'?m called) (\w+)$", t)
        if m:
            self.user_name = m.group(1).title()
            return
        m = re.match(r"remember (?:that )?(.+)", t)
        if m and not self._looks_like_fact(m.group(1)):
            self.notes.append(m.group(1).strip())
            return
        if t in FORGET:
            self.notes.clear()
            self.know = Knowledge()
            self.persona = None
            self.rules.clear()

    def _define_intent(self, text):
        """Exact dictionary lookup — definitions must be reliable, not generated."""
        t = text.lower().strip().rstrip("?.!")
        m = (re.match(r"^what does (\w+) (?:mean|stand for)$", t)
             or re.match(r"^what'?s (\w+) mean$", t)
             or re.match(r"^define (\w+)$", t)
             or re.match(r"^(?:what is |what'?s )?(?:the )?meaning of (\w+)$", t))
        if not m:
            return None
        w = m.group(1)
        if w in DEFS:
            return f"{w} means {DEFS[w]}."
        if any(s == w for s, _, _ in self.know.triples):    # else fall to taught knowledge
            return None
        return f"I don't have a definition for '{w}' yet."

    def _looks_like_fact(self, s):
        """So 'remember that a cat has fur' goes to the knowledge engine, not a flat note."""
        return bool(re.search(r"\b(is a|is an|are|has|have|can)\b", s.lower()))

    # --- neural reply, with persona + recent turns as context ------------
    def _generate(self, text):
        model, coder = self.voices[self.voice]
        # Keep only the last exchange as context — the real memory is the knowledge store, and
        # a long history derails this small model. Most turns generate cleanly from a fresh seed.
        # feed the standing rules (the model learned to obey in-context RULE: lines), but NOT
        # the chat history — feeding its own prior replies back made it perseverate on its own
        # thread ("won't be put off it"). Real continuity lives in the knowledge store instead.
        rule_lines = [f"RULE: {r}" for r in self.rules[-3:]]
        seed = ("\n".join(rule_lines) + "\n" if rule_lines else "") + f"USER: {text}\nBOT: "
        ids = coder.encode(seed) or [coder.stoi.get("\n", 0)]
        ban = [coder.stoi["<unk>"]] if "<unk>" in getattr(coder, "stoi", {}) else None
        out_ids = model.generate(torch.tensor([ids]), 50, temp=0.4, ban=ban)[0].tolist()
        gen = coder.decode(out_ids[len(ids):])    # decode ONLY the new tokens (robust)
        for cut in ("\nUSER", "\nBOT", "\nRULE", "USER:", "BOT:", "RULE:", '"', " '"):
            if cut in gen:                        # quotes (incl. dialogue ') stop book-bleed
                gen = gen.split(cut)[0]
        gen = gen.replace("\n", " ").strip()
        # keep it to a sentence or two so it can't ramble into novel prose
        if "reward:" not in gen:
            parts = re.split(r"(?<=[.!?])\s+", gen)
            gen = " ".join(parts[:2]).strip()
        return gen or "..."
