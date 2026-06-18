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
from gm.tools import Tools

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
GREET_IN = ("hi", "hello", "hey", "heya", "hiya", "howdy", "yo", "hi there", "hey there",
            "good morning", "good afternoon", "good evening", "greetings", "sup", "hello there")


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
        self.defs = {}                # word -> definition the USER taught it
        self._cooldown = 0            # rate-limit proactive questions (not after every turn)
        self._load()
        self.tools = Tools(self.know)  # executes the CALLs the model emits (see gm/tools.py)

    # --- persistence -----------------------------------------------------
    def _load(self):
        try:
            with open(self.path) as f:
                d = json.load(f)
            self.bot_name = d.get("bot_name")
            self.user_name = d.get("user_name")
            self.persona = d.get("persona")
            self.notes = d.get("notes", [])
            self.defs = d.get("defs", {})
            self.know.load(d.get("facts", []))
            self.asked = set(d.get("asked", []))
            self.history = [tuple(p) for p in d.get("history", [])][-20:]
        except (OSError, ValueError):
            pass

    def _save(self):
        try:
            with open(self.path, "w") as f:
                json.dump({"bot_name": self.bot_name, "user_name": self.user_name,
                           "persona": self.persona, "notes": self.notes, "defs": self.defs,
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
        # 1b) Learn a DEFINITION the user gives ("X means ...") — before the fact-engine, which
        #     would otherwise mis-store it. Stored so "what does X mean" returns it.
        if self._learn_definition(text):
            return self._react(text)         # natural acknowledgement from the model
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
        # 4) Respond through the model — which may CALL a tool (reward/knowledge) or just chat.
        return self._react(text)

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
            self.tools.know = self.know       # keep the tool dispatcher pointed at live memory
            self.persona = None
            self.rules.clear()

    _NOT_A_WORD = ("what", "it", "that", "this", "he", "she", "they", "there", "here",
                   "i", "you", "we", "who", "which")

    def _learn_definition(self, text):
        """Teach a definition for a word it doesn't know yet: 'X means ...', 'define X as ...',
        'X is defined as ...', 'X is short for ...'. A user-taught definition overrides the
        built-in dictionary. Returns True if a definition was stored."""
        t = text.strip().rstrip(".!")
        m = (re.match(r"(?:the word |the term )?(\w+) means (.+)$", t, re.I)
             or re.match(r"define (\w+) as (.+)$", t, re.I)
             or re.match(r"(\w+) is defined as (.+)$", t, re.I)
             or re.match(r"(?:the )?definition of (\w+) is (.+)$", t, re.I)
             or re.match(r"(\w+) is short for (.+)$", t, re.I))
        if not m:
            return False
        word, meaning = m.group(1).lower(), m.group(2).strip()
        if word in self._NOT_A_WORD or len(meaning) < 2:   # avoid "that means a lot" etc.
            return False
        self.defs[word] = meaning
        return True

    def _define_intent(self, text):
        """Exact dictionary lookup — definitions must be reliable, not generated. Checks what
        the USER taught first, then the built-in dictionary, then taught is-a knowledge."""
        t = text.lower().strip().rstrip("?.!")
        m = (re.match(r"^what does (\w+) (?:mean|stand for)$", t)
             or re.match(r"^what'?s (\w+) mean$", t)
             or re.match(r"^define (\w+)$", t)
             or re.match(r"^(?:what is |what'?s )?(?:the )?meaning of (\w+)$", t))
        if not m:
            return None
        w = m.group(1)
        if w in self._NOT_A_WORD:        # "what does that mean" is chat, not a lookup
            return None
        if w in self.defs:
            return f"{w} means {self.defs[w]}."
        if w in DEFS:
            return f"{w} means {DEFS[w]}."
        if any(s == w for s, _, _ in self.know.triples):    # else fall to taught knowledge
            return None
        return f"I don't have a definition for '{w}' yet - tell me '{w} means ...' and I'll learn it."

    def _looks_like_fact(self, s):
        """So 'remember that a cat has fur' goes to the knowledge engine, not a flat note."""
        return bool(re.search(r"\b(is a|is an|are|has|have|can)\b", s.lower()))

    # --- neural reply, with persona + recent turns as context ------------
    def _ban(self, coder):
        return [coder.stoi["<unk>"]] if "<unk>" in getattr(coder, "stoi", {}) else None

    def _pre(self):                               # standing RULE: lines fed before every turn
        return "".join(f"RULE: {r}\n" for r in self.rules[-3:])

    def _raw(self, seed, temp, top_k, max_new=50):
        """Sample the model from a seed and return ONLY the newly generated text (with newlines
        intact, so a CALL line can be parsed)."""
        model, coder = self.voices[self.voice]
        ids = coder.encode(seed) or [coder.stoi.get("\n", 0)]
        out = model.generate(torch.tensor([ids]), max_new, temp=temp, top_k=top_k,
                             ban=self._ban(coder))[0].tolist()
        return coder.decode(out[len(ids):])

    def _clean(self, gen, greet=False):
        """Trim a BOT reply to a sentence or two, kill transcript markers, book-quote bleed,
        and the LM's occasional looped clause."""
        for cut in ("\nUSER", "\nBOT", "\nRULE", "\nCALL", "\nRESULT",
                    "USER:", "BOT:", "RULE:", "CALL:", "RESULT:", '"', " '"):
            if cut in gen:
                gen = gen.split(cut)[0]
        gen = gen.replace("\n", " ").strip()
        if "reward:" not in gen:
            gen = re.sub(r"\b(.{8,40}?),?\s+and\s+\1\b", r"\1", gen, flags=re.I)
            parts, seen, kept = re.split(r"(?<=[.!?])\s+", gen), set(), []
            for p in parts:
                k = p.strip().lower()
                if k and k not in seen:
                    seen.add(k)
                    kept.append(p)
            gen = " ".join(kept[:1 if greet else 2]).strip()
        return gen

    def _generate(self, text):
        # Direct reply (no tool). Rules -> tight sampling so it obeys; greetings -> resample a
        # few times and keep the first that actually reads like a greeting.
        temp, top_k = (0.2, 3) if self.rules else (0.4, 40)
        greet = text.lower().strip().rstrip("?.!") in GREET_IN
        best = ""
        for _ in range(3 if greet else 1):
            gen = self._clean(self._raw(self._pre() + f"USER: {text}\nBOT: ", temp, top_k), greet)
            if not greet:
                return gen or "..."
            best = best or gen
            first = (gen.lower().split() or [""])[0].strip(".!,")
            if first in ("hi", "hello", "hey", "heya", "hiya", "howdy", "good", "yo"):
                return gen
        return best or "..."

    def _reward_intent(self, text):
        """INTERIM bridge: detect an explicit reward request by pattern and pull out the goal.
        Returns the goal string, "" for a goal-less request (ask), or None if not a reward ask.
        This exists only until the model is trained to emit `CALL: reward ...` itself; then the
        ReAct path below supersedes it and this can be deleted."""
        t = text.lower().strip().rstrip(".!?")
        if "reward" not in t:
            return None
        for pat in (r"reward(?:\s+(?:system|function|signal))?\s+for\s+(.+)$",
                    r"how should i reward\s+(.+)$", r"reward it for\s+(.+)$"):
            m = re.search(pat, t)
            if m:
                return m.group(1).strip()
        if re.search(r"(make|design|set up|give me|need|create|want|build).*\breward\b", t):
            return ""                              # reward request with no goal -> ask
        return None

    _MATH_WORDS = ("plus", "minus", "times", "divided", "multiplied", "squared", "cubed",
                   "power", "sqrt", "square root", "factorial", "percent", "subtract")

    def _math_intent(self, text):
        """INTERIM bridge: detect an arithmetic question and pull out the expression. Requires a
        digit AND an operator (symbol or math word) so 'i have 2 cats' isn't treated as math.
        Returns the expression, or None. Superseded once the model emits `CALL: calc ...`."""
        t = text.lower().strip().rstrip("?.")
        if not re.search(r"\d", t):
            return None
        has_op = (re.search(r"[+\-*/^%]", t) or re.search(r"\d\s*x\s*\d", t)
                  or re.search(r"\d\s*!", t) or any(w in t for w in self._MATH_WORDS))
        if not has_op:
            return None
        return re.sub(r"^(what'?s|what is|whats|calculate|compute|how much is|how many is|"
                      r"solve|tell me|please)\s+", "", t).strip() or t

    def _react(self, text):
        """ReAct: let the model decide whether this turn needs a TOOL. If it emits a CALL we
        RUN it (gm/tools.py) and feed the RESULT back for it to verbalise; otherwise it just
        replies. The logic lives in code; the net only routes to it and phrases the answer."""
        expr = self._math_intent(text)             # interim: route math to the calculator now
        if expr is not None:
            ans = self.tools.run("calc " + expr)
            if ans != "error":                     # if it doesn't parse, fall through to chat
                return ans
        goal = self._reward_intent(text)           # interim: route reward asks to the tool now
        if goal is not None:
            if not goal:
                return "a reward for what? tell me the goal and i'll build it."
            return f"reward: {self.tools.run('reward ' + goal)}"
        pre = self._pre()
        # Stage 1: continue from the bare USER line (no forced BOT:) — a trained model emits
        # either "CALL: ..." (needs a tool) or "BOT: ..." (just chat).
        head = self._raw(pre + f"USER: {text}\n", temp=0.2, top_k=5, max_new=24)
        m = re.match(r"\s*CALL:\s*([^\n]+)", head)
        if not m:
            return self._generate(text)           # no tool -> ordinary reply
        call = m.group(1).strip()
        result = self.tools.run(call)
        if call.split()[:1] == ["reward"]:        # reward spec is authoritative — return exactly
            return f"reward: {result}"
        # knowledge call: let the model phrase the engine's result, fall back to a plain one
        reply = self._clean(self._raw(
            pre + f"USER: {text}\nCALL: {call}\nRESULT: {result}\nBOT: ", temp=0.3, top_k=20))
        if reply:
            return reply
        return "hmm, I'm not sure." if result in ("error", "") else result
