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

# --- a grounded SENSE OF SELF: what it IS / CAN do / CANNOT do. Stated facts, not generated,
# so it stops confabulating a persona ("i'm going to the hospital"). Same idea as the tools:
# reliability from code, not from the net guessing. ---
SELF_IS = ("a small language model that runs right here on your computer. i was built from "
           "scratch - there's no big AI inside me, just a little neural network plus some "
           "reliable tools")
SELF_CAN = ["chat with you", "remember your name and facts you teach me", "learn word definitions",
            "do math", "tell the date and time", "answer questions about things you've taught me",
            "turn a goal into a reward spec"]
CAN_HINT = ("math", "calculat", "add", "subtract", "multipl", "divide", "count", "remember",
            "recall", "learn", "defin", "time", "date", "day", "year", "reward", "answer",
            "help", "chat", "talk", "chitchat")
CANNOT_HINT = ("see", "watch", "look", "hear", "listen", "go ", "going", "walk", "drive",
               "travel", "fly", "eat", "drink", "sleep", "feel", "touch", "smell", "taste",
               "browse", "internet", "web", "google", "online", "phone", "call", "email",
               "text someone", "buy", "cook", "sing", "move", "leave", "visit", "meet", "come over")
# worldly first-person claims a program can't truthfully make -> confabulation to suppress.
# (good first-person like "i'm good", "i like", "i can help", "i don't know" is NOT matched.)
CONFAB = [re.compile(p) for p in (
    r"\bi (?:'?m|am) going (to|back)\b(?!\s+(help|tell|show|try|make|answer|remember|be|do|give|"
    r"teach|guess|say|keep|let|find|get better|need))",
    r"\bi (went|drove|walked|flew|traveled|rode|ran|drove) (to|home|back|down|up|over|into)\b",
    r"\bi (have|had|'?ve got|got) (a|an|my|two|three|four|\d+) (brother|sister|mother|father|"
    r"mom|dad|son|daughter|wife|husband|family|dog|cat|car|house|home|job|boss|baby|child|kid)\b",
    r"\bmy (mother|father|mom|dad|wife|husband|brother|sister|son|daughter|family|house|car|"
    r"job|boss|teacher|doctor|parents|grandmother|grandfather|childhood)\b",
    r"\bwhen i was (young|little|a (kid|child|boy|girl|baby|student|teenager))\b",
    r"\bi was born\b", r"\bi grew up\b", r"\bi (live|lived) (in|at|near|on)\b",
    r"\bi (ate|drank|bought|sold|cooked|wore|met|visited|married|kissed)\b",
    r"\bi (?:'?m|am) (at|in) (the|my|a) (hospital|store|school|office|house|home|park|car|"
    r"kitchen|bed|doctor|work)\b",
    r"\bi (have|need|'?ve got) to (go|leave|run|get going|get back|head out|head home)\b",
)]


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
        self._trace = ""              # how the LAST answer was produced (for "why did you say that")
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
        # 0) "why did you say that?" -> explain how the LAST answer was actually produced.
        r = self._explain_intent(text)
        if r is not None:
            return r
        # 1) ANSWERS that must be accurate stay deterministic (this is "the logic"). Each route
        #    records WHY it answered (self._trace) so the bot can truthfully explain itself.
        for fn, why in ((self._recall, "that came from my saved memory (you told me earlier)."),
                        (self._self_intent, "that's a grounded fact about what i am, not a guess."),
                        (self._history_intent, "i read that back from this chat's transcript."),
                        (self._define_intent, "i looked that up in my dictionary or what you taught me."),
                        (self.know.ask, "i worked that out from the facts you've taught me.")):
            r = fn(text)
            if r is not None:
                self._trace = why
                return r
        # 1b) Learn a DEFINITION the user gives ("X means ...") — before the fact-engine, which
        #     would otherwise mis-store it. Stored so "what does X mean" returns it.
        if self._learn_definition(text):
            ack = self._react(text)          # natural acknowledgement from the model
            self._trace = "you taught me a definition, so i saved it to my dictionary."
            return ack
        # 2) Learn a fact: store it silently, react naturally. It does NOT pester with a
        #    question after every fact — it only wonders out loud when explicitly asked
        #    ("what are you curious about"), and only about things it genuinely can't infer.
        taught = self.know.teach(text)
        if taught is not None:
            if "contradict" in taught.lower() or "can't be right" in taught.lower():
                self._trace = "that clashed with a fact i already had, so i flagged it."
                return taught
            ack = self._generate(text)
            self._trace = "you taught me a fact, so i stored it in my knowledge."
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
        if t in RECALL_NAME_YOU:                  # "who/what are you" -> _self_intent (richer)
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

    # --- sense of SELF: answer questions about what it is / can do / has done, from grounded
    #     facts instead of letting the model invent a persona ----------------------------------
    def _self_intent(self, text):
        t = text.lower().strip().rstrip("?.!")
        me = self.bot_name or "Apollo"
        # identity: what/who are you, are you an ai/human/real/alive...
        if (t in ("what are you", "who are you", "what is this", "what're you")
                or re.search(r"\bare you (an? )?(ai|a\.i\.|robot|human|person|real|alive|a bot|"
                             r"a computer|a program|a machine|conscious|sentient|a chatbot)\b", t)
                or re.search(r"\bwhat (kind of (thing|program|ai) )?are you$", t)):
            base = f"i'm {me}, {SELF_IS}."
            if self.persona:
                base += f" right now i'm acting as {self.persona}."
            return base
        # capabilities (general)
        if re.search(r"\b(what can you do|what do you do|how can you help|what are you able to do|"
                     r"what are your (abilities|capabilities|skills)|what can i ask you)\b", t):
            return "i can " + ", ".join(SELF_CAN[:-1]) + ", and " + SELF_CAN[-1] + "."
        # "can you X" -> answer from real abilities, not a guess
        m = re.match(r"^(?:can|could|are you able to|do you know how to) (?:you )?(.+)$", t)
        if m:
            x = m.group(1).strip()
            if any(h in x for h in CAN_HINT):
                return "yes, that's something i can do - just ask."
            if any(h in x for h in CANNOT_HINT):
                return ("no - i can't do that. i'm just a program on your computer, with no body "
                        "and no internet. but i can chat, do math, remember things, and more.")
        # embodiment / having things
        m = re.search(r"\bdo you have (?:a |an |any )?(body|face|feelings|emotions|a family|"
                      r"parents|a (?:brother|sister|mother|father|mom|dad|pet|dog|cat|home|house|"
                      r"job|heart|soul|name)|friends|hands|eyes|ears|legs)\b", t)
        if m and "name" not in m.group(1):
            return ("no, i don't - i'm a program, not a living thing. i only exist here in this "
                    "chat, on your computer.")
        # location / origin
        if re.search(r"\bwhere (do you (live|come from)|are you (from|located)?)\b", t):
            return "i live right here on your computer - i don't exist anywhere else."
        # age / appearance
        if re.search(r"\bhow old are you\b|\bwhat'?s your age\b|\bwhen were you born\b", t):
            return ("i don't have an age - i'm a program. i only know what's happened since we "
                    "started talking.")
        if re.search(r"\bwhat do you look like\b|\bwhat'?s your appearance\b", t):
            return "i don't have a body or a face - i'm just text running on your computer."
        # episodic: what are you doing / what have you done
        if re.search(r"\bwhat (are you doing|have you done|did you do)\b", t):
            said = sum(1 for r, _ in self.session if r == "USER")
            tail = f" so far you've told me {said} thing(s) this chat." if said else ""
            return "right now i'm just here chatting with you." + tail
        return None

    def _explain_intent(self, text):
        """Explain how the LAST answer was actually produced — from the real code path, not a
        made-up story. The honest part: it distinguishes 'I looked this up' from 'I generated
        this (a guess)'. This is grounded introspection, not confabulated reasoning."""
        t = text.lower().strip().rstrip("?.!")
        triggers = ("why", "why is that", "how come", "how do you know", "how do you know that",
                    "how did you know", "how did you know that", "what made you say that",
                    "explain that", "explain yourself", "explain your answer", "says who",
                    "explain your reasoning", "explain your thinking", "how do you figure")
        if (t in triggers or t.startswith("why did you say") or t.startswith("why'd you say")
                or t.startswith("why do you say") or t.startswith("how do you know")):
            return ("here's how i got my last answer: " + self._trace) if self._trace \
                else "ask me something first, then i can explain how i answered it."
        return None

    def _confabulates(self, reply):
        """True if the reply is a worldly first-person claim a program can't truthfully make
        (the 'i'm going to the hospital' problem)."""
        low = reply.lower()
        return any(p.search(low) for p in CONFAB)

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
        # honest trace: a model-generated reply is NOT grounded in a tool or fact — it's the net
        # trying to sound natural, so the truthful explanation is that it's a guess.
        self._trace = ("i generated that with my language model - that's me trying to sound natural, "
                       "not something i looked up, so it's more of a guess than a fact.")
        temp, top_k = (0.2, 3) if self.rules else (0.4, 40)
        greet = text.lower().strip().rstrip("?.!") in GREET_IN
        best = ""
        for _ in range(3):                         # a few tries: skip confabulation / greeting whiffs
            gen = self._clean(self._raw(self._pre() + f"USER: {text}\nBOT: ", temp, top_k), greet)
            if not gen or self._confabulates(gen):  # don't let it invent a worldly self
                continue
            if greet:
                first = (gen.lower().split() or [""])[0].strip(".!,")
                if first not in ("hi", "hello", "hey", "heya", "hiya", "howdy", "good", "yo"):
                    best = best or gen
                    continue
            return gen
        # everything tried was empty/confabulated -> honest, grounded fallback
        return best or ("honestly, i'm just a small program here in this chat - i don't have a "
                        "life outside it. what would you like to talk about?")

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

    def _datetime_intent(self, text):
        """INTERIM bridge: detect a date/time/year question. Superseded once the model emits
        `CALL: date|time|year` itself."""
        t = text.lower().strip().rstrip("?.!")
        if "time is it" in t or re.search(r"what'?s the time|current time|the time now", t):
            return "time"
        if re.search(r"what year is it|what'?s the year|current year", t):
            return "year"
        if re.search(r"what day is it|what'?s the date|what is the date|what day is today|"
                     r"what'?s today'?s date|what is today'?s date|what'?s the day today", t):
            return "date"
        return None

    def _react(self, text):
        """ReAct: let the model decide whether this turn needs a TOOL. If it emits a CALL we
        RUN it (gm/tools.py) and feed the RESULT back for it to verbalise; otherwise it just
        replies. The logic lives in code; the net only routes to it and phrases the answer."""
        dt = self._datetime_intent(text)           # interim: route date/time to the clock now
        if dt is not None:
            self._trace = "i read the clock/calendar on your computer - that's a tool, not a guess."
            return f"it's {self.tools.run(dt)}."
        expr = self._math_intent(text)             # interim: route math to the calculator now
        if expr is not None:
            ans = self.tools.run("calc " + expr)
            if ans != "error":                     # if it doesn't parse, fall through to chat
                self._trace = f"i ran it through my calculator tool ({expr.strip()} = {ans}) - exact, not guessed."
                return ans
        goal = self._reward_intent(text)           # interim: route reward asks to the tool now
        if goal is not None:
            if not goal:
                return "a reward for what? tell me the goal and i'll build it."
            self._trace = ("you asked for a reward, so i ran my reward tool on the goal; it matched "
                           "a known pattern and built those terms in code - not a guess.")
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
