#!/usr/bin/env python3
"""Train the model to USE the logic engine as a TOOL, instead of us hand-coding a parser.

Each example teaches the full loop: turn natural language into a CALL the engine runs, then
turn the engine's RESULT into a natural reply. Inputs are phrased many ways so the model
learns to PARSE flexibly (the job the regex used to do, now learned). Every RESULT is computed
by the real engine (gm/know.py) so the data is always correct.

  USER: <natural language, many phrasings>
  CALL: store|find|ask|profile <args>
  RESULT: <what the engine returns>
  BOT: <natural reply generated from the result>

Output: data/tooluse/chat.txt
"""
import os
import random

from gm.know import Knowledge, sing
from gm.tools import build_reward, calc
from prep_rewards import GOALS, REQ, DIRECT, VAGUE

HERE = os.path.dirname(os.path.abspath(__file__))

TEACH = {
    "isa": ["a {s} is a {o}", "a {s} is a kind of {o}", "{s}s are {o}s",
            "every {s} is a {o}", "did you know a {s} is a {o}", "i think a {s} is a {o}"],
    "has": ["a {s} has {o}", "a {s} has a {o}", "{s}s have {o}", "a {s} comes with {o}",
            "{s}s usually have {o}", "remember that a {s} has {o}"],
    "can": ["a {s} can {o}", "{s}s can {o}", "a {s} is able to {o}",
            "a {s} knows how to {o}", "{s}s are able to {o}"],
}
CONFIRM = ["got it.", "okay, noted.", "i'll remember that.", "understood.", "good to know.",
           "okay, i've stored that.", "noted."]
Q_FIND = ["what has {o}", "what things have {o}", "which things have {o}",
          "name something with {o}", "what do you know that has {o}"]
Q_ASK_HAS = ["does a {s} have {o}", "do {s}s have {o}", "has a {s} got {o}",
             "is it true that a {s} has {o}"]
Q_ASK_ISA = ["is a {s} a {o}", "are {s}s {o}s", "is a {s} a kind of {o}",
             "would you say a {s} is a {o}"]
Q_PROFILE = ["what is a {s}", "tell me about a {s}", "describe a {s}",
             "what do you know about {s}s", "what's a {s}"]


def vlist(xs):
    xs = [f"a {x}" for x in xs]
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def main(n=12000, reward_n=9000, calc_n=9000, dt_n=9000, solve_n=6000, contrast_n=8000, seed=11):
    r = random.Random(seed)
    words = [w.strip().lower() for w in open(os.path.join(HERE, "common10k.txt"))]
    nouns = [w for w in words if w.isalpha() and 3 <= len(w) <= 9]
    verbs = ["run", "fly", "swim", "jump", "sing", "bark", "grow", "move", "shine",
             "float", "climb", "dig", "hunt", "sleep", "glow", "roll"]
    out = []

    for _ in range(n):
        k = Knowledge()
        cats = r.sample(nouns, 3)        # a small taxonomy: things -> category -> super
        thing, cat, sup = cats
        traits = r.sample(nouns, 2)
        act = r.choice(verbs)
        lines = []

        # teach facts via the tool
        def teach(s, rel, o):
            k._add(s, rel, o)
            nl = r.choice(TEACH[rel]).format(s=s, o=o)
            lines.extend([f"USER: {nl}", f"CALL: store {s} {rel} {o}",
                          "RESULT: ok", f"BOT: {r.choice(CONFIRM)}"])

        teach(thing, "isa", cat)
        teach(cat, "isa", sup)
        teach(cat, "has", traits[0])
        if r.random() < 0.6:
            teach(thing, "has", traits[1])
        if r.random() < 0.6:
            teach(cat, "can", act)

        # now ask the tool things (results computed by the real engine, incl. inheritance)
        qs = []
        o = traits[0]
        subs = k.subjects_with("has", o)
        qs.append((r.choice(Q_FIND).format(o=o), f"find has {o}",
                   ", ".join(subs) if subs else "none",
                   f"{vlist(subs)} {'has' if len(subs)==1 else 'have'} {o}." if subs
                   else f"nothing i know of has {o}."))

        yn = o in k.rel_objs(thing, "has")
        qs.append((r.choice(Q_ASK_HAS).format(s=thing, o=o), f"ask {thing} has {o}",
                   "yes" if yn else "no",
                   (f"yes, a {thing} has {o}." if yn else f"not that i know of.")))

        isayn = sup in k.ancestors(thing)
        qs.append((r.choice(Q_ASK_ISA).format(s=thing, o=sup), f"ask {thing} isa {sup}",
                   "yes" if isayn else "no",
                   (f"yes, a {thing} is a {sup}." if isayn else "not that i know of.")))

        parents = k.rel_objs(thing, "isa")
        prof = "isa " + ",".join(parents) if parents else "nothing"
        rep = (f"a {thing} is " + " and ".join(f"a {p}" for p in parents) + "."
               if parents else f"i don't know what a {thing} is.")
        qs.append((r.choice(Q_PROFILE).format(s=thing), f"profile {thing}", prof, rep))

        r.shuffle(qs)
        for nl, call, res, rep in qs[:r.randint(1, 3)]:
            lines.extend([f"USER: {nl}", f"CALL: {call}", f"RESULT: {res}", f"BOT: {rep}"])

        out.append("\n".join(lines))

    # ---- reward TOOL: route a goal to the reward builder, then verbalise the spec. The RESULT
    # is computed by the SAME build_reward() the runtime uses, so training and inference agree.
    # A big share uses RANDOM goals (verb + noun) so the model learns to CALL with the user's
    # literal goal for ANY request, instead of freelancing a spec on goals it never saw. ----
    goal_phrases = [g for g, _ in GOALS]
    nouns = [w.strip().lower() for w in open(os.path.join(HERE, "common10k.txt"))
             if w.strip().isalpha() and 3 <= len(w.strip()) <= 9]
    GERUNDS = ["moving", "fixing", "building", "carrying", "sorting", "catching", "throwing",
               "lifting", "pushing", "pulling", "drawing", "painting", "cleaning", "washing",
               "cooking", "planting", "driving", "flying", "jumping", "running", "climbing",
               "stacking", "balancing", "juggling", "dancing", "folding", "packing", "loading",
               "digging", "sweeping", "organizing", "guarding", "chasing", "delivering"]
    for _ in range(reward_n):
        pick = r.random()
        if pick < 0.45:                                   # random goal -> CALL generalizes
            g = f"{r.choice(GERUNDS)} the {r.choice(nouns)}"
            user, call = r.choice(REQ).format(g=g), f"reward {g}"
        elif pick < 0.8:                                  # curated domain goal
            g = r.choice(goal_phrases)
            user, call = r.choice(REQ).format(g=g), f"reward {g}"
        else:                                             # direct imperative
            d = r.choice(list(DIRECT))
            verb = r.choice(["make it", "teach it to", "i want it to", "get it to"])
            user, call = f"{verb} {d}", f"reward {d}"
        spec = build_reward(call.split(" ", 1)[1])
        out.append(f"USER: {user}\nCALL: {call}\nRESULT: {spec}\nBOT: reward: {spec}")
    # vague reward requests -> ASK for the goal (no tool call), so it doesn't guess blindly
    for _ in range(max(1, reward_n // 20)):
        u, b = r.choice(VAGUE)
        out.append(f"USER: {u}\nBOT: {b}")

    # ---- calculator TOOL: a tiny LM can't do reliable arithmetic, so it learns to CALL calc.
    # RESULT computed by the same calc() the runtime uses, so training and inference agree. ----
    ASK_MATH = ["what is {e}", "what's {e}", "calculate {e}", "{e}", "how much is {e}",
                "compute {e}", "solve {e}", "what does {e} equal"]
    ops = [("+", "plus"), ("-", "minus"), ("*", "times"), ("/", "divided by")]
    for _ in range(calc_n):
        sym, word = r.choice(ops)
        a, b = r.randint(2, 99), r.randint(2, 50)
        expr = f"{a} {sym} {b}" if r.random() < 0.5 else f"{a} {word} {b}"
        ans = calc(expr)
        if ans == "error":
            continue
        out.append(f"USER: {r.choice(ASK_MATH).format(e=expr)}\nCALL: calc {a} {sym} {b}\n"
                   f"RESULT: {ans}\nBOT: that's {ans}.")

    # ---- date/time TOOL: a model can't know the clock, so it learns to CALL date/time and
    # verbalise the RESULT. Results are VARIED (random dates/times) so it learns to echo the
    # tool's output, not memorize one date. ----
    DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
              "september", "october", "november", "december"]
    DATE_Q = ["what day is it", "what's the date", "what is today's date", "what day is it today",
              "what's today's date", "what is the date", "what day is today", "do you know the date",
              "tell me the date", "what's the day today", "today's date?", "what date is it"]
    TIME_Q = ["what time is it", "what's the time", "do you know the time", "what time is it now",
              "what is the time", "tell me the time", "got the time?", "the time please",
              "what time is it right now", "current time?"]
    YEAR_Q = ["what year is it", "what's the year", "what is the current year", "which year is it"]
    for _ in range(dt_n):
        pick = r.random()
        if pick < 0.5:
            d = (f"{r.choice(DAYS)}, {r.choice(MONTHS)} {r.randint(1, 28)}, "
                 f"{r.randint(2024, 2031)}").title()
            out.append(f"USER: {r.choice(DATE_Q)}\nCALL: date\nRESULT: {d}\nBOT: it's {d}.")
        elif pick < 0.85:
            tm = f"{r.randint(1, 12)}:{r.randint(0, 59):02d} {r.choice(['am', 'pm'])}"
            out.append(f"USER: {r.choice(TIME_Q)}\nCALL: time\nRESULT: {tm}\nBOT: it's {tm}.")
        else:
            y = str(r.randint(2024, 2031))
            out.append(f"USER: {r.choice(YEAR_Q)}\nCALL: year\nRESULT: {y}\nBOT: it's {y}.")

    # ---- solver TOOL: number-sequence puzzles. The brain learns to hand the sequence to the
    # solver (which SEARCHES for the rule) and verbalise the answer. RESULT computed by the real
    # solver, so train==runtime. ----
    from gm.solver import solve as _solve
    SEQ_Q = ["what comes next: {s}", "what's next in {s}", "solve this sequence: {s}",
             "finish the sequence {s}", "next number: {s}", "what comes after {s}",
             "figure out the pattern: {s}"]
    for _ in range(solve_n):
        kind = r.random()
        if kind < 0.32:                                   # arithmetic
            a, d = r.randint(1, 20), r.randint(2, 9)
            seq = [a + d * i for i in range(r.randint(4, 6))]
        elif kind < 0.58:                                 # geometric
            a, m = r.randint(1, 5), r.randint(2, 4)
            seq = [a * m ** i for i in range(r.randint(4, 5))]
        elif kind < 0.8:                                  # fibonacci-like
            seq = [r.randint(1, 4), r.randint(1, 5)]
            for _ in range(r.randint(3, 4)):
                seq.append(seq[-1] + seq[-2])
        else:                                             # quadratic (growing gaps)
            v, d = r.randint(1, 6), r.randint(1, 4)
            seq = []
            for i in range(r.randint(4, 6)):
                seq.append(v)
                v += d + 2 * i
        s = " ".join(map(str, seq))
        res = _solve(s)                                   # "NEXT  (rule: ...)"
        if "rule:" not in res:
            continue
        nxt = res.split("(")[0].strip()
        rule = res.split("rule:")[1].strip().rstrip(")")
        out.append(f"USER: {r.choice(SEQ_Q).format(s=s)}\nCALL: solve {s}\n"
                   f"RESULT: {res}\nBOT: {nxt} - {rule}.")

    # ---- CONTRAST: turns that LOOK tool-ish (numbers, goal words) but are NOT requests, so the
    # model learns WHEN to call vs. when to just chat. Same weight as the tool data => a real
    # decision boundary, not a reflex to CALL whenever it sees a number or a verb. ----
    things = ["cats", "dogs", "books", "plants", "friends", "cups", "songs", "siblings", "plants"]
    acts = ["jumping", "running", "cleaning my room", "cooking dinner", "painting", "swimming",
            "dancing", "reading", "hiking", "drawing", "studying", "gardening", "baking"]
    num_ctx = ["i have {n} {th}", "i'm {n} years old", "there were {n} people there",
               "i read {n} pages today", "we walked {n} miles", "i have {n} dollars left",
               "my team scored {n} points", "i slept {n} hours", "it's {n} degrees out",
               "i've got {n} {th}", "the recipe needs {n} eggs"]
    act_ctx = ["i went {a} today", "i love {a}", "i was {a} earlier", "i really enjoy {a}",
               "i'm thinking about {a} later", "{a} is my favorite thing", "i'm tired from {a}"]
    backs = ["nice!", "that's cool.", "sounds fun!", "good for you.", "oh nice.", "love that.",
             "that sounds nice.", "awesome, how was it?", "ha, that's great.", "neat!",
             "sounds like a good time.", "cool, tell me more.", "that's wonderful.",
             "oh that's lovely.", "sounds relaxing.", "i bet that felt good."]
    for _ in range(contrast_n):
        if r.random() < 0.5:
            u = r.choice(num_ctx).format(n=r.randint(2, 60), th=r.choice(things))
        else:
            u = r.choice(act_ctx).format(a=r.choice(acts))
        out.append(f"USER: {u}\nBOT: {r.choice(backs)}")

    r.shuffle(out)
    out_dir = os.path.join(HERE, "data", "tooluse")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chat.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(out) + "\n")
    print(f"[tooluse] {len(out):,} tool-use conversations -> data/tooluse/chat.txt "
          f"({os.path.getsize(os.path.join(out_dir,'chat.txt')):,} bytes)")


if __name__ == "__main__":
    main()
