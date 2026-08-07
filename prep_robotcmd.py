#!/usr/bin/env python3
"""Training data that teaches the mouth to be an HONEST intent parser, not a confabulator.

The failure this fixes (found on the real robot): the old model mapped ANY request onto its nearest
known landmark -- "go to the unicorn" -> "goto charger" -- because it was trained to always emit one
of a handful of memorised places. It said yes to everything, which is worse than saying no.

The new contract, aligned with how the robot actually works now (an object DETECTOR + CLIP ground the
object; the mouth just parses intent):

    goto <object phrase>   navigate to a described object -- the phrase is PASSED THROUGH verbatim so
                           perception/CLIP can ground ANY object; the mouth never has to know objects.
    go forward|back|left|right     a relative move
    stand | recover | cross | explore     fixed skills
    refuse                 an impossible / out-of-scope / nonsensical request -> say NO, don't invent
    <plain reply>          social chit-chat -> answer like a person, emit no command

Because `goto` PASSES THE OBJECT PHRASE THROUGH, the model generalises to objects it never saw in
training (it learns EXTRACTION, not a lookup table) -- and because impossible things map to `refuse`,
it stops confabulating. Every example is  USER: <request>  /  BOT: <output>.

    python3 prep_robotcmd.py            # -> data/robotcmd/chat.txt
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- objects the mouth learns to EXTRACT and pass through (it needn't know what they are). A wide,
# varied bank so it learns the pattern, not the list. CLIP grounds whatever comes out.
NOUNS = [
    "chair", "table", "ball", "cup", "mug", "bottle", "lamp", "book", "box", "plant", "flower pot",
    "teddy bear", "trash can", "bin", "sofa", "couch", "stool", "bench", "desk", "shelf", "cabinet",
    "door", "window", "wall", "corner", "charger", "dock", "stairs", "ramp", "rug", "mat", "plate",
    "bowl", "kettle", "toaster", "fridge", "oven", "sink", "bucket", "basket", "backpack", "bag",
    "shoe", "boot", "hat", "helmet", "clock", "phone", "laptop", "keyboard", "monitor", "speaker",
    "guitar", "drum", "pillow", "blanket", "towel", "vase", "candle", "mirror", "picture", "poster",
    "cone", "brick", "block", "barrel", "crate", "pallet", "ladder", "toolbox", "hammer", "wrench",
    "watering can", "broom", "mop", "umbrella", "fan", "heater", "radio", "television",
    "cactus", "fern", "pumpkin", "watermelon", "apple", "banana", "loaf of bread", "wine glass",
    "soccer ball", "basketball", "tennis ball", "beach ball", "yoga mat", "skateboard",
]
ADJ = ["", "", "", "red", "blue", "green", "yellow", "white", "black", "big", "small", "tall",
       "little", "round", "wooden", "metal", "plastic", "shiny", "old", "broken", "empty", "full"]
# descriptive references -> perception/CLIP resolves these too, so the mouth just passes them through
DESCRIPTIONS = [
    "thing you sit on", "thing you drink from", "thing you sleep on", "thing you read",
    "thing you throw", "thing you plug in to charge", "place you throw your rubbish",
    "biggest object in the room", "closest object", "nearest thing", "furthest object",
    "thing in the corner", "thing by the wall", "round red thing", "tall thing in the middle",
    "soft brown toy", "green leafy thing", "something to drink from", "somewhere to sit",
]
GOTO_VERBS = ["go to", "walk to", "walk over to", "head to", "head over to", "move to", "move over to",
              "get to", "come to", "navigate to", "make your way to", "go stand by", "go over to",
              "go and find", "find", "go look at", "take a look at", "bring yourself to", "approach"]

# ---- fixed non-object commands
FIXED = {
    "go forward": ["go forward", "move forward", "step forward", "go straight", "go ahead",
                   "forward a bit", "keep going forward", "advance", "move ahead"],
    "go back": ["go back", "back up", "step back", "move backward", "reverse", "back away", "retreat"],
    "go left": ["go left", "step left", "shuffle left", "veer left", "bear left", "sidestep left"],
    "go right": ["go right", "step right", "shuffle right", "veer right", "bear right", "sidestep right"],
    # TURN = rotate in place (distinct from go/step = translate). apollo used to have no turn verb, so
    # "turn around" leaked through as an object ("goto turn around"). These fix that at the source.
    "turn around": ["turn around", "turn back", "turn the other way", "about face", "spin around",
                    "face the other way", "do a 180", "rotate 180", "turn all the way around", "look behind you"],
    "turn left": ["turn left", "rotate left", "spin left", "face left", "turn to your left", "swivel left"],
    "turn right": ["turn right", "rotate right", "spin right", "face right", "turn to your right", "swivel right"],
    "stand": ["stand", "stop", "halt", "hold still", "stay there", "stand still", "freeze", "wait",
              "hold on", "don't move", "stay put"],
    "recover": ["get up", "recover", "stand back up", "get back on your feet", "you fell, get up",
                "stand up", "pick yourself up", "back on your feet"],
    "cross": ["cross the table", "cross the gap", "get across", "cross over", "walk across the table"],
    "explore": ["explore", "look around", "explore the room", "map the room", "scan the room",
                "have a look around", "check out the room", "survey the area", "go explore"],
}

# ---- REFUSE: things the robot genuinely CANNOT do (no gripper, can't fly, can't leave the room, isn't
# a chatbot-oracle). The mouth must say NO here instead of snapping to the nearest known skill.
IMPOSSIBLE = [
    "fly to the ceiling", "fly to the moon", "jump over the house", "teleport home", "swim to me",
    "climb the wall", "climb out the window", "dig a hole", "drive the car", "ride the bike",
    "pick up the cup", "grab the ball", "carry the box", "hold this", "throw the ball", "catch it",
    "open the door", "close the window", "pour the water", "make me a sandwich", "cook dinner",
    "make coffee", "wash the dishes", "fetch my keys", "bring me a drink", "hand me the remote",
    "paint the wall", "fix the sink", "water the plants", "feed the dog", "walk the dog",
    "play the guitar", "sing a song", "dance for me", "do a backflip", "juggle the balls",
    "read me a story", "call my mom", "order a pizza", "turn on the lights", "vacuum the floor",
    "lift the table", "push the sofa", "tidy up the room", "put the book on the shelf",
    "solve this equation", "what's the weather", "what time is it", "tell me a joke",
    "go to the unicorn", "go to the dragon", "walk to the spaceship", "find the treasure",
    "go outside", "go to the kitchen next door", "leave the room", "go to another building",
]
# gibberish / non-language -> also refuse (don't hallucinate a command out of noise)
GIBBERISH = ["asdf qwer", "blorp the fizzle", "xyzzy plugh", "go to the the the", "flurb",
             "run the mimsy borogoves", "zzzzz", "?!?!", "hgfd kjhg", "wibble wobble woo"]

# ---- social chit-chat -> a friendly reply, NOT a command and NOT a refusal
SOCIAL = [
    ("hello there", "hi!"), ("hi robot", "hello!"), ("how are you", "i'm good, thanks!"),
    ("what's your name", "i'm the robot."), ("thanks", "you're welcome!"), ("thank you", "any time!"),
    ("what's up", "not much!"), ("good job", "thank you!"), ("nice work", "thanks!"),
    ("good morning", "morning!"), ("goodnight", "goodnight!"), ("who made you", "my builder did."),
    ("what can you do", "i can walk around and go to things you point me at."),
    ("are you a robot", "yes, i am!"), ("i like you", "aw, thanks!"),
]

PREFIX = ["", "", "hey ", "ok ", "so ", "alright ", "please ", "can you ", "could you ",
          "i want you to ", "i need you to ", "would you ", "robot, ", "now ", "let's ", "go on, "]
SUFFIX = ["", "", " please", " now", " for me", " when you can", " thanks", " right now", " ok",
          " already", " would you"]


def _cap(s, r):
    return s[0].upper() + s[1:] if s and r.random() < 0.15 else s


def _wrap(r, text):
    return _cap((r.choice(PREFIX) + text + r.choice(SUFFIX)).strip(), r)


def main(seed=0, cap=16000):
    r = random.Random(seed)
    pairs = set()

    # goto <object phrase>  -- passthrough: the BOT output is the object phrase, article-stripped
    def obj_targets(n):
        got = []
        for _ in range(n):
            k = r.random()
            if k < 0.45:
                a = r.choice(ADJ); phrase = (a + " " + r.choice(NOUNS)).strip()
            elif k < 0.75:
                phrase = r.choice(NOUNS)
            else:
                phrase = r.choice(DESCRIPTIONS)
            got.append(phrase)
        return got
    for phrase in obj_targets(6000):
        verb = r.choice(GOTO_VERBS)
        art = r.choice(["the ", "the ", "a ", "that ", "the nearest ", ""])
        text = _wrap(r, f"{verb} {art}{phrase}")
        pairs.add((text, f"goto {phrase}"))

    # fixed commands
    for target, phrases in FIXED.items():
        for p in phrases:
            for _ in range(18):
                pairs.add((_wrap(r, p), target))

    # refusals (impossible + gibberish) -> BOT: refuse
    for p in IMPOSSIBLE:
        for _ in range(12):
            pairs.add((_wrap(r, p), "refuse"))
    for p in GIBBERISH:
        for _ in range(6):
            pairs.add((_cap((r.choice(PREFIX[:6]) + p).strip(), r), "refuse"))

    # ---- COMPOUND / MULTI-STEP. The human can chain sub-commands with ANY connector; apollo learns
    # to NORMALISE all of them ("then", "and then", "after that", "afterwards", "next", "followed by",
    # and the elliptical "once you reach X") into a canonical output where sub-commands are joined by
    # " then ". The bridge then splits the OUTPUT on " then " -- reliable precisely because apollo
    # always canonicalises, whatever wording the human used. Kept a single-command-dominant minority so
    # a lone command never gets turned into a spurious sequence.
    CONNECTORS = ["then", "and then", ", then", "after that", ", after that", "afterwards",
                  ", afterwards", "next", ", and then", "followed by", "and after that", ", next"]
    SEQ_ACTIONS = ["go forward", "go back", "go left", "go right",
                   "turn around", "turn left", "turn right", "stand"]

    def rand_phrase():
        k = r.random()
        if k < 0.45:
            return (r.choice(ADJ) + " " + r.choice(NOUNS)).strip()
        if k < 0.75:
            return r.choice(NOUNS)
        return r.choice(DESCRIPTIONS)

    def rand_atom(goto_ok=True):
        """(english fragment, canonical symbol) for one sub-command."""
        if goto_ok and r.random() < 0.5:
            phrase = rand_phrase()
            art = r.choice(["the ", "the ", "a ", "that ", ""])
            return f"{r.choice(GOTO_VERBS)} {art}{phrase}", f"goto {phrase}"
        target = r.choice(SEQ_ACTIONS)
        return r.choice(FIXED[target]), target

    def join_eng(engs):
        s = engs[0]
        for e in engs[1:]:
            c = r.choice(CONNECTORS)
            s += (c if c.startswith(",") else " " + c) + " " + e
        return s

    for _ in range(2000):                                   # 2- and 3-step chains, varied connectors
        n = 2 if r.random() < 0.72 else 3
        atoms = [rand_atom() for _ in range(n)]
        pairs.add((_wrap(r, join_eng([a[0] for a in atoms])),
                   " then ".join(a[1] for a in atoms)))

    ELLIPTIC = ["once you reach the {p}", "after you reach the {p}", "when you get to the {p}",
                "once you're at the {p}", "after you get to the {p}", "once you arrive at the {p}"]
    for _ in range(600):                                    # destination hidden in the connector clause
        phrase = rand_phrase()
        eng, sym = rand_atom(goto_ok=False)
        pairs.add((_wrap(r, f"{r.choice(ELLIPTIC).format(p=phrase)}, {eng}"),
                   f"goto {phrase} then {sym}"))

    for _ in range(400):                                    # any impossible part -> refuse the WHOLE thing
        eng, _sym = rand_atom()
        pairs.add((_wrap(r, join_eng([eng, r.choice(IMPOSSIBLE)])), "refuse"))

    pairs = list(pairs)
    r.shuffle(pairs)
    if len(pairs) > cap:
        pairs = pairs[:cap]
    out = [f"USER: {t}\nBOT: {c}" for t, c in pairs]

    # social chit-chat -> plain replies (kept modest so it doesn't dilute the command signal)
    for text, reply in SOCIAL:
        for pre in PREFIX[:5]:
            out.append(f"USER: {(pre + text).strip()}\nBOT: {reply}")

    r.shuffle(out)
    d = os.path.join(HERE, "data", "robotcmd")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "chat.txt"), "w") as f:
        f.write("\n\n".join(out) + "\n")

    uniq = len(set(l.split("\n")[0] for l in out))
    n_goto = sum(1 for l in out if "\nBOT: goto " in l)
    n_ref = sum(1 for l in out if l.endswith("BOT: refuse"))
    n_seq = sum(1 for l in out if " then " in l.split("\nBOT: ", 1)[-1])
    print(f"[robotcmd] {len(out):,} examples ({uniq:,} unique) -> data/robotcmd/chat.txt")
    print(f"           goto-passthrough {n_goto:,} | multi-step {n_seq:,} | refuse {n_ref:,} | "
          f"fixed/social the rest. Teaches intent-extraction, chaining + honest refusal, not a lookup.")


if __name__ == "__main__":
    main()
