#!/usr/bin/env python3
"""GENERATIVE problem-solving: "[obstacle] is in the way; here's what might work, and why." The point
is NOT a lookup table (that's what the retired plan tool was) — it's teaching the PATTERN so the model
GENERATES fitting solutions for obstacles it never saw, by recombining. The lever, same as the diverse
fables, is DIVERSITY: many obstacle types, each with property-conditioned options, so the model learns
that the right move DEPENDS on the obstacle's properties (a low log -> step over; a tall one -> climb;
a loose one -> roll aside). That conditional 'fit' is the heart of problem-solving, and it transfers.

No tool call — this is pure generated reasoning (understanding, not retrieval). Output: data/solving/chat.txt
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

# each obstacle: (description, goal, [ (action, CONDITION, reason) ... ]). CONDITION is a natural
# "it's ..." clause; a stated condition picks the fitting action -> teaches condition->solution
# reasoning (the heart of it: the right move DEPENDS on the obstacle's properties).
OBSTACLES = [
    ("a fallen log across the path", "get to the other side", [
        ("step over it", "low enough to step over", "i can clear it in a single stride"),
        ("climb over it", "too tall to step over", "climbing gets me over what i can't step across"),
        ("go around the end", "not very long", "if it doesn't stretch far, around is simplest"),
        ("roll it aside", "loose and not fixed down", "if it moves, i can just shift it out of the way")]),
    ("a locked door", "get inside", [
        ("look for a key nearby", "the kind that usually has a key close by", "doors often keep a key nearby"),
        ("try another entrance", "part of a big building", "a big building usually has another way in"),
        ("knock and ask to be let in", "occupied", "whoever's inside can open it for me"),
        ("check if a window is open", "on the ground floor", "a low window can be an easy way through")]),
    ("a wide river", "cross to the far bank", [
        ("wade across", "shallow and calm", "if it's shallow i can walk through safely"),
        ("look for a bridge upstream", "deep and fast", "deep water is dangerous, so i'd find a real crossing"),
        ("use stepping stones", "full of rocks", "rocks let me cross without getting swept away"),
        ("build or find a raft", "deep but slow", "on slow water a raft floats me across")]),
    ("a heavy box", "move it out of the way", [
        ("push it along the floor", "on a smooth floor", "sliding beats lifting a heavy load"),
        ("slide a board under it", "on a rough floor", "a board lets it glide where it won't slide"),
        ("empty it first, then move it", "full", "a lighter load is far easier to shift"),
        ("get help to lift it", "huge and bulky", "some things just need a second pair of hands")]),
    ("a tall wall", "get over it", [
        ("find footholds and climb", "rough with cracks and ledges", "grips let me pull myself up"),
        ("look for a gate or gap", "very long", "a long wall usually has an opening somewhere"),
        ("stack something to stand on", "just a bit too high", "a step up closes a small height gap"),
        ("go around it", "short", "if it doesn't stretch far, around is easiest")]),
    ("a gap in the floor", "get across", [
        ("step or jump across", "narrow", "a narrow gap is one easy stride"),
        ("lay a board across it", "wide", "a board turns a wide gap into a bridge"),
        ("climb down and back up", "shallow", "if it's shallow i can just go through it"),
        ("find another route", "deep and dangerous", "some gaps aren't worth the risk")]),
    ("thick mud", "get through", [
        ("step on the firmer patches", "patchy with dry spots", "solid ground keeps me from sinking"),
        ("lay branches down to walk on", "deep and sticky", "branches spread my weight over the mud"),
        ("go around it", "just a small patch", "avoiding it is easiest if it's small"),
        ("take slow, steady steps", "shallow", "moving slowly stops me from slipping")]),
    ("a steep hill", "get to the top", [
        ("climb straight up", "short", "a short slope is quickest head-on"),
        ("zigzag up the slope", "long and very steep", "switchbacks make a steep climb manageable"),
        ("use roots and rocks as handholds", "rocky", "grips keep me steady on the climb"),
        ("find a gentler path", "too steep to climb safely", "an easier route beats an exhausting one")]),
    ("a tangled rope", "get it free", [
        ("loosen it slowly", "loose with big loops", "gentle loosening undoes a light tangle"),
        ("find the loose end first", "a messy knot", "working from an end unwinds the rest"),
        ("cut it", "pulled tight and stuck", "a hopeless knot is faster to cut than to fight"),
        ("shake it out", "only lightly tangled", "a shake can drop a simple tangle apart")]),
    ("a dark room", "find your way", [
        ("feel along the wall", "small", "a wall guides me straight to the door"),
        ("look for a light switch", "indoors", "light solves the whole problem at once"),
        ("wait for your eyes to adjust", "a bit dim rather than pitch black", "eyes adapt and shapes appear"),
        ("use your phone as a torch", "one where i have my phone", "any light source makes it easy")]),
    ("a high shelf", "reach something on it", [
        ("stand on a sturdy stool", "just out of reach", "a step up closes the height gap"),
        ("use a long tool to reach", "a bit too high for a stool", "a reach-extender saves the climb"),
        ("ask a taller person", "one where someone's around", "a taller helper reaches it instantly"),
        ("use a grabber", "packed with fragile things", "a grabber avoids knocking anything over")]),
    ("a crowd blocking the way", "get through", [
        ("wait for it to thin out", "there for a moment", "crowds clear on their own with time"),
        ("politely ask people to move", "not too packed", "most people step aside if asked"),
        ("take a route around the edge", "loose in the middle", "the edges are usually clearer"),
        ("follow someone making a path", "one where someone's already moving through", "slipstreaming is easy")]),
    ("a stuck jar lid", "open it", [
        ("grip it with a cloth", "slippery", "a cloth gives me the grip i need"),
        ("run it under warm water", "a metal lid", "warmth expands the metal so it loosens"),
        ("tap the edge gently", "sealed tight by a vacuum", "a tap breaks the seal"),
        ("ask someone stronger", "just too tight for me", "a firmer grip finishes it")]),
    ("deep snow", "get across", [
        ("step in the existing footprints", "already tracked by someone", "packed prints hold my weight"),
        ("take wide steps to spread your weight", "fresh powder", "spreading out keeps me from sinking"),
        ("go around the deepest drifts", "patchy with shallow edges", "shallower snow is far easier"),
        ("wait for it to be cleared", "on a road", "sometimes waiting is the smart move")]),
    ("a broken bridge", "cross the gap", [
        ("use the beams still standing", "only partly broken", "a solid beam still carries me across"),
        ("find a shallow crossing below", "over low water", "the water below may be crossable"),
        ("lay a plank across the break", "broken by just a small gap", "a plank spans a small break"),
        ("take the long way around", "completely gone", "safety beats speed when there's no safe span")]),
    ("a thorny bush", "get past", [
        ("push the branches aside", "thin and sparse", "a sparse bush parts easily"),
        ("go around it", "small and off to one side", "a small bush is easy to avoid"),
        ("cover your arms and step through", "dense with no way around", "protection lets me push through"),
        ("cut a path", "thick and blocking the whole way", "clearing a path is worth it if it's dense")]),
    ("a stuck drawer", "open it", [
        ("wiggle it side to side", "just slightly jammed", "wiggling frees a light jam"),
        ("push it in, then pull evenly", "off its track", "resetting it lets it slide again"),
        ("clear whatever is caught inside", "something stuck at the back", "removing the snag lets it open"),
        ("pull with even force on both handles", "swollen from damp", "even force avoids yanking it crooked")]),
    ("a puddle across the path", "keep your feet dry", [
        ("step over it", "narrow", "a narrow puddle is one easy step"),
        ("walk around the edge", "wide but shallow at the sides", "the edges are usually shallower"),
        ("lay a board or stones to step on", "wide and deep", "stepping stones keep me above the water"),
        ("just walk through if you have boots", "one where i'm wearing boots", "boots make it a non-issue")]),
    ("a heavy curtain of vines", "get through", [
        ("part them with your hands", "loose and hanging free", "loose vines push aside easily"),
        ("find a thinner spot to pass", "patchy", "a gap saves the effort"),
        ("cut through them", "thick and woven together", "dense vines are faster to cut"),
        ("duck underneath", "hanging high", "if they hang high i can pass below")]),
    ("a jammed window", "open it", [
        ("push up firmly and evenly", "painted shut", "even pressure breaks a paint seal"),
        ("tap around the frame first", "stuck at the edges", "loosening the edges frees the sash"),
        ("clear the track of grit", "gritty in the runner", "a clean track lets it glide"),
        ("try a different window", "swollen and truly stuck", "another window may open freely")]),
    ("a fast-moving stream", "cross it", [
        ("wade at a wide, calm spot", "wide and shallow", "wide water usually runs slower and shallower"),
        ("cross on stepping stones", "dotted with rocks", "rocks give footing above the current"),
        ("find a log or bridge", "deep and fast", "fast deep water is too risky to wade"),
        ("cross where it splits into channels", "braided into small channels", "small channels are each easy")]),
    ("a pile of rubble", "get past", [
        ("climb over the stable parts", "settled and solid", "settled rubble holds my weight"),
        ("clear a path through it", "loose and shifting", "loose rubble is safer moved than climbed"),
        ("go around it", "piled to one side", "a route around avoids the hazard"),
        ("test each step before trusting it", "uncertain and mixed", "testing footing prevents a fall")]),
    ("a closed heavy gate", "get through", [
        ("lift the latch and push", "just latched", "a latch is the simplest fix"),
        ("look for another opening", "chained shut", "a chained gate needs a different way in"),
        ("push slowly with your whole body", "heavy but unlocked", "weight, not force, swings a heavy gate"),
        ("call out for someone to open it", "barred from the other side", "someone on the far side can free it")]),
    ("a slippery slope", "get down safely", [
        ("go down sideways in small steps", "steep and smooth", "sideways steps keep me from sliding"),
        ("hold onto plants or rocks", "dotted with handholds", "grips slow a slippery descent"),
        ("sit and slide carefully", "short and clear below", "a controlled slide is safe on a short drop"),
        ("find a drier, rougher line down", "wet in places", "rough dry ground has far more grip")]),
    ("a low doorway", "get through", [
        ("duck your head", "just a bit low", "a quick duck clears a small overhang"),
        ("crouch and step through", "quite low", "crouching gets me under a low frame"),
        ("turn sideways as you crouch", "low and narrow", "turning fits me through a tight, low gap"),
        ("crawl through", "very low", "crawling is the only way under a very low opening")]),
    ("a flooded underpass", "get to the other side", [
        ("wade through", "shallow, below the knee", "shallow water is safe to wade"),
        ("find the higher path around", "deep and rising", "deep water means i take the high route"),
        ("wait for the water to drain", "draining already", "a draining flood clears on its own"),
        ("use the raised walkway if there is one", "one with a ledge along the side", "a ledge stays dry")]),
    ("a tall fence", "get to the other side", [
        ("climb over using the rails", "made of rails i can grip", "rails give hand and footholds"),
        ("find the gate", "very long", "a long fence has a gate somewhere"),
        ("look for a gap at the bottom", "loose along the ground", "a loose bottom edge lets me slip under"),
        ("go around the end", "short", "around beats over when it's short")]),
    ("a knotted shoelace", "get it undone", [
        ("pull the loose loops open", "loosely knotted", "loose loops come apart with a tug"),
        ("work the knot with your nails", "a small tight knot", "picking loosens a stubborn knot"),
        ("loosen it from the top eyelets down", "cinched tight", "freeing slack from above loosens the knot"),
        ("cut and re-lace it", "fused solid", "a hopeless knot is faster to replace")]),
    ("a spilled load of groceries", "gather them up", [
        ("pick up the rolling items first", "rolling away", "catching the movers stops them scattering"),
        ("stack them back into the bag", "mostly in one spot", "if they're together, just re-bag them"),
        ("use a nearby cart or box", "too many to carry", "a container beats many trips"),
        ("check for anything broken first", "with something fragile", "handling breakage first avoids a mess")]),
    ("a foggy path", "find your way", [
        ("follow the edge of the trail", "a clear path underfoot", "the trail guides me even when i can't see far"),
        ("move slowly and listen", "thick fog", "slowing down and listening keeps me oriented"),
        ("wait for it to lift", "clearing already", "fog often lifts within the hour"),
        ("use landmarks you can still see", "patchy with gaps", "near landmarks keep me on course")]),
    ("a heavy sliding door", "open it", [
        ("push it along its track", "on a smooth track", "sliding needs a push, not a lift"),
        ("clear the track first", "gritty in the groove", "a clean groove lets it roll"),
        ("lift slightly as you slide", "dropped off its rollers", "a small lift reseats it"),
        ("get help for a wide one", "very large and heavy", "a big door is easier with two")]),
]
REQ_OPEN = ["{o} is in the way and i need to {g}. what do i do?", "how do i get past {o}?",
            "there's {o} - how do i {g}?", "i need to {g} but {o} is blocking me. what should i do?",
            "help me get past {o}.", "what's the best way past {o}?",
            "i'm trying to {g} and {o} is in the way. any ideas?"]
REQ_COND = ["{o} is in my way, and it's {c}. how do i {g}?",
            "there's {o} - it's {c}. what should i do?",
            "i need to {g}. {o} is blocking me and it's {c}. any ideas?",
            "{o} is in the way. it's {c} - what's the move?"]
GHINT = ["i'll", "the best move is to", "i'd", "let me", "i think i should"]


def main(n=15000, seed=61):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        obj, goal, opts = r.choice(OBSTACLES)
        if r.random() < 0.62:
            # CONDITION-DRIVEN: a stated condition selects the fitting action (the key reasoning)
            action, cond, why = r.choice(opts)
            req = r.choice(REQ_COND).format(o=obj, g=goal, c=cond)
            reply = (f"{obj.capitalize()} is blocking me. since it's {cond}, {r.choice(GHINT)} "
                     f"{action} - {why}.")
        else:
            # OPEN: list a few candidate options, then reason to a sensible pick
            picks = r.sample(opts, min(3, len(opts)))
            names = [p[0] for p in picks]
            chosen = picks[0]
            options_txt = ", ".join(names[:-1]) + ", or " + names[-1]
            req = r.choice(REQ_OPEN).format(o=obj, g=goal)
            reply = (f"{obj.capitalize()} is in the way. i could {options_txt}. "
                     f"{r.choice(GHINT).capitalize()} {chosen[0]} first - {chosen[2]}.")
        out.append(f"USER: {req}\nBOT: {reply}")
    r.shuffle(out)
    out_dir = os.path.join(HERE, "data", "solving")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chat.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(out) + "\n")
    print(f"[solving] {len(out):,} generative problem-solving turns ({len(OBSTACLES)} obstacle types) "
          f"-> data/solving/chat.txt ({os.path.getsize(os.path.join(out_dir,'chat.txt')):,} bytes)")


if __name__ == "__main__":
    main()
