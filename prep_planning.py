#!/usr/bin/env python3
"""Teach GROUNDED MEANS-ENDS REASONING: goal -> objective -> best action. This is the reasoning the
user actually wants ("i want to walk forward" -> work out what forward means -> figure the action ->
decide), NOT verbal syllogisms. The model already designs the objective (reward) well; here it learns
to ROUTE an action-goal to the `plan` tool (which grounds objective->action) and VERBALIZE the
decision. Every RESULT is computed by the real plan() tool, so the reasoning shown is always correct.

Output: data/planning/chat.txt
"""
import os
import random

from gm.tools import plan, GOAL_ACTIONS

HERE = os.path.dirname(os.path.abspath(__file__))

# goal phrases the model should route to plan (drawn from the same domain as REWARD_LIB, plus
# natural object phrasings) — kept concrete and motor/embodied, matching the tool's action map.
GOALS = [
    "walk forward", "run fast", "jump", "jump as high as you can", "climb the wall", "crawl forward",
    "balance on one foot", "stand still", "swim across", "move forward", "pick up the box",
    "grab the cup", "lift the crate", "stack the blocks", "place the vase on the shelf",
    "carry the tray", "pour the water", "open the door", "sort the toys", "clean the floor",
    "sweep the room", "reach the target", "avoid the obstacles", "follow the line", "find the exit",
    "explore the room", "win the game", "climb to the top", "walk without falling",
    "pick up the ball without dropping it", "reach the door without hitting anything",
]
REQ = ["how do i {g}", "how should i {g}", "what should i do to {g}", "i want to {g}",
       "help me {g}", "what's the best way to {g}", "figure out how to {g}", "i need to {g}",
       "how would you {g}", "what do i do to {g}", "teach me to {g}", "plan how to {g}"]
THINK = ["let me reason it out.", "let me work through it.", "let me think about the best move.",
         "okay, let me plan this.", "let me figure out the action."]


def _aim_words(spec):
    """A short natural read of the objective, e.g. '+forward_velocity +upright -fall' ->
    'forward speed while staying balanced'. Keeps the chain readable; the spec itself is also shown."""
    pos = [t.lstrip("+").split("(")[0] for t in spec.split() if t.startswith("+")]
    nice = {"forward_velocity": "forward speed", "jump_height": "height", "height_gained": "height",
            "upright": "staying upright", "stability": "staying stable", "grasp_success": "a firm grip",
            "object_height": "raising it", "reached_target": "reaching it", "score": "the best score",
            "area_cleaned": "a clean surface", "opened": "getting it open", "on_path": "staying on the path"}
    parts = [nice[p] for p in pos if p in nice][:2]
    if not parts:
        return "the goal"
    return parts[0] + (" while " + parts[1] if len(parts) > 1 else "")


def main(n=12000, seed=29):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        g = r.choice(GOALS)
        res = plan(g)
        if res == "error":
            continue
        spec, _, action = res.partition("::")
        spec, action = spec.strip(), action.strip()
        req = r.choice(REQ).format(g=g)
        # the model verbalizes the decomposition in words, CALLs plan (grounded), then decides.
        out.append(
            f"USER: {req}\n"
            f"BOT: {r.choice(THINK)} first i'll work out what i'm optimizing, then the action.\n"
            f"CALL: plan {g}\n"
            f"RESULT: {spec} :: {action}\n"
            f"BOT: to {g}, i'm aiming for {_aim_words(spec)} ({spec}), so the best move is to {action}.")
    r.shuffle(out)
    out_dir = os.path.join(HERE, "data", "planning")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chat.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(out) + "\n")
    print(f"[planning] {len(out):,} grounded plan chains ({len(GOALS)} goals x {len(GOAL_ACTIONS)} "
          f"action templates) -> data/planning/chat.txt "
          f"({os.path.getsize(os.path.join(out_dir, 'chat.txt')):,} bytes)")


if __name__ == "__main__":
    main()
