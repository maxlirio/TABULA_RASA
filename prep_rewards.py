#!/usr/bin/env python3
"""Teach LANGUAGE -> REWARD across many domains and phrasings, so it generalizes beyond
locomotion. The model sees a goal stated many ways ("design a reward system for X",
"make a reward for X", "make it X") and learns to emit a reward spec: a signed list of
relevant terms. Domains/terms vary so it learns the OPERATION, not specific rewards.

These are EXAMPLES of the breadth, not the limit — the variety is what lets it generalize.
Output: data/rewards/chat.txt
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

# (gerund-form goal, reward spec) across domains
GOALS = [
    # locomotion
    ("moving forward efficiently", "+forward_velocity +upright -fall -energy_cost"),
    ("walking smoothly", "+forward_velocity +upright -fall -jerk"),
    ("running fast", "+forward_velocity(strong) +upright -fall"),
    ("jumping well", "+jump_height +stable_landing +upright -fall -energy_cost"),
    ("jumping high", "+jump_height(strong) +stable_landing -fall"),
    ("balancing", "+upright -lateral_deviation -fall -energy_cost"),
    ("climbing stairs", "+height_gained +upright -fall -energy_cost"),
    ("swimming forward", "+forward_velocity -drag -energy_cost"),
    ("standing still", "+upright -movement -fall"),
    # manipulation
    ("grabbing the object", "+grasp_success -drops -collisions"),
    ("lifting the box", "+box_height +grasp_success -drops"),
    ("stacking the blocks", "+blocks_stacked +stability -drops -collisions"),
    ("placing the cup on the shelf", "+placed_correctly -drops -collisions"),
    ("carrying the tray without spilling", "+distance_carried -spills -tilt"),
    ("opening the door", "+door_opened -collisions -force_used"),
    ("pouring water into the glass", "+water_in_glass -spilled -overflow"),
    # organizing / sorting
    ("organizing the files", "+files_in_correct_folder +tidiness -misplaced -time"),
    ("sorting the items by color", "+items_correctly_sorted -mistakes -time"),
    ("tidying the desk", "+items_put_away +tidiness -clutter_left"),
    ("alphabetizing the list", "+correct_order -out_of_order -time"),
    ("grouping similar objects", "+correct_groups -wrong_groups"),
    # cleaning
    ("cleaning the room", "+area_cleaned -mess_remaining -time"),
    ("sweeping the floor", "+floor_cleaned -dirt_left -time"),
    ("washing the dishes", "+dishes_cleaned -dishes_left -water_wasted"),
    # navigation
    ("reaching the target", "+reached_target -distance -time -collisions"),
    ("avoiding the obstacles", "+obstacles_avoided -collisions"),
    ("following the path", "+on_path -deviation -time"),
    ("exploring the room", "+area_explored -repeated_areas -time"),
    ("finding the exit", "+reached_exit -distance -time"),
    # games / objectives
    ("winning the game", "+score +wins -losses"),
    ("scoring points", "+points_scored -turnovers"),
    ("beating the opponent", "+score_difference +wins -losses"),
    ("surviving as long as possible", "+time_alive -damage_taken"),
    # building / making
    ("building a tall tower", "+tower_height +stability -blocks_fallen"),
    ("building a stable bridge", "+bridge_stability +span -collapses -materials_used"),
    ("drawing a circle", "+shape_accuracy -deviation"),
    # generic quality / efficiency
    ("finishing the task quickly", "+task_progress -time -energy_cost"),
    ("doing it accurately", "+accuracy -errors"),
    ("using less energy", "+task_progress -energy_cost"),
]
REQ = [
    "design a reward system for {g}", "design a reward for {g}",
    "make a reward function for {g}", "make a reward for {g}",
    "give me a reward for {g}", "how should i reward {g}",
    "what's a good reward for {g}", "i need a reward system for {g}",
    "set up a reward for {g}", "reward it for {g}",
]
# direct imperative style too ("make it jump high")
DIRECT = {
    "move forward efficiently": "+forward_velocity +upright -fall -energy_cost",
    "walk smoothly": "+forward_velocity +upright -fall -jerk",
    "jump high": "+jump_height(strong) +stable_landing -fall",
    "run fast": "+forward_velocity(strong) +upright -fall",
    "balance": "+upright -lateral_deviation -fall -energy_cost",
    "stack the blocks": "+blocks_stacked +stability -drops",
    "reach the target": "+reached_target -distance -time -collisions",
    "clean the room": "+area_cleaned -mess_remaining -time",
}
VAGUE = [
    ("design a reward system", "for what task? tell me the goal and i'll build the reward."),
    ("make a reward", "a reward for what? give me the goal."),
    ("make it better", "better at what - the goal decides the reward."),
]


def main(n=700, seed=2):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        for g, spec in GOALS:
            out.append(f"USER: {r.choice(REQ).format(g=g)}\nBOT: reward: {spec}\n")
        for g, spec in DIRECT.items():
            verb = r.choice(["make it", "teach it to", "i want it to", "get it to"])
            out.append(f"USER: {verb} {g}\nBOT: reward: {spec}\n")
        for u, b in VAGUE:
            out.append(f"USER: {u}\nBOT: {b}\n")
    r.shuffle(out)
    out_dir = os.path.join(HERE, "data", "rewards")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chat.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"[rewards] {len(out):,} reward turns ({len(GOALS)} domains x {len(REQ)} phrasings) "
          f"-> data/rewards/chat.txt")


if __name__ == "__main__":
    main()
