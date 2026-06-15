#!/usr/bin/env python3
"""Build consistent-frame training pairs that map a PLAIN-ENGLISH locomotion goal to a
reward spec for a virtual humanoid. This is the "language -> reward" job: you say
"make it move forward efficiently", it emits a reward made of known terms.

The reward is a small DSL: a signed list of terms from a fixed library, with optional
(strong)/(weak) emphasis. A separate compiler (you write later) turns this into the actual
reward function; RL then optimises the policy. The model only has to learn goal -> which
terms, their sign, and rough emphasis -- and to ask for clarity when the goal is vague.

Term library (the ceiling on what it can express):
  forward_velocity, distance, upright, fall, energy_cost, jerk, lateral_deviation, height, alive
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

# ways to phrase "I want the humanoid to ..."
LEAD = ["make it", "teach it to", "i want it to", "get it to", "have it", "train it to"]
# core forward goal + how modifiers attach extra reward terms
SPEED = {"": "", "quickly": "(strong)", "fast": "(strong)", "slowly": "(weak)",
         "at a steady pace": ""}
EFF = ["efficiently", "using less energy", "without wasting energy"]
SMOOTH = ["smoothly", "gently", "without jerking"]


def spec(*terms):
    return "reward: " + " ".join(terms)


def frames(r):
    lead = r.choice(LEAD)
    speed_word = r.choice(list(SPEED))
    s = SPEED[speed_word]
    out = []
    base = f"+forward_velocity{s} +upright -fall"

    # move forward (plain / speed)
    g = f"{lead} move forward {speed_word}".strip()
    out.append((g, spec("+forward_velocity" + s, "+upright", "-fall")))
    # efficiently -> add energy penalty
    out.append((f"{lead} move forward {r.choice(EFF)}",
                spec("+forward_velocity", "+upright", "-fall", "-energy_cost")))
    # smoothly -> add jerk penalty
    out.append((f"{lead} walk {r.choice(SMOOTH)}",
                spec("+forward_velocity", "+upright", "-fall", "-jerk")))
    # straight line -> penalise drift
    out.append((f"{lead} go forward in a straight line",
                spec("+forward_velocity", "-lateral_deviation", "+upright", "-fall")))
    # without falling -> strong fall penalty
    out.append((f"{lead} move forward without falling",
                spec("+forward_velocity", "+upright", "-fall(strong)")))
    # as far as possible -> distance
    out.append((f"{lead} walk as far as possible",
                spec("+distance", "+upright", "-fall")))
    # balance / stand still
    out.append((r.choice([f"{lead} stand still", f"{lead} balance in place",
                          f"{lead} keep its balance"]),
                spec("+upright", "-lateral_deviation", "-fall", "-energy_cost")))
    # combined: forward + efficient + smooth
    out.append((f"{lead} move forward {r.choice(EFF)} and {r.choice(SMOOTH)}",
                spec("+forward_velocity", "+upright", "-fall", "-energy_cost", "-jerk")))

    # vague goals -> ask ONE clarifying question, stay on topic
    out.append(("make it better",
                "better at what - moving forward, balancing, or using less energy?"))
    out.append((f"{lead} move",
                "which way should it move - forward, or just balance in place?"))
    out.append((f"{lead} do well",
                "what counts as doing well here - speed, distance, or staying upright?"))
    return out


def main(n=900, seed=2):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        for u, b in frames(r):
            out.append(f"USER: {u}\nBOT: {b}\n")
    r.shuffle(out)
    out_dir = os.path.join(HERE, "data", "rewards")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "chat.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"[rewards] {len(out):,} turns -> {path} ({os.path.getsize(path):,} bytes)")


if __name__ == "__main__":
    main()
