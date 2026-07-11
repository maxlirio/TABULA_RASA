#!/usr/bin/env python3
"""One-shot Kaggle/T4 runner: finds the attached datasets, builds the corpus (v5 + Wikipedia +
tools + extra reward-design), and continues training the 157M model. Run from the repo root in a
notebook that has tabula-corpus-v5, tabula-warmstart, and wikitext-103 attached:
    !cd TABULA_RASA && python t4_run.py            # defaults: 250MB wiki, batch 16
    !cd TABULA_RASA && python t4_run.py 150 8       # if out-of-memory: less wiki, smaller batch
"""
import glob
import os
import shutil
import subprocess
import sys

import torch

WIKI_MB = sys.argv[1] if len(sys.argv) > 1 else "600"  # ALL of wikitext-103 (~530MB) -> max unique data
BATCH = sys.argv[2] if len(sys.argv) > 2 else "8"      # smaller batch: the 280M model needs the memory
ITERS = sys.argv[3] if len(sys.argv) > 3 else "80000"  # bigger model + more data -> train longer
# BIGGER MODEL: 1024-wide, 16-layer (~280M params). This does NOT match the 768/12 warmstart, so
# train_lm falls back to random init (from scratch) for run #1 - expected; it BUILDS the new base.
# Run #2+ warm-start from run #1's output (same arch) to accumulate training across runs.
N_EMBD, N_LAYER, N_HEAD = "1024", "16", "16"
WARM = (len(sys.argv) <= 4 or sys.argv[4] != "nowarm")

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
env = dict(os.environ, PYTHONPATH=os.getcwd(), PYTHONUNBUFFERED="1")

# find the datasets under whichever root exists (Kaggle mounts /kaggle/input; Colab uses /content)
DATA_ROOTS = ["/kaggle/input", "/content", os.getcwd()]


def find(pattern, must_contain=None):
    for root in DATA_ROOTS:
        hits = [h for h in glob.glob(f"{root}/**/{pattern}", recursive=True)
                if (must_contain is None or must_contain in h)]
        if hits:
            return hits
    return []


corp = find("chat.txt", "tabula-corpus-v5")
assert corp, "tabula-corpus-v5 dataset not found (attach on Kaggle / download to /content on Colab)"
warm = find("apollo.pt", "warmstart")            # 'warmstart' in path -> never picks our own output
assert warm, "tabula-warmstart dataset not found"
wiki = find("wiki.train.tokens")
assert wiki, "wikitext-103 dataset not found"
os.makedirs("data/mixed", exist_ok=True)
shutil.copy(corp[0], "data/mixed/chat.txt")
print("v5", round(os.path.getsize(corp[0]) / 1e6), "| warm", round(os.path.getsize(warm[0]) / 1e6),
      "| wiki", round(os.path.getsize(wiki[0]) / 1e6), "MB", flush=True)

for cmd in (["prep_wiki.py", WIKI_MB], ["prep_tooluse.py"], ["prep_reasoning.py"],
            ["prep_rules.py"], ["prep_reward_design.py"], ["prep_dialogue2.py"],
            ["prep_stories2.py"], ["prep_solving.py"]):   # generative: diverse fables + problem-solving
    subprocess.run([sys.executable] + cmd, check=True, env=env)


def append(path, times):
    if not os.path.exists(path):
        return
    data = "\n\n".join(b.strip() + " ■" for b in open(path).read().split("\n\n") if b.strip())
    with open("data/mixed/chat.txt", "a") as f:
        for _ in range(times):
            f.write("\n\n" + data)


def strip_all_reward_blocks():
    """Remove EVERY old reward-design block from the base corpus so the ONLY reward data the model
    sees is the clean, uniform new set appended right after. The v5 base was built before the
    spec-format fix; a compound-only scrub was asymmetric (roles like reduce/increase/avoid had no
    underscore in their old spec, so they survived and got over-represented ~1.7x, biasing the model
    to those roles). Dropping any block whose reward: line is a signed spec removes ALL old reward
    examples at once, leaving reward-design to come solely from the fresh uniform append."""
    import re as _re
    spec = _re.compile(r"reward:\s*[+\-]", _re.I)
    blocks = [b for b in open("data/mixed/chat.txt").read().split("\n\n") if b.strip()]
    kept = [b for b in blocks if not spec.search(b)]
    with open("data/mixed/chat.txt", "w") as f:
        f.write("\n\n".join(kept) + "\n")
    print(f"stripped old reward blocks from base: {len(blocks) - len(kept):,} dropped, "
          f"{len(kept):,} kept", flush=True)


append("data/wiki/chat.txt", 1)
append("data/tooluse/chat.txt", 3)
append("data/reasoning/chat.txt", 2)
append("data/rules/chat.txt", 2)
append("data/dialogue2/chat.txt", 10)        # sentiment-correct chat + greetings
append("data/stories2/chat.txt", 20)         # DIVERSE real fables -> generative storytelling (weaving)
append("data/solving/chat.txt", 6)           # generative problem-solving (obstacle -> fitting options)
strip_all_reward_blocks()                   # remove ALL old reward data BEFORE adding the clean set
append("data/reward_design/chat.txt", 16)   # HEAVY upweight: at 4% exposure reward-design drowned in
                                            # both prior runs; the proxy learned it only at ~100%.
                                            # ~20-25% exposure is the lever (verified by preflight B4).
print("final corpus MB:", round(os.path.getsize("data/mixed/chat.txt") / 1e6), flush=True)

warm_arg = warm[0] if WARM else ""     # "" -> train_lm falls back to random init (from scratch)
print("warm-start:", "ON " + warm[0] if WARM else "OFF (from scratch)", flush=True)
# write the model to whichever working dir is writable (Kaggle: /kaggle/working; Colab: /content)
out_dir = next((d for d in ("/kaggle/working", "/content") if os.path.isdir(d) and os.access(d, os.W_OK)),
               os.getcwd())
out_path = os.path.join(out_dir, "apollo.pt")
print("output model ->", out_path, flush=True)
subprocess.run([sys.executable, "-u", "train_lm.py", "mixed", out_path,
                "Apollo", ITERS, "8", N_EMBD, N_LAYER, "256", N_HEAD, BATCH, warm_arg, "30"],
               env=env, check=True)
print("TRAINING COMPLETE ->", out_path, flush=True)
