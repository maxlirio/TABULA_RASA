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

WIKI_MB = sys.argv[1] if len(sys.argv) > 1 else "250"
BATCH = sys.argv[2] if len(sys.argv) > 2 else "16"
ITERS = sys.argv[3] if len(sys.argv) > 3 else "48000"  # last run used only 2.6h/12h -> go longer
# 4th arg "nowarm" trains from scratch (fallback if warm-start's old-format prior still biases specs)
WARM = (len(sys.argv) <= 4 or sys.argv[4] != "nowarm")

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
env = dict(os.environ, PYTHONPATH=os.getcwd(), PYTHONUNBUFFERED="1")

corp = [c for c in glob.glob("/kaggle/input/**/chat.txt", recursive=True) if "tabula-corpus-v5" in c]
assert corp, "attach the tabula-corpus-v5 dataset"
warm = glob.glob("/kaggle/input/**/apollo.pt", recursive=True)
assert warm, "attach the tabula-warmstart dataset"
wiki = glob.glob("/kaggle/input/**/wiki.train.tokens", recursive=True)
assert wiki, "attach the wikitext-103 dataset"
os.makedirs("data/mixed", exist_ok=True)
shutil.copy(corp[0], "data/mixed/chat.txt")
print("v5", round(os.path.getsize(corp[0]) / 1e6), "| warm", round(os.path.getsize(warm[0]) / 1e6),
      "| wiki", round(os.path.getsize(wiki[0]) / 1e6), "MB", flush=True)

for cmd in (["prep_wiki.py", WIKI_MB], ["prep_tooluse.py"], ["prep_reasoning.py"],
            ["prep_rules.py"], ["prep_reward_design.py"]):
    subprocess.run([sys.executable] + cmd, check=True, env=env)


def append(path, times):
    if not os.path.exists(path):
        return
    data = "\n\n".join(b.strip() + " ■" for b in open(path).read().split("\n\n") if b.strip())
    with open("data/mixed/chat.txt", "a") as f:
        for _ in range(times):
            f.write("\n\n" + data)


append("data/wiki/chat.txt", 1)
append("data/tooluse/chat.txt", 3)
append("data/reasoning/chat.txt", 2)
append("data/rules/chat.txt", 2)
append("data/reward_design/chat.txt", 3)  # upweighted: specs need many passes to sharpen the format


def scrub_old_reward_format():
    """The v5 BASE corpus was assembled before the spec-format fix, so it carries OLD glued-format
    reward examples ("reward: +trash_collected ...") whose object-glued tokens now prune to <unk>.
    Left in, they CONTRADICT the new "+collected trash" examples under the identical reasoning
    templates and the model learns neither. Drop every block whose reward: line has an X_Y compound
    so there is ONE consistent target. (Reasoning is plain English -> unaffected.)"""
    import re as _re
    glued = _re.compile(r"reward:[^\n]*[+\-][a-z]+_[a-z]+", _re.I)
    blocks = [b for b in open("data/mixed/chat.txt").read().split("\n\n") if b.strip()]
    kept = [b for b in blocks if not glued.search(b)]
    with open("data/mixed/chat.txt", "w") as f:
        f.write("\n\n".join(kept) + "\n")
    print(f"scrubbed old-format reward blocks: {len(blocks) - len(kept):,} dropped, "
          f"{len(kept):,} kept", flush=True)


scrub_old_reward_format()
print("final corpus MB:", round(os.path.getsize("data/mixed/chat.txt") / 1e6), flush=True)

warm_arg = warm[0] if WARM else ""     # "" -> train_lm falls back to random init (from scratch)
print("warm-start:", "ON " + warm[0] if WARM else "OFF (from scratch)", flush=True)
subprocess.run([sys.executable, "-u", "train_lm.py", "mixed", "/kaggle/working/apollo.pt",
                "Apollo", ITERS, "8", "768", "12", "256", "12", BATCH, warm_arg, "30"],
               env=env, check=True)
print("TRAINING COMPLETE", flush=True)
