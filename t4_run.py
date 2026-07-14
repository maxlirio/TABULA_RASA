#!/usr/bin/env python3
"""One-shot Kaggle/T4 runner. Run from the repo root in a notebook with tabula-corpus-v5,
tabula-warmstart-280m, and wikitext-103 attached:
    !cd TABULA_RASA && python t4_run.py            # defaults below
    !cd TABULA_RASA && python t4_run.py 80 4       # if out-of-memory: less wiki, smaller batch

RUN #3. It warm-starts run #1's weights (same 1024/16 arch, so every tensor carries over) and
re-feeds the skills at exposures that can actually be learned, keeping v5 + a wiki slice as a REPLAY
stream so the fluency the base run paid 9.5 hours for isn't forgotten.

WHAT RUN #2 ACTUALLY TAUGHT US (it trained 40k iters and shipped a copy of its own input):
  - Its premise was half wrong. "The model stopped calling its tools" was a STALE PROBE talking: the
    tool probe still demanded a CALL on reward prompts, years after 69e4375 retired the lookup plan
    tool and made reward design generative. Run #1 scores 1.00 on the corrected probe -- its tools
    were never broken. The old probe was pinned at 0.68 for all 201 evals, so half the
    checkpoint-selection metric was a CONSTANT.
  - The saver then compared every trained checkpoint against iter 1 -- which, on a warm start, IS
    THE INPUT MODEL. Nothing beat it, so it staged iter 1 and discarded the trained weights unsaved.
  - And rules DROWNED, exactly as tooluse had: 1.2% of exposure, probe 0.71 -> 0.50. Same disease,
    new organ. The gate had floors only for the skills that had already failed.

So this run: rules 3x -> 14x (1.2% -> 5.2%), a corrected tool probe, a STORY probe (the headline
goal, previously unmeasured, graded at the shell's own decoding), and a saver that treats the warm
baseline as something to BEAT and always writes the final model.
"""
import glob
import os
import shutil
import subprocess
import sys

import torch

from corpus_scrub import markup_count, scrub_markup

WIKI_MB = sys.argv[1] if len(sys.argv) > 1 else "120"  # a REPLAY slice, not the whole encyclopedia:
#                                                        run #1 already put Wikipedia in the weights
BATCH = sys.argv[2] if len(sys.argv) > 2 else "8"      # the 280M model needs the memory
ITERS = sys.argv[3] if len(sys.argv) > 3 else "40000"  # continuation, not base-building (~5h on T4)
# SAME 1024/16 architecture as run #1, so the warm-start matches tensor-for-tensor and this is a
# true continuation. MIN_FREQ 0 = reuse run #1's vocabulary verbatim (see train_lm): rebuilding it
# from this smaller corpus would prune the rare Wikipedia words the base model learned, throwing
# away trained embedding rows for words that simply aren't frequent in a skills-heavy mix.
N_EMBD, N_LAYER, N_HEAD, MIN_FREQ = "1024", "16", "16", "0"
WARM = (len(sys.argv) <= 4 or sys.argv[4] != "nowarm")

def check_gpu():
    """Fail NOW, not 3 minutes from now. Kaggle's PyTorch ships sm_70+ kernels only, so a P100
    (sm_60) reports cuda available and then dies on the first kernel launch -- but only AFTER the
    corpus is assembled, so it surfaces as a cryptic CUDA trace that reads like a data bug. The
    Kaggle API cannot request a GPU type (enable_gpu is the only knob and it defaults to a P100),
    so the T4 is a UI choice and this guard is the only thing that catches getting it wrong."""
    assert torch.cuda.is_available(), "No GPU. Set the notebook Accelerator to GPU T4 x2."
    name, cap = torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
    print(f"GPU: {name} (sm_{cap[0]}{cap[1]})", flush=True)
    assert cap >= (7, 0), (f"{name} is sm_{cap[0]}{cap[1]}; this PyTorch is sm_70+ only and cannot "
                           f"execute a single kernel on it. Set the Accelerator to GPU T4 x2.")
    (torch.zeros(8, 8, device="cuda") @ torch.zeros(8, 8, device="cuda")).sum().item()   # prove it


check_gpu()
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
warm = find("apollo.pt", "warmstart-280m")       # run #1's weights; 'warmstart' in the path means
#                                                  we can never accidentally pick up our own output
assert warm, "tabula-warmstart-280m dataset not found (upload run #1's apollo_280m.pt as apollo.pt)"
wiki = find("wiki.train.tokens")
assert wiki, "wikitext-103 dataset not found"
os.makedirs("data/mixed", exist_ok=True)
shutil.copy(corp[0], "data/mixed/chat.txt")
print("v5", round(os.path.getsize(corp[0]) / 1e6), "| warm", round(os.path.getsize(warm[0]) / 1e6),
      "| wiki", round(os.path.getsize(wiki[0]) / 1e6), "MB", flush=True)

# NOTE: prep_bigwiki / prep_bookcorpus are deliberately NOT run. Those are the base-building corpora
# from run #1; they are already in the warm-started weights, and re-feeding 600MB of them would bury
# the skills all over again (that is exactly what went wrong).
for cmd in (["prep_wiki.py", WIKI_MB], ["prep_tooluse.py"], ["prep_reasoning.py"],
            ["prep_rules.py"], ["prep_reward_design.py"], ["prep_dialogue2.py"],
            ["prep_stories2.py"], ["prep_solving.py"]):   # generative: fables + problem-solving
    subprocess.run([sys.executable] + cmd, check=False, env=env)   # check=False: one flaky download
    #                                                                shouldn't abort the whole run


def append(path, times):
    if not os.path.exists(path):
        return
    data = "\n\n".join(b.strip() + " ■" for b in open(path).read().split("\n\n") if b.strip())
    with open("data/mixed/chat.txt", "a") as f:
        for _ in range(times):
            f.write("\n\n" + data)


def scrub_base_markup():
    """Runs on the v5 BASE only (before any skill data is appended). v5 is 35% of exposure and 14% of
    its blocks carry _italic_ / [Illustration] Gutenberg typesetting. Run #1 memorized junk that
    reliably co-occurred with prose -- a fable index -- and this is the same junk in the biggest
    source we feed. Scrubbing at assembly costs seconds and needs no re-upload of the v5 dataset.
    Deliberately NOT applied to the skill sources: they contain no underscores or brackets today, and
    if one ever adopts them as SYNTAX (a tool call, a reward spec) this must not silently mangle it."""
    text = open("data/mixed/chat.txt").read()
    before, total = markup_count(text)
    open("data/mixed/chat.txt", "w").write(scrub_markup(text))
    after, _ = markup_count(open("data/mixed/chat.txt").read())
    print(f"scrubbed Gutenberg markup from v5 base: {before:,}/{total:,} blocks carried it "
          f"-> {after:,} remain", flush=True)


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


# Exposure is the lever this project keeps re-learning: a skill the model sees at a few percent of
# its diet is a skill it does not learn, no matter how long you train. v5 + the wiki slice are the
# REPLAY stream (~50%) that keeps run #1's fluency; everything else is a skill being re-taught.
# Targets (preflight B/B4 verify them on the real assembled corpus, so they can't silently drift):
#   reward_design ~21%   tooluse ~15%   stories (dialogue2 + stories2) ~8%   solving ~3%
scrub_base_markup()                          # v5 base only — must precede the skill appends
append("data/wiki/chat.txt", 1)
append("data/tooluse/chat.txt", 12)          # 2% in run #1 -> the model stopped calling its tools
append("data/reasoning/chat.txt", 6)
append("data/rules/chat.txt", 14)       # 3x = 1.2% and the rules skill DROWNED in run
                                             # #2 (probe 0.71 -> 0.50). 14x = 5.2%.
append("data/dialogue2/chat.txt", 12)        # sentiment chat + COMPLETE template stories (the arc)
append("data/stories2/chat.txt", 20)         # diverse real fables, now scrubbed of index junk
append("data/solving/chat.txt", 8)           # generative problem-solving (obstacle -> fitting options)
strip_all_reward_blocks()                    # remove ALL old reward data BEFORE adding the clean set
append("data/reward_design/chat.txt", 20)    # the proven recipe: ~20%+ exposure is what makes reward
                                             # design stick (4% drowned it in two separate full runs)
print("final corpus MB:", round(os.path.getsize("data/mixed/chat.txt") / 1e6), flush=True)

warm_arg = warm[0] if WARM else ""     # "" -> train_lm falls back to random init (from scratch)
print("warm-start:", "ON " + warm[0] if WARM else "OFF (from scratch)", flush=True)
# write the model to whichever working dir is writable (Kaggle: /kaggle/working; Colab: /content)
out_dir = next((d for d in ("/kaggle/working", "/content") if os.path.isdir(d) and os.access(d, os.W_OK)),
               os.getcwd())
out_path = os.path.join(out_dir, "apollo.pt")
print("output model ->", out_path, flush=True)
subprocess.run([sys.executable, "-u", "train_lm.py", "mixed", out_path,
                "Apollo", ITERS, "8", N_EMBD, N_LAYER, "256", N_HEAD, BATCH, warm_arg, MIN_FREQ],
               env=env, check=True)
print("TRAINING COMPLETE ->", out_path, flush=True)
