#!/usr/bin/env python3
"""Pre-flight GATE: before spending hours on a T4 run, prove in MINUTES that the data is actually
trainable. It catches the class of bug that has repeatedly only surfaced at the END of long runs —
the data asking the model to emit tokens the pipeline makes impossible (compound spec tokens pruned
to <unk>), a missing stop token, or a degenerate distribution.

Three checks, no big model, tiny text output (no flaky downloads):
  A. VOCAB COVERAGE  — does every reward-spec token survive min_freq pruning? (the bug that bit us)
  B. DISTRIBUTION    — corpus size, per-source share, duplication, stop-token coverage
  C. OVERFIT PROXY   — can a SMALL model learn to emit the target spec format? if it can't even
                       overfit, no amount of scale will. (the "memorization test", as a gate)

Run on Kaggle with tabula-corpus-v5 + wikitext-103 attached (P100 is plenty):
    python preflight.py            # full corpus, min_freq 30, wiki 250MB (matches t4_run.py)
"""
import glob
import os
import re
import subprocess
import sys
import time
from collections import Counter

import torch

from gm.lm import CharLM, WordCoder

MIN_FREQ = int(sys.argv[1]) if len(sys.argv) > 1 else 30
WIKI_MB = sys.argv[2] if len(sys.argv) > 2 else "250"
# append weights MUST mirror t4_run.py so the vocab we test is the vocab the real run will use
WEIGHTS = [("wiki", 1), ("tooluse", 3), ("reasoning", 2), ("rules", 2), ("reward_design", 3)]
_TOK = re.compile(r"\n|[+\-]?[A-Za-z][A-Za-z_]*(?:\([a-z]+\))?|[0-9]+|[^\sA-Za-z0-9]")
EOS = "■"
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()


def stamp(msg):
    print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------- 0. assemble the real corpus
env = dict(os.environ, PYTHONPATH=os.getcwd(), PYTHONUNBUFFERED="1")
corp = [c for c in glob.glob("/kaggle/input/**/chat.txt", recursive=True) if "tabula-corpus-v5" in c]
assert corp, "attach the tabula-corpus-v5 dataset"
wiki = glob.glob("/kaggle/input/**/wiki.train.tokens", recursive=True)
assert wiki, "attach the wikitext-103 dataset"
os.makedirs("data/mixed", exist_ok=True)
import shutil
shutil.copy(corp[0], "data/mixed/chat.txt")
for cmd in (["prep_wiki.py", WIKI_MB], ["prep_tooluse.py"], ["prep_reasoning.py"],
            ["prep_rules.py"], ["prep_reward_design.py"]):
    subprocess.run([sys.executable] + cmd, check=True, env=env)
stamp("prep scripts done")

def _append(name, times):                                     # same as t4_run.py append()
    p = f"data/{name}/chat.txt"
    if not os.path.exists(p):
        return ""
    data = "\n\n".join(b.strip() + " " + EOS for b in open(p).read().split("\n\n") if b.strip())
    with open("data/mixed/chat.txt", "a") as f:
        for _ in range(times):
            f.write("\n\n" + data)
    return data

# EXACT mirror of t4_run.py assembly order: append non-reward sources, strip ALL old reward blocks
# from the base, THEN append the clean uniform reward_design. (Order matters — stripping before the
# reward_design append is what makes the new set the sole, balanced source.)
sources = {"v5": open("data/mixed/chat.txt").read()}
for name, times in [("wiki", 1), ("tooluse", 3), ("reasoning", 2), ("rules", 2)]:
    sources[name] = _append(name, times)
import re as _scrub_re
_spec = _scrub_re.compile(r"reward:\s*[+\-]", _scrub_re.I)
_blocks0 = [b for b in open("data/mixed/chat.txt").read().split("\n\n") if b.strip()]
_kept0 = [b for b in _blocks0 if not _spec.search(b)]
with open("data/mixed/chat.txt", "w") as f:
    f.write("\n\n".join(_kept0) + "\n")
stamp(f"stripped {len(_blocks0) - len(_kept0):,} old reward blocks from base")
sources["reward_design"] = _append("reward_design", 3)
text = open("data/mixed/chat.txt").read()
stamp(f"corpus assembled: {len(text)/1e6:.0f} MB")

coder = WordCoder.from_text(text, min_freq=MIN_FREQ)
vocab = coder.stoi
stamp(f"tokenizer: vocab {len(coder.tokens):,} at min_freq {MIN_FREQ}")

report = []   # (label, ok, detail)


# ---------------------------------------------------------------- A. reward-spec vocab coverage
import prep_reward_design as P
# common objects that should appear in wiki/v5 (so they survive pruning) — keep this list common
TEST_OBJECTS = ["trash", "kitchen", "phone", "battery", "garden", "dishes", "laundry",
                "leaves", "engine", "data", "files", "noise"]
# the function words are what the FIX is about: each is shared across thousands of designs, so it
# must survive. objects are informational (some niche ones may be <unk>, same as in the reasoning).
fn_tokens, obj_tokens = set(), set()
for verbs, why, terms in P.ROLES:
    spec = terms.format(o="OBJ")
    for tok in _TOK.findall(spec):
        (obj_tokens if tok == "OBJ" else fn_tokens).add(tok)
missing_fn = sorted(t for t in fn_tokens if t not in vocab)
obj_present = sum(o in vocab for o in TEST_OBJECTS)
obj_cov = obj_present / len(TEST_OBJECTS)
# the OLD glued format, for contrast — these SHOULD all be missing (that was the bug)
old_glued = [f"+{o}_collected" for o in TEST_OBJECTS[:5]] + [f"+{o}_level" for o in TEST_OBJECTS[:5]]
old_present = sum(t in vocab for t in old_glued)
okA = not missing_fn and obj_cov >= 0.9 and old_present == 0
report.append(("A. reward-spec coverage", okA,
               f"function-words in vocab: {len(fn_tokens) - len(missing_fn)}/{len(fn_tokens)} "
               f"(missing: {missing_fn or 'none'}); "
               f"common objects in vocab: {obj_present}/{len(TEST_OBJECTS)} ({obj_cov:.0%}); "
               f"[contrast] old glued format present: {old_present}/{len(old_glued)} (want 0)"))


# ---------------------------------------------------------------- B. distribution sanity
total_mb = len(text) / 1e6
shares = {n: len(s) / len(text) for n, s in sources.items()}
# weight the appended sources by their repeat count to reflect TRAINING exposure
exposure = {"v5": len(sources["v5"])}
for n, t in WEIGHTS:
    if n in sources:
        exposure[n] = len(sources[n]) * t
exp_tot = sum(exposure.values())
exposure = {n: v / exp_tot for n, v in exposure.items()}
blocks = [b for b in text.split("\n\n") if b.strip()]
dup = 1 - len(set(blocks)) / len(blocks)
eos_in_vocab = EOS in vocab
eos_cov = sum(b.rstrip().endswith(EOS) for b in blocks) / len(blocks)
okB = eos_in_vocab and dup < 0.6 and max(exposure.values()) < 0.85
report.append(("B. distribution", okB,
               f"{total_mb:.0f} MB, {len(blocks):,} blocks; exposure-share "
               + ", ".join(f"{n} {exposure[n]:.0%}" for n in exposure)
               + f"; dup blocks {dup:.1%}; stop-token in vocab {eos_in_vocab}, "
               f"blocks ending with stop {eos_cov:.0%}"))


# ---------------------------------------------------------------- B2. format-conflict scan
# The proxy (below) trains from scratch on clean signal, so it can't see CONTRADICTIONS in the
# full corpus — e.g. old glued-format reward lines ("reward: +trash_collected") left over from an
# earlier corpus build, competing with the new "reward: +collected trash" under identical reasoning.
# This scan reads the assembled corpus directly and fails if a meaningful fraction of reward: lines
# still use the old X_Y compound format. (This is the check that would have caught the 62K-line
# contamination that a from-scratch proxy passed straight over.)
import re as _re
_glued = _re.compile(r"reward:[^\n]*[+\-][a-z]+_[a-z]+", _re.I)
_newfmt = _re.compile(r"reward:[^\n]*[+\-](collected|remaining|amount|cleanliness|sorted|built|"
                      r"intact|recalled|fixed|level|charge|health|reached|contact)\s", _re.I)
rblocks = [b for b in text.split("\n\n") if "reward:" in b]
old_fmt = sum(bool(_glued.search(b)) for b in rblocks)
new_fmt = sum(bool(_newfmt.search(b)) for b in rblocks)
conflict = old_fmt / max(old_fmt + new_fmt, 1)
okB2 = conflict < 0.02      # a tiny residue is ok; anything real means two contradictory targets
report.append(("B2. reward-format consistency", okB2,
               f"reward blocks: {new_fmt:,} new-format vs {old_fmt:,} OLD glued-format "
               f"({conflict:.1%} contamination); the corpus must teach ONE spec format. "
               f"NOTE: the real run WARM-STARTS from apollo_v5 (old-format prior) — clean data is "
               f"what lets the new format win over it."))


# ---------------------------------------------------------------- B3. reward-role balance
# The model picks the wrong ROLE ("picking up trash" -> "the goal is MORE trash") when some roles
# are over-represented. This counts each role's distinctive reasoning phrase across the corpus and
# fails if the distribution is lopsided (which is what an asymmetric scrub silently produced).
ROLE_SIGS = {"collect": "so reward how much", "reduce": "the goal is LESS",
             "increase": "the goal is MORE", "avoid": "exactly what we don't want",
             "reach": "so reward reaching", "clean": "a clean", "sort": "order is the goal",
             "build": "so reward progress and stability", "protect": "safe is the goal",
             "learn": "so reward recall", "fix": "so reward it fixed",
             "balance": "so reward staying level", "charge": "so reward the charge",
             "tend": "reward its health"}
rc = {k: text.count(v) for k, v in ROLE_SIGS.items()}
lo, hi = min(rc.values()), max(rc.values())
ratio = hi / max(lo, 1)
okB3 = ratio < 1.4       # roughly uniform; asymmetric-scrub skew was ~1.7x
report.append(("B3. reward-role balance", okB3,
               f"role counts (max/min ratio {ratio:.2f}, want <1.4): "
               + ", ".join(f"{k} {rc[k]:,}" for k in sorted(rc, key=rc.get, reverse=True)[:6])
               + f", ... min={lo:,}"))


# ---------------------------------------------------------------- C. tiny-overfit proxy
# Can a SMALL model learn to EMIT the new spec format? Train only on the reward-design signal
# (with the REAL full vocab so token ids match the big run), overfit ~2500 iters, then inspect
# what it generates. If it produces clean "+collected trash -remaining trash -time" specs, the
# format is learnable and the 8h run will do it better. If not, stop before the 8h run.
stamp("proxy: training small model on reward-design signal ...")
sig = sources.get("reward_design", "")
ids = torch.tensor(coder.encode(sig), dtype=torch.long)
block = 96
model = CharLM(len(coder.tokens), n_embd=256, n_head=8, n_layer=4, block_size=block, drop=0.0).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
bs = 64
unk = coder.stoi.get("<unk>")
for it in range(1, 2501):
    ix = torch.randint(len(ids) - block - 1, (bs,))
    x = torch.stack([ids[i:i + block] for i in ix]).to(dev)
    y = torch.stack([ids[i + 1:i + block + 1] for i in ix]).to(dev)
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if it % 500 == 0:
        stamp(f"  proxy iter {it}  loss {loss.item():.3f}")

PROMPTS = ["picking up trash", "keeping the kitchen clean", "charging my phone",
           "collecting the leaves", "reducing noise", "fixing the engine"]
model = model.to("cpu").eval()        # gen_ids builds CPU seed tensors; keep model+input on one device
ban = [unk] if unk is not None else None
clean = 0
proxy_lines = []
try:
    for g in PROMPTS:
        seed = coder.encode(f"USER: design a reward for {g}\nBOT: ")
        out = model.gen_ids(list(seed), 40, temp=0.1, top_k=1, ban=ban)
        txt = coder.decode(out).split(EOS)[0].split("\n")[0].strip()
        spec = txt.split("reward:")[-1].strip() if "reward:" in txt else txt
        obj = g.split()[-1]
        # "clean" = spec mentions the actual object as a standalone token and has >=2 signed terms
        toks = _TOK.findall(spec)
        signed = [t for t in toks if t[0] in "+-"]
        has_obj = obj in toks and "<unk>" not in spec
        ok = has_obj and len(signed) >= 2
        clean += ok
        proxy_lines.append(f"    [{'OK' if ok else '  '}] {g:24} -> {txt[:90]}")
    okC = clean >= len(PROMPTS) * 0.7
    detailC = (f"{clean}/{len(PROMPTS)} prompts produced a clean spec (object as standalone token "
               f"+ >=2 signed terms)\n" + "\n".join(proxy_lines))
except Exception as e:
    okC, detailC = False, f"proxy generation errored: {e!r}"
report.append(("C. overfit proxy emits clean specs", okC, detailC))


# ---------------------------------------------------------------- summary
print("\n" + "=" * 78)
print("PRE-FLIGHT REPORT".center(78))
print("=" * 78)
for label, ok, detail in report:
    print(f"\n[{'PASS' if ok else 'FAIL'}] {label}\n    {detail}")
allok = all(ok for _, ok, _ in report)
print("\n" + "=" * 78)
print(("ALL CHECKS PASS — safe to launch the T4 run." if allok
       else "ONE OR MORE CHECKS FAILED — fix the data before the T4 run.").center(78))
print("=" * 78, flush=True)
sys.exit(0 if allok else 1)
