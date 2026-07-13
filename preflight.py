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

Runs anywhere the inputs can be found — Kaggle (/kaggle/input), Colab (/content), or a laptop (the
repo's own data/ dir), on CUDA, Apple MPS, or CPU:
    python preflight.py            # min_freq 0 (reuse the warm vocab), wiki 120MB — matches t4_run.py

NOT on a Kaggle P100. Kaggle's PyTorch now ships sm_70+ kernels only, so a P100 (sm_60) cannot
execute ANY kernel -- "no kernel image is available for execution on the device". The API cannot
request a GPU type (enable_gpu only, and its default is a P100), so an API-pushed kernel is a P100
and will die. Select T4 in the Kaggle UI, or just run this locally. _pick_device below PROBES the
device rather than trusting is_available(), so this fails in seconds with a clear message instead of
three minutes in with a CUDA stack trace.
"""
import glob
import os
import re
import subprocess
import sys
import time
from collections import Counter

import torch

from corpus_scrub import markup_count, scrub_markup
from gm.lm import CharLM, WordCoder

MIN_FREQ = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # 0 = reuse run #1's vocab, as t4_run does
WIKI_MB = sys.argv[2] if len(sys.argv) > 2 else "120"
# Append weights MUST mirror t4_run.py or this gate is testing a corpus nobody will train on. ONE
# list, used for both the assembly and the exposure maths (they were two lists, which is how they
# drift apart). reward_design stays last: it is appended AFTER the old-reward strip.
WEIGHTS = [("wiki", 1), ("tooluse", 12), ("reasoning", 6), ("rules", 3), ("dialogue2", 12),
           ("stories2", 20), ("solving", 8), ("reward_design", 20)]
_TOK = re.compile(r"\n|[+\-]?[A-Za-z][A-Za-z_]*(?:\([a-z]+\))?|[0-9]+|[^\sA-Za-z0-9]")
_re_attr = re.compile(r"\([A-Z][a-z]+\)")     # "(French)" source tags -> a fable index, not a story
EOS = "■"
t0 = time.time()


def stamp(msg):
    print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)


def _pick_device():
    """PROBE the accelerator, don't trust is_available(). A Kaggle P100 reports cuda available and
    then fails on the first real kernel launch (sm_60 dropped from their PyTorch build) -- three
    minutes of corpus assembly wasted before a cryptic CUDA trace. A tiny matmul settles it now."""
    for name in ("cuda", "mps"):
        try:
            if not getattr(torch, name).is_available():
                continue
        except Exception:
            continue
        try:
            (torch.zeros(8, 8, device=name) @ torch.zeros(8, 8, device=name)).sum().item()
            return name
        except Exception as e:
            print(f"[preflight] {name} is present but CANNOT execute kernels ({type(e).__name__}: "
                  f"{str(e).splitlines()[0]}) — falling back", flush=True)
    return "cpu"


dev = _pick_device()


# ---------------------------------------------------------------- 0. assemble the real corpus
env = dict(os.environ, PYTHONPATH=os.getcwd(), PYTHONUNBUFFERED="1")
# Same root auto-detection as t4_run.py: Kaggle mounts /kaggle/input, Colab uses /content, and a
# laptop just has the repo. WIKI_DIR lets a local run point at a downloaded wikitext-103.
DATA_ROOTS = ["/kaggle/input", "/content", os.environ.get("WIKI_DIR", ""), os.getcwd()]


def find(pattern, must_contain=None):
    for root in DATA_ROOTS:
        if not root or not os.path.isdir(root):
            continue
        hits = [h for h in glob.glob(f"{root}/**/{pattern}", recursive=True)
                if must_contain is None or must_contain in h]
        if hits:
            return hits
    return []


# the v5 BASE corpus: the Kaggle dataset, or the repo's own copy when running locally
corp = find("chat.txt", "tabula-corpus-v5") or find("chat.txt", "mixed_v5")
assert corp, "tabula-corpus-v5 not found (attach the dataset, or keep data/mixed_v5/chat.txt)"
wiki = find("wiki.train.tokens")
assert wiki, "wikitext-103 not found (attach the dataset, or set WIKI_DIR=/path/to/wikitext)"
warm_ck = find("apollo.pt", "warmstart") or find("apollo_280m.pt")   # supplies the vocab at min_freq 0
os.makedirs("data/mixed", exist_ok=True)
import shutil
shutil.copy(corp[0], "data/mixed/chat.txt")
stamp(f"inputs: v5 {corp[0]} | wiki {wiki[0]} | warm {warm_ck[0] if warm_ck else 'NONE'} | dev {dev}")
# pass the wiki path explicitly — prep_wiki only globs /kaggle/input on its own
for cmd in (["prep_wiki.py", WIKI_MB, wiki[0]], ["prep_tooluse.py"], ["prep_reasoning.py"],
            ["prep_rules.py"], ["prep_reward_design.py"], ["prep_dialogue2.py"],
            ["prep_stories2.py"], ["prep_solving.py"]):   # no bigwiki/bookcorpus: see t4_run.py
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
# mirror t4_run: scrub the v5 BASE before any skill data is appended (see corpus_scrub)
_base = open("data/mixed/chat.txt").read()          # read BEFORE opening for write ("w" truncates)
_pre_scrub, _pre_total = markup_count(_base)
with open("data/mixed/chat.txt", "w") as f:
    f.write(scrub_markup(_base))
stamp(f"scrubbed Gutenberg markup from v5 base: {_pre_scrub:,}/{_pre_total:,} blocks carried it")

sources = {"v5": open("data/mixed/chat.txt").read()}
for name, times in WEIGHTS:
    if name == "reward_design":       # appended below, after the strip
        continue
    sources[name] = _append(name, times)
import re as _scrub_re
_spec = _scrub_re.compile(r"reward:\s*[+\-]", _scrub_re.I)
_blocks0 = [b for b in open("data/mixed/chat.txt").read().split("\n\n") if b.strip()]
_kept0 = [b for b in _blocks0 if not _spec.search(b)]
with open("data/mixed/chat.txt", "w") as f:
    f.write("\n\n".join(_kept0) + "\n")
stamp(f"stripped {len(_blocks0) - len(_kept0):,} old reward blocks from base")
sources["reward_design"] = _append("reward_design", dict(WEIGHTS)["reward_design"])
text = open("data/mixed/chat.txt").read()
stamp(f"corpus assembled: {len(text)/1e6:.0f} MB")

# MIN_FREQ 0 = reuse the warm checkpoint's vocabulary, exactly as train_lm.py does for a
# continuation run. Testing a freshly-built vocab here would be testing the wrong tokenizer.
if MIN_FREQ == 0:
    assert warm_ck, "min_freq 0 needs the warm checkpoint (it supplies the vocab)"
    coder = WordCoder(torch.load(warm_ck[0], map_location="cpu")["tokens"])
    stamp(f"tokenizer: REUSING warm checkpoint vocab, {len(coder.tokens):,} tokens")
else:
    coder = WordCoder.from_text(text, min_freq=MIN_FREQ)
    stamp(f"tokenizer: vocab {len(coder.tokens):,} at min_freq {MIN_FREQ}")
vocab = coder.stoi

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
okB = eos_in_vocab and dup < 0.75 and max(exposure.values()) < 0.85   # dup high is OK: we upweight
report.append(("B. distribution", okB,
               f"{total_mb:.0f} MB, {len(blocks):,} blocks; exposure-share "
               + ", ".join(f"{n} {exposure[n]:.0%}" for n in exposure)
               + f"; dup blocks {dup:.1%}; stop-token in vocab {eos_in_vocab}, "
               f"blocks ending with stop {eos_cov:.0%}"))

# B4. reward-design EXPOSURE floor — the lever these runs revealed. At 4% exposure reward-design
# drowned (both a warm-started and a from-scratch full run failed); the proxy learned it only near
# 100%. Fail if reward-design is too small a fraction of training exposure to actually be learned.
rd_exp = exposure.get("reward_design", 0)
okB4 = rd_exp >= 0.15          # back to the PROVEN floor. It was relaxed to 8% for the run #1
#                               fluency phase, which is precisely when reward design degraded again.
report.append(("B4. reward-design exposure", okB4,
               f"reward_design is {rd_exp:.0%} of training exposure (want >=15%: 22% produced clean "
               f"specs on every goal, 8% degraded them, 4% drowned them completely)"))

# B5. the OTHER skills need a learnable share too — run #1 proved the point in the negative: tooluse
# fell to 2% and the model simply stopped calling its tools (invented a time instead of asking the
# clock), while stories at 2% collapsed into reciting fable-index lines. Same disease as reward
# design, different organ. So gate every skill we are paying to re-teach, not just the reward one.
SKILL_FLOORS = {"tooluse": 0.10, "solving": 0.02}
story_exp = exposure.get("dialogue2", 0) + exposure.get("stories2", 0)   # two sources, one skill
skill_bad = [f"{n} {exposure.get(n, 0):.0%} (want >={f:.0%})"
             for n, f in SKILL_FLOORS.items() if exposure.get(n, 0) < f]
if story_exp < 0.05:
    skill_bad.append(f"stories {story_exp:.0%} (want >=5%, dialogue2 + stories2)")
okB5 = not skill_bad
report.append(("B5. skill exposure floors", okB5,
               f"tooluse {exposure.get('tooluse', 0):.0%}, stories {story_exp:.0%}, "
               f"solving {exposure.get('solving', 0):.0%}"
               + (f"; TOO LOW: {', '.join(skill_bad)}" if skill_bad else " — all learnable")))

# B6. story-source sanity: the fable scrape is the one source that can silently come back EMPTY (a
# flaky gutendex call) or CONTAMINATED (tables of contents parsed as 'stories' — run #1 memorized
# "Lion and the Bull, The. La Fontaine (French)" and recited it when asked for a story).
_s2 = "data/stories2/chat.txt"
_s2_blocks = [b for b in open(_s2).read().split("\n\n") if b.strip()] if os.path.exists(_s2) else []
_idx = sum(1 for b in _s2_blocks if b.count(", The.") + b.count(", A.") >= 2 or b.count("(") >= 3
           and _re_attr.search(b)) if _s2_blocks else 0
okB6 = len(_s2_blocks) >= 500 and _idx / max(len(_s2_blocks), 1) < 0.01
report.append(("B6. story source", okB6,
               f"{len(_s2_blocks):,} fable blocks fetched, {_idx} index-like "
               f"(want >=500 blocks and ~0 index junk)"))

# B7. typesetting markup — the SAME disease as the memorized fable index, and the reason B6 exists.
# [Illustration] and _italic_ markers ride along inside good prose (14% of v5's blocks, 10%/22% of
# the raw fable scrape). The model cannot tell "junk that reliably appears in books" from "language",
# so it learns to emit it. Assert the assembled corpus is clean AFTER the scrub, not before.
_mk_bad, _mk_tot = markup_count(text)
okB7 = _mk_bad / max(_mk_tot, 1) < 0.005
report.append(("B7. typesetting markup", okB7,
               f"{_mk_bad:,}/{_mk_tot:,} blocks ({_mk_bad / max(_mk_tot, 1):.2%}) still carry "
               f"[bracket] or _underscore_ markup after the scrub (want <0.5%)"))


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
torch.manual_seed(1)               # deterministic proxy so C isn't flaky run-to-run
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

# (prompt phrase, its actual object) — object is NOT always the last word ("keeping the kitchen clean")
PROMPTS = [("picking up trash", "trash"), ("keeping the kitchen clean", "kitchen"),
           ("charging my phone", "phone"), ("collecting the leaves", "leaves"),
           ("reducing noise", "noise"), ("fixing the engine", "engine")]
model = model.to("cpu").eval()        # gen_ids builds CPU seed tensors; keep model+input on one device
ban = [unk] if unk is not None else None
clean = 0
proxy_lines = []
try:
    for g, obj in PROMPTS:
        seed = coder.encode(f"USER: design a reward for {g}\nBOT: ")
        out = model.gen_ids(list(seed), 60, temp=0.1, top_k=1, ban=ban)   # 60 tokens: reach the spec
        txt = coder.decode(out).split(EOS)[0].split("\n")[0].strip()
        spec = txt.split("reward:")[-1].strip() if "reward:" in txt else txt
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
