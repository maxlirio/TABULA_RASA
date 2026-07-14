#!/usr/bin/env python3
"""Train the from-scratch character LM on the downloaded books. Checkpoints to lm.pt and
prints a sample every so often so you can watch it learn — random noise at first, then
gradually English-shaped text. Nothing it writes is hardcoded; it's learned from the data.
"""
import glob
import os
import random
import re
import sys
import time

import torch

from gm.lm import CharLM, WordCoder, save


def load_corpus(subdir="."):
    parts = []
    base = os.path.join(os.path.dirname(__file__), "data", subdir)
    files = sorted(glob.glob(os.path.join(base, "pg*.txt")))
    chat = os.path.join(base, "chat.txt")          # turn-structured dialogue corpus
    if os.path.exists(chat):
        return open(chat, encoding="utf-8", errors="ignore").read()
    for f in files:
        s = open(f, encoding="utf-8", errors="ignore").read()
        m = re.search(r"\*\*\* ?START OF.*?\*\*\*", s, re.S)
        if m:
            s = s[m.end():]
        m = re.search(r"\*\*\* ?END OF", s)
        if m:
            s = s[:m.start()]
        # normalise smart quotes/dashes to shrink the vocabulary
        s = (s.replace("“", '"').replace("”", '"').replace("‘", "'")
             .replace("’", "'").replace("—", "-").replace("–", "-")
             .replace("\r", ""))
        parts.append(s)
    return "\n".join(parts)


def warm_start(model, coder, warm_path, name):
    """Continue training ON TOP of an existing checkpoint instead of from scratch.
    Copies every architecture-matching tensor (the transformer blocks, positions, norms)
    wholesale, and transplants the token-embedding ROWS by matching token strings, so all
    prior learning is preserved; only genuinely-new vocab words start randomly. Returns the
    number of embedding rows carried over (0 means it fell back to random init)."""
    import os as _os
    if not warm_path or not _os.path.exists(warm_path):
        print(f"[{name}] warm-start: no checkpoint at {warm_path!r} - training from scratch")
        return 0
    old = torch.load(warm_path, map_location="cpu")
    ost = old["state"]
    n_layer_old = len({k.split(".")[1] for k in ost if k.startswith("blocks.")})
    if (ost["tok.weight"].shape[1] != model.tok.weight.shape[1]      # n_embd
            or old.get("block_size") != model.block_size
            or n_layer_old != len(model.blocks)
            or old.get("mode") != "word"):
        print(f"[{name}] warm-start: architecture mismatch - training from scratch")
        return 0
    msd = model.state_dict()
    src, copied = {}, 0
    for k, v in ost.items():                       # carry over everything vocab-independent
        if k in msd and msd[k].shape == v.shape and k not in ("tok.weight", "head.weight"):
            src[k] = v
            copied += 1
    new_emb = msd["tok.weight"].clone()            # transplant embedding rows by token string
    old_stoi = {t: i for i, t in enumerate(old["tokens"])}
    old_emb, moved = ost["tok.weight"], 0
    for t, i_new in coder.stoi.items():
        j = old_stoi.get(t)
        if j is not None:
            new_emb[i_new] = old_emb[j]
            moved += 1
    src["tok.weight"] = new_emb
    src["head.weight"] = new_emb                    # weights are tied
    model.load_state_dict(src, strict=False)
    model.head.weight = model.tok.weight            # keep the tie after loading
    print(f"[{name}] warm-start: copied {copied} tensors, transplanted {moved:,}/"
          f"{len(coder.stoi):,} embedding rows ({moved/len(coder.stoi):.0%} of new vocab)")
    return moved


def main(subdir="modern", ckpt="apollo.pt", name="Apollo", iters=2500, threads=None,
         n_embd=192, n_layer=4, block=128, n_head=6, batch=None, warm=None, min_freq=8):
    torch.manual_seed(1)
    torch.set_num_threads(threads or os.cpu_count() or 4)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    text = load_corpus(subdir)
    if min_freq == 0 and warm and os.path.exists(warm):
        # CONTINUATION run: reuse the warm checkpoint's exact vocabulary instead of rebuilding it
        # from this corpus. A skills-focused run #2 corpus no longer contains full Wikipedia, so
        # rebuilding would prune the rare words the base model already learned - and every pruned
        # row is learning thrown away. Identical tokens => 100% of the embedding rows transplant.
        coder = WordCoder(torch.load(warm, map_location="cpu")["tokens"])
        print(f"[{name}] vocab: reusing the warm checkpoint's {len(coder.tokens):,} tokens")
    else:
        coder = WordCoder.from_text(text, min_freq=min_freq)  # prune rare words (big corpus -> ~62k)
    data = torch.from_numpy(coder.encode_array(text)).long()  # chunked encode: avoids OOM on GB corpora
    n = int(0.95 * len(data))
    train, val = data[:n], data[n:]
    print(f"[{name}] corpus: {len(data):,} tokens, vocab {len(coder.tokens):,}, device {device}")

    # Train to a STAGING file so the live model (what talk.py loads) is never a half-baked
    # checkpoint. Back up the current good model, then swap the new one in only at the end.
    final_path = os.path.join(os.path.dirname(__file__), ckpt)
    stage_path = final_path + ".training"
    if os.path.exists(final_path):
        try:
            import shutil
            shutil.copyfile(final_path, final_path + ".bak")
        except OSError:
            pass

    # big batch on a real GPU keeps it busy (the whole point of the T4); small on Mac/CPU
    batch = batch or (64 if device == "cuda" else (32 if block <= 128 else 24))
    model = CharLM(len(coder.tokens), n_embd=n_embd, n_head=n_head,
                   n_layer=n_layer, block_size=block, drop=0.2)
    warmed = warm_start(model, coder, warm, name)   # continue on top of an existing model
    model = model.to(device)
    kind = "warm-started" if warmed else "random init"
    print(f"[{name}] model: {sum(p.numel() for p in model.parameters()):,} params ({kind})")
    # fine-tune more gently when continuing from a trained model; full LR for a fresh start
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4 if warmed else 3e-4)

    def get_batch(d):
        ix = torch.randint(len(d) - block - 1, (batch,))
        x = torch.stack([d[i:i + block] for i in ix])
        y = torch.stack([d[i + 1:i + block + 1] for i in ix])
        return x.to(device), y.to(device)

    # --- generalization probe: can it follow NOVEL in-context rules over HELD-OUT words? ---
    # We test several rule TYPES, not just substitution, so a high score means broad
    # instruction-following (the foundation for the tools/ReAct phase) rather than one
    # memorized pattern. Each probe is (seed, expected-first-words); checked greedily.
    hp = os.path.join(os.path.dirname(__file__), "rules_holdout.json")
    probes, unk_id = [], coder.stoi.get("<unk>")
    if os.path.exists(hp):
        import json as _json
        hw = [w for w in _json.load(open(hp)).get("holdout", []) if w in coder.stoi]
        rp = random.Random(0)
        for _ in range(7):
            if len(hw) >= 4:
                x, y, w, z = rp.sample(hw, 4)
                # 1) substitution: replace one word with another
                probes.append((f"RULE: say {x} instead of {y}\nUSER: {y}\nBOT: ", [x]))
                # 2) constant-answer override: ignore content, emit a fixed word
                probes.append((f"RULE: no matter what i say, just reply {x}\nUSER: {y}\nBOT: ", [x]))
                # 3) echo: copy the input back
                probes.append((f"RULE: repeat exactly what i say\nUSER: {x}\nBOT: ", [x]))
                # 4) conditional: pick the branch that matches the input
                probes.append((f"RULE: if i say {y} reply {x}, if i say {z} reply {w}\n"
                               f"USER: {z}\nBOT: ", [w]))

    def rule_gen():
        if not probes:
            return 0.0
        ban = [unk_id] if unk_id is not None else None
        hits = 0
        for seed, expect in probes:
            ids = coder.encode(seed)
            o = model.generate(torch.tensor([ids]).to(device), len(expect) + 2, temp=0.1,
                               top_k=1, ban=ban)[0].tolist()
            g = coder.decode(o[len(ids):]).split("\n")[0].strip().split()
            hits += (g[:len(expect)] == expect)
        return hits / len(probes)

    # --- tool-use probe: does it know WHEN to call a tool? Recall (emit CALL when a tool is the
    # only way to know) AND precision (do NOT emit CALL otherwise).
    #
    # REWARD PROMPTS ARE NEGATIVES. They used to be positives, from back when a lookup "plan tool"
    # answered them -- but 69e4375 RETIRED that tool and made reward design generative, and the
    # reward_design corpus (20% of exposure) now teaches the model to answer them DIRECTLY. The
    # probe kept demanding a CALL, so it scored the correct behaviour as failure and pinned itself
    # at 0.68 for an entire 40k-iter run -- half the checkpoint-selection metric, frozen. A probe
    # must track the corpus it grades; when a skill is retired, its probe is retired with it. ---
    tool_pos = ["what day is it", "what time is it", "what's the date", "what year is it",
                "what comes next: 3 6 9 12", "finish the sequence 2 4 8 16",
                "what is 47 times 6", "what is 15% of 200"]      # only a tool can know these
    tool_neg = ["i have three cats at home", "i went running this morning", "how are you today",
                "i love a sunny day", "tell me a story", "my favorite color is blue",
                "i'm feeling a bit tired", "we have 5 people coming over",
                # generative reward design: answer it, do NOT call a tool
                "design a reward system for winning the race", "make a reward for stacking the cups",
                "how should i reward feeding the dog", "give me a reward for painting the fence",
                "code a reward for winning the game", "write a reward function for walking"]

    def _calls(text):
        ids = coder.encode(f"USER: {text}\n")
        ban = [unk_id] if unk_id is not None else None
        o = model.generate(torch.tensor([ids]).to(device), 4, temp=0.1,
                           top_k=1, ban=ban)[0].tolist()
        return coder.decode(o[len(ids):]).strip().lower().startswith("call:")

    def tool_gen():
        hits = sum(_calls(q) for q in tool_pos)              # clock/calc/solve -> should CALL
        hits += sum(not _calls(q) for q in tool_neg)         # chat + reward design -> should NOT
        return hits / (len(tool_pos) + len(tool_neg))

    # --- story probe: run #2's HEADLINE goal, and nothing measured it -- so the saver would happily
    # discard a model that told perfect stories. Run #1's failure mode is degeneration: it produced
    # real narrative that looped ("Venus was very gracious about it, and changed her into a woman"
    # x3) and never reached an ending. Distinct-4 catches exactly that: a looping story reuses its
    # 4-grams, a healthy one barely repeats. Cheap, continuous, and it fails the thing we saw. ---
    story_reqs = ["tell me a story", "tell me a bedtime story", "spin me a tale", "tell me a tale"]

    def story_gen():
        scores = []
        for q in story_reqs:
            ids = coder.encode(f"USER: {q}\nBOT: ")
            # temp 0.4 / top_k 40 = EXACTLY what gm/chat.py decodes with. Degeneration is a property
            # of the decoding regime, not the weights: this model scores 0.99 at temp 0.8 (probe sees
            # nothing), 0.89 at the shell's settings, and 0.67 greedy ("I was a great man, and I was
            # a great man"). Grade the regime the user actually experiences, or the probe saturates
            # and drives nothing -- which is how `tool` sat frozen for a whole run.
            o = model.generate(torch.tensor([ids]).to(device), 120, temp=0.4, top_k=40,
                               ban=[unk_id] if unk_id is not None else None)[0].tolist()
            w = coder.decode(o[len(ids):]).split("■")[0].split("USER:")[0].split()
            if len(w) < 24:                      # too short to be a story at all
                scores.append(0.0)
                continue
            g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
            scores.append(len(set(g)) / len(g))  # distinct-4: 1.0 = no repetition, low = looping
        return sum(scores) / len(scores)

    t0 = time.time()
    # (skills, -val): maximize the skills we are paying to teach, then minimize val loss.
    #
    # On a WARM-START run, iter 1 IS THE INPUT MODEL. Run #2 learned this the expensive way: run #1
    # already scored well, fine-tuning transiently disturbs the probes (normal), nothing in 40,000
    # iterations beat the incumbent, so the saver staged iter 1 and os.replace swapped it in --
    # 4.7 GPU-hours whose entire output was a copy of the input, with the trained model discarded
    # unsaved. So when warm-starting, iter 1 is a BASELINE to print and beat, never a candidate to
    # save. And the last model is now ALWAYS written, so a run can never again produce nothing.
    best = (-1.0, float("inf"))
    baseline = None
    last_path = final_path.replace(".pt", "_last.pt")
    amp = (device == "cuda")                  # fp16 mixed precision — big speedup on tensor-core GPUs (T4)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    for it in range(1, iters + 1):
        x, y = get_batch(train)
        with torch.autocast(device_type="cuda" if amp else "cpu",
                            dtype=torch.float16, enabled=amp):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if it == 1 or it % 200 == 0:
            model.eval()
            with torch.no_grad():
                vl = sum(model(*get_batch(val))[1].item() for _ in range(5)) / 5  # avg val
                gen = rule_gen()
                tool = tool_gen()
                story = story_gen()
                seed = torch.tensor([[coder.stoi.get("\n", 0)]]).to(device)
                sample = coder.decode(model.generate(seed, 40, temp=0.8)[0][1:])
            skills = gen + tool + story
            star = ""
            if it == 1 and warmed:
                # the incumbent: what the warm checkpoint already scores. Print it, don't save it.
                baseline = skills
                star = "  <- WARM BASELINE (not a candidate; the run must BEAT this)"
            elif (skills, -vl) > best:        # keep the checkpoint best at the SKILLS we want
                best = (skills, -vl)
                model.to("cpu")
                save(model, coder, stage_path)
                model.to(device)
                star = "  <- saved (best)"
            print(f"\n--- [{name}] iter {it}  train {loss.item():.3f}  val {vl:.3f}  "
                  f"gen {gen:.2f}  tool {tool:.2f}  story {story:.2f}  skills {skills:.2f}"
                  f"  {time.time() - t0:.0f}s{star} ---")
            print(sample.replace("\n", " "))
            model.train()

    # ALWAYS write the final model, whatever the probes said. A run that trains for hours and
    # writes nothing is the worst possible outcome, and it is what run #2 did.
    model.to("cpu")
    save(model, coder, last_path)
    model.to(device)
    if os.path.exists(stage_path):
        os.replace(stage_path, final_path)    # swap the best model in atomically
        print(f"\nDONE. best -> {ckpt}   last -> {os.path.basename(last_path)} ({name})")
    else:
        save(model, coder, final_path)        # nothing ever beat the baseline
        print(f"\nDONE. NO checkpoint beat the warm baseline ({baseline:.2f}); "
              f"wrote the FINAL model to {ckpt} anyway. last -> {os.path.basename(last_path)}")
    if baseline is not None:
        print(f"[{name}] warm baseline skills {baseline:.2f} -> best trained {best[0]:.2f} "
              f"({'IMPROVED' if best[0] > baseline else 'DID NOT IMPROVE'})")


if __name__ == "__main__":
    # args: subdir ckpt name iters [threads] [n_embd] [n_layer] [block] [n_head] [batch] [warm.pt]
    #       [min_freq]  (min_freq 0 + a warm checkpoint = reuse that checkpoint's vocabulary)
    a = sys.argv[1:]
    main(a[0] if len(a) > 0 else "modern",
         a[1] if len(a) > 1 else "apollo.pt",
         a[2] if len(a) > 2 else "Apollo",
         int(a[3]) if len(a) > 3 else 2500,
         int(a[4]) if len(a) > 4 else None,
         int(a[5]) if len(a) > 5 else 192,
         int(a[6]) if len(a) > 6 else 4,
         int(a[7]) if len(a) > 7 else 128,
         int(a[8]) if len(a) > 8 else 6,
         int(a[9]) if len(a) > 9 else None,
         a[10] if len(a) > 10 else None,
         int(a[11]) if len(a) > 11 else 8)
