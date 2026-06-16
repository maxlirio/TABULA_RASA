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


def main(subdir="modern", ckpt="apollo.pt", name="Apollo", iters=2500, threads=None,
         n_embd=192, n_layer=4, block=128, n_head=6, batch=None):
    torch.manual_seed(1)
    torch.set_num_threads(threads or os.cpu_count() or 4)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    text = load_corpus(subdir)
    coder = WordCoder.from_text(text, min_freq=16)  # prune rare words (big corpus -> ~62k vocab)
    data = torch.tensor(coder.encode(text), dtype=torch.long)
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
                   n_layer=n_layer, block_size=block, drop=0.2).to(device)
    print(f"[{name}] model: {sum(p.numel() for p in model.parameters()):,} params (random init)")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def get_batch(d):
        ix = torch.randint(len(d) - block - 1, (batch,))
        x = torch.stack([d[i:i + block] for i in ix])
        y = torch.stack([d[i + 1:i + block + 1] for i in ix])
        return x.to(device), y.to(device)

    # --- generalization probe: can it follow rules over HELD-OUT (never-trained) words? ---
    hp = os.path.join(os.path.dirname(__file__), "rules_holdout.json")
    probe_pairs, unk_id = [], coder.stoi.get("<unk>")
    if os.path.exists(hp):
        import json as _json
        hw = [w for w in _json.load(open(hp)).get("holdout", []) if w in coder.stoi]
        rp = random.Random(0)
        while len(probe_pairs) < 20 and len(hw) >= 2:
            a, b = rp.choice(hw), rp.choice(hw)
            if a != b:
                probe_pairs.append((a, b))

    def rule_gen():
        if not probe_pairs:
            return 0.0
        ban = [unk_id] if unk_id is not None else None
        hits = 0
        for x, y in probe_pairs:
            ids = coder.encode(f"RULE: say {x} instead of {y}\nUSER: {y}\nBOT: ")
            o = model.generate(torch.tensor([ids]).to(device), 3, temp=0.1,
                               top_k=1, ban=ban)[0].tolist()
            g = coder.decode(o[len(ids):]).split("\n")[0].strip().split()
            hits += (g[:1] == [x])
        return hits / len(probe_pairs)

    t0 = time.time()
    best = (-1.0, float("inf"))               # (generalization, -val): maximize gen, then min val
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
                seed = torch.tensor([[coder.stoi.get("\n", 0)]]).to(device)
                sample = coder.decode(model.generate(seed, 40, temp=0.8)[0][1:])
            star = ""
            if (gen, -vl) > best:             # keep the checkpoint that GENERALIZES best
                best = (gen, -vl)
                model.to("cpu")
                save(model, coder, stage_path)
                model.to(device)
                star = "  <- saved (best gen)"
            print(f"\n--- [{name}] iter {it}  train {loss.item():.3f}  val {vl:.3f}  "
                  f"gen {gen:.2f}  {time.time() - t0:.0f}s{star} ---")
            print(sample.replace("\n", " "))
            model.train()
    os.replace(stage_path, final_path)        # swap the best model in atomically
    print(f"\nDONE. saved {ckpt} ({name})")


if __name__ == "__main__":
    # args: subdir ckpt name iters [threads] [n_embd] [n_layer] [block] [n_head]
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
         int(a[9]) if len(a) > 9 else None)
