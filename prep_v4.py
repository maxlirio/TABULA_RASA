#!/usr/bin/env python3
"""CORPUS V4 = v3 (conversation-majority + light frames + book base) PLUS:
  - reward_design (model learns to DESIGN rewards by reasoning, x2 — the new core)
  - stories (long-reply data so it stops the one-line dodge)
Run prep_reward_design.py and prep_stories.py first. Output: data/mixed_v4/chat.txt
"""
import os, random
HERE = os.path.dirname(os.path.abspath(__file__))
EOS = " ■"

def blocks(sub):
    p = os.path.join(HERE, "data", sub, "chat.txt")
    if not os.path.exists(p): return []
    return [b.strip() for b in open(p, encoding="utf-8", errors="ignore").read().split("\n\n") if b.strip()]

def sample_to(r, items, budget):
    r.shuffle(items); kept=size=0; out=[]
    for b in items:
        if size > budget: break
        out.append(b); size += len(b)+2
    return out

def main(books_budget=90_000_000, dlg_modern_budget=22_000_000, seed=14):
    r = random.Random(seed)
    conversation = (blocks("dialogue_clean")*3 + blocks("convo")*6 + blocks("dialogue_classical")*2
                    + blocks("chitchat")*2 + sample_to(r, blocks("dialogue_modern"), dlg_modern_budget))
    frames = blocks("commands") + blocks("rules") + blocks("misc") + blocks("dictionary") + blocks("knowledge")
    base = sample_to(r, blocks("books") + blocks("modern") + blocks("classical"), books_budget)
    reward_design = blocks("reward_design") * 2          # the new core skill: reasoned reward DESIGN
    stories = blocks("stories")                          # long-reply storytelling
    allb = [b.strip()+EOS for b in (conversation+frames+base+reward_design+stories) if b.strip()]
    r.shuffle(allb)
    out_dir = os.path.join(HERE, "data", "mixed_v4"); os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "chat.txt")
    open(path,"w",encoding="utf-8").write("\n\n".join(allb)+"\n")
    def mb(x): return sum(len(b)+2 for b in x)/1e6
    print(f"[v4] conversation={mb(conversation):.0f} frames={mb(frames):.0f} books={mb(base):.0f} "
          f"reward_design={mb(reward_design):.0f} stories={mb(stories):.0f} MB -> {os.path.getsize(path)/1e6:.0f}MB")

if __name__ == "__main__":
    main()
