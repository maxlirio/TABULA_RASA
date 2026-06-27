#!/usr/bin/env python3
"""CORPUS V5 — bigger + larger vocabulary. More books (more unique words + more tokens to feed a
larger model), conversation-majority chat, the EXPANDED reasoned reward_design (x2), and MORE
storytelling (x2, with the convo dodge removed). Run prep_reward_design.py + prep_stories.py first.
Output: data/mixed_v5/chat.txt
"""
import os, random
HERE = os.path.dirname(os.path.abspath(__file__)); EOS = " ■"
def blocks(s):
    p=os.path.join(HERE,"data",s,"chat.txt")
    return [b.strip() for b in open(p,encoding="utf-8",errors="ignore").read().split("\n\n") if b.strip()] if os.path.exists(p) else []
def sample_to(r,it,b):
    r.shuffle(it); k=[]; sz=0
    for x in it:
        if sz>b: break
        k.append(x); sz+=len(x)+2
    return k
def main(books_budget=240_000_000, dlg_modern_budget=25_000_000, seed=14):
    r=random.Random(seed)
    conversation=(blocks("dialogue_clean")*3+blocks("convo")*6+blocks("dialogue_classical")*2
                  +blocks("chitchat")*2+sample_to(r,blocks("dialogue_modern"),dlg_modern_budget))
    frames=blocks("commands")+blocks("rules")+blocks("misc")+blocks("dictionary")+blocks("knowledge")
    base=sample_to(r,blocks("books")+blocks("modern")+blocks("classical"),books_budget)
    reward_design=blocks("reward_design")*2
    stories=blocks("stories")*2
    allb=[b.strip()+EOS for b in (conversation+frames+base+reward_design+stories) if b.strip()]
    r.shuffle(allb)
    d=os.path.join(HERE,"data","mixed_v5"); os.makedirs(d,exist_ok=True)
    open(os.path.join(d,"chat.txt"),"w",encoding="utf-8").write("\n\n".join(allb)+"\n")
    mb=lambda x:sum(len(b)+2 for b in x)/1e6
    print(f"[v5] conv={mb(conversation):.0f} frames={mb(frames):.0f} books={mb(base):.0f} "
          f"reward_design={mb(reward_design):.0f} stories={mb(stories):.0f} -> {os.path.getsize(os.path.join(d,'chat.txt'))/1e6:.0f}MB")
if __name__=="__main__": main()
