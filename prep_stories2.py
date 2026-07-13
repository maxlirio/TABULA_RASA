#!/usr/bin/env python3
"""DIVERSE story data — so the model learns MANY story ideas to recombine, not one template frame.

The template generator (prep_dialogue2) taught the model a single skeleton with swappable names, so
every story came out the same shape. The fix isn't a cleverer template — it's genuinely diverse,
COMPLETE, SHORT real stories. Fables and folk tales are ideal: hundreds of tiny self-contained
stories, each a different situation, characters, and moral. Trained on these, the model has a real
palette to blend into varied (and sometimes novel-feeling) output instead of one frame.

Honest ceiling: a 123M from-scratch model RECOMBINES what it saw; it won't truly invent. But with
diverse input its stories vary a lot instead of being the same tale with a new name. Some archaic
fable flavor will leak (we de-SHOUT the openings and normalize quotes, but don't fully modernize).

Sources: public-domain fable/short-tale collections on Project Gutenberg (training data, not
pretrained weights — consistent with the from-scratch rule). Output: data/stories2/chat.txt
"""
import json
import os
import random
import re
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
# seed IDs of fable / short-tale collections (many small complete stories each)
SOURCE_IDS = [21, 11339, 19994, 28, 49010, 2591]   # Aesop x4, more fables, Grimm (short ones kept)


def _discover(target=40):
    """Auto-find more fable / folk-tale collections via the Gutendex API, so the corpus has more
    UNIQUE stories (diversity is what teaches weaving; repetition just memorizes). Robust to failure."""
    ids = []
    for subj in ("fables", "folk tales", "fairy tales"):
        for page in (1, 2):
            try:
                url = f"https://gutendex.com/books?languages=en&topic={subj.replace(' ','%20')}&page={page}"
                j = json.loads(urllib.request.urlopen(
                    urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20).read())
                for b in j.get("results", []):
                    if b.get("id"):
                        ids.append(b["id"])
            except Exception:
                break
            if len(ids) >= target:
                break
    return ids[:target]
REQ = ["tell me a story", "tell me a story", "can you tell me a story", "tell me a little story",
       "i'd love to hear a story", "tell me a bedtime story", "spin me a tale", "tell me a tale",
       "read me a story", "make up a story", "story time!", "tell me a short story"]


def _fetch(gid):
    for url in (f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
                f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return urllib.request.urlopen(req, timeout=25).read().decode("utf-8", "ignore")
        except Exception:
            continue
    return ""


def _clean(s):
    s = (s.replace("\r", "").replace("“", '"').replace("”", '"').replace("‘", "'")
         .replace("’", "'").replace("—", "-").replace("–", "-"))
    s = re.sub(r"\b[A-Z][A-Z']{1,}\b", lambda m: m.group(0).lower(), s)   # de-SHOUT the openings
    s = re.sub(r"\s+", " ", s).strip()
    # capitalize sentence starts (the de-SHOUT lowercased them)
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)


def _is_title(ln):
    s = ln.strip()
    if not (3 < len(s) < 60) or s.endswith((".", "!", "?", '"', ",", ";", ":")):
        return False
    words = [w for w in s.split() if w.isalpha()]
    return len(words) >= 2 and sum(w[0].isupper() for w in words) / len(words) >= 0.7


_ATTR = re.compile(r"\([A-Z][a-z]+\)")   # "(French)" / "(Greek)" source-attribution tags


def _is_index(body):
    """Index / table-of-contents runs slip past the title splitter as a fake 'story' (the 280M run
    memorized 'Lion and the Bull, The. La Fontaine (French)' verbatim). Real prose never repeats
    inverted-title commas or attribution tags."""
    inverted = body.count(", The.") + body.count(", A.") + body.count(", An.")
    return inverted >= 2 or len(_ATTR.findall(body)) >= 2


def _split_stories(txt):
    m = re.search(r"\*\*\* ?START OF.*?\*\*\*", txt, re.S)
    if m:
        txt = txt[m.end():]
    m = re.search(r"\*\*\* ?END OF", txt)
    if m:
        txt = txt[:m.start()]
    out, cur, have_title = [], [], False
    for ln in txt.replace("\r", "").split("\n"):
        if _is_title(ln):
            if have_title and cur:
                body = _clean(" ".join(cur))
                # SHORT + complete: fits the 256-token window and has a real arc
                if 160 < len(body) < 1100 and body.count(".") >= 2 and not _is_index(body):
                    out.append(body)
            cur, have_title = [], True
        else:
            cur.append(ln)
    return out


def main(seed=53):
    r = random.Random(seed)
    ids = list(dict.fromkeys(SOURCE_IDS + _discover()))     # seed IDs + auto-discovered collections
    stories = []
    for gid in ids:
        txt = _fetch(gid)
        got = _split_stories(txt) if txt else []
        if got:
            stories += got
            print(f"[stories2] source {gid}: {len(got)} stories", flush=True)
    # dedupe by a prefix key (some collections overlap)
    seen, uniq = set(), []
    for s in stories:
        k = s[:60].lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    if not uniq:
        print("[stories2] no stories fetched (offline?) — skipping"); return
    out = [f"USER: {r.choice(REQ)}\nBOT: {s}" for s in uniq for _ in range(3)]  # a few requests each
    r.shuffle(out)
    out_dir = os.path.join(HERE, "data", "stories2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chat.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(out) + "\n")
    print(f"[stories2] {len(uniq):,} UNIQUE stories -> {len(out):,} training pairs "
          f"-> data/stories2/chat.txt ({os.path.getsize(os.path.join(out_dir,'chat.txt')):,} bytes)")


if __name__ == "__main__":
    main()
