"""A small character-level language model, built and trained FROM SCRATCH (random init,
no pretrained weights). This is the part that learns the *shape* of language from a large
body of text, so what it produces emerges from data rather than from rules anyone typed.

It's a tiny GPT-style transformer in PyTorch. On a few MB of books and a CPU it won't be
smart — it learns spelling, grammar-ish structure, and word habits, and babbles in that
style. But nothing it writes is hardcoded; it's all learned. That's the whole point.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, drop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True, dropout=drop)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                 nn.Linear(4 * n_embd, n_embd), nn.Dropout(drop))
        self.drop = nn.Dropout(drop)
        self.register_buffer("mask", torch.triu(torch.ones(block_size, block_size), 1).bool())

    def forward(self, x):
        t = x.size(1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=self.mask[:t, :t], need_weights=False)
        x = x + self.drop(a)
        x = x + self.mlp(self.ln2(x))
        return x


class CharLM(nn.Module):
    def __init__(self, vocab, n_embd=192, n_head=6, n_layer=4, block_size=128, drop=0.1):
        super().__init__()
        self.block_size = block_size
        self.vocab = vocab
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, drop)
                                     for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab)
        self.apply(self._init)
        self.head.weight = self.tok.weight       # weight tying (regularizes + fewer params)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        t = idx.size(1)
        x = self.drop(self.tok(idx) + self.pos(torch.arange(t, device=idx.device)))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab), targets.view(-1))
        return logits, loss

    def gen_ids(self, ids, n, temp=0.4, top_k=40, ban=None, stop=None,
                rep_penalty=1.0, no_repeat=0):
        """Backend-agnostic interface used by chat.py: take a list of token ids, return ONLY the
        newly generated ids. (The NumPy model in gm/lm_np.py exposes the same method.)
        `stop`: a token id (the learned EOS ■) that ends generation early — most replies are short,
        so this avoids grinding out the full `n` tokens only to throw most away."""
        dev = next(self.parameters()).device
        out = self.generate(torch.tensor([ids], device=dev), n, temp=temp, top_k=top_k,
                            ban=ban, stop=stop, rep_penalty=rep_penalty, no_repeat=no_repeat,
                            gen_from=len(ids))[0].tolist()
        return out[len(ids):]

    @torch.no_grad()
    def generate(self, idx, n, temp=0.8, top_k=40, ban=None, stop=None,
                 rep_penalty=1.0, no_repeat=0, gen_from=None):
        # rep_penalty>1 down-weights tokens already produced; no_repeat>0 forbids repeating any
        # n-gram of that size. Both act ONLY on tokens generated in THIS call (from gen_from), so the
        # prompt is never penalised. Off by default (1.0 / 0) -- probes and training are unchanged;
        # chat turns them on, which is what kills the "i'm sorry. i'm sorry. i'm sorry" degeneration.
        self.eval()
        start = idx.size(1) if gen_from is None else gen_from
        for _ in range(n):
            cond = idx[:, -self.block_size:]
            logits, _ = self(cond)
            logits = logits[:, -1, :] / max(temp, 1e-6)
            if ban:
                logits[:, ban] = -float("inf")     # never emit these tokens (e.g. <unk>)
            if (rep_penalty != 1.0 or no_repeat) and idx.size(1) > start:
                for b in range(idx.size(0)):
                    gen = idx[b, start:].tolist()
                    if rep_penalty != 1.0:
                        for t in set(gen):
                            lo = logits[b, t]
                            logits[b, t] = lo / rep_penalty if lo > 0 else lo * rep_penalty
                    if no_repeat and len(gen) >= no_repeat - 1:
                        prefix = tuple(gen[-(no_repeat - 1):]) if no_repeat > 1 else ()
                        seq = idx[b].tolist()
                        for i in range(len(seq) - no_repeat + 1):
                            if tuple(seq[i:i + no_repeat - 1]) == prefix:
                                logits[b, seq[i + no_repeat - 1]] = -float("inf")
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if stop is not None and nxt.numel() == 1 and nxt.item() == stop:
                break                              # hit the learned EOS -> stop early
        return idx


class Coder:
    """Maps characters <-> integers for the model."""
    mode = "char"

    def __init__(self, chars):
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos.get(int(i), "") for i in ids)


# A word/symbol tokenizer learned from your own corpus (still from-scratch). Predicting whole
# words instead of characters means the model can only ever emit REAL words — no letter-salad
# babble — and 128 tokens of context covers far more conversation than 128 characters.
import re as _re
from collections import Counter as _Counter

_TOK = _re.compile(r"\n|[+\-]?[A-Za-z][A-Za-z_]*(?:\([a-z]+\))?|[0-9]+|[^\sA-Za-z0-9]")
_NOSPACE = set(",.!?:;)%'")


def _detok(tokens):
    out = []
    for t in tokens:
        if t == "\n":
            out.append("\n")
        elif (not out or out[-1].endswith(("\n", "'", "(")) or t[:1] in _NOSPACE):
            out.append(t)
        else:
            out.append(" " + t)
    return "".join(out)


class WordCoder:
    """Maps word/symbol tokens <-> integers."""
    mode = "word"

    def __init__(self, tokens):
        self.tokens = tokens
        self.stoi = {t: i for i, t in enumerate(tokens)}
        self.itos = {i: t for i, t in enumerate(tokens)}
        self.unk = self.stoi.get("<unk>", 0)

    @classmethod
    def from_text(cls, text, min_freq=1):
        # count in CHUNKS: findall over a multi-GB corpus builds a ~400M-string list (~tens of GB)
        # and OOMs. Chunk on newlines so only a slice is materialized at a time (findall stays C-fast).
        counts = _Counter()
        n, i, step = len(text), 0, 50_000_000
        while i < n:
            j = text.find("\n", min(i + step, n)) if i + step < n else n
            if j == -1:
                j = n
            counts.update(_TOK.findall(text[i:j]))
            i = j
        vocab = ["<unk>", "\n"] + sorted(t for t, c in counts.items()
                                         if c >= min_freq and t != "\n")
        return cls(vocab)

    @property
    def chars(self):                # compat with code that reads .chars
        return self.tokens

    def encode(self, s):
        return [self.stoi.get(t, self.unk) for t in _TOK.findall(s)]

    def encode_array(self, text):
        """Memory-efficient encode of a huge corpus -> numpy int32 array, built in chunks (a Python
        list of 400M ints is ~3GB+; encoding chunk-by-chunk into numpy keeps peak memory low)."""
        import numpy as np
        parts, n, i, step = [], len(text), 0, 50_000_000
        while i < n:
            j = text.find("\n", min(i + step, n)) if i + step < n else n
            if j == -1:
                j = n
            g = self.stoi.get
            parts.append(np.fromiter((g(t, self.unk) for t in _TOK.findall(text[i:j])),
                                     dtype=np.int32))
            i = j
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)

    def decode(self, ids):
        return _detok([self.itos.get(int(i), "") for i in ids])


def save(model, coder, path):
    d = {"vocab": model.vocab, "block_size": model.block_size,
         "n_head": model.blocks[0].attn.num_heads, "state": model.state_dict()}
    if getattr(coder, "mode", "char") == "word":
        d["mode"], d["tokens"] = "word", coder.tokens
    else:
        d["mode"], d["chars"] = "char", coder.chars
    torch.save(d, path)


def load(path, map_location="cpu"):
    d = torch.load(path, map_location=map_location)
    coder = WordCoder(d["tokens"]) if d.get("mode") == "word" else Coder(d["chars"])
    # infer dims from the saved tensors
    st = d["state"]
    n_embd = st["tok.weight"].shape[1]
    n_layer = len({k.split(".")[1] for k in st if k.startswith("blocks.")})
    n_head = d.get("n_head")
    if not n_head:                       # older checkpoints: pick a head count that divides
        n_head = next(h for h in (8, 6, 4, 3, 2, 1) if n_embd % h == 0)
    model = CharLM(d["vocab"], n_embd=n_embd, n_layer=n_layer,
                   block_size=d["block_size"], n_head=n_head)
    model.load_state_dict(st)
    model.eval()
    return model, coder
