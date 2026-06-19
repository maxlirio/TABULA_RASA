"""PyTorch-FREE inference for the from-scratch brain — a NumPy reimplementation of the exact
same transformer (gm/lm.py), so it can run in a tiny portable bundle with no PyTorch install.
Weights are exported once (export_np.py) to a .npz; this loads them and runs generation with the
same interface chat.py expects (a `coder` + `model.gen_ids(...)`)."""
import json
import os
import re
from collections import Counter as _Counter

import numpy as np

np.seterr(all="ignore")          # inference-only: ignore harmless fp under/overflow in softmax/erf

# ---- tokenizer (identical rules to gm/lm.py WordCoder, but torch-free) ----
_TOK = re.compile(r"\n|[+\-]?[A-Za-z][A-Za-z_]*(?:\([a-z]+\))?|[0-9]+|[^\sA-Za-z0-9]")
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
    mode = "word"

    def __init__(self, tokens):
        self.tokens = tokens
        self.stoi = {t: i for i, t in enumerate(tokens)}
        self.itos = {i: t for i, t in enumerate(tokens)}
        self.unk = self.stoi.get("<unk>", 0)

    @property
    def chars(self):
        return self.tokens

    def encode(self, s):
        return [self.stoi.get(t, self.unk) for t in _TOK.findall(s)]

    def decode(self, ids):
        return _detok([self.itos.get(int(i), "") for i in ids])


# ---- math ----
_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911)


def _erf(x):                                   # Abramowitz-Stegun 7.1.26 (~1e-7), matches torch GELU
    s = np.sign(x)
    t = 1.0 / (1.0 + _A[5] * np.abs(x))
    y = 1.0 - (((((_A[4] * t + _A[3]) * t) + _A[2]) * t + _A[1]) * t + _A[0]) * t * np.exp(-x * x)
    return s * y


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class NpModel:
    def __init__(self, w, n_head, n_layer, block_size):
        self.w = w
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.E = w["tok.weight"].shape[1]
        self.hd = self.E // n_head

    def _forward(self, ids):
        w = self.w
        T = len(ids)
        x = w["tok.weight"][ids] + w["pos.weight"][:T]            # (T, E)
        cmask = np.triu(np.full((T, T), -1e9, dtype=np.float32), 1)
        for i in range(self.n_layer):
            p = f"blocks.{i}."
            h = _ln(x, w[p + "ln1.weight"], w[p + "ln1.bias"])
            qkv = h @ w[p + "attn.in_proj_weight"].T + w[p + "attn.in_proj_bias"]   # (T, 3E)
            q, k, v = qkv[:, :self.E], qkv[:, self.E:2 * self.E], qkv[:, 2 * self.E:]
            q = q.reshape(T, self.n_head, self.hd).transpose(1, 0, 2)  # (H, T, hd)
            k = k.reshape(T, self.n_head, self.hd).transpose(1, 0, 2)
            v = v.reshape(T, self.n_head, self.hd).transpose(1, 0, 2)
            att = (q @ k.transpose(0, 2, 1)) / np.sqrt(self.hd) + cmask    # (H, T, T)
            att = _softmax(att, axis=-1)
            o = (att @ v).transpose(1, 0, 2).reshape(T, self.E)           # (T, E)
            o = o @ w[p + "attn.out_proj.weight"].T + w[p + "attn.out_proj.bias"]
            x = x + o
            h = _ln(x, w[p + "ln2.weight"], w[p + "ln2.bias"])
            h = _gelu(h @ w[p + "mlp.0.weight"].T + w[p + "mlp.0.bias"])
            h = h @ w[p + "mlp.2.weight"].T + w[p + "mlp.2.bias"]
            x = x + h
        return _ln(x, w["ln.weight"], w["ln.bias"])                  # (T, E)

    def logits_last(self, ids):
        h = self._forward(ids)[-1]                                   # (E,)
        return self.w["tok.weight"] @ h + self.w["head.bias"]        # (V,) tied head

    def gen_ids(self, ids, n, temp=0.4, top_k=40, ban=None):
        ids = list(ids)
        new, rng = [], np.random
        for _ in range(n):
            z = self.logits_last(ids[-self.block_size:]).astype(np.float64)
            z = z / max(temp, 1e-6)
            if ban:
                z[ban] = -np.inf
            if top_k and top_k < z.size:
                kth = np.partition(z, -top_k)[-top_k]
                z[z < kth] = -np.inf
            if top_k == 1:
                nxt = int(np.argmax(z))
            else:
                p = _softmax(z)
                nxt = int(rng.choice(z.size, p=p))
            ids.append(nxt)
            new.append(nxt)
        return new


def load_np(npz_path, json_path=None):
    json_path = json_path or os.path.splitext(npz_path)[0] + ".json"
    meta = json.load(open(json_path))
    npz = np.load(npz_path)
    w = {k: npz[k].astype(np.float32) for k in npz.files}            # f16 on disk -> f32 compute
    model = NpModel(w, meta["n_head"], meta["n_layer"], meta["block_size"])
    return model, WordCoder(meta["tokens"])
