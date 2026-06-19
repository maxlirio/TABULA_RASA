#!/usr/bin/env python3
"""Export a trained apollo.pt (PyTorch) to a PyTorch-FREE bundle: apollo.npz (weights, float16)
+ apollo.json (tokens + config). gm/lm_np.py runs these with no torch installed."""
import json
import os
import sys

import numpy as np
import torch

from gm.lm import load

HERE = os.path.dirname(os.path.abspath(__file__))


def main(ckpt="apollo.pt", out="apollo"):
    model, coder = load(os.path.join(HERE, ckpt))
    sd = model.state_dict()
    arrays = {k: v.detach().cpu().numpy().astype(np.float16)
              for k, v in sd.items() if not k.endswith(".mask")}   # mask recomputed at runtime
    np.savez(os.path.join(HERE, out + ".npz"), **arrays)
    meta = {"tokens": list(coder.tokens), "n_head": model.blocks[0].attn.num_heads,
            "n_layer": len(model.blocks), "block_size": model.block_size, "vocab": model.vocab}
    json.dump(meta, open(os.path.join(HERE, out + ".json"), "w"))
    npz_mb = os.path.getsize(os.path.join(HERE, out + ".npz")) / 1e6
    print(f"[export] {out}.npz ({npz_mb:.0f}MB, float16) + {out}.json "
          f"({len(coder.tokens):,} tokens, {len(model.blocks)} layers)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "apollo.pt")
