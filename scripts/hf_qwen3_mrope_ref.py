#!/usr/bin/env python3
"""Export HF Qwen3-ASR MRoPE cos/sin for parity with Burn thinker."""
from __future__ import annotations

import json
import math
import sys


def hf_mrope_cos_sin(seq: int, head_dim: int, theta: float, mrope_section: list[int]) -> tuple[list[float], list[float]]:
    half = head_dim // 2
    inv_freq = [1.0 / (theta ** (i / head_dim)) for i in range(0, head_dim, 2)]

    def freqs_for_pos(pos: int) -> list[float]:
        freqs_t = [pos * f for f in inv_freq]
        freqs_h = freqs_t.copy()
        freqs_w = freqs_t.copy()
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            for idx in range(offset, length, 3):
                freqs_t[idx] = freqs_h[idx] if dim == 1 else freqs_w[idx]
        emb = freqs_t + freqs_t
        return [math.cos(x) for x in emb], [math.sin(x) for x in emb]

    cos_all: list[float] = []
    sin_all: list[float] = []
    for pos in range(seq):
        c, s = freqs_for_pos(pos)
        cos_all.extend(c)
        sin_all.extend(s)
    return cos_all, sin_all


def main() -> int:
    seq = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    head_dim = 128
    theta = 1_000_000.0
    section = [24, 20, 20]
    cos, sin = hf_mrope_cos_sin(seq, head_dim, theta, section)
    out = {
        "seq": seq,
        "head_dim": head_dim,
        "theta": theta,
        "mrope_section": section,
        "cos": cos[: head_dim * 2],
        "sin": sin[: head_dim * 2],
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())