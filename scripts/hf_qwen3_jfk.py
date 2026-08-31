#!/usr/bin/env python3
"""Compare HF Qwen3-ASR output on samples/jfk.wav (requires: pip install qwen-asr)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JFK = ROOT / "samples" / "jfk.wav"


def main() -> int:
    if not JFK.exists():
        print(f"missing {JFK}", file=sys.stderr)
        return 1

    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError:
        print("install: pip install qwen-asr", file=sys.stderr)
        return 1

    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B",
        dtype="auto",
        device_map="auto",
    )
    results = model.transcribe(str(JFK), language=None)
    out = {
        "language": results[0].language,
        "text": results[0].text,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())