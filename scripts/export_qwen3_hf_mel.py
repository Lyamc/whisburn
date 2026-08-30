#!/usr/bin/env python3
"""Export HF Qwen3 mel stats for parity (local bundle)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from qwen_asr import Qwen3ASRModel

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    wav, _ = sf.read(ROOT / "samples" / "jfk.wav")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    model = Qwen3ASRModel.from_pretrained(
        str(ROOT / "models" / "qwen3-asr-0.6b"),
        dtype="float32",
        device_map="cpu",
    )
    fe = model.processor.feature_extractor
    m = fe(
        wav.astype(np.float32),
        sampling_rate=16000,
        return_tensors="np",
        padding=False,
        truncation=False,
    )
    mel = m["input_features"][0]  # [128, T]
    out = {
        "frames": int(mel.shape[1]),
        "mean": float(mel.mean()),
        "std": float(mel.std()),
        "first_mel0": mel[0, :8].tolist(),
        "first_mel63": mel[63, :8].tolist(),
    }
    out_path = ROOT / "temp_dump_qwen3-asr-0.6b" / "hf_mel_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())