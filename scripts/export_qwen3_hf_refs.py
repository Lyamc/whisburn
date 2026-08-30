#!/usr/bin/env python3
"""Export HF Qwen3 audio encoder outputs for Rust parity tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from qwen_asr import Qwen3ASRModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "temp_dump_qwen3-asr-0.6b"


def main() -> int:
    wav, _ = sf.read(ROOT / "samples" / "jfk.wav")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    model = Qwen3ASRModel.from_pretrained(
        str(ROOT / "models" / "qwen3-asr-0.6b"),
        dtype="float32",
        device_map="cpu",
    )
    proc = model.processor
    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=[text], audio=[wav.astype(np.float32)], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    import torch

    with torch.no_grad():
        af = model.model.thinker.get_audio_features(
            inputs["input_features"],
            feature_attention_mask=inputs["feature_attention_mask"],
        )
        gen = model.model.generate(**inputs, max_new_tokens=32, do_sample=False)
    new_ids = gen.sequences[0, inputs["input_ids"].shape[1] :].tolist()

    OUT.mkdir(parents=True, exist_ok=True)
    af_np = af.float().cpu().numpy()
    af_np.astype(np.float32).tofile(OUT / "hf_audio_features.bin")
    meta = {
        "mel_frames": int(inputs["input_features"].shape[-1]),
        "audio_tokens": int(af_np.shape[0]),
        "hidden": int(af_np.shape[1]),
        "af_mean": float(af_np.mean()),
        "af_std": float(af_np.std()),
        "af_first8": af_np[0, :8].tolist(),
        "new_token_ids": new_ids,
    }
    (OUT / "hf_refs.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())