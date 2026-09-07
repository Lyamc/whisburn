#!/usr/bin/env python3
"""Pack Silero VAD ONNX/JIT weights into canonical safetensors."""
import json
import sys
from pathlib import Path

import numpy as np


def _save(tensors: dict, out: Path):
    try:
        from safetensors.numpy import save_file
    except ImportError:
        sys.exit("pip install safetensors numpy")
    save_file({k: np.ascontiguousarray(v.astype(np.float32)) for k, v in tensors.items()}, str(out))


def collect_tensors(graph, raw: dict):
    from onnx import numpy_helper

    for init in graph.initializer:
        raw[init.name] = numpy_helper.to_array(init)
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.dims:
                    raw[node.name or node.output[0]] = numpy_helper.to_array(attr.t)
        if node.op_type == "If":
            # Prefer the 16 kHz then_branch (sr == 16000).
            then_g = None
            else_g = None
            for attr in node.attribute:
                if attr.name == "then_branch" and attr.HasField("g"):
                    then_g = attr.g
                elif attr.name == "else_branch" and attr.HasField("g"):
                    else_g = attr.g
            if then_g is not None:
                collect_tensors(then_g, raw)
            elif else_g is not None:
                collect_tensors(else_g, raw)


def from_onnx(path: Path) -> dict:
    import onnx

    model = onnx.load(str(path))
    raw = {}
    collect_tensors(model.graph, raw)
    if not raw:
        raise SystemExit(f"no tensors in {path}")
    return map_silero(raw)


def map_silero(raw: dict) -> dict:
    by_shape = {}
    for name, arr in raw.items():
        by_shape.setdefault(tuple(arr.shape), []).append((name, arr))

    def take(*shapes):
        for s in shapes:
            items = by_shape.get(s)
            if items:
                return items.pop(0)[1]
        raise KeyError(f"no tensor with shape in {shapes}")

    stft = take((258, 1, 256), (258, 256, 1))
    if stft.shape == (258, 256, 1):
        stft = np.transpose(stft, (0, 2, 1))
    enc = [
        take((128, 129, 3), (128, 3, 129)),
        take((64, 128, 3), (64, 3, 128)),
        take((64, 64, 3), (64, 3, 64)),
        take((128, 64, 3), (128, 3, 64)),
    ]
    for i, w in enumerate(enc):
        if w.ndim == 3 and w.shape[1] == 3:
            enc[i] = np.transpose(w, (0, 2, 1))
    enc_b = [take((128,)), take((64,)), take((64,)), take((128,))]
    w_ih = take((512, 128))
    w_hh = take((512, 128))
    b_ih = take((512,))
    try:
        b_hh = take((512,))
    except KeyError:
        b_hh = np.zeros(512, np.float32)
    dec_w = take((1, 128, 1), (1, 1, 128), (1, 128))
    dec_b = take((1,))
    out = {
        "stft.conv.weight": stft,
        "decoder.lstm.weight_ih_l0": w_ih,
        "decoder.lstm.weight_hh_l0": w_hh,
        "decoder.lstm.bias_ih_l0": b_ih,
        "decoder.lstm.bias_hh_l0": b_hh,
        "decoder.output.weight": dec_w.reshape(-1),
        "decoder.output.bias": dec_b,
    }
    for i, (w, b) in enumerate(zip(enc, enc_b)):
        out[f"encoder.{i}.conv.weight"] = w
        out[f"encoder.{i}.conv.bias"] = b
    return out


def main():
    if len(sys.argv) < 3:
        print("usage: pack_silero_vad.py <silero_vad.onnx> <out_dir>")
        sys.exit(2)
    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = from_onnx(src)
    _save(tensors, out_dir / "model.safetensors")
    (out_dir / "vad_runtime.json").write_text(
        json.dumps(
            {
                "kind": "silero-v5",
                "sample_rate": 16000,
                "window": 512,
                "context": 64,
                "inference_status": "ready",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / ".burn_version").write_text("0.21.0-vad\n", encoding="utf-8")
    print("wrote", out_dir / "model.safetensors", "tensors", len(tensors))


if __name__ == "__main__":
    main()
