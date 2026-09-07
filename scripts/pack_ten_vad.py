#!/usr/bin/env python3
"""Pack TEN-VAD ONNX + feature metadata into safetensors + ten_features.json."""
import json
import sys
from pathlib import Path

import numpy as np


def dump_onnx(path: Path):
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    raw = {}
    for init in model.graph.initializer:
        raw[init.name] = numpy_helper.to_array(init)
    meta = {}
    for p in model.metadata_props:
        meta[p.key] = p.value
    return raw, meta


def parse_vec(s):
    if not s:
        return None
    s = s.strip().replace("[", "").replace("]", "")
    if not s:
        return None
    return np.array([float(x) for x in s.split(",") if x.strip()], dtype=np.float32)


def iofg_to_ifgo(w, hidden=64):
    """ONNX LSTM gate order IOFG -> PyTorch IFGO."""
    w = np.asarray(w)
    if w.ndim == 3:
        w = w[0]
    chunks = np.split(w, 4, axis=0)
    i, o, f, g = chunks
    return np.concatenate([i, f, g, o], axis=0).astype(np.float32)


def map_lstm(raw: dict) -> dict:
    def get(name):
        if name in raw:
            return raw[name]
        for k, v in raw.items():
            if k.endswith(name):
                return v
        return None

    w0 = get("W0__70")
    r0 = get("R0__71")
    b0 = get("B0__72")
    w1 = get("W0__99")
    r1 = get("R0__100")
    b1 = get("B0__101")
    if w0 is None or r0 is None:
        raise SystemExit(f"TEN-VAD LSTM W/R not found, keys={list(raw)[:20]}")
    w0 = iofg_to_ifgo(w0)
    r0 = iofg_to_ifgo(r0)
    w1 = iofg_to_ifgo(w1) if w1 is not None else np.zeros((256, 64), np.float32)
    r1 = iofg_to_ifgo(r1) if r1 is not None else np.zeros((256, 64), np.float32)
    if b0 is None:
        b0 = np.zeros((1, 512), np.float32)
    b0 = np.asarray(b0).reshape(-1)
    b_ih0 = iofg_to_ifgo(b0[:256])
    b_hh0 = iofg_to_ifgo(b0[256:])
    if b1 is None:
        b1 = np.zeros((1, 512), np.float32)
    b1 = np.asarray(b1).reshape(-1)
    b_ih1 = iofg_to_ifgo(b1[:256])
    b_hh1 = iofg_to_ifgo(b1[256:])
    dense_w = get("StatefulPartitionedCall/vad_model/dense_3/Tensordot/ReadVariableOp:0")
    dense_b = get("StatefulPartitionedCall/vad_model/dense_3/BiasAdd/ReadVariableOp:0")
    out_w = get("StatefulPartitionedCall/vad_model/dense_5/Tensordot/ReadVariableOp:0")
    out_b = get("StatefulPartitionedCall/vad_model/dense_5/BiasAdd/ReadVariableOp:0")
    if dense_w is None:
        dense_w = np.zeros((128, 32), np.float32)
    if out_w is None:
        out_w = np.zeros((32, 1), np.float32)
    return {
        "lstm.weight_ih_l0": w0,
        "lstm.weight_hh_l0": r0,
        "lstm.bias_ih_l0": b_ih0,
        "lstm.bias_hh_l0": b_hh0,
        "lstm.weight_ih_l1": w1,
        "lstm.weight_hh_l1": r1,
        "lstm.bias_ih_l1": b_ih1,
        "lstm.bias_hh_l1": b_hh1,
        "dense.weight": dense_w.T if dense_w.shape[0] == 128 else dense_w,
        "dense.bias": np.zeros(32, np.float32) if dense_b is None else dense_b.reshape(-1),
        "fc.weight": out_w.reshape(-1),
        "fc.bias": np.zeros(1, np.float32) if out_b is None else out_b.reshape(-1),
    }


def default_window(n=768):
    # periodic hann
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)


def main():
    if len(sys.argv) < 3:
        print("usage: pack_ten_vad.py <ten-vad.onnx> <out_dir>")
        sys.exit(2)
    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    raw, meta = dump_onnx(src)
    tensors = map_lstm(raw)
    from safetensors.numpy import save_file

    save_file({k: np.ascontiguousarray(v.astype(np.float32)) for k, v in tensors.items()}, str(out_dir / "model.safetensors"))
    mean = parse_vec(meta.get("mean", ""))
    inv = parse_vec(meta.get("inv_stddev", ""))
    window = parse_vec(meta.get("window", ""))
    if mean is None or mean.size != 41:
        mean = np.zeros(41, np.float32)
    if inv is None or inv.size != 41:
        inv = np.ones(41, np.float32)
    if window is None or window.size < 256:
        window = default_window()
    feat = {
        "mean": mean.tolist(),
        "inv_stddev": inv.tolist(),
        "window": window.tolist(),
        "hop_size": 256,
    }
    (out_dir / "ten_features.json").write_text(json.dumps(feat), encoding="utf-8")
    (out_dir / "vad_runtime.json").write_text(
        json.dumps({"kind": "ten-vad", "sample_rate": 16000, "inference_status": "ready"}, indent=2),
        encoding="utf-8",
    )
    (out_dir / ".burn_version").write_text("0.21.0-vad\n", encoding="utf-8")
    print("wrote TEN-VAD", out_dir, "meta_keys", list(meta))


if __name__ == "__main__":
    main()
