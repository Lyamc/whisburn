#!/usr/bin/env python3
"""Extract a NVIDIA .nemo Parakeet TDT checkpoint into HF-style safetensors.

Inference stays in Burn: this only lifts NeMo/PyTorch tensors into the same
`model.safetensors` + `config.json` + `tokenizer.json` layout v3 already uses.
No ONNX Runtime.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import struct
import sys
import tarfile
import tempfile
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

NEMO_TO_HF = {
    "encoder.pre_encode.out": "encoder.subsampling.linear",
    "encoder.pre_encode.conv": "encoder.subsampling.layers",
    "decoder.prediction.embed": "decoder.embedding",
    "decoder.prediction.dec_rnn.lstm": "decoder.lstm",
    "joint.enc": "encoder_projector",
    "joint.pred": "decoder.decoder_projector",
    "joint.joint_net.2": "joint.head",
}

ATTN_RENAME = {
    "linear_q": "q_proj",
    "linear_k": "k_proj",
    "linear_v": "v_proj",
    "linear_out": "o_proj",
    "linear_pos": "relative_k_proj",
    "pos_bias_u": "bias_u",
    "pos_bias_v": "bias_v",
}


def map_nemo_key(key: str) -> str | None:
    key = key.strip()
    if key.startswith("module."):
        key = key[len("module.") :]
    if key.startswith("model."):
        key = key[len("model.") :]
    if key.startswith("preprocessor.") or key.startswith("spec_augmentation."):
        return None
    if "num_batches_tracked" in key or "inv_freq" in key:
        return None

    key = key.replace("conv.batch_norm.", "conv.norm.")

    for src, dst in sorted(NEMO_TO_HF.items(), key=lambda kv: -len(kv[0])):
        if key == src or key.startswith(src + "."):
            key = dst + key[len(src) :]
            break

    if ".self_attn." in key:
        for src, dst in ATTN_RENAME.items():
            key = key.replace(f".self_attn.{src}", f".self_attn.{dst}")

    return key


class _Storage:
    def __init__(self, data: bytes, dtype: np.dtype):
        self.data = data
        self.dtype = np.dtype(dtype)


class _Tensor:
    def __init__(self, storage: _Storage, offset: int, size, stride):
        self.storage = storage
        self.offset = int(offset)
        self.size = tuple(int(x) for x in size)
        self.stride = tuple(int(x) for x in stride)

    def as_f32(self) -> np.ndarray:
        itemsize = self.storage.dtype.itemsize
        n = len(self.storage.data) // itemsize
        flat = np.frombuffer(self.storage.data, dtype=self.storage.dtype, count=n)
        if not self.size:
            return np.array(flat[self.offset], dtype=np.float32)
        numel = int(np.prod(self.size))
        # Fast path: contiguous row-major
        expected = 1
        contig = True
        for dim, st in zip(reversed(self.size), reversed(self.stride)):
            if st != expected:
                contig = False
                break
            expected *= dim
        if contig:
            start = self.offset
            return np.asarray(flat[start : start + numel].reshape(self.size), dtype=np.float32)
        idx = np.array(self.offset, dtype=np.int64)
        grid = np.stack(np.meshgrid(*[np.arange(s) for s in self.size], indexing="ij"), axis=-1)
        for axis, st in enumerate(self.stride):
            idx = idx + grid[..., axis] * st
        return np.asarray(flat[idx], dtype=np.float32)


_DTYPE_MAP = {
    "FloatStorage": np.dtype("<f4"),
    "HalfStorage": np.dtype("<f2"),
    "DoubleStorage": np.dtype("<f8"),
    "BFloat16Storage": np.dtype("<u2"),
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "ByteStorage": np.dtype("u1"),
    "BoolStorage": np.dtype("u1"),
}


class _Unpickler(pickle.Unpickler):
    def __init__(self, fh, zipf: zipfile.ZipFile | None, prefix: str):
        super().__init__(fh)
        self.zipf = zipf
        self.prefix = prefix

    def persistent_load(self, saved_id):
        if not isinstance(saved_id, tuple) or not saved_id:
            raise pickle.UnpicklingError(f"bad persistent id: {saved_id!r}")
        kind = saved_id[0]
        if kind == "storage" or kind == b"storage":
            # ('storage', storage_type, key, location, numel)
            storage_type, key, _location, numel = saved_id[1], saved_id[2], saved_id[3], saved_id[4]
            type_name = getattr(storage_type, "__name__", str(storage_type))
            dtype = _DTYPE_MAP.get(type_name, np.dtype("<f4"))
            if self.zipf is not None:
                name = f"{self.prefix}data/{key}"
                if name not in self.zipf.namelist():
                    # some archives use the key as a full path
                    matches = [n for n in self.zipf.namelist() if n.endswith(f"/{key}") or n.endswith(f"data/{key}")]
                    if not matches:
                        raise FileNotFoundError(f"storage {key} not in zip")
                    name = matches[0]
                raw = self.zipf.read(name)
            else:
                raw = b"\x00" * (int(numel) * dtype.itemsize)
            return _Storage(raw, dtype)
        raise pickle.UnpicklingError(f"unknown persistent id {saved_id!r}")

    def find_class(self, module, name):
        if module == "torch._utils" and name in {
            "_rebuild_tensor_v2",
            "_rebuild_tensor",
            "_rebuild_parameter",
        }:
            return _rebuild_tensor_v2
        if module == "torch._tensor" and name == "_rebuild_from_type_v2":
            return lambda func, type_obj, args, state: func(*args)
        if module.startswith("torch") and name.endswith("Storage"):
            return lambda *a, **k: type(name, (), {"__name__": name})
        if module == "torch" and name in {"FloatTensor", "HalfTensor", "BFloat16Tensor", "DoubleTensor"}:
            return _Tensor
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module in {"torch", "torch.nn.modules.container", "numpy.core.multiarray"}:
            if name in {"scalar", "reconstruct"}:
                return lambda *a, **k: a[0] if a else None
        try:
            return super().find_class(module, name)
        except Exception:
            return lambda *a, **k: None


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
    return _Tensor(storage, storage_offset, size, stride)


def load_state_dict(ckpt_path: Path) -> dict:
    if zipfile.is_zipfile(ckpt_path):
        zipf = zipfile.ZipFile(ckpt_path)
        pkls = [n for n in zipf.namelist() if n.endswith("data.pkl")]
        if not pkls:
            raise RuntimeError(f"no data.pkl in {ckpt_path}")
        pkl_name = pkls[0]
        prefix = pkl_name[: -len("data.pkl")]
        with zipf.open(pkl_name) as fh:
            obj = _Unpickler(fh, zipf, prefix).load()
        zipf.close()
    else:
        with ckpt_path.open("rb") as fh:
            obj = _Unpickler(fh, None, "").load()
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        return obj
    raise RuntimeError(f"unexpected checkpoint type: {type(obj)}")


def tensor_to_f32(value) -> np.ndarray:
    if isinstance(value, _Tensor):
        arr = value.as_f32()
    elif isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=np.float32)
    else:
        arr = np.asarray(value, dtype=np.float32)
    return np.ascontiguousarray(arr)


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, dict] = {}
    body = bytearray()
    offset = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr.astype("<f4", copy=False))
        nbytes = int(arr.nbytes)
        header[name] = {
            "dtype": "F32",
            "shape": [int(x) for x in arr.shape],
            "data_offsets": [offset, offset + nbytes],
        }
        body.extend(arr.tobytes())
        offset += nbytes
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # 8-byte aligned header
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(body)


def vocab_to_tokenizer_json(vocab_path: Path, out_path: Path) -> int:
    vocab: dict[str, int] = {}
    for line in vocab_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if " " in line:
            tok, _, idx = line.rpartition(" ")
            vocab[tok] = int(idx)
        else:
            vocab[line] = len(vocab)
    unk = "<unk>" if "<unk>" in vocab else next(iter(vocab))
    payload = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "unk_token": unk, "vocab": vocab},
    }
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    return len(vocab)


def parse_yaml_ints(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip().strip("\"'")
        if val.isdigit() or (val.startswith("-") and val[1:].isdigit()):
            out[key] = int(val)
    return out


def write_config(out_path: Path, yaml_text: str, vocab_size: int) -> None:
    vals = parse_yaml_ints(yaml_text)
    d_model = vals.get("d_model", 1024)
    n_layers = vals.get("n_layers", 24)
    n_heads = vals.get("n_heads", 8)
    n_mels = vals.get("features", vals.get("n_mels", vals.get("feat_in", 128)))
    blank = max(vocab_size - 1, 0)
    n_extra = vals.get("num_extra_outputs", 5)
    durations = list(range(n_extra)) if n_extra > 0 else [0, 1, 2, 3, 4]
    config = {
        "architectures": ["ParakeetForTDT"],
        "blank_token_id": blank,
        "pad_token_id": blank,
        "decoder_hidden_size": 640,
        "durations": durations,
        "encoder_config": {
            "hidden_size": d_model,
            "num_attention_heads": n_heads,
            "num_hidden_layers": n_layers,
            "num_mel_bins": n_mels,
        },
        "model_type": "parakeet_tdt",
        "vocab_size": vocab_size,
    }
    out_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def extract_nemo(nemo_path: Path, dest: Path) -> tuple[Path, str]:
    dest.mkdir(parents=True, exist_ok=True)
    yaml_text = ""
    ckpt: Path | None = None
    with tarfile.open(nemo_path, "r:*") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if not member.isfile():
                continue
            if name.endswith((".ckpt", ".pt", ".pth", ".bin")) or name in {
                "model_weights.ckpt",
                "model_weights.pt",
            }:
                extracted = dest / "model_weights.ckpt"
                with tar.extractfile(member) as src, extracted.open("wb") as out:
                    out.write(src.read())
                ckpt = extracted
            elif name in {"model_config.yaml", "model_config.yml"}:
                with tar.extractfile(member) as src:
                    yaml_text = src.read().decode("utf-8", errors="replace")
                (dest / "model_config.yaml").write_text(yaml_text, encoding="utf-8")
            elif name.endswith("tokenizer.model") or name.endswith("vocab.txt") or name.endswith("tokenizer.vocab") or name.endswith("tokenizer.json"):
                dest_name = "tokenizer.model" if name.endswith("tokenizer.model") else (
                    "tokenizer.json" if name.endswith("tokenizer.json") else "vocab.txt"
                )
                with tar.extractfile(member) as src:
                    (dest / dest_name).write_bytes(src.read())
    if ckpt is None:
        raise FileNotFoundError(f"no weight checkpoint inside {nemo_path}")
    return ckpt, yaml_text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--vocab", type=Path, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    work = Path(tempfile.mkdtemp(prefix="nemo-parakeet-"))
    try:
        ckpt, yaml_text = extract_nemo(args.nemo, work)
        print(f"loaded checkpoint {ckpt} ({ckpt.stat().st_size} bytes)", file=sys.stderr)
        state = load_state_dict(ckpt)
        mapped: dict[str, np.ndarray] = {}
        skipped = 0
        unmapped: list[str] = []
        for raw_key, value in state.items():
            if not isinstance(raw_key, str):
                continue
            hf = map_nemo_key(raw_key)
            if hf is None:
                skipped += 1
                continue
            try:
                arr = tensor_to_f32(value)
            except Exception as exc:
                print(f"skip {raw_key}: {exc}", file=sys.stderr)
                skipped += 1
                continue
            if hf in {
                None,
            }:
                continue
            # Keep unmapped-but-converted keys so convert.rs can report them.
            mapped[hf] = arr
            if not (
                hf.startswith("encoder.")
                or hf.startswith("decoder.")
                or hf.startswith("joint.")
                or hf.startswith("encoder_projector.")
            ):
                unmapped.append(hf)

        if not mapped:
            raise SystemExit("no tensors extracted from .nemo checkpoint")

        write_safetensors(args.out / "model.safetensors", mapped)
        vocab_path = args.vocab or args.out / "vocab.txt"
        if not vocab_path.exists() and (work / "vocab.txt").exists():
            vocab_path = work / "vocab.txt"
        vocab_size = 1025
        if vocab_path.exists():
            vocab_size = vocab_to_tokenizer_json(vocab_path, args.out / "tokenizer.json")
        write_config(args.out / "config.json", yaml_text, vocab_size)
        print(
            f"wrote {len(mapped)} tensors, skipped {skipped}, odd-keys {len(unmapped)} -> {args.out}",
            file=sys.stderr,
        )
        if unmapped[:12]:
            print("odd keys:", ", ".join(unmapped[:12]), file=sys.stderr)
        return 0
    finally:
        # keep work dir small: checkpoint is multi-GB
        import shutil

        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
