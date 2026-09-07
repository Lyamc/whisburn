#!/usr/bin/env python3
"""Fuse WeSpeaker / ResNet speaker checkpoints to embedding.safetensors."""
import json
import sys
from pathlib import Path

import numpy as np


def load_state(path: Path) -> dict:
    if path.suffix == ".nemo" or path.name.endswith(".nemo"):
        import tarfile
        import tempfile

        with tarfile.open(path, "r:*") as tar:
            names = tar.getnames()
            ckpt = next((n for n in names if n.endswith((".ckpt", ".pt", "model_weights.ckpt"))), None)
            if ckpt is None:
                raise SystemExit(f"no ckpt in {path}: {names[:20]}")
            with tempfile.TemporaryDirectory() as td:
                tar.extract(ckpt, path=td)
                return load_state(Path(td) / ckpt)
    if path.suffix == ".safetensors" or path.name.endswith(".safetensors"):
        from safetensors.numpy import load_file

        return load_file(str(path))
    import io
    import pickle
    import torch

    class Dummy:
        def __init__(self, *a, **k):
            self.__dict__.update(k)

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            self._state = state

        def state_dict(self):
            if isinstance(getattr(self, "_state", None), dict):
                return self._state
            return {k: v for k, v in self.__dict__.items() if torch.is_tensor(v)}

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("torch"):
                try:
                    return super().find_class(module, name)
                except Exception:
                    return Dummy
            return Dummy

    import sys
    import types

    class FakeMod(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__path__ = []
            self.__package__ = name

        def __getattr__(self, item):
            if item.startswith("__"):
                raise AttributeError(item)
            full = f"{self.__name__}.{item}"
            if item[:1].islower():
                if full not in sys.modules:
                    sys.modules[full] = FakeMod(full)
                setattr(self, item, sys.modules[full])
                return sys.modules[full]
            def _init(self, *a, **k):
                self.__dict__.update(k)

            def _setstate(self, s):
                if isinstance(s, dict):
                    self.__dict__.update(s)

            cls = type(
                item,
                (object,),
                {
                    "__init__": _init,
                    "__setstate__": _setstate,
                    "__module__": self.__name__,
                },
            )
            setattr(self, item, cls)
            return cls

    import importlib.machinery

    class Finder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in ("pyannote", "lightning", "pytorch_lightning"):
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            return FakeMod(spec.name)

        def exec_module(self, module):
            return None

    sys.meta_path.insert(0, Finder())
    for name in (
        "pyannote",
        "pyannote.audio",
        "pyannote.audio.core",
        "pyannote.audio.core.model",
        "pyannote.audio.core.task",
        "pyannote.audio.models",
        "pyannote.audio.models.embedding",
        "pyannote.audio.models.segmentation",
        "lightning",
        "pytorch_lightning",
    ):
        sys.modules[name] = FakeMod(name)

    try:
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    except Exception:
        pass
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        with open(path, "rb") as f:
            obj = Unpickler(f).load()

    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "ema"):
            inner = obj.get(k)
            if isinstance(inner, dict):
                obj = inner
                break
            if hasattr(inner, "state_dict"):
                obj = inner.state_dict()
                break
        out = {}
        for k, v in obj.items():
            if torch.is_tensor(v):
                out[str(k).replace("module.", "")] = v.detach().cpu().numpy()
            elif hasattr(v, "detach"):
                out[str(k).replace("module.", "")] = v.detach().cpu().numpy()
        if out:
            return out
    raise SystemExit(f"unsupported checkpoint {path}")


def fuse_conv_bn(conv_w, conv_b, bn_w, bn_b, bn_m, bn_v, eps=1e-5):
    std = np.sqrt(bn_v + eps)
    scale = bn_w / std
    # conv_w [O,I,K,K]
    fused_w = conv_w * scale.reshape(-1, 1, 1, 1)
    fused_b = (conv_b if conv_b is not None else np.zeros_like(bn_m)) - bn_m
    fused_b = fused_b * scale + bn_b
    return fused_w.astype(np.float32), fused_b.astype(np.float32)


def pick(state, *suffixes):
    for k, v in state.items():
        for s in suffixes:
            if k.endswith(s) or k == s:
                return v
    return None


def main():
    if len(sys.argv) < 3:
        print("usage: pack_wespeaker.py <ckpt> <out_dir>")
        sys.exit(2)
    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(src)
    keys = list(state)
    print("keys", len(keys), "sample", keys[:12])

    tensors = {}

    def conv_pair(prefix_out, conv_s, bn_s, default_stride_unused=1):
        w = pick(state, conv_s + ".weight")
        if w is None:
            return False
        b = pick(state, conv_s + ".bias")
        gamma = pick(state, bn_s + ".weight")
        beta = pick(state, bn_s + ".bias")
        mean = pick(state, bn_s + ".running_mean")
        var = pick(state, bn_s + ".running_var")
        if gamma is None:
            tensors[prefix_out + ".weight"] = w.astype(np.float32)
            tensors[prefix_out + ".bias"] = (
                b.astype(np.float32) if b is not None else np.zeros(w.shape[0], np.float32)
            )
            return True
        fw, fb = fuse_conv_bn(w, b, gamma, beta, mean, var)
        tensors[prefix_out + ".weight"] = fw
        tensors[prefix_out + ".bias"] = fb
        return True

    # stem
    if not conv_pair("stem.conv", "conv1", "bn1"):
        conv_pair("stem.conv", "resnet.conv1", "resnet.bn1")

    layout = [(0, 3), (1, 4), (2, 6), (3, 3)]
    for li, n in layout:
        for bi in range(n):
            base = f"layer{li}.{bi}"
            alt = f"resnet.layer{li}.{bi}"
            conv_pair(f"{base}.conv1", f"{base}.conv1", f"{base}.bn1") or conv_pair(
                f"{base}.conv1", f"{alt}.conv1", f"{alt}.bn1"
            )
            conv_pair(f"{base}.conv2", f"{base}.conv2", f"{base}.bn2") or conv_pair(
                f"{base}.conv2", f"{alt}.conv2", f"{alt}.bn2"
            )
            conv_pair(f"{base}.down", f"{base}.downsample.0", f"{base}.downsample.1") or conv_pair(
                f"{base}.down", f"{alt}.downsample.0", f"{alt}.downsample.1"
            )

    embed = pick(state, "seg.1.weight", "fc.weight", "embedding.weight", "linear.weight", "seg_1.weight")
    embed_b = pick(state, "seg.1.bias", "fc.bias", "embedding.bias", "linear.bias", "seg_1.bias")
    if embed is None:
        # last 2D weight that looks like [256, 512] or [192, *]
        cands = [(k, v) for k, v in state.items() if getattr(v, "ndim", 0) == 2 and v.shape[0] in (128, 192, 256, 512)]
        if not cands:
            raise SystemExit("no embedding linear found")
        embed = cands[-1][1]
        print("using embedding", cands[-1][0], embed.shape)
    tensors["embed.weight"] = embed.astype(np.float32)
    tensors["embed.bias"] = (
        embed_b.astype(np.float32)
        if embed_b is not None
        else np.zeros(embed.shape[0], np.float32)
    )

    from safetensors.numpy import save_file

    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, str(out_dir / "embedding.safetensors"))
    (out_dir / "diarize_runtime.json").write_text(
        json.dumps({"kind": "wespeaker-resnet34", "embed_dim": int(embed.shape[0]), "inference_status": "ready"}, indent=2),
        encoding="utf-8",
    )
    (out_dir / ".burn_version").write_text("0.21.0-diarize\n", encoding="utf-8")
    print("wrote", len(tensors), "tensors to", out_dir)


if __name__ == "__main__":
    main()
