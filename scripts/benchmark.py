#!/usr/bin/env python3
"""Run one transcription per downloaded model and write benchmark.csv."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "benchmark.csv"
AUDIO = ROOT / "samples" / "jfk.wav"
MODELS_DIR = ROOT / "models"
BIN = ROOT / "target" / "release" / "whisburn"

VRAM_ESTIMATE_MB = {
    "tiny": 400,
    "tiny_en": 400,
    "base": 600,
    "base_en": 600,
    "small": 1200,
    "small_en": 1200,
    "medium": 2500,
    "medium_en": 2500,
    "large-v3-turbo": 6000,
    "distil-medium-en": 1500,
    "distil-large-v3": 3000,
    "parakeet-tdt-0.6b-v2": 1200,
    "parakeet-tdt-0.6b-v3": 1200,
    "parakeet-ctc-0.6b": 1200,
    "parakeet-ctc-1.1b": 2200,
    "moonshine-tiny": 300,
    "moonshine-base": 500,
    "t-one": 500,
    "qwen3-asr-0.6b": 2500,
    "qwen3-asr-1.7b": 4500,
    "vibevoice-asr": 8000,
    "bitnet-asr": 2500,
}

# Burn-ready ASR models to include. Already-downloaded ones are reused;
# missing ones are fetched via `whisburn models download`.
BENCHMARK_MODELS = [
    "tiny_en",
    "tiny",
    "base_en",
    "base",
    "small_en",
    "small",
    "medium_en",
    "medium",
    "distil-medium-en",
    "distil-large-v3",
    "large-v3-turbo",
    "parakeet-tdt-0.6b-v3",
    "parakeet-tdt-0.6b-v2",
    "parakeet-ctc-0.6b",
    "parakeet-ctc-1.1b",
    "moonshine-tiny",
    "moonshine-base",
    "t-one",
    "qwen3-asr-0.6b",
    "qwen3-asr-1.7b",
    "vibevoice-asr",
    "bitnet-asr",
]

ENV = os.environ.copy()
ENV["VK_ICD_FILENAMES"] = "/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json"
ENV["VK_DRIVER_FILES"] = ENV["VK_ICD_FILENAMES"]
ENV["XDG_DATA_DIRS"] = "/run/opengl-driver/share:" + ENV.get("XDG_DATA_DIRS", "")
loader = "/nix/store/fg2v4x22xgnld8y34p8nwgz9hhzbpw2b-vulkan-loader-1.4.341.0/lib"
ENV["LD_LIBRARY_PATH"] = f"/run/opengl-driver/lib:{loader}:" + ENV.get("LD_LIBRARY_PATH", "")
ENV["WGPU_BACKEND"] = "vulkan"
ENV["RUST_LOG"] = ENV.get("RUST_LOG", "warn")


def audio_length_secs(path: Path) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
            text=True,
        ).strip()
        return float(out)
    except Exception:
        return 11.0


def model_size_mb(name: str) -> float:
    path = MODELS_DIR / name
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return round(total / (1024 * 1024), 1)


def nvidia_snapshot() -> tuple[float | None, float | None]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        ).strip().split(",")
        return float(out[0].strip()), float(out[1].strip())
    except Exception:
        return None, None


def proc_rss_mb(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                kb = float(line.split()[1])
                return kb / 1024.0
    except Exception:
        return None
    return None


def proc_cpu_times(pid: int) -> tuple[float, float] | None:
    try:
        clk = os.sysconf("SC_CLK_TCK")
        parts = Path(f"/proc/{pid}/stat").read_text().split()
        utime = int(parts[13]) / clk
        stime = int(parts[14]) / clk
        wall = time.time()
        return utime + stime, wall
    except Exception:
        return None


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    parts = re.findall(r".+?(?:[.!?;]+(?:\s|$)|$)", text)
    out = []
    for part in parts:
        s = part.strip()
        if not s:
            continue
        if re.fullmatch(r"[,.\s]+", s):
            out.append(s)
            continue
        out.append(s)
    return out or [text]


def sentences_from_output(raw: str) -> tuple[str, str]:
    data = None
    raw = raw.strip()
    start = raw.find("{")
    if start >= 0:
        try:
            data = json.loads(raw[start:])
        except json.JSONDecodeError:
            data = None
    blob = raw
    if isinstance(data, dict):
        sents = data.get("sentences") or []
        joined = " ".join(str(s.get("text", "")).strip() for s in sents if str(s.get("text", "")).strip())
        blob = joined or str(data.get("text") or "")
        if not blob:
            segs = data.get("segments") or []
            blob = " ".join(str(s.get("text", "")).strip() for s in segs if str(s.get("text", "")).strip())
    texts = split_sentences(blob)
    if not texts:
        return "", ""
    return texts[0], texts[-1]


def parse_cli_elapsed(stderr: str) -> float | None:
    m = re.search(r"Transcription finished in ([0-9.]+)s", stderr)
    if m:
        return float(m.group(1))
    return None


class Sampler:
    def __init__(self, pid: int):
        self.pid = pid
        self.stop = threading.Event()
        self.gpu_util: list[float] = []
        self.gpu_mem: list[float] = []
        self.rss: list[float] = []
        self.cpu_pct: list[float] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._prev_cpu = proc_cpu_times(pid)

    def start(self) -> None:
        self._thread.start()

    def finish(self) -> None:
        self.stop.set()
        self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self.stop.is_set():
            gpu_u, gpu_m = nvidia_snapshot()
            if gpu_u is not None:
                self.gpu_util.append(gpu_u)
            if gpu_m is not None:
                self.gpu_mem.append(gpu_m)
            rss = proc_rss_mb(self.pid)
            if rss is not None:
                self.rss.append(rss)
            now = proc_cpu_times(self.pid)
            if now and self._prev_cpu:
                d_cpu = now[0] - self._prev_cpu[0]
                d_wall = now[1] - self._prev_cpu[1]
                if d_wall > 0:
                    self.cpu_pct.append(max(0.0, 100.0 * d_cpu / d_wall))
            if now:
                self._prev_cpu = now
            self.stop.wait(0.2)


def avg(xs: list[float]) -> float | None:
    if not xs:
        return None
    return round(sum(xs) / len(xs), 1)


def peak(xs: list[float]) -> float | None:
    if not xs:
        return None
    return round(max(xs), 1)


def model_is_ready(name: str) -> bool:
    d = MODELS_DIR / name
    if not d.is_dir():
        return False
    if name.startswith("qwen3-asr-"):
        return (d / "model.safetensors").exists() and (d / "qwen3_runtime.json").exists()
    if name.startswith("moonshine"):
        return (d / "model.safetensors").exists() and (d / "moonshine_runtime.json").exists()
    if name == "t-one":
        return (d / "model.safetensors").exists() and (d / "tone_runtime.json").exists()
    if name in ("vibevoice-asr", "bitnet-asr"):
        return (d / "model.safetensors.index.json").exists() and (d / "vibevoice_runtime.json").exists()
    return (d / "model.mpk").exists() or (d / "model.mpk.gz").exists() or (d / f"{name}.mpk").exists()


def ensure_model(name: str) -> bool:
    if model_is_ready(name):
        return True
    print(f"downloading/converting {name} …", flush=True)
    cmd = [str(BIN), "models", "download", name, "--verbose"]
    env = ENV.copy()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-1500:] if proc.stdout else "")
        print(proc.stderr[-1500:] if proc.stderr else "")
        print(f"skip {name}: download failed ({proc.returncode})", flush=True)
        return False
    return model_is_ready(name)


def available_models() -> list[str]:
    names = []
    for name in BENCHMARK_MODELS:
        if ensure_model(name):
            names.append(name)
        else:
            print(f"not ready: {name}", flush=True)
    return names


def run_one(model: str, audio_secs: float, device: str | None = None) -> dict:
    out_path = ROOT / "verify-outputs" / f"benchmark_{model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BIN),
        "transcribe",
        "--input",
        str(AUDIO),
        "--model",
        model,
        "--format",
        "json",
        "--sentences",
        "--language",
        "en",
        "--verbose",
    ]
    if device is None and model == "vibevoice-asr":
        device = "cpu"
    if device:
        cmd.extend(["--device", device])
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    sampler = Sampler(proc.pid)
    sampler.start()
    stdout, stderr = proc.communicate()
    wall = time.time() - t0
    sampler.finish()
    out_path.write_text((stdout or "") + "\n" + (stderr or "")[-4000:], encoding="utf-8")
    if proc.returncode != 0:
        blob = (stderr or stdout or "").lower()
        if device is None and any(
            s in blob for s in ("out of memory", "oom", "failed to allocate", "no possible adapter")
        ):
            print(f"  GPU failed for {model}, retrying --device cpu …", flush=True)
            return run_one(model, audio_secs, device="cpu")
        err = (stderr or stdout or f"exit {proc.returncode}").strip().replace("\n", " ")[-240:]
        return {
            "model": model,
            "model_size_mb": model_size_mb(model),
            "vram_estimate_mb": VRAM_ESTIMATE_MB.get(model, ""),
            "peak_rss_mb": peak(sampler.rss),
            "peak_gpu_memory_mb": peak(sampler.gpu_mem),
            "avg_cpu_pct": avg(sampler.cpu_pct),
            "avg_gpu_pct": avg(sampler.gpu_util),
            "completion_time_s": round(wall, 3),
            "wall_time_s": round(wall, 3),
            "audio_file": str(AUDIO.relative_to(ROOT)),
            "audio_length_s": round(audio_secs, 3),
            "first_sentence": f"ERROR: {err}",
            "last_sentence": "",
        }
    first, last = sentences_from_output(stdout)
    cli_elapsed = parse_cli_elapsed(stderr)
    return {
        "model": model,
        "model_size_mb": model_size_mb(model),
        "vram_estimate_mb": VRAM_ESTIMATE_MB.get(model, ""),
        "peak_rss_mb": peak(sampler.rss),
        "peak_gpu_memory_mb": peak(sampler.gpu_mem),
        "avg_cpu_pct": avg(sampler.cpu_pct),
        "avg_gpu_pct": avg(sampler.gpu_util),
        "completion_time_s": round(cli_elapsed if cli_elapsed is not None else wall, 3),
        "wall_time_s": round(wall, 3),
        "audio_file": str(AUDIO.relative_to(ROOT)),
        "audio_length_s": round(audio_secs, 3),
        "first_sentence": first,
        "last_sentence": last,
    }


FIELDS = [
    "model",
    "model_size_mb",
    "vram_estimate_mb",
    "peak_rss_mb",
    "peak_gpu_memory_mb",
    "avg_cpu_pct",
    "avg_gpu_pct",
    "completion_time_s",
    "wall_time_s",
    "audio_file",
    "audio_length_s",
    "first_sentence",
    "last_sentence",
]


def load_existing_rows() -> list[dict]:
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keep = []
    for row in rows:
        first = (row.get("first_sentence") or "").strip()
        if first.startswith("ERROR:"):
            continue
        if row.get("model"):
            keep.append(row)
    return keep


def write_csv(rows: list[dict]) -> None:
    by_name = {row["model"]: row for row in rows if row.get("model")}
    ordered = []
    for name in BENCHMARK_MODELS:
        if name in by_name:
            ordered.append(by_name.pop(name))
    ordered.extend(by_name.values())
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(ordered)
    print(f"wrote {CSV_PATH} ({len(ordered)} rows)", flush=True)


def skip_row(model: str, audio_secs: float, reason: str) -> dict:
    return {
        "model": model,
        "model_size_mb": model_size_mb(model) if (MODELS_DIR / model).exists() else "",
        "vram_estimate_mb": VRAM_ESTIMATE_MB.get(model, ""),
        "peak_rss_mb": "",
        "peak_gpu_memory_mb": "",
        "avg_cpu_pct": "",
        "avg_gpu_pct": "",
        "completion_time_s": "",
        "wall_time_s": "",
        "audio_file": str(AUDIO.relative_to(ROOT)),
        "audio_length_s": round(audio_secs, 3),
        "first_sentence": f"SKIP: {reason}",
        "last_sentence": "",
    }


def main() -> int:
    import sys

    if not BIN.exists():
        raise SystemExit(f"missing binary: {BIN}")
    if not AUDIO.exists():
        raise SystemExit(f"missing audio: {AUDIO}")
    audio_secs = audio_length_secs(AUDIO)
    rows = load_existing_rows()
    done = {row["model"] for row in rows}
    print(f"existing rows: {sorted(done) or 'none'}", flush=True)

    force = "--force" in sys.argv[1:]
    requested = [a for a in sys.argv[1:] if not a.startswith("-")]
    wanted = requested or BENCHMARK_MODELS
    if force:
        done.difference_update(wanted)
    pending = [name for name in wanted if name not in done]
    ready = []
    for name in pending:
        if ensure_model(name):
            ready.append(name)
        else:
            print(f"not ready: {name}", flush=True)
            rows.append(skip_row(name, audio_secs, "download failed or HF artifacts unavailable"))
            write_csv(rows)

    for i, model in enumerate(ready, 1):
        print(f"[{i}/{len(ready)}] {model} …", flush=True)
        row = run_one(model, audio_secs)
        rows.append(row)
        print(
            f"  size={row['model_size_mb']}MB rss={row['peak_rss_mb']} "
            f"gpu_mem={row['peak_gpu_memory_mb']} gpu={row['avg_gpu_pct']}% "
            f"cpu={row['avg_cpu_pct']}% t={row['completion_time_s']}s",
            flush=True,
        )
        print(f"  first={row['first_sentence'][:80]!r}", flush=True)
        print(f"  last={row['last_sentence'][:80]!r}", flush=True)
        write_csv(rows)

    write_csv(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
