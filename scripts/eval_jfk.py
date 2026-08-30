#!/usr/bin/env python3
"""Transcribe samples/jfk.wav with every local Burn-ready ASR model and grade quality."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "release" / "o3whisburn"
AUDIO = ROOT / "samples" / "jfk.wav"
OUT_DIR = ROOT / "verify-outputs"
REPORT = OUT_DIR / "jfk_eval.json"

REF_WORDS = [
    "and", "so", "my", "fellow", "americans",
    "ask", "not", "what", "your", "country", "can", "do", "for", "you",
    "ask", "what", "you", "can", "do", "for", "your", "country",
]

MODELS = [
    "tiny_en", "tiny", "base_en", "base", "small_en", "small",
    "medium_en", "medium", "distil-medium-en", "distil-large-v3", "large-v3-turbo",
    "parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v2", "parakeet-ctc-0.6b", "parakeet-ctc-1.1b",
    "moonshine-tiny", "moonshine-base",
    "t-one",
    "qwen3-asr-0.6b", "qwen3-asr-1.7b",
    "bitnet-asr",
    "vibevoice-asr",
]

ENV = os.environ.copy()
ENV["VK_ICD_FILENAMES"] = "/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json"
ENV["VK_DRIVER_FILES"] = ENV["VK_ICD_FILENAMES"]
ENV["XDG_DATA_DIRS"] = "/run/opengl-driver/share:" + ENV.get("XDG_DATA_DIRS", "")
LOADER = "/nix/store/fg2v4x22xgnld8y34p8nwgz9hhzbpw2b-vulkan-loader-1.4.341.0/lib"
ENV["LD_LIBRARY_PATH"] = f"/run/opengl-driver/lib:{LOADER}:" + ENV.get("LD_LIBRARY_PATH", "")
ENV["WGPU_BACKEND"] = "vulkan"
ENV["RUST_LOG"] = "warn"

WORD_RE = re.compile(r"[a-z0-9']+", re.I)
SPEAKER_RE = re.compile(
    r"\b(speaker|spk|kennedy|jfk|john f\.? kennedy)\b",
    re.I,
)


def words(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def content_ok(hyp: list[str]) -> tuple[bool, str]:
    """Required words in order; extra function words allowed, extra content words fail."""
    i = 0
    extra = []
    filler = {
        "a", "an", "the", "uh", "um", "and", "so", "to", "of", "in", "on",
        "is", "are", "be", "that", "this", "it",
    }
    for w in hyp:
        if i < len(REF_WORDS) and w == REF_WORDS[i]:
            i += 1
            continue
        if i < len(REF_WORDS) and w in filler:
            continue
        if w in filler or w in REF_WORDS:
            continue
        extra.append(w)
    if i < len(REF_WORDS):
        missing = REF_WORDS[i:]
        return False, f"missing from {missing[0]!r}: {' '.join(missing[:8])}"
    if extra:
        return False, "extra words: " + " ".join(extra[:8])
    return True, "all required words in order"


def has_punctuation(text: str) -> bool:
    return bool(re.search(r"[,:;—.!?]", text))


def sentence_split_ok(text: str) -> bool:
    """Boundary between the two 'ask' clauses (period/semicolon/dash after first you)."""
    t = " ".join(text.split())
    if re.search(r"for you[.!?;:—–-]\s+Ask\b", t, re.I):
        return True
    if re.search(r"for you[.!?;]\s+", t, re.I) and re.search(r"\bAsk what you can do\b", t, re.I):
        return True
    return False


def has_speaker(text: str, raw_json: dict | None) -> bool:
    if SPEAKER_RE.search(text or ""):
        return True
    if not raw_json:
        return False
    blob = json.dumps(raw_json)
    if SPEAKER_RE.search(blob):
        return True
    for seg in raw_json.get("segments") or []:
        if isinstance(seg, dict) and seg.get("speaker"):
            return True
    return False


def grade(text: str, raw_json: dict | None) -> tuple[str, str]:
    hyp = words(text)
    ok, detail = content_ok(hyp)
    if not ok:
        return "Poor", detail
    punct = has_punctuation(text)
    split = sentence_split_ok(text)
    speaker = has_speaker(text, raw_json)
    if speaker and punct and split:
        return "Excellent", "words + punctuation + sentence split + speaker"
    if punct:
        if split:
            return "Great", "words + punctuation + sentence split (no speaker)"
        return "Great", "words + punctuation (single-sentence punctuation)"
    return "Good", "all words, no punctuation"


def run_model(name: str) -> dict:
    out_path = OUT_DIR / f"jfk_{name}.json"
    cmd = [
        str(BIN), "transcribe",
        "--input", str(AUDIO),
        "--model", name,
        "--format", "json",
        "--sentences",
        "--language", "en",
    ]
    timeout = 3600 if name == "vibevoice-asr" else (900 if name == "bitnet-asr" else 300)
    if name == "vibevoice-asr":
        cmd.extend(["--device", "cpu"])
    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=str(ROOT), env=ENV,
        capture_output=True, text=True,
        timeout=timeout,
    )
    # Retry on CPU if GPU OOM / wgpu panic.
    combined = (proc.stderr or "") + (proc.stdout or "")
    if proc.returncode != 0 and name != "vibevoice-asr" and re.search(
        r"out of memory|OOM|Failed to allocate|wgpu", combined, re.I
    ):
        print(f"  GPU failed, retrying {name} on CPU", flush=True)
        cmd_cpu = [c for c in cmd if c != "--device"] + ["--device", "cpu"]
        # strip previous --device gpu if any
        proc = subprocess.run(
            cmd_cpu, cwd=str(ROOT), env=ENV,
            capture_output=True, text=True,
            timeout=timeout,
        )
    wall = time.time() - t0
    stdout = proc.stdout or ""
    payload = None
    text = ""
    err = None
    if proc.returncode != 0:
        err = (proc.stderr or stdout)[-2000:]
    else:
        start = stdout.find("{")
        if start >= 0:
            try:
                payload, _ = json.JSONDecoder().raw_decode(stdout[start:])
                text = payload.get("text") or ""
                if not text and payload.get("sentences"):
                    text = " ".join(s.get("text", "") for s in payload["sentences"])
            except json.JSONDecodeError as e:
                err = f"json parse: {e}"
                text = stdout[-1500:]
        else:
            text = stdout.strip()
            err = "no json object in stdout"
    grade_label, reason = grade(text, payload) if text and not err else ("Poor", err or "empty")
    rec = {
        "model": name,
        "ok": proc.returncode == 0 and not err,
        "wall_s": round(wall, 2),
        "grade": grade_label,
        "reason": reason,
        "text": text,
        "error": err,
    }
    out_path.write_text(json.dumps({"record": rec, "payload": payload, "stderr": (proc.stderr or "")[-4000:]}, indent=2))
    return rec


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for name in MODELS:
        model_dir = ROOT / "models" / name
        if not model_dir.is_dir():
            print(f"SKIP {name}: not downloaded", flush=True)
            results.append({"model": name, "ok": False, "grade": "Poor", "reason": "not downloaded", "wall_s": 0, "text": ""})
            continue
        print(f"\n=== {name} ===", flush=True)
        rec = run_model(name)
        results.append(rec)
        print(f"  {rec['wall_s']}s  {rec['grade']}: {rec['reason']}", flush=True)
        print(f"  {rec['text'][:240]!r}", flush=True)
        REPORT.write_text(json.dumps({"ref_words": REF_WORDS, "results": results}, indent=2))
    print("\n==== SUMMARY ====")
    for r in results:
        print(f"{r['model']:24} {r.get('wall_s', 0):8}s  {r['grade']:10}  {r.get('text', '')[:80]}")
    REPORT.write_text(json.dumps({"ref_words": REF_WORDS, "results": results}, indent=2))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
