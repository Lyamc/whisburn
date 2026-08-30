#!/usr/bin/env python3
"""Exercise the web frontend's HTTP contract for every available model × setting combo.

Builds the same `/v1/transcribe/stream` and `/v1/transcode` URLs the UI uses,
loads `GET /` to confirm branding + batch controls, and writes a JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "http://127.0.0.1:8787"
SAMPLE = ROOT / "samples" / "jfk.wav"
MODELS_DIR = ROOT / "models"
REPORT_PATH = ROOT / "verify-outputs" / "web_frontend_matrix.json"


def request(url: str, data: bytes | None = None, headers: dict | None = None, timeout: int = 300):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, dict(resp.headers), resp.read()


def multipart(field: str, filename: str, payload: bytes, boundary: str = "----o3whisburn-matrix") -> tuple[bytes, str]:
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + payload + f"\r\n--{boundary}--\r\n".encode("utf-8")
    return body, f"multipart/form-data; boundary={boundary}"


def parse_sse_done(raw: bytes) -> dict:
    last = {}
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        try:
            last = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
    return last


def available_models(catalog: list[dict]) -> list[str]:
    ready = []
    for item in catalog:
        if item.get("category") != "Asr" or not item.get("burn_ready"):
            continue
        name = item["name"]
        if (MODELS_DIR / name).is_dir():
            ready.append(name)
    return ready


def validate_transcript(fmt: str, text: str, timestamps: bool, diarize: bool, sentences: bool) -> str | None:
    if not text or not text.strip():
        return "empty transcript"
    if fmt == "json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return f"invalid json: {e}"
        if not isinstance(data, dict):
            return "json is not an object"
        if sentences and "sentences" not in data:
            return "json missing sentences field"
        blob = json.dumps(data)
        if diarize and "SPEAKER_" not in blob:
            return "diarize json missing SPEAKER_ labels"
        return None
    if fmt == "srt":
        if "-->" not in text:
            return "srt missing cue timestamps"
        if diarize and "SPEAKER_" not in text:
            return "diarize srt missing SPEAKER_ labels"
        return None
    if fmt == "vtt":
        if "WEBVTT" not in text and "-->" not in text:
            return "vtt missing cues"
        if diarize and "SPEAKER_" not in text:
            return "diarize vtt missing SPEAKER_ labels"
        return None
    if fmt == "txt":
        if timestamps and "-->" not in text and "[" not in text:
            return "timestamped txt missing time markers"
        if timestamps and diarize and "SPEAKER_" not in text:
            return "diarize txt missing SPEAKER_ labels"
        return None
    return f"unknown format {fmt}"


def transcribe_url(base: str, model: str, fmt: str, language: str, timestamps: str, sentences: str, diarize: bool, file_extension: str = "") -> str:
    params = [
        f"model={model}",
        f"format={fmt}",
        f"language={language}",
        f"timestamps={timestamps}",
        f"sentences={sentences}",
    ]
    if diarize:
        params.append("task=diarize")
    if file_extension:
        params.append(f"file_extension={file_extension}")
    return f"{base}/v1/transcribe/stream?{'&'.join(params)}"


def write_report(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp.replace(REPORT_PATH)


def load_prior_ok(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    prior = {}
    for row in data.get("results") or []:
        label = row.get("label")
        if label and row.get("ok"):
            prior[label] = row
    return prior


def run_matrix(base: str, include: list[str] | None, exclude: set[str], skip_completed: bool, timeout: int) -> int:
    status, _, html = request(base + "/", timeout=30)
    page = html.decode("utf-8", errors="replace")
    ui_checks = {
        "branding o3whisburn": "o3whisburn" in page,
        "no leftover o2whisburn": "o2whisburn" not in page,
        "batch queue": "Batch queue" in page,
        "multi-file input": 'id="file-input"' in page and "multiple" in page,
        "folder picker": 'id="btn-output-folder"' in page,
        "stream endpoint in UI": "/v1/transcribe/stream" in page,
        "transcode endpoint in UI": "/v1/transcode" in page,
        "matrix helper": "runFrontendMatrix" in page,
    }
    failed_ui = [name for name, ok in ui_checks.items() if not ok]
    if failed_ui:
        print("UI HTML checks failed:", ", ".join(failed_ui))
        return 1
    print(f"GET / ok ({status}) — branding and batch UI present")

    _, _, models_raw = request(base + "/v1/models", timeout=30)
    catalog = json.loads(models_raw.decode("utf-8"))
    models = available_models(catalog)
    if include:
        wanted = set(include)
        models = [m for m in models if m in wanted]
    models = [m for m in models if m not in exclude]
    if not models:
        print("No downloaded burn-ready ASR models found under ./models")
        return 1
    print("Available models:", ", ".join(models))

    prior_ok = load_prior_ok(REPORT_PATH) if skip_completed else {}
    if prior_ok:
        print(f"Skipping {len(prior_ok)} previously passing labels from {REPORT_PATH.name}")

    wav = SAMPLE.read_bytes()
    body, content_type = multipart("audio", "jfk.wav", wav)

    formats = ["json", "txt", "srt", "vtt"]
    timestamps_vals = ["true", "false"]
    diarize_vals = [False, True]
    sentences_vals = ["false", "true"]
    languages = ["en", "auto"]

    combos = list(product(models, formats, timestamps_vals, diarize_vals, sentences_vals, languages))
    results = []
    failures = 0
    skipped = 0
    start_all = time.time()

    for i, (model, fmt, ts, dia, sent, lang) in enumerate(combos, 1):
        url = transcribe_url(base, model, fmt, lang, ts, sent, dia)
        label = f"{model}|{fmt}|ts={ts}|dia={int(dia)}|sent={sent}|lang={lang}"
        if label in prior_ok:
            results.append(prior_ok[label])
            skipped += 1
            print(f"[{i}/{len(combos)}] skip {label}")
            continue
        t0 = time.time()
        try:
            status, _headers, raw = request(url, data=body, headers={"Content-Type": content_type}, timeout=timeout)
            ev = parse_sse_done(raw)
            text = (ev.get("partial_text") or "") if isinstance(ev, dict) else ""
            phase = ev.get("phase") if isinstance(ev, dict) else None
            err = None
            if status != 200:
                err = f"HTTP {status}"
            elif phase == "error":
                err = ev.get("task_label") or "stream error"
            elif phase != "done":
                err = f"missing done event (phase={phase!r})"
            else:
                err = validate_transcript(fmt, text, ts == "true", dia, sent == "true")
            ms = int((time.time() - t0) * 1000)
            ok = err is None
            if not ok:
                failures += 1
            results.append({"ok": ok, "label": label, "ms": ms, "bytes": len(text), "error": err, "preview": text[:160]})
            mark = "ok" if ok else "FAIL"
            print(f"[{i}/{len(combos)}] {mark} {label} ({ms}ms){'' if ok else ' — ' + str(err)}")
        except Exception as e:
            failures += 1
            ms = int((time.time() - t0) * 1000)
            results.append({"ok": False, "label": label, "ms": ms, "error": str(e), "preview": ""})
            print(f"[{i}/{len(combos)}] FAIL {label} ({ms}ms) — {e}")
        write_report({
            "base": base,
            "models": models,
            "excluded": sorted(exclude),
            "total": len(results),
            "failed": failures,
            "skipped": skipped,
            "elapsed_secs": round(time.time() - start_all, 1),
            "ui_checks": ui_checks,
            "results": results,
            "partial": True,
        })

    transcode_cases = [
        ("wav", None),
        ("opus", "medium"),
        ("opus", "low"),
    ]
    for fmt, bitrate in transcode_cases:
        qs = f"format={fmt}&download=true"
        if bitrate:
            qs += f"&bitrate={bitrate}"
        url = f"{base}/v1/transcode?{qs}"
        label = f"transcode|{fmt}|{bitrate or '-'}"
        t0 = time.time()
        try:
            status, headers, raw = request(url, data=body, headers={"Content-Type": content_type}, timeout=120)
            ms = int((time.time() - t0) * 1000)
            err = None
            if status != 200:
                err = f"HTTP {status}"
            elif len(raw) < 256:
                err = f"tiny payload ({len(raw)} bytes)"
            cd = headers.get("Content-Disposition") or headers.get("content-disposition") or ""
            if "attachment" not in cd:
                err = err or "missing Content-Disposition attachment"
            ok = err is None
            if not ok:
                failures += 1
            results.append({"ok": ok, "label": label, "ms": ms, "bytes": len(raw), "error": err})
            print(f"[transcode] {'ok' if ok else 'FAIL'} {label} ({ms}ms, {len(raw)} bytes){'' if ok else ' — ' + str(err)}")
        except Exception as e:
            failures += 1
            results.append({"ok": False, "label": label, "ms": int((time.time() - t0) * 1000), "error": str(e)})
            print(f"[transcode] FAIL {label} — {e}")

    for ext in ["txt", "srt", "vtt", "json"]:
        url = transcribe_url(base, models[0], "txt", "en", "true", "false", False, file_extension=ext)
        label = f"file_extension|{ext}"
        t0 = time.time()
        try:
            status, _, raw = request(url, data=body, headers={"Content-Type": content_type}, timeout=300)
            ev = parse_sse_done(raw)
            ok = status == 200 and ev.get("phase") == "done" and bool(ev.get("partial_text"))
            err = None if ok else f"phase={ev.get('phase')}"
            if not ok:
                failures += 1
            results.append({"ok": ok, "label": label, "ms": int((time.time() - t0) * 1000), "error": err})
            print(f"[ext] {'ok' if ok else 'FAIL'} {label}")
        except Exception as e:
            failures += 1
            results.append({"ok": False, "label": label, "error": str(e)})
            print(f"[ext] FAIL {label} — {e}")

    report = {
        "base": base,
        "models": models,
        "excluded": sorted(exclude),
        "total": len(results),
        "failed": failures,
        "skipped": skipped,
        "elapsed_secs": round(time.time() - start_all, 1),
        "ui_checks": ui_checks,
        "results": results,
    }
    write_report(report)
    print(f"\n{len(results) - failures}/{len(results)} passed in {report['elapsed_secs']}s ({skipped} skipped)")
    print(f"report: {REPORT_PATH}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--models", default="", help="Comma-separated model names (default: all downloaded burn-ready ASR)")
    parser.add_argument(
        "--exclude",
        default="vibevoice-asr,bitnet-asr",
        help="Comma-separated models to skip. Default excludes vibevoice-asr and bitnet-asr (slow greedy STT).",
    )
    parser.add_argument("--no-skip-completed", action="store_true", help="Re-run labels that already passed")
    parser.add_argument("--timeout", type=int, default=900, help="Per-request timeout in seconds")
    args = parser.parse_args()
    if not SAMPLE.exists():
        print(f"missing sample: {SAMPLE}", file=sys.stderr)
        return 1
    include = [m.strip() for m in args.models.split(",") if m.strip()] or None
    exclude = {m.strip() for m in args.exclude.split(",") if m.strip()}
    return run_matrix(args.base.rstrip("/"), include, exclude, not args.no_skip_completed, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
