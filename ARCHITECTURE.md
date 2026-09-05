# Architecture

## Overview

whisburn is a Rust workspace that separates **types and tasks**, **audio I/O**, **inference**, **model lifecycle**, **HTTP serving**, and **CLI** into distinct crates. Inference runs on [Burn](https://github.com/tracel-ai/burn) 0.21 with the WGPU backend.

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ whisburn-cli│────▶│whisburn-server│────▶│ whisburn-models │
└──────┬──────┘     └──────┬───────┘     └────────┬────────┘
       │                   │                       │
       │              GET /  (web UI)              │
       └───────────────────┼───────────────────────┘
                           ▼
                  ┌─────────────────┐
                  │ whisburn-engine │◀── Burn 0.21 / WGPU
                  └────────┬────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      ┌──────────────┐          ┌──────────────┐
      │whisburn-audio│          │whisburn-core │
      └──────────────┘          └──────────────┘
```

## Crate responsibilities

### whisburn-core

Shared domain types with no Burn or I/O dependencies:

- `SpeechTask` — transcribe, translate, diarize, TTS, STS
- `TaskOptions` — language, beam size, timestamps, orchestration flags
- `TranscriptResult` / `TranscriptSegment` — structured output
- `OutputFormat` — json, txt, srt, vtt (transcript artifacts)
- `Pipeline` — composable multi-step processing (orchestration)

### whisburn-audio

Audio decoding, transcript encoding, and audio export:

- **Decode** — Symphonia (WAV, MP3, FLAC, OGG, AAC, M4A, MKV, …) → mono PCM `f32` @ 16 kHz
- **Resample** — rubato to model sample rates
- **Encode** — `TranscriptResult` → JSON, plain text, SRT, VTT
- **Transcode** — PCM → normalized WAV (`hound`); PCM → Opus-in-Ogg via ffmpeg (`libopus`, VOIP profile)
- **Export types** — `AudioExportFormat` (wav, opus), `OpusBitrate` (low 32 kbps, medium 64 kbps)

### whisburn-engine

Burn model definitions and inference:

- **Whisper** — encoder/decoder, mel preprocessing, beam search decode
- **Parakeet TDT** — Conformer encoder, TDT greedy decode, NeMo mel preprocessing
- **Qwen3-ASR** — audio tower (conv + transformer encoder) + Qwen3 thinker (GQA, MRoPE), greedy decode
- **VibeVoice-ASR** — safetensors bundle download, waveform prep, `SpeechConnector` scaffold; Qwen2.5-7B decoder WIP
- **T-one** — stub
- **Model registry** — static `MODEL_REGISTRY` with HF IDs and `burn_ready` flags
- **Loader** — reads `.mpk` + `.cfg` + `tokenizer.json` from model cache; Qwen3 loads `model.safetensors` directly

Key transcription paths:

**Whisper**

1. `waveform_to_mel_tensor` — STFT, mel filterbank, Whisper dynamic-range normalization
2. `mels_to_text/` — encoder forward, beam search decoder with HF `suppress_tokens`
3. `trim_trailing_hallucination` — post-decode cleanup

**Parakeet TDT**

1. `waveform_to_mel_tensor` + `audio::prep_audio` — NeMo mel with mask-aware norm
2. `Parakeet::forward` / `decode_tdt_greedy` — Conformer encode + TDT joint network
3. `decode/parakeet.rs` — token filter (dedup, leading-garbage drop) and BPE decode

**Qwen3-ASR**

1. `prep_audio` — 128-bin mel, drop last frame (1100 frames for JFK)
2. `Qwen3AudioTower` — chunked conv stack, sinusoidal PE, full-sequence audio encoder attention
3. `build_asr_prompt` — chat template with `<|audio_pad|>` placeholders
4. `Qwen3Thinker::generate_greedy` — prefix forward + KV-cache decode with MRoPE

### whisburn-models

Model download and conversion:

- Resolves HuggingFace source per registry entry (always upstream HF for Whisper)
- Downloads `model.safetensors`, `config.json`, `tokenizer.json`
- Converts weights: HF layout → npy dump → `NamedMpkGzFileRecorder` (Burn 0.21.0)
- `ModelManager` — async ensure + `process_waveform` used by server and CLI

### whisburn-server

Axum HTTP service with embedded web UI:

- `AppState` holds `Arc<ModelManager>`, upload size limit, default model
- **Routes** — health, model list/ensure, transcribe, transcode, transcode stream, web UI (`/`, `/app`)
- **Upload** — shared multipart parser (`upload.rs`); field `audio` or `file`
- **Downloads** — `Content-Disposition: attachment` with sanitized filename stem
- **Transcode stream** — async ffmpeg pipe → chunked `audio/ogg` (`tokio_util::io::ReaderStream`)
- Configurable upload size (default 512 MB), CORS, request tracing, prefetch on startup

### whisburn-cli

Thin clap front-end delegating to `whisburn-server` and `whisburn-models`. Supports `transcribe`, `translate`, `diarize`, and `--orchestrate` scout/expert passes.

## Transcription data flow

```
audio file / bytes
    │
    ▼
decode_to_mono_pcm()          [whisburn-audio]
    │
    ▼
waveform_to_mel_tensor()      [whisburn-engine::mel]
    │
    ▼
prep_audio()                  [whisburn-engine::audio]
    │  (Whisper: no ×32768 scale; NeMo models use int16 scale)
    ▼
forward_encoder()             [whisburn-engine::model::Whisper]
    │
    ▼
beam_search + forward_decoder()
    │  load_whisper_decode_config()  — suppress_tokens from config.json
    │  build_logit_mask()            — HF generation constraints
    │  truncate at first EOS
    ▼
decode (skip_special=true)
    │
    ▼
encode_result()               [whisburn-audio] → JSON / SRT / …
```

## HTTP server data flow

```
Browser or curl
    │
    ├─ GET / ──────────────────▶ embedded index.html (upload UI)
    │
    ├─ POST /v1/transcribe ────▶ multipart audio
    │       │
    │       ▼
    │   decode_bytes_to_mono_pcm()
    │       │
    │       ▼
    │   ModelManager::process_waveform()
    │       │
    │       ▼
    │   encode_result(format) + Content-Disposition
    │
    ├─ POST /v1/transcode ─────▶ multipart audio
    │       │
    │       ▼
    │   decode → transcode_waveform(wav | opus)
    │
    └─ POST /v1/transcode/stream
            │
            ▼
        decode → encode_wav → ffmpeg pipe → chunked audio/ogg
```

## Audio transcode flow (Opus)

```
uploaded bytes (any Symphonia format)
    │
    ▼
decode_bytes_to_mono_pcm()     16 kHz mono f32
    │
    ▼
encode_wav()                   in-memory WAV (RIFF)
    │
    ▼
ffmpeg -i pipe:0               libopus, VOIP application
    -c:a libopus -b:a 32k|64k
    -f ogg pipe:1
    │
    ▼
full buffer (/v1/transcode) or stdout stream (/v1/transcode/stream)
```

WAV export skips ffmpeg and writes 16-bit PCM directly via `hound`.

## Model conversion flow

```
HuggingFace repo (openai/whisper-tiny.en, …)
    │
    ▼
model.safetensors + config.json + tokenizer.json
    │
    ▼
convert_whisper_from_hf()     [whisburn-models::convert::whisper]
    │  F16→F32, burn_linear_layout transpose [out,in]→[in,out]
    ▼
encoder/*.npy + decoder/*.npy + shapes.json
    │
    ▼
save_model_from_npy()         [whisburn-models::convert::burn_mpk]
    │  WGPU load + NamedMpkGzFileRecorder
    ▼
model.mpk + config.cfg + .burn_version (0.21.0)
```

### Parakeet conversion flow

```
HuggingFace repo (nvidia/parakeet-tdt-0.6b-v3)
    │
    ▼
model.safetensors + config.json + tokenizer.json
    │
    ▼
convert_parakeet_from_hf()    [whisburn-models::convert::parakeet]
    │  HF NeMo keys → Burn Conformer + TDT paths
    │  parakeet_decode.json for runtime TDT metadata
    ▼
encoder/*.npy + decoder/*.npy + shapes.json
    │
    ▼
save_model_from_npy()         [burn_mpk — safetensors removed before this step]
    ▼
model.mpk + config.cfg + parakeet_decode.json + .burn_version
```

### Qwen3-ASR (no mpk conversion)

```
HuggingFace repo (Qwen/Qwen3-ASR-0.6B)
    │
    ▼
model.safetensors + tokenizer.json + qwen3_runtime.json
    │
    ▼
load_qwen3_weights()          [engine::model::qwen3::weights]
    │  audio_tower + thinker from safetensors at runtime
    ▼
greedy decode (no .mpk intermediate)
```

## Design decisions

### Burn 0.21, not Gadersd bundles

[Gadersd/whisper-burn](https://huggingface.co/Gadersd/whisper-burn) ships pre-converted `.mpk.gz` for Burn **0.8**. This workspace uses Burn **0.21.0**; weights are converted locally from upstream HF safetensors. Existing 0.16.1 `.mpk` bundles must be reconverted (`models download <name> --force`).

### English-only Whisper prompts

Models ending in `_en` or containing `.en` skip `<|lang|>` and `<|task|>` prompt tokens, matching OpenAI English-only checkpoints.

### Beam search stopping

Decoder beam search prefers **finished** hypotheses (sequences ending in EOS) over higher-probability unfinished beams, preventing trailing hallucinations after the true transcript.

### Functional style

- Pure conversion helpers in `whisburn-models::convert` (`dtype`, `mapping`, `layout` modules)
- Iterator-based download cleanup and HF tensor mapping (`try_for_each`, `fold`, `successors`)
- Decode filters as composable predicates (`filter`, `fold`, `skip`)
- Pipeline composition via `whisburn-core::Pipeline`
- Minimal shared mutable state; `ModelManager` is the primary runtime cache

### Module size convention

No single `.rs` source file exceeds **240 lines**. Large modules are split into directories (`download/`, `convert/parakeet/`, `transcribe/decode/`, `transcribe/mels_to_text/`).

### Opus via ffmpeg (not pure Rust)

Opus encoding uses an ffmpeg subprocess piping in-memory WAV. This keeps the server binary free of native `libopus` link dependencies while supporting browser-friendly `audio/ogg` preview. A future pure-Rust encoder could replace this path.

## Extension points

| Goal | Where to start |
|------|----------------|
| New ASR model | `registry.rs`, `convert/`, `engine::model`, `transcribe/decode/` |
| New speech task | `whisburn-core::SpeechTask`, server query params, CLI flags |
| New transcript format | `whisburn-core::OutputFormat`, `whisburn-audio::encode` |
| New audio export format | `whisburn-audio::transcode`, server `/v1/transcode` handlers |
| Live transcript streaming | `streaming_mode` in engine CLI path; add SSE/WebSocket route |
| Orchestration | `whisburn-core::Pipeline`, `TaskOptions::orchestrate` |