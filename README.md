# whisburn

Burn-based speech processing in Rust: transcription, translation, diarization, and TTS — with automatic model download and conversion.

**whisburn** is a rewrite of [whisper-burn](https://github.com/Gadersd/whisper-burn) as a multi-crate workspace targeting **Burn 0.21**. Whisper, **Parakeet TDT v3**, and **Qwen3-ASR** are working end-to-end; additional backends are registered and being ported.

The default build uses **native TLS** (Schannel on Windows, Security framework on macOS, OpenSSL elsewhere) and a **pure-Rust** tokenizer regex backend. You do **not** need a C compiler or the `cc` crate to build.

## Features

- **CLI** — transcribe local audio, download models, start a server
- **Web UI** — browser upload, transcribe, and download (JSON, TXT, SRT, VTT) at `http://localhost:8787/`
- **Batch processing** — drop multiple files; each transcript/transcode downloads as soon as that file finishes (optional folder save)
- **HTTP API** — multipart upload for transcription and audio transcoding
- **Audio transcoding** — normalized WAV export; Opus-in-Ogg (32/64 kbps) via ffmpeg
- **Opus streaming** — chunked `audio/ogg` for in-browser preview
- **Auto model management** — downloads from HuggingFace, converts safetensors → npy → Burn `.mpk`
- **Burn inference** — WGPU backend (CPU/GPU via Burn)
- **Functional pipeline** — composable task types in `whisburn-core`; conversion and decode paths favor iterators, folds, and pure helpers over imperative loops

## Quick start

### Prerequisites

- [Rust](https://rustup.rs/) (2021 edition)
- A WGPU-capable GPU driver (falls back to software / ndarray where supported)
- Optional: `HF_TOKEN` for gated HuggingFace models
- Optional: [ffmpeg](https://ffmpeg.org/) with `libopus` for Opus transcoding and streaming on the server

No C toolchain is required for a normal `cargo build`. On Windows, use the MSVC target and MSVC `link.exe` (the project `.cargo/config.toml` overrides a user-level `lld-link`, which fails against recent VS 2022 CRTs).

### Build

```bash
cd whisburn
cargo build -p whisburn-cli
```

Release binary:

```bash
cargo build -p whisburn-cli --release
# → target/release/whisburn
```

### Download a model

```bash
cargo run -p whisburn-cli -- models download tiny_en --verbose
```

### Transcribe

```bash
cargo run -p whisburn-cli -- transcribe --input path/to/audio.wav --model tiny_en
```

Output defaults to JSON. Use `--format txt`, `srt`, or `vtt` for other formats.

### Start the server

```bash
cargo run -p whisburn-cli -- serve --port 8787 --model qwen3-asr-0.6b
```

`serve` is the API **and** the web UI. It binds the API under `/v1/` and serves the upload page at `/`. `whisburn serve` opens **http://127.0.0.1:8787/** in your browser (pass `--no-open` to skip). Use `http://127.0.0.1:8787/`, not `http://0.0.0.0:8787/`.

The UI auto-transcodes uploads to Opus for in-browser playback (requires ffmpeg). For the Iced desktop client, run `whisburn app`.

**curl example:**

```bash
curl -X POST "http://localhost:8787/v1/transcribe?model=tiny_en&format=json&download=true" \
  -F "audio=@samples/jfk.wav" \
  -o jfk.json
```

**Transcode to Opus:**

```bash
curl -X POST "http://localhost:8787/v1/transcode?format=opus&bitrate=medium&download=true" \
  -F "audio=@samples/jfk.wav" \
  -o jfk.ogg
```

## Workspace layout

| Crate | Role |
|-------|------|
| [`whisburn-core`](crates/whisburn-core) | Shared types, `SpeechTask`, pipeline traits |
| [`whisburn-audio`](crates/whisburn-audio) | Decode (Symphonia), resample, encode, transcode (WAV/Opus) |
| [`whisburn-engine`](crates/whisburn-engine) | Burn inference (Whisper, Parakeet TDT, Qwen3-ASR, stubs for T-one/VibeVoice) |
| [`whisburn-models`](crates/whisburn-models) | Registry, HF download, safetensors conversion |
| [`whisburn-server`](crates/whisburn-server) | Axum HTTP API + embedded web UI |
| [`whisburn-cli`](crates/whisburn-cli) | `serve`, `transcribe`, `models` commands (binary: `whisburn`) |
| [`whisburn-app`](crates/whisburn-app) | Optional Iced desktop / WASM client |

See [ARCHITECTURE.md](ARCHITECTURE.md) for data flow and crate dependencies.

## CLI reference

```
whisburn serve       Start HTTP server (web UI at /)
whisburn transcribe  Transcribe a local audio file
whisburn models      Download or list models
```

### `serve`

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `8787` | Listen port |
| `--model` | `tiny_en` | Default model for `/v1/transcribe` |
| `--device` | auto | Burn device override |
| `--hf-token` | `$HF_TOKEN` | HuggingFace API token |
| `--no-open` | off | Do not open the web UI in a browser |
| `-v, --verbose` | | Verbose logging |

### `transcribe`

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --input` | *required* | Input audio path |
| `-m, --model` | `tiny_en` | Model name |
| `-l, --language` | `en` | Source language |
| `--task` | `transcribe` | `transcribe`, `translate`, `diarize` |
| `--format` | `json` | `json`, `txt`, `srt`, `vtt` |
| `--timestamps` | auto | Segment timestamps (off for `*_en` by default) |
| `--orchestrate` | `false` | Scout + expert two-pass transcription |
| `--scout-model` | `tiny_en` | Fast model for orchestration scout pass |
| `--beam-size` | `5` | Decoder beam width |
| `--max-tokens` | `448` | Max decoder tokens per chunk |
| `--sentences` | `false` | Group output segments into sentences |
| `--device` | auto | Burn device override |
| `-v, --verbose` | | Verbose logging |
| `--debug` | | Debug mel / chunk logging |

All major flags also read `WHISBURN_*` environment variables.

### `models download`

| Flag | Description |
|------|-------------|
| `--force` | Clear stale artifacts and reconvert |
| `--hf-token` | HuggingFace API token |
| `-v, --verbose` | Verbose logging |

### `models list`

Prints all registered models with category, VRAM estimate, and `burn_ready` status.

### `verify`

Exercise multiple models × formats × settings and write artifacts for inspection:

```bash
cargo run -p whisburn-cli -- verify --models "tiny_en,base_en" --sentences --diarize --out-dir verify-outputs
```

Useful to confirm json/txt/srt + diarization labels + sentence grouping work across Whisper / Parakeet / Qwen3 etc.

## HTTP API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI (upload, transcribe, download) |
| `GET` | `/app` | Same as `/` |
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | List registered models |
| `POST` | `/v1/models/{name}/ensure` | Download and prepare a model |
| `POST` | `/v1/transcribe` | Transcribe uploaded audio (multipart) |
| `POST` | `/v1/transcode` | Transcode audio to WAV or Opus-in-Ogg |
| `POST` | `/v1/transcode/stream` | Stream Opus-in-Ogg (chunked) for playback |

Upload field name: `audio` or `file`.

Successful responses set `Content-Disposition: attachment` when `download=true` (default), so browsers save files with sensible names (e.g. `jfk.srt`).

### `/v1/transcribe` query parameters

| Param | Default | Description |
|-------|---------|-------------|
| `model` | server default | Model name |
| `task` | `transcribe` | `transcribe`, `translate`, … |
| `language` | `en` | Source language (`auto` for Qwen3) |
| `format` | `json` | `json`, `txt`, `text`, `srt`, `vtt` |
| `timestamps` | `true` | Include segment timestamps |
| `orchestrate` | `false` | Scout + expert two-pass transcription |
| `sentences` | `false` | Group into sentences (timed outputs + json.sentences) |
| `download` | `true` | Set `Content-Disposition: attachment` |

### `/v1/transcode` and `/v1/transcode/stream` query parameters

| Param | Default | Description |
|-------|---------|-------------|
| `format` | `opus` | `wav` or `opus` / `ogg` |
| `bitrate` | `medium` | `low` (32 kbps) or `medium` / `med` (64 kbps); Opus only |
| `download` | `true` | Attachment header (`stream` uses `inline` for `<audio>` playback) |

**Input formats** (decoded via Symphonia): WAV, MP3, FLAC, OGG, AAC, M4A, MKV, and more. **Opus output** requires ffmpeg with `libopus` on the server PATH.

## Supported models

Whisper models (`tiny` through `large-v3-turbo`, distil variants) are **Burn-ready** and download from upstream HuggingFace repos. Pre-converted [Gadersd/whisper-burn](https://huggingface.co/Gadersd/whisper-burn) bundles target Burn 0.8 and are **not** used.

| Status | Models |
|--------|--------|
| Working | Whisper ASR (`tiny_en`, `base`, `small`, …) |
| Working | Parakeet TDT (`parakeet-tdt-0.6b-v3`, `parakeet-tdt-0.6b-v2`) |
| Working | Qwen3-ASR (`qwen3-asr-0.6b`, `qwen3-asr-1.7b`) |
| Registered, conversion WIP | Parakeet CTC, T-one |
| Working | VibeVoice-ASR-BitNet (`bitnet-asr`, Qwen2.5-1.5B ternary I2_S decoder) |
| Experimental, not recommended | VibeVoice-ASR 7B (`vibevoice-asr`: GGUF Q4 → INT8 attn/MLP, tiled f32 lm_head; not a ggml 1:1) |
| Planned | VAD, diarization, TTS |

See [MODELS.md](MODELS.md) for the conversion pipeline and registry details.

## Cross compilation

```bash
# CPU-friendly cross to static linux (great for containers)
rustup target add x86_64-unknown-linux-musl
cargo build -p whisburn-cli --target x86_64-unknown-linux-musl --release

# Run with CPU fallback on the target
WHISBURN_DEVICE=cpu ./target/x86_64-unknown-linux-musl/release/whisburn transcribe ...
```

See [`.cargo/config.toml`](.cargo/config.toml) and [`scripts/cross-build.sh`](scripts/cross-build.sh) for more targets and helpers.

Notes:
- GPU acceleration (wgpu) requires appropriate drivers + graphics stack on the **target** machine.
- `WHISBURN_DEVICE=cpu` selects wgpu's CPU path (portable, no native GPU required at runtime for basic use).
- Full CUDA/Metal cross builds need matching native toolchains.
- Prefer targets where **native TLS** is available; rustls/`ring` is not used by default (avoids a C build dependency).

## Testing

Integration tests (require downloaded models and bundled `samples/jfk.wav`):

```bash
cargo test -p whisburn-engine --test jfk_mel
cargo test -p whisburn-engine --test parakeet_parity
cargo test -p whisburn-engine --test qwen3_greedy_parity
cargo test -p whisburn-engine --test qwen3_audio_hf_mel
```

Download models before parity tests:

```bash
cargo run -p whisburn-cli -- models download parakeet-tdt-0.6b-v3 --verbose
cargo run -p whisburn-cli -- models download qwen3-asr-0.6b --verbose
```

Server API tests (no model weights required):

```bash
cargo test -p whisburn-server
```

Model source resolution:

```bash
cargo test -p whisburn-models
```

## Environment variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token for gated models |
| `WHISBURN_MODELS_DIR` | Model cache directory (default: `./models`) |
| `WHISBURN_CONFIG` | Directory containing `settings.toml` |
| `WHISBURN_DEFAULT_MODEL` | Default model for serve/transcribe |
| `WHISBURN_PORT` | Server listen port (default 8787) |
| `WHISBURN_DEVICE` | Device override (`cpu`, `0`, `1`, …) |
| `WHISBURN_PRELOAD_MODELS` | Comma list or `all` for startup preload |
| `WHISBURN_VERBOSE`, `WHISBURN_DEBUG` | Verbose/debug flags |
| `RUST_LOG` | Log filter (e.g. `whisburn=debug`) |

CLI flags that support `--flag` also read the matching env (clap `env`). Priority: CLI > env > config file.

## Samples

Bundled test audio lives in [`samples/`](samples/) (no external reference repos required).

## License

MIT — see [Cargo.toml](Cargo.toml) workspace metadata.
