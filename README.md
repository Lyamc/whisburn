# o3whisburn

Burn-based speech processing in Rust: transcription, translation, diarization, and TTS — with automatic model download and conversion.

**o3whisburn** is a rewrite of [whisper-burn](https://github.com/Gadersd/whisper-burn) as a multi-crate workspace targeting **Burn 0.16**. Whisper, **Parakeet TDT v3**, and **Qwen3-ASR** are working end-to-end; additional backends are registered and being ported.

## Features

- **CLI** — transcribe local audio, download models, start a server
- **Web UI** — browser upload, transcribe, and download (JSON, TXT, SRT, VTT) at `http://localhost:8787/`
- **Batch processing** — drop multiple files; each transcript/transcode downloads as soon as that file finishes (optional folder save)
- **HTTP API** — multipart upload for transcription and audio transcoding
- **Audio transcoding** — normalized WAV export; Opus-in-Ogg (32/64 kbps) via ffmpeg
- **Opus streaming** — chunked `audio/ogg` for in-browser preview
- **Auto model management** — downloads from HuggingFace, converts safetensors → npy → Burn `.mpk`
- **Burn inference** — WGPU backend (CPU/GPU via Burn)
- **Functional pipeline** — composable task types in `o3whisburn-core`; conversion and decode paths favor iterators, folds, and pure helpers over imperative loops

## Quick start

### Prerequisites

- [Rust](https://rustup.rs/) (2021 edition)
- A WGPU-capable GPU driver (falls back to software rendering where supported)
- Optional: `HF_TOKEN` for gated HuggingFace models
- Optional: [ffmpeg](https://ffmpeg.org/) with `libopus` for Opus transcoding and streaming on the server

### Build

```bash
cargo build -p o3whisburn-cli
```

### Download a model

```bash
cargo run -p o3whisburn-cli -- models download tiny_en --verbose
```

### Transcribe

```bash
cargo run -p o3whisburn-cli -- transcribe --input path/to/audio.wav --model tiny_en
```

Output defaults to JSON. Use `--format txt`, `srt`, or `vtt` for other formats.

### Start the server

```bash
cargo run -p o3whisburn-cli -- serve --port 8787 --model qwen3-asr-0.6b
```

Open **http://localhost:8787/** in a browser to upload audio, pick an output format, and download the transcript. The UI also auto-transcodes uploads to Opus for in-browser playback (requires ffmpeg).

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
| [`o3whisburn-core`](crates/o3whisburn-core) | Shared types, `SpeechTask`, pipeline traits |
| [`o3whisburn-audio`](crates/o3whisburn-audio) | Decode (Symphonia), resample, encode, transcode (WAV/Opus) |
| [`o3whisburn-engine`](crates/o3whisburn-engine) | Burn inference (Whisper, Parakeet TDT, Qwen3-ASR, stubs for T-one/VibeVoice) |
| [`o3whisburn-models`](crates/o3whisburn-models) | Registry, HF download, safetensors conversion |
| [`o3whisburn-server`](crates/o3whisburn-server) | Axum HTTP API + embedded web UI |
| [`o3whisburn-cli`](crates/o3whisburn-cli) | `serve`, `transcribe`, `models` commands |

See [ARCHITECTURE.md](ARCHITECTURE.md) for data flow and crate dependencies.

## CLI reference

```
o3whisburn serve       Start HTTP server (web UI at /)
o3whisburn transcribe  Transcribe a local audio file
o3whisburn models      Download or list models
```

### `serve`

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `8787` | Listen port |
| `--model` | `tiny_en` | Default model for `/v1/transcribe` |
| `--device` | auto | Burn device override |
| `--hf-token` | `$HF_TOKEN` | HuggingFace API token |
| `--prefetch` | `true` | Download default model on startup |
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

All major flags also read `O3WHISBURN_*` environment variables.

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
cargo run -p o3whisburn-cli -- verify --models "tiny_en,base_en" --sentences --diarize --out-dir verify-outputs
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
| Working | VibeVoice-ASR (`vibevoice-asr`, Qwen2.5-7B INT8 decoder + dual encoders) |
| Working | VibeVoice-ASR-BitNet (`bitnet-asr`, Qwen2.5-1.5B ternary I2_S decoder) |
| Planned | VAD, diarization, TTS |

See [MODELS.md](MODELS.md) for the conversion pipeline and registry details.

## Cross compilation

```bash
# CPU-friendly cross to static linux (great for containers)
rustup target add x86_64-unknown-linux-musl
cargo build -p o3whisburn-cli --target x86_64-unknown-linux-musl --release

# Run with CPU fallback on the target
O3WHISBURN_DEVICE=cpu ./target/x86_64-unknown-linux-musl/release/o3whisburn transcribe ...
```

See [`.cargo/config.toml`](.cargo/config.toml) and [`scripts/cross-build.sh`](scripts/cross-build.sh) for more targets and helpers.

Notes:
- GPU acceleration (wgpu) requires appropriate drivers + graphics stack on the **target** machine.
- `O3WHISBURN_DEVICE=cpu` selects wgpu's CPU path (portable, no native GPU required at runtime for basic use).
- Full CUDA/Metal cross builds need matching native toolchains.

## Testing

Integration tests (require downloaded models and bundled `samples/jfk.wav`):

```bash
cargo test -p o3whisburn-engine --test jfk_mel
cargo test -p o3whisburn-engine --test parakeet_parity
cargo test -p o3whisburn-engine --test qwen3_greedy_parity
cargo test -p o3whisburn-engine --test qwen3_audio_hf_mel
```

Download models before parity tests:

```bash
cargo run -p o3whisburn-cli -- models download parakeet-tdt-0.6b-v3 --verbose
cargo run -p o3whisburn-cli -- models download qwen3-asr-0.6b --verbose
```

Server API tests (no model weights required):

```bash
cargo test -p o3whisburn-server
```

Model source resolution:

```bash
cargo test -p o3whisburn-models
```

## Environment variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token for gated models |
| `O3WHISBURN_MODELS_DIR` | Model cache directory (default: `./models`) |
| `O3WHISBURN_CONFIG` | Directory containing `settings.toml` |
| `O3WHISBURN_DEFAULT_MODEL` | Default model for serve/transcribe |
| `O3WHISBURN_PORT` | Server listen port (default 8787) |
| `O3WHISBURN_DEVICE` | Device override (`cpu`, `0`, `1`, …) |
| `O3WHISBURN_PRELOAD_MODELS` | Comma list or `all` for startup preload |
| `O3WHISBURN_VERBOSE`, `O3WHISBURN_DEBUG` | Verbose/debug flags |
| `RUST_LOG` | Log filter (e.g. `o3whisburn=debug`) |

CLI flags that support `--flag` also read the matching env (clap `env`). Priority: CLI > env > config file.

## Samples

Bundled test audio lives in [`samples/`](samples/) (no external reference repos required).

## License

MIT — see [Cargo.toml](Cargo.toml) workspace metadata.