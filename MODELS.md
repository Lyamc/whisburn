# Models

## Registry

All models are declared in `crates/whisburn-engine/src/model/registry.rs`. Each entry includes:

- **name** — CLI/API identifier (e.g. `tiny_en`)
- **hf_id** — upstream HuggingFace repo
- **burn_ready** — whether download + Burn inference is implemented

List models:

```bash
cargo run -p whisburn-cli -- models list
```

## Download

```bash
cargo run -p whisburn-cli -- models download <name> [--force] [--verbose]
```

| Flag | Effect |
|------|--------|
| `--force` | Deletes stale `encoder/`, `decoder/`, `shapes.json`, and old `.mpk` artifacts, then reconverts |
| `--verbose` | Logs download and conversion progress |

The CLI and server call the same `download_model()` function in `whisburn-models`.

### Cache location

Default: `./models/<name>/`

Override:

```bash
export WHISBURN_MODELS_DIR=/data/whisburn-models
```

### Authentication

Set `HF_TOKEN` (or pass `--hf-token`) for gated HuggingFace repositories.

## Whisper (working)

### Source

Whisper models always download from **upstream HuggingFace** (e.g. `openai/whisper-tiny.en`). Pre-converted bundles on `Gadersd/whisper-burn` target Burn 0.8 and are incompatible with this workspace.

### Conversion pipeline

1. **Download** — `model.safetensors`, `config.json`, `tokenizer.json`, `preprocessor_config.json`
2. **Convert** (`convert/whisper.rs`):
   - Map HF tensor names to Burn module paths
   - Cast F16 weights to F32
   - Transpose linear layers: HF `[out, in]` → Burn 0.16 `[in, out]`
   - Write 1D flattened `.npy` files + `shapes.json`
3. **Record** (`convert/burn_mpk.rs`):
   - Load npy dump on WGPU
   - Save `model.mpk` via `NamedMpkGzFileRecorder`
   - Write `config.cfg` and `.burn_version` (`0.16.1`)

### Ready bundle contents

```
models/tiny_en/
├── model.mpk              # Burn weights
├── config.cfg             # Architecture config
├── config.json            # HF generation config (suppress_tokens, etc.)
├── tokenizer.json
├── .burn_version          # "0.16.1"
└── preprocessor_config.json
```

### Inference notes

- Mel spectrograms use Whisper dynamic-range normalization (not int16 ×32768 scaling)
- `suppress_tokens` and `begin_suppress_tokens` are loaded from `config.json`
- English-only models (`*_en`) omit language/task prompt tokens
- Decoder stops at first EOS; finished beam hypotheses are preferred

### Available Whisper models

| Name | HuggingFace | VRAM (est.) |
|------|-------------|-------------|
| `tiny` | openai/whisper-tiny | 400 MB |
| `tiny_en` | openai/whisper-tiny.en | 400 MB |
| `base` | openai/whisper-base | 600 MB |
| `base_en` | openai/whisper-base.en | 600 MB |
| `small` | openai/whisper-small | 1200 MB |
| `small_en` | openai/whisper-small.en | 1200 MB |
| `medium` | openai/whisper-medium | 2500 MB |
| `medium_en` | openai/whisper-medium.en | 2500 MB |
| `large-v3-turbo` | openai/whisper-large-v3-turbo | 6000 MB |
| `distil-medium-en` | distil-whisper/distil-medium.en | 1500 MB |
| `distil-large-v3` | distil-whisper/distil-large-v3 | 3000 MB |

## Parakeet TDT (working)

### Source

Parakeet TDT **v3** downloads transformers `model.safetensors` from `nvidia/parakeet-tdt-0.6b-v3`. **v2** is only published as a NeMo `.nemo` archive (`nvidia/parakeet-tdt-0.6b-v2`); we extract tensors into the same HF-style layout and convert them to Burn `.mpk`. Inference is Burn only — not ONNX Runtime (unlike [parakeet-rs](https://github.com/altunenes/parakeet-rs)).

### Conversion pipeline

1. **Download** — v3: `model.safetensors` + `config.json` + `tokenizer.json`. v2: official `.nemo` (weights lifted to the same layout; `vocab.txt` → `tokenizer.json`).
2. **Convert** (`convert/parakeet/`):
   - Map HF NeMo tensor names to Burn Conformer + TDT decoder paths
   - Transpose linear layers where Burn 0.16 layout differs
   - Write `parakeet_decode.json` (blank token, vocab, duration indices)
   - Run `save_model_from_npy()` and remove npy intermediates
3. **Record** — `model.mpk`, `config.cfg`, `.burn_version` (`0.16.1`)

### Ready bundle contents

```
models/parakeet-tdt-0.6b-v3/
├── model.mpk
├── config.cfg
├── config.json
├── tokenizer.json
├── parakeet_decode.json
└── .burn_version
```

### Inference notes

- Mel spectrograms use NeMo-style preprocessing (mask-aware utterance norm, int16 waveform scale)
- TDT greedy decode aligns with HF `ParakeetTDTGenerationMixin` (no RNN-T `max_symbols_per_step`)
- Post-decode token filter collapses CTC duplicates and drops leading garbage tokens
- Encoder has **no** spurious top-level `LayerNorm` (HF Parakeet has no `encoder/ln`)

### Parakeet models

| Name | HuggingFace | Status |
|------|-------------|--------|
| `parakeet-tdt-0.6b-v3` | nvidia/parakeet-tdt-0.6b-v3 | `burn_ready: true` |
| `parakeet-tdt-0.6b-v2` | nvidia/parakeet-tdt-0.6b-v2 | `burn_ready: true` |
| `parakeet-ctc-0.6b` | nvidia/parakeet-ctc-0.6b | `burn_ready: true` (encoder + CTC greedy) |
| `parakeet-ctc-1.1b` | nvidia/parakeet-ctc-1.1b | `burn_ready: true` (42-layer encoder + CTC greedy) |
| `t-one` | t-tech/T-one | `burn_ready: true` (8 kHz Conformer CTC, Russian, greedy) |

## Offline summarizer (Qwen3-0.6B)

Toggle **Summarize transcript** in the web UI (or `whisburn transcribe --summarize`) to run a small local English LLM after ASR. The transcript still downloads as usual; a sidecar `{stem}_summary.txt` is written next to it.

- **Model** — `qwen3-0.6b` (`Qwen/Qwen3-0.6B`), ~0.6B parameters, ~1.8 GB VRAM. Downloaded on first use.
- **Long files** — 4-hour calls are split into ~1800-word chunks, summarized, then merged. First load can take a minute while weights hit the GPU.
- **English only** — prompts and chunking assume English speech.

```bash
cargo run -p whisburn-cli -- models download qwen3-0.6b
cargo run -p whisburn-cli -- transcribe -i samples/jfk.wav --model tiny_en --format txt --summarize
```

## Qwen3-ASR (working)

[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR/) is a speech-to-text model with an audio encoder (thinker audio tower) and Qwen3 text decoder. Burn inference loads weights directly from `model.safetensors` (no `.mpk` conversion).

### Source

| Name | HuggingFace | VRAM (est.) |
|------|-------------|-------------|
| `qwen3-asr-0.6b` | Qwen/Qwen3-ASR-0.6B | 2500 MB |
| `qwen3-asr-1.7b` | Qwen/Qwen3-ASR-1.7B | 4500 MB |

### Download

```bash
cargo run -p whisburn-cli -- models download qwen3-asr-0.6b --verbose
```

Requires ~2 GB disk (0.6B) or ~4 GB (1.7B) and a stable network connection.

### Bundle layout

```
models/qwen3-asr-0.6b/
├── model.safetensors
├── config.json
├── preprocessor_config.json
├── tokenizer_config.json
├── generation_config.json
├── chat_template.json
├── vocab.json
├── merges.txt
├── qwen3_runtime.json      # Burn runtime metadata
└── .burn_version           # "0.16.1-qwen3"
```

### Inference notes

- Mel spectrograms use Whisper-style 128-bin preprocessing (not NeMo); last mel frame is dropped to match HF frame counts
- Audio tower: chunked conv (`n_window=50`), full-sequence encoder attention, `proj1/gelu/proj2` output (1024-dim for 0.6B)
- Thinker: GQA with per-head KV `repeat_interleave`, MRoPE (`mrope_section`), greedy decode with KV cache
- Chat prompt matches HF `chat_template.json` (`<|im_start|>`, `<|audio_pad|>`, etc.)
- Long audio is split at low-energy boundaries (HF-style, up to 1200 s per chunk)
- Language: auto-detect by default (`--language auto`). Pass a supported ISO code (e.g. `--language zh`) to force `language Chinese<asr_text>` in the assistant prompt
- Output is parsed from HF-style `language {Name}<asr_text>{transcript}`

Weights load directly from `thinker.audio_tower.*` and `thinker.model.*` keys in `model.safetensors` (no `.mpk` conversion).

### Transcribe

```bash
cargo run -p whisburn-cli -- transcribe -i samples/jfk.wav -m qwen3-asr-0.6b --language auto --verbose
```

Via HTTP (web UI at `http://localhost:8787/` or curl):

```bash
curl -X POST "http://localhost:8787/v1/transcribe?model=qwen3-asr-0.6b&language=auto&format=txt&download=true" \
  -F "audio=@samples/jfk.wav" -o jfk.txt
```

### Parity tests

HF reference dumps live in `temp_dump_qwen3-asr-0.6b/` (generate with `scripts/export_qwen3_hf_refs.py`):

```bash
cargo test -p whisburn-engine --test qwen3_thinker_parity
cargo test -p whisburn-engine --test qwen3_greedy_parity
cargo test -p whisburn-engine --test qwen3_audio_hf_mel
cargo test -p whisburn-engine --test qwen3_mel
```

## VibeVoice-ASR (Burn INT8 decoder)

Microsoft [VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) is a **9B-parameter** speech-to-text model (Qwen2.5-7B + dual causal conv encoders). It uses **raw 24 kHz waveform** input (not mel spectrograms).

The 7B language-model weights are **per-channel INT8** in Burn (not GGUF/ggml). That keeps inference on the Burn path while cutting decoder RAM ~4× versus f32. On-disk HuggingFace shards stay f32 (~16 GB); they are quantized while loading. GGUF runtimes (llama.cpp, VibeASR.cpp) are a different stack and are not used.

### Source

Downloads from **`microsoft/VibeVoice-ASR-HF`** (Transformers port with `tokenizer.json` and `processor_config.json`).

### Bundle layout

```
models/vibevoice-asr/
├── model-00001-of-00008.safetensors   # ~17 GB total (8 shards)
├── ...
├── model.safetensors.index.json
├── config.json
├── tokenizer.json
├── processor_config.json
├── vibevoice_runtime.json             # Burn runtime metadata
└── .burn_version                      # "0.16.1-vibevoice-stt"
```

### Download

```bash
cargo run -p whisburn-cli -- models download vibevoice-asr --verbose
```

Requires ~20 GB disk and a stable network connection. Speech encoders, connectors, and the **Qwen2.5-7B greedy STT decoder** are loaded from safetensors. The 7B decoder is ~20 GB in f32 — use `WHISBURN_DEVICE=cpu` (or `--device cpu`) if GPU VRAM is under ~20 GB.

### Architecture notes

| Component | HF prefix | Burn status |
|-----------|-----------|-------------|
| Acoustic encoder | `acoustic_tokenizer_encoder.*` | Loaded at runtime |
| Semantic encoder | `semantic_tokenizer_encoder.*` | Loaded at runtime |
| Speech connectors | `multi_modal_projector.{acoustic,semantic}_*` | Loaded at runtime |
| Qwen2.5-7B LLM | `language_model.model.*` | Greedy decode |
| LM head | `language_model.lm_head.weight` | Loaded |

Speech compression: **3200×** (7.5 Hz frames at 24 kHz). Audio normalized to **-25 dBFS**.

## VibeVoice-ASR-BitNet (Burn ternary 1.5B)

Microsoft [VibeVoice-ASR-BitNet](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet) is the 1.5B edge variant of VibeVoice-ASR. The official runtime is ggml (`VibeASR.cpp`); this project loads the **SafeTensors** checkpoint and runs it in Burn:

- Qwen2.5-1.5B decoder projections use BitNet-style **absmean ternary** `{-1,0,+1}` (I2_S).
- Dual 24 kHz VAE encoders stay f32 (decoder-only tensors are skipped).
- GGUF files in the HF repo are **not** downloaded.

### Source

Downloads from **`microsoft/VibeVoice-ASR-BitNet`**.

### Download

```bash
cargo run -p whisburn-cli -- models download bitnet-asr --verbose
```

Requires ~12 GB disk. The 1.5B ternary decoder is much smaller than the 7B INT8 path and can run on a 12 GB GPU.

## Planned models

| Name | Category | `burn_ready` |
|------|----------|--------------|
| `moonshine-tiny` | ASR | greedy STT (raw 16 kHz) |
| `moonshine-base` | ASR | greedy STT (raw 16 kHz, 61M) |
| `vibevoice-asr` | ASR | greedy STT (7B; prefer CPU if VRAM < 20 GB) |
| `bitnet-asr` | ASR | greedy STT (1.5B ternary I2_S, Burn) |
| `silero-vad` | VAD | false |
| `ten-vad` | VAD | false |
| `diarization-3.1` | Diarization | false |
| `nemo-diarization` | Diarization | false |

## Server audio transcoding

The HTTP server can transcode any uploaded audio to normalized WAV or Opus-in-Ogg without running ASR:

| Endpoint | Output | Notes |
|----------|--------|-------|
| `POST /v1/transcode?format=wav` | 16-bit mono WAV @ 16 kHz | Pure Rust (`hound`); no ffmpeg |
| `POST /v1/transcode?format=opus&bitrate=low\|medium` | Opus in Ogg container | Requires ffmpeg + `libopus` |
| `POST /v1/transcode/stream?bitrate=…` | Chunked `audio/ogg` | For in-browser `<audio>` preview |

The web UI at `/` uses the stream endpoint automatically when a file is selected.

## Disk usage

Conversion is disk-intensive. A single Whisper tiny model may temporarily need several GB during safetensors → npy → mpk conversion. After conversion, intermediate files can be removed with `--force` re-download or manual cleanup.

Build artifacts in `target/` can also be large (~10 GB for a full debug build). Safe to delete when not actively compiling:

```bash
# Windows PowerShell
Remove-Item -Recurse -Force target
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Garbled transcription | Rebuild CLI after code changes; verify mel stats (~min −0.5, max ~1.5 for Whisper) |
| `unknown model` | Run `models list`; name must match registry exactly |
| `Burn backend is not yet implemented` | Model has `burn_ready: false` |
| Stale weights after upgrade | `models download <name> --force --verbose` |
| `incompatible burn version` | Delete model dir and reconvert; check `.burn_version` is `0.16.1` |
| Opus preview fails in web UI | Install ffmpeg with `libopus`; WAV transcode still works without it |
| Qwen3 garbage tokens | Rebuild after engine fixes; run `qwen3_greedy_parity` against HF refs |