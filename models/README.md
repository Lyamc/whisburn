# Model cache

Downloaded and converted models are stored here by default. Override the location with:

```bash
export WHISBURN_MODELS_DIR=/path/to/models
```

## Prepare a model

```bash
cargo run -p whisburn-cli -- models download tiny_en --verbose
```

Use `--force` to clear stale conversion artifacts and rebuild the Burn bundle.

Each model directory typically contains:

| File | Purpose |
|------|---------|
| `model.mpk` | Burn weight record (Burn 0.21.0) — Whisper, Parakeet |
| `model.safetensors` | Direct runtime weights — Qwen3-ASR (no mpk) |
| `config.cfg` / `{name}.cfg` | Model architecture config |
| `config.json` | HuggingFace generation config (suppress tokens, etc.) |
| `tokenizer.json` | GPT-2 BPE tokenizer (Whisper, Parakeet) or Qwen tokenizer assets |
| `parakeet_decode.json` | Parakeet TDT decode metadata (blank, vocab, durations) |
| `qwen3_runtime.json` | Qwen3-ASR Burn runtime metadata |
| `vibevoice-asr-q4_k.gguf` | VibeVoice 7B Q4_K weights (dequantized to Burn INT8 at load; experimental) |
| `vibevoice_runtime.json` | VibeVoice Burn runtime metadata |
| `.burn_version` | Records the Burn version used for conversion |

Intermediate files (`model.safetensors`, `encoder/*.npy`, `decoder/*.npy`) are removed after Whisper/Parakeet conversion. Qwen3 keeps `model.safetensors` as the runtime weight source.

### Whisper example

```bash
cargo run -p whisburn-cli -- models download tiny_en --verbose
cargo run -p whisburn-cli -- transcribe -i samples/jfk.wav -m tiny_en -v
```

### Parakeet example

```bash
cargo run -p whisburn-cli -- models download parakeet-tdt-0.6b-v3 --verbose
cargo run -p whisburn-cli -- transcribe -i samples/jfk.wav -m parakeet-tdt-0.6b-v3 -v
```

### Qwen3-ASR example

```bash
cargo run -p whisburn-cli -- models download qwen3-asr-0.6b --verbose
cargo run -p whisburn-cli -- transcribe -i samples/jfk.wav -m qwen3-asr-0.6b --language auto -v
```

See [MODELS.md](../MODELS.md) for the full pipeline.