# Audio samples

Test and demo audio bundled with whisburn (no external reference repos required).

| File | Description |
|------|-------------|
| `jfk.wav` | JFK inaugural excerpt (~11 s); used by integration and parity tests |

## CLI

```bash
cargo run -p whisburn-cli -- transcribe --input samples/jfk.wav --model tiny_en
cargo run -p whisburn-cli -- transcribe -i samples/jfk.wav -m qwen3-asr-0.6b --language auto --verbose
```

## HTTP server / web UI

```bash
cargo run -p whisburn-cli -- serve --port 8787 --model qwen3-asr-0.6b
```

Open http://localhost:8787/, upload `jfk.wav`, choose a download format (JSON, TXT, SRT, VTT), and click **Transcribe & download**.

**curl:**

```bash
curl -X POST "http://localhost:8787/v1/transcribe?model=qwen3-asr-0.6b&format=srt&download=true" \
  -F "audio=@samples/jfk.wav" -o jfk.srt
```