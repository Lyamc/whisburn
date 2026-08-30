"""HF Parakeet TDT baseline for JFK sample."""
import json
import sys

def main():
    try:
        import torch
        import soundfile as sf
        from transformers import AutoModelForTDT, AutoProcessor
    except ImportError as e:
        print(f"MISSING_DEPS: {e}", file=sys.stderr)
        sys.exit(2)

    wav_path = "samples/jfk.wav"
    model_id = "nvidia/parakeet-tdt-0.6b-v3"

    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForTDT.from_pretrained(model_id)
    model.eval()

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, return_dict_in_generate=True)

    text = processor.decode(out.sequences, skip_special_tokens=True)
    print("HF_TEXT:", text[0] if isinstance(text, list) else text)

    # Mel stats for parity
    feats = inputs.input_features
    print(f"HF_MEL: shape={tuple(feats.shape)} max={feats.max().item():.6f} mean={feats.mean().item():.6f}")

    # First 20 token ids
    seq = out.sequences[0].tolist()
    print("HF_TOKENS_FIRST20:", seq[:20])
    print("HF_TOKEN_COUNT:", len(seq))

if __name__ == "__main__":
    main()