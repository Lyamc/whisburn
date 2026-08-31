"""Compare HF vs local mel pipeline stats for JFK."""
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor

wav_path = "samples/jfk.wav"
audio, sr = sf.read(wav_path)
if audio.ndim > 1:
    audio = audio[:, 0]

processor = AutoProcessor.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
hf = inputs.input_features[0].numpy()

# Reproduce HF steps manually for diff debugging
wave = torch.tensor(audio, dtype=torch.float32)[None, :]
pre = torch.cat([wave[:, :1], wave[:, 1:] - 0.97 * wave[:, :-1]], dim=1)
window = torch.hann_window(400, periodic=False)
stft = torch.stft(pre, 512, hop_length=160, win_length=400, window=window, return_complex=True, center=True, pad_mode="constant")
mag = torch.view_as_real(stft)
mag = torch.sqrt(mag.pow(2).sum(-1)).pow(2)
mel_filters = processor.feature_extractor.mel_filters
log_mel = torch.log(mel_filters @ mag + 2**-24).permute(0, 2, 1)
n_frames = log_mel.shape[1]
mean = log_mel.sum(dim=1, keepdim=True) / n_frames
var = ((log_mel - mean) ** 2).sum(dim=1, keepdim=True) / max(n_frames - 1, 1)
norm = (log_mel - mean) / (var.sqrt() + 1e-5)
manual = norm[0].numpy()

print(f"HF processor: shape={hf.shape} max={hf.max():.6f} min={hf.min():.6f} mean={hf.mean():.6f}")
print(f"Manual repro: shape={manual.shape} max={manual.max():.6f} min={manual.min():.6f} mean={manual.mean():.6f}")
print(f"Max abs diff processor vs manual: {np.abs(hf-manual).max():.6f}")

# Export slice for rust test if needed
np.save("temp_dump_parakeet-tdt-0.6b-v3/hf_mel.npy", hf)
print("Wrote temp_dump_parakeet-tdt-0.6b-v3/hf_mel.npy")