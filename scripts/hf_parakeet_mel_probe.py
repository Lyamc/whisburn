"""Probe HF Parakeet logits with mask-correct mel."""
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForTDT, AutoProcessor

audio, sr = sf.read("samples/jfk.wav")
if audio.ndim > 1:
    audio = audio[:, 0]

processor = AutoProcessor.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
model = AutoModelForTDT.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
model.eval()

inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
hf_mel = inputs.input_features
print("processor mel max", hf_mel.max().item())

# Mask-correct mel (matches updated rust pipeline)
wave = torch.tensor(audio, dtype=torch.float32)[None, :]
pre = torch.cat([wave[:, :1], wave[:, 1:] - 0.97 * wave[:, :-1]], 1)
window = torch.hann_window(400, periodic=False)
stft = torch.stft(pre, 512, hop_length=160, win_length=400, window=window, return_complex=True, center=True, pad_mode="constant")
mag = torch.sqrt(torch.view_as_real(stft).pow(2).sum(-1)).pow(2)
mel_f = processor.feature_extractor.mel_filters
log_mel = torch.log(mel_f @ mag + 2**-24).permute(0, 2, 1)
fl = int(inputs.attention_mask.sum().item())
mask = torch.arange(log_mel.shape[1])[None, :] < fl
masked = log_mel * mask.unsqueeze(-1)
mean = masked.sum(1, keepdim=True) / fl
var = ((masked - mean) ** 2 * mask.unsqueeze(-1)).sum(1) / max(fl - 1, 1)
std = torch.sqrt(var).unsqueeze(1)
rust_style = (log_mel - mean) / (std + 1e-5)
rust_style = rust_style * mask.unsqueeze(-1)
print("rust-style mel max", rust_style.max().item(), "diff vs processor", (hf_mel - rust_style).abs().max().item())

# Greedy first steps with processor mel
with torch.no_grad():
    enc_out = model.get_audio_features(input_features=hf_mel, attention_mask=inputs.attention_mask)
    cache = None
    token = torch.tensor([[2]], dtype=torch.long)
    frame = 0
    valid = int(enc_out.attention_mask.sum().item())
    for step in range(12):
        enc_frame = enc_out.pooler_output[:, frame:frame+1, :]
        enc_slice = model._prepare_encoder_outputs_for_generation(enc_out, torch.tensor([frame]))
        # manual single step
        dec_out = model.decoder(token, cache=cache if cache else model.decoder.forward.__self__)
        break