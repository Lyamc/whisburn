"""Export HF mel + encoder outputs for rust parity tests."""
from pathlib import Path
import os
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForTDT, AutoProcessor

# Avoid shadowing HF repo id with models/parakeet-tdt-0.6b-v3 in project root.
os.chdir(Path(__file__).resolve().parent)

root = Path(__file__).resolve().parent.parent
out = root / "temp_dump_parakeet-tdt-0.6b-v3"
out.mkdir(parents=True, exist_ok=True)

audio, sr = sf.read(root / "samples/jfk.wav")
if audio.ndim > 1:
    audio = audio[:, 0]

processor = AutoProcessor.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
model = AutoModelForTDT.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
model.eval()

inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
with torch.no_grad():
    enc = model.get_audio_features(
        input_features=inputs.input_features,
        attention_mask=inputs.attention_mask,
    )

mel = inputs.input_features[0].numpy().astype(np.float32)  # [frames, mels]
pooler = enc.pooler_output[0].numpy().astype(np.float32)  # [enc, 640]
pre_proj = enc.last_hidden_state[0].numpy().astype(np.float32)  # [enc, 1024]
# Burn expects [mels, frames] contiguous for [1, mels, frames].
mel_bct = np.transpose(mel, (1, 0)).astype(np.float32)
np.save(out / "hf_mel_bct.npy", mel_bct.reshape(-1))
np.save(out / "hf_enc_pooler.npy", pooler.reshape(-1))
np.save(out / "hf_enc_preproj.npy", pre_proj.reshape(-1))
print(
    "saved",
    mel.shape,
    pooler.shape,
    "mel_max",
    mel.max(),
    "preproj0",
    np.linalg.norm(pre_proj[0]),
    "pooler0",
    np.linalg.norm(pooler[0]),
)