"""Manual HF TDT greedy steps for parity debugging."""
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
with torch.no_grad():
    enc = model.get_audio_features(
        input_features=inputs.input_features,
        attention_mask=inputs.attention_mask,
    )
valid = int(enc.attention_mask.sum().item())
print(f"enc frames={enc.pooler_output.shape[1]} valid={valid}")

from transformers.models.parakeet.generation_parakeet import ParakeetRNNTDecoderCache

cache = ParakeetRNNTDecoderCache(model.config)
token_id = model.config.pad_token_id
frame = 0
blank = model.config.blank_token_id
vocab = model.config.vocab_size

for step in range(15):
    enc_frame = enc.pooler_output[:, frame : frame + 1, :]
    dec_in = torch.tensor([[token_id]], dtype=torch.long)
    dec = model.decoder(dec_in, cache=cache)
    logits = model.joint(
        encoder_hidden_states=enc_frame.unsqueeze(2),
        decoder_hidden_states=dec.unsqueeze(1),
    ).squeeze(2)[0, 0]
    tok = int(logits[:vocab].argmax().item())
    dur = int(logits[vocab:].argmax().item())
    if tok == blank and dur == 0:
        dur = 1
    print(f"step {step}: frame={frame} token={tok} dur={dur}")
    token_id = tok
    frame += dur
    if frame >= valid:
        break