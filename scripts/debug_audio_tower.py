#!/usr/bin/env python3
"""Bisect Qwen3 audio tower vs Rust."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from qwen_asr import Qwen3ASRModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "temp_dump_qwen3-asr-0.6b"


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt((d * d).mean()))


def main() -> int:
    mel = np.fromfile(OUT / "hf_mel.bin", dtype=np.float32)
    hf_out = np.fromfile(OUT / "hf_audio_features.bin", dtype=np.float32)
    mel = mel.reshape(128, -1)
    print("mel shape", mel.shape, "hf_out", hf_out.shape)

    model = Qwen3ASRModel.from_pretrained(
        str(ROOT / "models" / "qwen3-asr-0.6b"),
        dtype="float32",
        device_map="cpu",
    )
    tower = model.model.thinker.audio_tower
    print("attn impl", tower.config._attn_implementation)

    # HF audio_tower expects [n_mels, frames] per utterance (not batched 3D).
    feat = torch.from_numpy(mel).float()
    feat_len = torch.tensor([mel.shape[1]], dtype=torch.long)

    with torch.no_grad():
        full = tower(feat, feature_lens=feat_len).last_hidden_state.numpy().reshape(-1)
    print("full rmse", rmse(full, hf_out))

    # Re-run forward pieces manually (mirror HF forward)
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import _get_feat_extract_output_lengths

    feature_lens = feat_len
    aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
    chunk_num = torch.ceil(feature_lens / (tower.n_window * 2)).long()
    chunk_lengths = torch.tensor(
        [tower.n_window * 2] * chunk_num.sum(), dtype=torch.long, device=feature_lens.device
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % (tower.n_window * 2)
    chunk_lengths[chunk_lengths == 0] = tower.n_window * 2
    print("chunk_lengths", chunk_lengths.tolist(), "aftercnn", aftercnn_lens.tolist())

    chunk_list = feat.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
    padded_mask_after_cnn = torch.nn.utils.rnn.pad_sequence(
        [torch.ones(length, dtype=torch.bool) for length in feature_lens_after_cnn],
        batch_first=True,
    )
    padded_feature = padded_feature.unsqueeze(1)

    padded_embeds = []
    for chunk in padded_feature.split(tower.conv_chunksize, dim=0):
        x = F.gelu(tower.conv2d1(chunk))
        x = F.gelu(tower.conv2d2(x))
        x = F.gelu(tower.conv2d3(x))
        padded_embeds.append(x)
    padded_embed = torch.cat(padded_embeds, dim=0)
    b, c, f, t = padded_embed.size()
    print("conv out b,c,f,t", b, c, f, t)

    correct = tower.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
    wrong = tower.conv_out(padded_embed.transpose(1, 3).contiguous().view(b, t, c * f))
    pe = tower.positional_embedding.positional_embedding[: correct.shape[1], :].unsqueeze(0)
    correct = correct + pe
    wrong = wrong + pe

    def flatten_hidden(padded, mask):
        hs = padded[mask]
        return hs

    hs_correct = flatten_hidden(correct, padded_mask_after_cnn)
    hs_wrong = flatten_hidden(wrong, padded_mask_after_cnn)
    print(
        "post-conv+pe rmse correct vs wrong",
        rmse(hs_correct.detach().numpy(), hs_wrong.detach().numpy()),
    )

    # encoder layers
    window_aftercnn = padded_mask_after_cnn.shape[-1] * (tower.n_window_infer // (tower.n_window * 2))
    cu_chunk_lens = [0]
    for cnn_len in aftercnn_lens:
        cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
        remainder = cnn_len % window_aftercnn
        if remainder != 0:
            cu_chunk_lens += [remainder]
    cu_seqlens = torch.tensor(cu_chunk_lens, dtype=torch.int32).cumsum(-1)
    print("cu_seqlens", cu_seqlens.tolist(), "window_aftercnn", window_aftercnn)

    hidden = hs_correct
    for i, layer in enumerate(tower.layers):
        hidden = layer(hidden, cu_seqlens)[0]
        if i in (0, 31):
            print(f"layer{i} hidden std", float(hidden.std()))

    hidden = tower.ln_post(hidden)
    hidden = tower.proj2(tower.act(tower.proj1(hidden)))
    print("manual rmse", rmse(hidden.detach().numpy().reshape(-1), hf_out))

    # block mask vs no mask on layer 0 only
    seq = hs_correct.shape[0]
    min_val = torch.finfo(hs_correct.dtype).min
    block = torch.full([1, 1, seq, seq], min_val)
    for i in range(1, len(cu_seqlens)):
        s, e = cu_seqlens[i - 1], cu_seqlens[i]
        block[..., s:e, s:e] = 0
    layer0 = tower.layers[0]
    h0_nomask = layer0(hs_correct.clone(), cu_seqlens)[0]
    # force mask via internal attn eager
    h_ln = layer0.self_attn_layer_norm(hs_correct)
    q = layer0.self_attn.q_proj(h_ln).reshape(seq, -1, layer0.self_attn.head_dim).transpose(0, 1).unsqueeze(0)
    k = layer0.self_attn.k_proj(h_ln).reshape(seq, -1, layer0.self_attn.head_dim).transpose(0, 1).unsqueeze(0)
    v = layer0.self_attn.v_proj(h_ln).reshape(seq, -1, layer0.self_attn.head_dim).transpose(0, 1).unsqueeze(0)
    scores = torch.matmul(q, k.transpose(2, 3)) * layer0.self_attn.scaling
    scores_block = scores + block
    attn_block = torch.softmax(scores_block, dim=-1, dtype=torch.float32)
    out_block = torch.matmul(attn_block, v).transpose(1, 2).reshape(seq, -1)
    out_block = layer0.self_attn.out_proj(out_block)
    h0_block = hs_correct + out_block
    print("layer0 nomask vs block rmse", rmse(h0_nomask.detach().numpy(), h0_block.detach().numpy()))

    hs_correct.detach().numpy().astype(np.float32).tofile(OUT / "hf_audio_pre_layers.bin")
    ln0 = tower.layers[0].self_attn_layer_norm(hs_correct)
    ln0.detach().numpy().astype(np.float32).tofile(OUT / "hf_audio_layer0_attn_ln.bin")
    attn0 = tower.layers[0].self_attn(hidden_states=ln0, cu_seqlens=cu_seqlens)
    attn0.detach().numpy().astype(np.float32).tofile(OUT / "hf_audio_layer0_attn.bin")

    sa = tower.layers[0].self_attn
    seq_length = ln0.shape[0]
    q = sa.q_proj(ln0).reshape(seq_length, sa.num_heads, -1).transpose(0, 1).unsqueeze(0)
    k = sa.k_proj(ln0).reshape(seq_length, sa.num_heads, -1).transpose(0, 1).unsqueeze(0)
    v = sa.v_proj(ln0).reshape(seq_length, sa.num_heads, -1).transpose(0, 1).unsqueeze(0)
    scores = torch.matmul(q, k.transpose(2, 3)) * sa.scaling
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn, v).transpose(1, 2).reshape(seq_length, -1)
    rust_style = sa.out_proj(out)
    print("hf attn vs rust-style manual rmse", rmse(attn0.detach().numpy(), rust_style.detach().numpy()))
    q.detach().numpy().astype(np.float32).tofile(OUT / "hf_audio_layer0_q.bin")
    k.detach().numpy().astype(np.float32).tofile(OUT / "hf_audio_layer0_k.bin")
    scores.detach().numpy().astype(np.float32).tofile(OUT / "hf_audio_layer0_scores.bin")
    h0_sdpa = tower.layers[0](hs_correct, cu_seqlens)[0]
    h0_sdpa.detach().numpy().astype(np.float32).tofile(OUT / "hf_audio_post_layer0.bin")

    saved = tower.config._attn_implementation
    tower.config._attn_implementation = "eager"
    h0_eager = tower.layers[0](hs_correct, cu_seqlens)[0]
    tower.config._attn_implementation = saved
    print("layer0 sdpa vs eager rmse", rmse(h0_sdpa.detach().numpy(), h0_eager.detach().numpy()))

    # SDPA (reference) vs eager+block mask full tower
    def run_layers(hidden, impl: str, use_block_mask: bool):
        saved = tower.config._attn_implementation
        tower.config._attn_implementation = impl
        try:
            h = hidden
            seq_len = h.shape[0]
            mask = None
            if use_block_mask:
                min_val = torch.finfo(h.dtype).min
                mask = torch.full([1, 1, seq_len, seq_len], min_val)
                for i in range(1, len(cu_seqlens)):
                    s, e = int(cu_seqlens[i - 1]), int(cu_seqlens[i])
                    mask[..., s:e, s:e] = 0
            for layer in tower.layers:
                if impl == "eager" and use_block_mask:
                    h = layer(h, cu_seqlens, attention_mask=mask)[0]
                else:
                    h = layer(h, cu_seqlens)[0]
            h = tower.ln_post(h)
            h = tower.proj2(tower.act(tower.proj1(h)))
            return h.detach().numpy().reshape(-1)
        finally:
            tower.config._attn_implementation = saved

    sdpa_out = run_layers(hs_correct, "sdpa", False)
    eager_block = run_layers(hs_correct, "eager", True)
    eager_full = run_layers(hs_correct, "eager", False)
    print("sdpa vs eager_block rmse", rmse(sdpa_out, eager_block))
    print("sdpa vs eager_full rmse", rmse(sdpa_out, eager_full))
    print("post-conv+pe first8", hs_correct[0, :8].detach().numpy().tolist())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())