use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::model::vibevoice::{
    build_asr_prompt, load_vibevoice_runtime, normalize_dbfs, VibeVoiceASR,
};
use crate::token::Gpt2Tokenizer;

pub fn decode_vibevoice<B: Backend>(
    model: &VibeVoiceASR<B>,
    bpe: &Gpt2Tokenizer,
    waveform: &[f32],
    sample_rate: usize,
    model_name: &str,
    verbose: bool,
) -> (String, Vec<usize>) {
    let runtime = load_vibevoice_runtime(model_name);
    let target_rate = runtime.sample_rate;

    let waveform = if sample_rate == target_rate {
        waveform.to_vec()
    } else {
        o3whisburn_audio::resample_mono(waveform, sample_rate, target_rate)
            .unwrap_or_else(|_| waveform.to_vec())
    };

    let normalized = normalize_dbfs(&waveform, runtime.target_dbfs);
    let prompt = build_asr_prompt(&runtime, normalized.len(), None);
    let device = model.decoder.lm_head.device();
    let audio = Tensor::<B, 1>::from_floats(normalized.as_slice(), &device)
        .reshape([1, 1, normalized.len()]);
    let speech = model.encode_speech(audio);
    let [_, speech_tokens, _] = speech.dims();

    if verbose {
        println!(
            "DEBUG VibeVoice: {} samples @ {}Hz, {} speech tokens, {} prompt pads",
            normalized.len(),
            target_rate,
            speech_tokens,
            prompt.vae_token_count
        );
    }

    let pad_count = speech_tokens.max(1);
    let pads = runtime.audio_pad_token.repeat(pad_count);
    let prompt_text = format!(
        "<|im_start|>system\n{}<|im_end|>\n\
         <|im_start|>user\n{}{}{}\n{}<|im_end|>\n\
         <|im_start|>assistant\n",
        prompt.system_prompt,
        runtime.audio_bos_token,
        pads,
        runtime.audio_eos_token,
        prompt.user_suffix
    );
    let prefix_ids = bpe.encode_with_special_tokens(&prompt_text, false);
    let pad_id = bpe
        .token_to_id(&runtime.audio_pad_token)
        .or_else(|| bpe.token_to_id("<|box_start|>"))
        .unwrap_or(0);
    let pad_positions = prefix_ids.iter().filter(|&&id| id == pad_id).count();
    let eos_ids: Vec<usize> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|t| bpe.token_to_id(t))
        .collect();

    if verbose {
        println!(
            "DEBUG VibeVoice: pad_id={pad_id}, encoded pads={pad_positions}/{}, prefix={}, eos={eos_ids:?}",
            pad_count,
            prefix_ids.len()
        );
    }

    let tokens = model.decoder.generate_greedy(
        &prefix_ids,
        speech,
        pad_id,
        &eos_ids,
        256,
        &device,
    );
    let text = unwrap_vibevoice_text(&bpe.decode(&tokens, true).unwrap_or_default());
    (text, tokens)
}

fn unwrap_vibevoice_text(text: &str) -> String {
    let trimmed = text.trim();
    let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return text.to_string();
    };
    if let Some(items) = value.as_array() {
        let parts: Vec<String> = items
            .iter()
            .filter_map(|item| item.get("Content").and_then(|c| c.as_str()).map(str::to_string))
            .collect();
        if !parts.is_empty() {
            return parts.join(" ");
        }
    }
    if let Some(content) = value.get("Content").and_then(|c| c.as_str()) {
        return content.to_string();
    }
    text.to_string()
}