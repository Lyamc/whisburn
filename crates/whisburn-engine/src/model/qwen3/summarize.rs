use burn::tensor::{Device, DeviceKind};
use std::sync::Arc;


use whisburn_core::{WhisburnError, WhisburnResult};

use crate::token::Gpt2Tokenizer;

use super::thinker::Qwen3ThinkerConfig;
use super::{load_qwen3_lm_weights, load_qwen3_runtime, Qwen3Thinker};

pub const DEFAULT_SUMMARIZER_MODEL: &str = "qwen3-0.6b";

/// ~600 English words ≈ 800 tokens. A 1,800-word chunk's prefill still pushed
/// available memory under earlyoom's 10% line once the summarizer was already resident.
pub const CHUNK_WORDS: usize = 600;
const CHUNK_MAX_NEW_TOKENS: usize = 256;
const MERGE_MAX_NEW_TOKENS: usize = 384;

pub type SummarizeProgress = Arc<dyn Fn(usize, usize, &str) + Send + Sync>;

const SYSTEM_PROMPT: &str = "You summarize English speech transcripts. Be concise. Use short bullet points. Cover: what it is about, key points, decisions, and action items. Do not invent facts.";

pub fn chunk_transcript(text: &str, max_words: usize) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= max_words {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < words.len() {
        let mut end = (start + max_words).min(words.len());
        if end < words.len() {
            // Prefer a sentence boundary in the last quarter of the window.
            let search_from = start + (max_words * 3 / 4);
            if let Some(rel) = words[search_from..end]
                .iter()
                .rposition(|w| w.ends_with('.') || w.ends_with('?') || w.ends_with('!'))
            {
                end = search_from + rel + 1;
            }
        }
        chunks.push(words[start..end].join(" "));
        start = end;
    }
    chunks
}

pub fn strip_think(text: &str) -> String {
    let mut out = text.to_string();
    loop {
        let Some(a) = out.find("<think>") else { break };
        let close = "</think>";
        if let Some(rel) = out[a..].find(close) {
            out.replace_range(a..a + rel + close.len(), "");
        } else {
            out.replace_range(a.., "");
            break;
        }
    }
    out.trim().to_string()
}

fn build_prompt(user: &str) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user}\n/no_think<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn generate_completion(
    thinker: &Qwen3Thinker,
    tokenizer: &Gpt2Tokenizer,
    prompt: &str,
    max_new: usize,
    device: &Device,
) -> WhisburnResult<String> {
    let ids = tokenizer.encode_with_special_tokens(prompt, false);
    let eos = {
        let mut ids = vec![151_643usize, 151_645];
        if let Some(id) = tokenizer.token_to_id("<|endoftext|>") {
            ids.push(id);
        }
        if let Some(id) = tokenizer.token_to_id("<|im_end|>") {
            ids.push(id);
        }
        ids
    };
    let gen = thinker.generate_greedy_text(&ids, &eos, max_new, device);
    let raw = tokenizer
        .decode(&gen, true)
        .map_err(|e| WhisburnError::Inference(e.to_string()))?;
    Ok(strip_think(&raw))
}

pub fn summarize_with_thinker(
    thinker: &Qwen3Thinker,
    tokenizer: &Gpt2Tokenizer,
    text: &str,
    device: &Device,
    progress: Option<SummarizeProgress>,
) -> WhisburnResult<String> {
    let chunks = chunk_transcript(text, CHUNK_WORDS);
    if chunks.is_empty() {
        return Ok(String::new());
    }

    let n = chunks.len();
    let mut partials = Vec::with_capacity(n);
    for (i, chunk) in chunks.iter().enumerate() {
        if let Some(cb) = &progress {
            cb(
                i,
                n + usize::from(n > 1),
                &format!("Summarizing section {} of {n}", i + 1),
            );
        }
        let user = if n == 1 {
            format!("Summarize this English transcript:\n\n{chunk}")
        } else {
            format!(
                "This is section {} of {n} of a long English transcript. Summarize only this section:\n\n{chunk}",
                i + 1
            )
        };
        let part = generate_completion(
            thinker,
            tokenizer,
            &build_prompt(&user),
            CHUNK_MAX_NEW_TOKENS,
            device,
        )?;
        trim_allocator();
        if !part.is_empty() {
            partials.push(part);
        }
    }

    if partials.len() <= 1 {
        return Ok(partials.pop().unwrap_or_default());
    }

    // Merge in groups that stay inside the same word budget as a transcript
    // chunk, so the merge prompt cannot grow into a second full-length prefill.
    let mut guard = 0usize;
    while partials.len() > 1 {
        guard += 1;
        if guard > partials.len() + 4 {
            break;
        }
        if let Some(cb) = &progress {
            cb(n, n + 1, "Merging section summaries");
        }
        let groups = group_partials(&partials, CHUNK_WORDS);
        if groups.len() == 1 {
            let merged = merge_summary_group(thinker, tokenizer, &partials, device)?;
            return Ok(merged);
        }
        let mut next = Vec::with_capacity(groups.len());
        for group in groups {
            if group.len() == 1 {
                next.push(partials[group[0]].clone());
            } else {
                let subset: Vec<String> = group.iter().map(|&i| partials[i].clone()).collect();
                next.push(merge_summary_group(thinker, tokenizer, &subset, device)?);
            }
        }
        if next.len() >= partials.len() {
            let merged = merge_summary_group(thinker, tokenizer, &partials, device)?;
            return Ok(merged);
        }
        partials = next;
    }
    Ok(partials.pop().unwrap_or_default())
}

fn merge_summary_group(
    thinker: &Qwen3Thinker,
    tokenizer: &Gpt2Tokenizer,
    partials: &[String],
    device: &Device,
) -> WhisburnResult<String> {
    let joined = partials
        .iter()
        .enumerate()
        .map(|(i, p)| format!("Section {}:\n{p}", i + 1))
        .collect::<Vec<_>>()
        .join("\n\n");
    let merge_user = format!(
        "Combine these section summaries of a long English conversation into one concise summary with: overview, key points, decisions, action items.\n\n{joined}"
    );
    let merged = generate_completion(
        thinker,
        tokenizer,
        &build_prompt(&merge_user),
        MERGE_MAX_NEW_TOKENS,
        device,
    )?;
    trim_allocator();
    Ok(merged)
}

/// Groups whose combined word count stays within `max_words`.
pub fn group_partials(partials: &[String], max_words: usize) -> Vec<Vec<usize>> {
    let mut groups = Vec::new();
    let mut cur = Vec::new();
    let mut words = 0usize;
    for (i, part) in partials.iter().enumerate() {
        let w = part.split_whitespace().count().max(1);
        if !cur.is_empty() && words.saturating_add(w) > max_words {
            groups.push(std::mem::take(&mut cur));
            words = 0;
        }
        cur.push(i);
        words = words.saturating_add(w);
    }
    if !cur.is_empty() {
        groups.push(cur);
    }
    groups
}

fn trim_allocator() {
    #[cfg(target_os = "linux")]
    unsafe {
        extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        let _ = malloc_trim(0);
    }
}

pub fn load_summarizer(
    model_name: &str,
    device: &Device,
    verbose: bool,
) -> WhisburnResult<(Gpt2Tokenizer, Qwen3Thinker)> {
    let runtime = load_qwen3_runtime(model_name);
    let cfg = Qwen3ThinkerConfig {
        hidden_size: runtime.text_hidden_size,
        n_layers: runtime.text_layers,
        n_heads: runtime.text_heads,
        n_kv_heads: runtime.text_kv_heads,
        head_dim: runtime.text_head_dim,
        intermediate_size: runtime.text_intermediate_size,
        vocab_size: runtime.vocab_size,
        rms_norm_eps: runtime.rms_norm_eps,
        rope_theta: runtime.rope_theta,
        mrope_section: runtime.mrope_section,
    };
    let thinker = cfg.init(device);
    let thinker = load_qwen3_lm_weights(
        thinker,
        &whisburn_core::resolve_model_dir(model_name),
        device,
        verbose,
    )
    .map_err(|e| WhisburnError::Model(e.to_string()))?;
    let tokenizer = Gpt2Tokenizer::new(model_name)
        .map_err(|e| WhisburnError::Model(format!("summarizer tokenizer: {e}")))?;
    Ok((tokenizer, thinker))
}

#[cfg(test)]
mod tests {
    use super::{chunk_transcript, group_partials, strip_think, CHUNK_WORDS};

    #[test]
    fn chunks_long_english_on_sentence_boundaries() {
        let mut text = String::new();
        for i in 0..4000 {
            text.push_str(&format!("This is sentence number {i}. "));
        }
        let chunks = chunk_transcript(&text, CHUNK_WORDS);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|c| !c.is_empty()));
        let total: usize = chunks.iter().map(|c| c.split_whitespace().count()).sum();
        assert_eq!(total, text.split_whitespace().count());
    }

    #[test]
    fn short_text_is_one_chunk() {
        let chunks = chunk_transcript("Hello there. This is short.", 1800);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn strips_think_blocks() {
        let raw = "<think>\nplan\n</think>\n- Point one\n- Point two";
        assert_eq!(strip_think(raw), "- Point one\n- Point two");
    }

    #[test]
    fn merge_groups_stay_within_the_word_budget() {
        let partials: Vec<String> = (0..5)
            .map(|i| format!("section {i} {}", "word ".repeat(100)))
            .collect();
        let groups = group_partials(&partials, 250);
        assert!(groups.len() >= 3);
        let flat: Vec<usize> = groups.iter().flatten().copied().collect();
        assert_eq!(flat, vec![0, 1, 2, 3, 4]);
        for group in &groups {
            let words: usize = group
                .iter()
                .map(|&i| partials[i].split_whitespace().count())
                .sum();
            assert!(words <= 250);
        }
    }
}
