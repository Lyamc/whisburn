use std::path::Path;

use hf_hub::api::sync::Api;

use crate::convert::granite::prepare_granite_bundle;
use crate::sources::DownloadSource;

use super::fetch::{fetch_optional_hf_file, fetch_required_hf_file};
use super::options::DownloadOptions;

const REQUIRED: &[&str] = &["config.json", "model.safetensors", "tokenizer.json"];
const OPTIONAL: &[&str] = &[
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
];

pub fn download_granite_hf(
    api: &Api,
    source: &DownloadSource,
    dir: &Path,
    name: &str,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    let repo = api.model(source.repo_id().to_string());
    if options.verbose {
        tracing::info!(
            "downloading Granite FP8 weights from {} (about 4 GB)",
            source.repo_id()
        );
    }
    for file in REQUIRED {
        fetch_required_hf_file(&repo, source.repo_id(), file, dir, options)?;
    }
    for file in OPTIONAL {
        let _ = fetch_optional_hf_file(&repo, source.repo_id(), file, dir, options)?;
    }
    prepare_granite_bundle(dir, name, options)
}
