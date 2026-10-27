use burn::tensor::Device;

use burn::config::Config;
use burn::module::Module;
use whisburn_engine::model::load::{init_shapes, load_model_from_npy};

use crate::download::DownloadOptions;
use crate::paths::model_dir;

pub fn save_model_from_npy(name: &str, options: &DownloadOptions) -> anyhow::Result<()> {
    let dump_dir = model_dir(name);
    if !dump_dir.join("encoder").exists() && !dump_dir.join("decoder").exists() {
        anyhow::bail!("no npy dump at {} for conversion", dump_dir.display());
    }

    options.report(crate::download::PrepProgress::convert(
        format!("Building Burn weight file for '{name}'"),
        0.8,
    ));
    if options.verbose {
        tracing::info!("converting npy dump to burnpack for '{name}'");
    }

    // Packing reads weights back to the host. Flex does that without a GPU adapter.
    let device = Device::flex();
    let dump_str = dump_dir
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("invalid model path"))?;
    init_shapes(dump_str);
    let (model, model_config) = load_model_from_npy(dump_str, &device)
        .map_err(|e| anyhow::anyhow!("npy load failed for '{name}': {e}"))?;

    let model_output = dump_dir.join("model.bpk");
    model.save_file(&model_output)?;

    let cfg_path = dump_dir.join(format!("{name}.cfg"));
    model_config.save(&cfg_path)?;

    let alias_cfg = dump_dir.join("config.cfg");
    if !alias_cfg.exists() {
        model_config.save(&alias_cfg)?;
    }

    options.report(crate::download::PrepProgress::convert(
        format!("Burn bundle saved for '{name}'"),
        1.0,
    ));
    if options.verbose {
        tracing::info!("burn bundle saved to {}", dump_dir.display());
    }

    Ok(())
}
