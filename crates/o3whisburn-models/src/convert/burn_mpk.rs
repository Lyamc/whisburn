use std::path::Path;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::config::Config;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder, Recorder};
use o3whisburn_engine::model::load::{init_shapes, load_model_from_npy};

use crate::download::DownloadOptions;
use crate::paths::model_dir;

pub fn save_model_from_npy(name: &str, options: &DownloadOptions) -> anyhow::Result<()> {
    let dump_dir = model_dir(name);
    if !dump_dir.join("encoder").exists() && !dump_dir.join("decoder").exists() {
        anyhow::bail!(
            "no npy dump at {} for conversion",
            dump_dir.display()
        );
    }

    if options.verbose {
        tracing::info!("converting npy dump to burn mpk for '{name}'");
    }

    let device = WgpuDevice::DefaultDevice;
    let dump_str = dump_dir
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("invalid model path"))?;
    init_shapes(dump_str);
    let (model, model_config) =
        load_model_from_npy::<Wgpu>(dump_str, &device).map_err(|e| {
            anyhow::anyhow!("npy load failed for '{name}': {e}")
        })?;

    let model_output = dump_dir.join("model");
    NamedMpkGzFileRecorder::<FullPrecisionSettings>::new()
        .record(model.into_record(), model_output.clone().into())?;

    let cfg_path = dump_dir.join(format!("{name}.cfg"));
    model_config.save(&cfg_path)?;

    let alias_cfg = dump_dir.join("config.cfg");
    if !alias_cfg.exists() {
        model_config.save(&alias_cfg)?;
    }

    decompress_gz_to_aliases(&dump_dir, name)?;

    if options.verbose {
        tracing::info!("burn bundle saved to {}", dump_dir.display());
    }

    Ok(())
}

fn decompress_gz_to_aliases(dir: &Path, _name: &str) -> anyhow::Result<()> {
    use flate2::read::GzDecoder;
    use std::fs::File;
    use std::io::copy;

    let gz = dir.join("model.mpk.gz");
    if !gz.exists() {
        return Ok(());
    }

    let mpk = dir.join("model.mpk");
    // Always refresh the uncompressed alias — a leftover model.mpk is
    // preferred by the loader and would keep serving pre-fix weights.
    let mut decoder = GzDecoder::new(File::open(&gz)?);
    let mut out = File::create(&mpk)?;
    copy(&mut decoder, &mut out)?;

    Ok(())
}