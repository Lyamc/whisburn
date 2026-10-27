use std::fs;
use std::io::{copy, Read};
use std::path::Path;

const STALE_FILES: &[&str] = &[
    "model.mpk",
    "model.mpk.gz",
    "model.bpk",
    "model.bpk.gz",
    "config.cfg",
    "shapes.json",
    ".burn_version",
    "parakeet_decode.json",
    "actual_n_vocab.npy",
    "mean.npy",
    "std.npy",
];

const STALE_SUBDIRS: &[&str] = &[
    "encoder",
    "decoder",
    "ctc_linear",
    "final_proj",
    "joint_pred",
];

pub fn clear_stale_artifacts(dir: &Path) {
    STALE_FILES
        .iter()
        .map(|file| dir.join(file))
        .chain(STALE_SUBDIRS.iter().map(|subdir| dir.join(subdir)))
        .for_each(|path| {
            if path.is_dir() {
                let _ = fs::remove_dir_all(path);
            } else {
                let _ = fs::remove_file(path);
            }
        });

    fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            let ext = path.extension().and_then(|e| e.to_str());
            ext == Some("mpk")
                || ext == Some("bpk")
                || ext == Some("cfg")
                || path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.ends_with(".mpk.gz") || n.ends_with(".bpk.gz"))
        })
        .for_each(|path| {
            let _ = fs::remove_file(path);
        });
}

pub fn decompress_mpk_gz_variants(dir: &Path, model_name: &str, verbose: bool) -> anyhow::Result<()> {
    use std::fs::File;

    [
        dir.join(format!("{model_name}.bpk.gz")),
        dir.join("model.bpk.gz"),
    ]
        .iter()
        .filter(|gz_path| gz_path.exists())
        .try_for_each(|gz_path| {
            let unpacked_name = gz_path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.trim_end_matches(".gz"))
                .unwrap_or("model.bpk");

            let unpacked_path = dir.join(unpacked_name);
            if unpacked_path.exists() {
                return Ok(());
            }

            if verbose {
                tracing::info!(
                    "decompressing {} -> {}",
                    gz_path.display(),
                    unpacked_path.display()
                );
            }

            let gz_file = File::open(gz_path)?;
            let mut decoder = flate2::read::GzDecoder::new(gz_file);
            let mut unpacked_file = File::create(&unpacked_path)?;
            copy(&mut decoder, &mut unpacked_file)?;

            let alias = dir.join("model.bpk");
            if unpacked_path != alias && !alias.exists() {
                fs::copy(&unpacked_path, &alias)?;
            }
            Ok(())
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn decompress_unpacks_burnpack_and_leaves_named_mpk_gzip() {
        let dir = std::env::temp_dir().join(format!("whisburn-gz-{}-bpk", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let write_gz = |name: &str, bytes: &[u8]| {
            let file = fs::File::create(dir.join(name)).unwrap();
            let mut enc = flate2::write::GzEncoder::new(file, flate2::Compression::default());
            enc.write_all(bytes).unwrap();
            enc.finish().unwrap();
        };
        write_gz("model.bpk.gz", b"pack");
        write_gz("model.mpk.gz", b"old");

        decompress_mpk_gz_variants(&dir, "tiny_en", false).unwrap();

        assert_eq!(fs::read(dir.join("model.bpk")).unwrap(), b"pack");
        assert!(!dir.join("model.mpk").is_file());
        assert!(dir.join("model.mpk.gz").is_file());
        let _ = fs::remove_dir_all(&dir);
    }
}

pub fn save_response_gz<R: Read>(
    mut reader: R,
    output_dir: &Path,
    filename: &str,
    verbose: bool,
) -> anyhow::Result<()> {
    use indicatif::{ProgressBar, ProgressStyle};

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner} {msg}")
            .unwrap(),
    );
    pb.set_message(format!("decompressing {filename}"));

    let output_path = if filename.ends_with(".gz") {
        output_dir.join(filename.trim_end_matches(".gz"))
    } else {
        output_dir.join(filename)
    };

    let mut decoder = flate2::read::GzDecoder::new(&mut reader);
    let mut out = std::fs::File::create(&output_path)?;
    copy(&mut decoder, &mut out)?;
    pb.finish_and_clear();

    if verbose {
        tracing::info!("wrote {}", output_path.display());
    }

    Ok(())
}