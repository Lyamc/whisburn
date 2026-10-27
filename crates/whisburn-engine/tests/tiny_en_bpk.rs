use std::path::Path;

use burn::tensor::Device;
use whisburn_engine::model::load_model;
use whisburn_engine::model::Model;

/// Loads a local `whisburn-server/models/tiny_en/model.bpk` through `load_model`.
/// The burnpack is about 145 MB, over GitHub's file limit, so a clone without it skips.
#[test]
fn shipped_tiny_en_burnpack_loads() {
    let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../whisburn-server/models");
    let weights = models.join("tiny_en/model.bpk");
    if !weights.is_file() {
        eprintln!("skip: tiny_en burnpack missing at {}", weights.display());
        return;
    }
    let server = models.parent().expect("server crate dir");
    std::env::set_current_dir(server).expect("chdir to whisburn-server");
    let device = Device::flex();
    let (_bpe, _cfg, model) = load_model("tiny_en", &device, false).expect("load shipped tiny_en");
    assert!(matches!(model, Model::Whisper(_)));
}
