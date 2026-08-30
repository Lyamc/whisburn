use burn::tensor::{backend::Backend, Tensor};
use crate::audio::AudioProcessor;
use std::ops::Div;
use std::sync::Arc;

use crate::model::load::helpers::load_tensor;
use std::path::Path;

pub fn waveform_to_mel_tensor<B: Backend>(
    waveform: Vec<f32>,
    sample_rate: usize,
    window_length_samples: usize,
    device: &B::Device,
    n_mels: usize,
    is_nemo: bool,
    model_name: &str,
) -> impl Iterator<Item = Tensor<B, 3>> {
    let chunk_overlap = sample_rate * 3;
    let n_samples_per_tensor = window_length_samples;
    let shift = n_samples_per_tensor.saturating_sub(chunk_overlap).max(1);
    let iter_len = waveform.len().saturating_sub(1).div(shift) + 1;
    let device_clone = device.clone();

    let mut processor = AudioProcessor::new(&device_clone, sample_rate as f64, n_mels, is_nemo, model_name);
    
    if is_nemo {
        let mean_path = format!("models/{}/mean.npy", model_name);
        let std_path = format!("models/{}/std.npy", model_name);
        if Path::new(&mean_path).exists() && Path::new(&std_path).exists() {
            let mean = load_tensor::<B, 2>("mean", &format!("models/{}", model_name), &device_clone).unwrap();
            let std = load_tensor::<B, 2>("std", &format!("models/{}", model_name), &device_clone).unwrap();
            processor = processor.with_stats(mean, std);
        }
    }

    let processor_arc = Arc::new(processor);

    (0..iter_len).into_iter().map(move |i| {
        let processor = processor_arc.clone();
        let start = i * shift;
        let end = (start + n_samples_per_tensor).min(waveform.len());

        let slice = &waveform[start..end];

        let waveform_tensor = Tensor::<B, 1>::from_floats(
            slice,
            &device_clone,
        );

        let mels = processor.prep_audio(waveform_tensor.unsqueeze());

        mels
    })
}
