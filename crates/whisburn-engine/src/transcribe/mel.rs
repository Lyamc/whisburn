use burn::tensor::{Device, DeviceKind};
use burn::tensor::{Tensor};
use crate::audio::AudioProcessor;
use std::ops::Div;
use std::sync::Arc;

use crate::model::load::helpers::load_tensor;

/// How many encoder windows `waveform_to_mel_tensor` yields.
///
/// Overlap is 3 seconds. The count is computed up front so decoding can stream
/// one window at a time instead of allocating every spectrogram first.
pub fn mel_window_count(waveform_len: usize, window_length_samples: usize, sample_rate: usize) -> usize {
    let chunk_overlap = sample_rate.saturating_mul(3);
    let shift = window_length_samples.saturating_sub(chunk_overlap).max(1);
    waveform_len.saturating_sub(1).div(shift) + 1
}

pub fn waveform_to_mel_tensor(
    waveform: Vec<f32>,
    sample_rate: usize,
    window_length_samples: usize,
    device: &Device,
    n_mels: usize,
    is_nemo: bool,
    model_name: &str,
) -> impl Iterator<Item = Tensor< 3>> {
    let n_samples_per_tensor = window_length_samples;
    let chunk_overlap = sample_rate.saturating_mul(3);
    let shift = n_samples_per_tensor.saturating_sub(chunk_overlap).max(1);
    let iter_len = mel_window_count(waveform.len(), window_length_samples, sample_rate);
    let device_clone = device.clone();

    let mut processor = AudioProcessor::new(&device_clone, sample_rate as f64, n_mels, is_nemo, model_name);
    
    if is_nemo {
        let dir = whisburn_core::resolve_model_dir(model_name);
        let mean_path = dir.join("mean.npy");
        let std_path = dir.join("std.npy");
        if mean_path.exists() && std_path.exists() {
            let dir_str = dir.to_string_lossy();
            let mean = load_tensor::< 2>("mean", dir_str.as_ref(), &device_clone).unwrap();
            let std = load_tensor::< 2>("std", dir_str.as_ref(), &device_clone).unwrap();
            processor = processor.with_stats(mean, std);
        }
    }

    let processor_arc = Arc::new(processor);

    (0..iter_len).into_iter().map(move |i| {
        let processor = processor_arc.clone();
        let start = i * shift;
        let end = (start + n_samples_per_tensor).min(waveform.len());

        let slice = &waveform[start..end];

        let waveform_tensor = Tensor::< 1>::from_floats(
            slice,
            &device_clone,
        );

        let mels = processor.prep_audio(waveform_tensor.unsqueeze());

        mels
    })
}

#[cfg(test)]
mod tests {
    use super::mel_window_count;

    #[test]
    fn long_audio_is_many_windows_not_one_buffer() {
        let sample_rate = 16_000;
        let window = sample_rate * 30;
        // A clip that is exactly one window still yields the overlap tail.
        assert_eq!(mel_window_count(window, window, sample_rate), 2);
        // 1.5 h at a 30 s Whisper window with 3 s overlap: about 200 steps, not one buffer.
        let hour_and_half = sample_rate * 5_400;
        assert_eq!(mel_window_count(hour_and_half, window, sample_rate), 200);
        assert!(mel_window_count(hour_and_half, window, sample_rate) > mel_window_count(window, window, sample_rate));
    }
}
