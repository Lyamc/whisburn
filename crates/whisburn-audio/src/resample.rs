use rubato::{FftFixedIn, Resampler};

use whisburn_core::WhisburnError;

pub fn resample_mono(
    input: &[f32],
    from_rate: usize,
    to_rate: usize,
) -> Result<Vec<f32>, WhisburnError> {
    if from_rate == to_rate {
        return Ok(input.to_vec());
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let chunk_size = 1024;
    let mut resampler = FftFixedIn::<f32>::new(from_rate, to_rate, chunk_size, 1, 1)
        .map_err(|e| WhisburnError::Audio(e.to_string()))?;

    let mut output = Vec::with_capacity(input.len() * to_rate / from_rate + chunk_size);
    let mut pos = 0;

    while pos < input.len() {
        let end = (pos + chunk_size).min(input.len());
        let mut chunk = input[pos..end].to_vec();
        if chunk.len() < chunk_size {
            chunk.resize(chunk_size, 0.0);
        }

        let resampled = resampler
            .process(&[chunk], None)
            .map_err(|e| WhisburnError::Audio(e.to_string()))?;

        let valid = if end == input.len() {
            resampled[0].len().saturating_sub(chunk_size - (end - pos))
        } else {
            resampled[0].len()
        };

        output.extend_from_slice(&resampled[0][..valid.min(resampled[0].len())]);
        pos = end;
    }

    Ok(output)
}