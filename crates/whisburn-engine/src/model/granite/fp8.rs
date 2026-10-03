//! NVIDIA FP8 E4M3 (finite, no infinities). Exponent bias is 7. The all-ones
//! mantissa (`0x7F` / `0xFF`) is NaN. Every other code is finite, including
//! exponent 15, whose maximum magnitude is 448.

pub fn f8_e4m3_to_f32(b: u8) -> f32 {
    let sign = (b & 0x80) != 0;
    let exp = (b >> 3) & 0x0f;
    let mant = b & 0x07;
    let mag = if exp == 0 {
        (mant as f32) * 2f32.powi(-9)
    } else if exp == 0x0f && mant == 0x07 {
        f32::NAN
    } else {
        (1.0 + (mant as f32) / 8.0) * 2f32.powi((exp as i32) - 7)
    };
    if sign { -mag } else { mag }
}

#[cfg(test)]
mod tests {
    use super::f8_e4m3_to_f32;

    #[test]
    fn e4m3_known_values() {
        assert_eq!(f8_e4m3_to_f32(0x00), 0.0);
        assert_eq!(f8_e4m3_to_f32(0x80), -0.0);
        assert!((f8_e4m3_to_f32(0x38) - 1.0).abs() < 1e-6);
        assert!((f8_e4m3_to_f32(0x40) - 2.0).abs() < 1e-6);
        assert!((f8_e4m3_to_f32(0x7e) - 448.0).abs() < 1e-3);
        assert!(f8_e4m3_to_f32(0x7f).is_nan());
        assert!((f8_e4m3_to_f32(0xb8) + 1.0).abs() < 1e-6);
    }
}
