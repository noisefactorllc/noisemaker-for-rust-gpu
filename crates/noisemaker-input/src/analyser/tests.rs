//! Behavioural tests of the analyser. Bit-exactness against Chromium's own
//! AnalyserNode is gated by parity/check_audio_analyzer.mjs.

use super::*;

fn sine(frames: usize, cycles_per_frame: f64, amplitude: f64) -> Vec<f32> {
    (0..frames)
        .map(|i| {
            (amplitude * (2.0 * std::f64::consts::PI * cycles_per_frame * i as f64).sin()) as f32
        })
        .collect()
}

#[test]
fn defaults_match_analyser_node() {
    let analyser = AudioAnalyzer::new();
    assert_eq!(analyser.fft_size(), 2048);
    assert_eq!(analyser.frequency_bin_count(), 1024);
    assert_eq!(analyser.smoothing_time_constant(), 0.8);
    assert_eq!(
        (analyser.min_decibels(), analyser.max_decibels()),
        (-100.0, -30.0)
    );
}

#[test]
fn settings_are_validated_like_analyser_node() {
    let mut analyser = AudioAnalyzer::new();
    assert_eq!(
        analyser.set_fft_size(16),
        Err(AnalyserError::FftSizeOutOfRange(16))
    );
    assert_eq!(
        analyser.set_fft_size(65536),
        Err(AnalyserError::FftSizeOutOfRange(65536))
    );
    assert_eq!(
        analyser.set_fft_size(300),
        Err(AnalyserError::FftSizeNotPowerOfTwo(300))
    );
    assert!(analyser.set_fft_size(256).is_ok());
    assert_eq!(analyser.frequency_bin_count(), 128);
    assert!(analyser.set_min_decibels(-30.0).is_err());
    assert!(analyser.set_max_decibels(-100.0).is_err());
    assert!(analyser.set_min_decibels(f64::NEG_INFINITY).is_err());
    assert!(analyser.set_smoothing_time_constant(1.5).is_err());
    assert!(analyser.set_smoothing_time_constant(f64::NAN).is_err());
    assert!(analyser.set_decibel_range(-90.0, -10.0).is_ok());
    assert!(analyser.set_decibel_range(-10.0, -10.0).is_err());
}

#[test]
fn silence_reads_as_minus_infinity_and_mid_scale_bytes() {
    let mut analyser = AudioAnalyzer::for_input_manager(0.8);
    analyser.write(&[0.0; 256], 1);
    let mut float = vec![0.0f32; 128];
    analyser.get_float_frequency_data(&mut float);
    assert!(float.iter().all(|&db| db == f32::NEG_INFINITY));
    let mut bytes = vec![1u8; 128];
    analyser.get_byte_frequency_data(&mut bytes);
    assert!(bytes.iter().all(|&b| b == 0));
    let mut time = vec![0u8; 256];
    analyser.get_byte_time_domain_data(&mut time);
    assert!(time.iter().all(|&b| b == 128));
}

#[test]
fn input_joins_one_render_quantum_at_a_time() {
    let mut analyser = AudioAnalyzer::for_input_manager(0.0);
    analyser.write(&[0.5; 100], 1);
    assert_eq!(analyser.render_quanta(), 0);
    assert_eq!(analyser.pending_frames(), 100);
    let mut time = vec![0.0f32; 256];
    analyser.get_float_time_domain_data(&mut time);
    assert!(time.iter().all(|&v| v == 0.0));
    analyser.write(&[0.5; 28], 1);
    assert_eq!(analyser.render_quanta(), 1);
    analyser.get_float_time_domain_data(&mut time);
    assert!(time[..128].iter().all(|&v| v == 0.0));
    assert!(time[128..].iter().all(|&v| v == 0.5));
}

#[test]
fn analysis_runs_once_per_render_quantum() {
    let mut analyser = AudioAnalyzer::for_input_manager(0.5);
    analyser.write(&sine(256, 8.0 / 256.0, 0.5), 1);
    let mut first = vec![0.0f32; 128];
    analyser.get_float_frequency_data(&mut first);
    let mut second = vec![0.0f32; 128];
    analyser.get_float_frequency_data(&mut second);
    assert_eq!(first, second, "no new quantum, no second smoothing step");
    analyser.write(&sine(128, 8.0 / 256.0, 0.5), 1);
    analyser.get_float_frequency_data(&mut second);
    assert_ne!(first, second);
}

#[test]
fn a_full_scale_sine_peaks_near_zero_decibels_at_its_bin() {
    let mut analyser = AudioAnalyzer::for_input_manager(0.0);
    // Bin 8 of a 256-point FFT, full scale.
    analyser.write(&sine(1024, 8.0 / 256.0, 1.0), 1);
    let mut float = vec![0.0f32; 128];
    analyser.get_float_frequency_data(&mut float);
    let peak = float
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    assert_eq!(peak.0, 8);
    // The Blackman window's coherent gain is 0.42: 20*log10(0.42/2) dB.
    assert!(
        (peak.1 - (20.0 * (0.42f32 / 2.0).log10())).abs() < 0.01,
        "{}",
        peak.1
    );
}

#[test]
fn stereo_down_mix_averages_channels() {
    let mut analyser = AudioAnalyzer::for_input_manager(0.0);
    let frames: Vec<f32> = (0..128).flat_map(|_| [0.25f32, 0.75]).collect();
    analyser.write(&frames, 2);
    let mut time = vec![0.0f32; 256];
    analyser.get_float_time_domain_data(&mut time);
    assert!(time[128..].iter().all(|&v| v == 0.5));
}

#[test]
fn discrete_layouts_keep_the_first_channel() {
    let mut analyser = AudioAnalyzer::for_input_manager(0.0);
    let frames: Vec<f32> = (0..128).flat_map(|_| [0.25f32, 0.5, 1.0]).collect();
    analyser.write(&frames, 3);
    let mut time = vec![0.0f32; 256];
    analyser.get_float_time_domain_data(&mut time);
    assert!(time[128..].iter().all(|&v| v == 0.25));
}

#[test]
fn surround_down_mix_follows_the_speakers_rule() {
    let mut analyser = AudioAnalyzer::for_input_manager(0.0);
    let frame = [0.5f32, 0.25, 0.125, 0.9, 0.5, 0.25];
    let frames: Vec<f32> = (0..128).flat_map(|_| frame).collect();
    analyser.write(&frames, 6);
    let mut time = vec![0.0f32; 256];
    analyser.get_float_time_domain_data(&mut time);
    let half = 0.5f32.sqrt();
    let expected = (0.5 * half + 0.25 * half) + 0.125 + 0.5 * 0.5 + 0.25 * 0.5;
    assert!(
        (time[200] - expected).abs() < 1e-6,
        "{} vs {expected}",
        time[200]
    );
}

#[test]
fn changing_the_fft_size_clears_magnitudes_without_a_new_analysis() {
    let mut analyser = AudioAnalyzer::for_input_manager(0.0);
    analyser.write(&sine(512, 8.0 / 256.0, 0.5), 1);
    let mut bytes = vec![0u8; 128];
    analyser.get_byte_frequency_data(&mut bytes);
    assert!(bytes.iter().any(|&b| b > 0));
    analyser.set_fft_size(512).unwrap();
    let mut bytes = vec![0u8; 256];
    analyser.get_byte_frequency_data(&mut bytes);
    assert!(
        bytes.iter().all(|&b| b == 0),
        "same render quantum: no analysis yet"
    );
    analyser.write(&sine(128, 8.0 / 256.0, 0.5), 1);
    analyser.get_byte_frequency_data(&mut bytes);
    assert!(bytes.iter().any(|&b| b > 0));
}

#[test]
fn byte_time_domain_clips_and_truncates() {
    let mut analyser = AudioAnalyzer::for_input_manager(0.0);
    let mut frames = vec![0.0f32; 128];
    frames[0] = 1.5;
    frames[1] = -1.5;
    frames[2] = 0.99;
    frames[3] = f32::NAN;
    analyser.write(&frames, 1);
    let mut time = vec![0u8; 256];
    analyser.get_byte_time_domain_data(&mut time);
    assert_eq!(&time[128..132], &[255, 0, 254, 0]);
}

#[test]
fn the_portable_flush_model_flushes_denormal_inputs_and_tiny_results() {
    use super::portable::{Flush, add, mul, vsma};
    let tiny = f32::from_bits(1); // smallest denormal
    let min = f32::MIN_POSITIVE;
    for flush in [Flush::Arm, Flush::X86] {
        assert_eq!(mul(tiny, 1.0, flush).to_bits(), 0);
        assert_eq!(mul(-tiny, 1.0, flush).to_bits(), (-0.0f32).to_bits());
        assert_eq!(
            mul(min, 0.5, flush).to_bits(),
            0,
            "a denormal result is flushed"
        );
        assert_eq!(
            add(min, -min * 0.5, flush).to_bits(),
            0,
            "a denormal difference is flushed"
        );
        assert_eq!(add(min * 2.0, -min, flush), min, "{flush:?}");
        let mut dest = [1.0f32, tiny, 0.0];
        vsma(&[tiny, 2.0, min], 0.5, &mut dest, flush);
        assert_eq!(dest, [1.0, 1.0, 0.0]);
    }
    assert_eq!(mul(tiny, 1.0, Flush::None), tiny);
    // An exact product within half an (unbounded-exponent) ulp below the
    // normal range: ARM flushes it (tiny before rounding), x86 keeps it (it
    // rounds up to the smallest normal). (1 - 2^-23)(1 + 2^-23) = 1 - 2^-46.
    let a = f32::from_bits(0x3f7f_fffe);
    let b = f32::from_bits(0x0080_0001);
    assert_eq!(mul(a, b, Flush::Arm), 0.0);
    assert_eq!(mul(a, b, Flush::X86), f32::MIN_POSITIVE);
    assert_eq!(mul(a, b, Flush::None), f32::MIN_POSITIVE);
}

/// With scales of 0.5 and 0.25 every product is exact, so Chromium's macOS
/// vDSP down-mix and the portable one agree bit for bit in the normal range.
/// (Denormals differ by design: under FPCR.FZ, Apple silicon flushes denormal
/// results but not denormal inputs, while the portable model follows the
/// architecture's default for the other ARM and x86 CPUs, which flush both.
/// macOS itself runs vDSP in Chromium's mode, so it never uses the model.)
#[cfg(target_os = "macos")]
#[test]
fn the_portable_down_mix_matches_vdsp_for_exact_scales() {
    use super::portable::{self, Flush};
    let mut seed = 0x1234_5678u32;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    for _ in 0..200 {
        let samples: Vec<Vec<f32>> = (0..4)
            .map(|_| {
                (0..128)
                    .map(|_| match next() % 4 {
                        0 => f32::from_bits((next() & 0x807f_ffff) | 0x0200_0000), // small normals
                        1 => 0.0,
                        _ => (next() as f32 / u32::MAX as f32) * 2.0 - 1.0,
                    })
                    .collect()
            })
            .collect();
        for (channels, scale) in [(2, 0.5f32), (4, 0.25)] {
            let sources: Vec<&[f32]> = samples[..channels].iter().map(Vec::as_slice).collect();
            let mut native = [0.0f32; 128];
            down_mix(&sources, &mut native);
            let mut model = [0.0f32; 128];
            for source in &sources {
                portable::vsma(source, scale, &mut model, Flush::Arm);
            }
            let bits = |v: &[f32; 128]| v.map(f32::to_bits);
            assert_eq!(bits(&native), bits(&model), "{channels} channels");
        }
    }
}
