//! A Web Audio `AnalyserNode` for hosts that capture audio themselves.
//!
//! This is Chromium's implementation, the browser the reference host runs on
//! (Chromium 153: `third_party/blink/renderer/modules/webaudio/realtime_analyser.cc`,
//! `analyser_handler.cc`, `platform/audio/audio_bus.cc`, `fft_frame.cc` and
//! `rustfft_ffi.rs`), reproduced operation for operation:
//!
//! - Input arrives in 128-frame render quanta. Each quantum is down-mixed to
//!   mono with the "speakers" rules of `AudioBus::SumFrom` (1, 2, 4 and 6
//!   channels; other counts keep channel 1) and appended to a 65536-sample ring.
//!   The down-mix runs, as on Chromium's audio thread, with denormals flushed
//!   to zero; on macOS it calls the same Accelerate `vDSP_vsma`/`vDSP_vadd`
//!   routines Chromium's `vector_math` uses there.
//! - Frequency data is analysed at most once per render quantum
//!   (`last_analysis_time_`): the last `fftSize` samples get a Blackman window
//!   (alpha 0.16, computed in double and applied in float), a real FFT, and
//!   magnitudes `|X[k]| / fftSize` smoothed with `smoothingTimeConstant`.
//! - The FFT is Chromium's `WebAudioRustFft` path (enabled by default since
//!   Chromium 153): a half-size complex `rustfft` 6.4.1 transform from a fresh
//!   `FftPlanner`, then the even-size real reconstruction of `rustfft_ffi.rs`.
//! - Float data is `20 * log10f(magnitude)`; byte data scales the decibels
//!   between `minDecibels` and `maxDecibels`, truncating like `ClampTo`.
//!
//! Time is counted in render quanta: two reads without a new quantum see the
//! same analysis, like two reads at the same `AudioContext.currentTime`.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, OnceLock, RwLock};

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::audio::FrequencyDataSource;

/// Frames per Web Audio render quantum.
pub const RENDER_QUANTUM_FRAMES: usize = 128;
/// Smallest `fftSize`.
pub const MIN_FFT_SIZE: u32 = 32;
/// Largest `fftSize`.
pub const MAX_FFT_SIZE: u32 = 32768;
/// Default `fftSize`.
pub const DEFAULT_FFT_SIZE: u32 = 2048;
/// Default `smoothingTimeConstant`.
pub const DEFAULT_SMOOTHING_TIME_CONSTANT: f64 = 0.8;
/// Default `minDecibels`.
pub const DEFAULT_MIN_DECIBELS: f64 = -100.0;
/// Default `maxDecibels`.
pub const DEFAULT_MAX_DECIBELS: f64 = -30.0;
/// Ring capacity (`kInputBufferSize = kMaxFFTSize * 2`).
const INPUT_BUFFER_SIZE: usize = MAX_FFT_SIZE as usize * 2;

/// Why an analyser setting was rejected (the `IndexSizeError`s and IDL
/// `TypeError`s of `AnalyserNode`).
#[derive(Clone, Debug, PartialEq)]
pub enum AnalyserError {
    /// `fftSize` outside 32..=32768.
    FftSizeOutOfRange(u32),
    /// `fftSize` not a power of two.
    FftSizeNotPowerOfTwo(u32),
    /// A non-finite decibel or smoothing value (IDL `double` conversion).
    NonFinite(f64),
    /// `minDecibels` not below `maxDecibels`.
    MinDecibels {
        /// The rejected value.
        value: f64,
        /// The current `maxDecibels`.
        max: f64,
    },
    /// `maxDecibels` not above `minDecibels`.
    MaxDecibels {
        /// The rejected value.
        value: f64,
        /// The current `minDecibels`.
        min: f64,
    },
    /// `minDecibels >= maxDecibels` in a combined update.
    DecibelRange {
        /// The rejected minimum.
        min: f64,
        /// The rejected maximum.
        max: f64,
    },
    /// `smoothingTimeConstant` outside 0..=1.
    Smoothing(f64),
}

impl fmt::Display for AnalyserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnalyserError::FftSizeOutOfRange(size) => write!(
                f,
                "The value provided ({size}) is outside the range [{MIN_FFT_SIZE}, {MAX_FFT_SIZE}]."
            ),
            AnalyserError::FftSizeNotPowerOfTwo(size) => {
                write!(f, "The value provided ({size}) is not a power of two.")
            }
            AnalyserError::NonFinite(value) => {
                write!(f, "The provided double value ({value}) is non-finite.")
            }
            AnalyserError::MinDecibels { value, max } => write!(
                f,
                "The minDecibels provided ({value}) is greater than or equal to the maximum bound ({max})."
            ),
            AnalyserError::MaxDecibels { value, min } => write!(
                f,
                "The maxDecibels provided ({value}) is less than or equal to the minimum bound ({min})."
            ),
            AnalyserError::DecibelRange { min, max } => write!(
                f,
                "maxDecibels ({max}) must be greater than or equal to minDecibels ( {min})."
            ),
            AnalyserError::Smoothing(value) => write!(
                f,
                "The smoothing value provided ({value}) is outside the range [0, 1]."
            ),
        }
    }
}

impl std::error::Error for AnalyserError {}

/// A 16-byte aligned render quantum, like Chromium's `AudioFloatArray`.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct Quantum([f32; RENDER_QUANTUM_FRAMES]);

impl Quantum {
    const ZERO: Quantum = Quantum([0.0; RENDER_QUANTUM_FRAMES]);
}

/// Collects interleaved frames into planar render quanta and down-mixes them.
#[derive(Clone)]
struct QuantumAssembler {
    /// Channel count of the frames in `planar`.
    channels: usize,
    /// One planar buffer per channel; frames sit at their quantum position.
    planar: Vec<Quantum>,
    /// Frames of the current quantum already down-mixed into `mono`.
    mixed: usize,
    /// Frames buffered in `planar` after `mixed`.
    buffered: usize,
    /// The down-mixed current quantum.
    mono: Quantum,
}

impl QuantumAssembler {
    fn new() -> Self {
        QuantumAssembler {
            channels: 0,
            planar: Vec::new(),
            mixed: 0,
            buffered: 0,
            mono: Quantum::ZERO,
        }
    }

    /// Down-mixes the buffered planar frames into `mono`.
    fn flush(&mut self) {
        if self.buffered == 0 {
            return;
        }
        let range = self.mixed..self.mixed + self.buffered;
        let sources: Vec<&[f32]> = self.planar[..self.channels]
            .iter()
            .map(|q| &q.0[range.clone()])
            .collect();
        down_mix(&sources, &mut self.mono.0[range.clone()]);
        self.mixed += self.buffered;
        self.buffered = 0;
    }
}

/// Chromium's `AudioBus::SumFrom` from an input bus into a zeroed mono bus
/// (`RealtimeAnalyser::WriteInput`): speakers down-mix for 2, 4 and 6
/// channels, a copy of channel 1 otherwise.
fn down_mix(sources: &[&[f32]], dest: &mut [f32]) {
    match sources.len() {
        2 => {
            dest.fill(0.0);
            with_denormals_disabled(|| {
                vsma(sources[0], 0.5, dest);
                vsma(sources[1], 0.5, dest);
            });
        }
        4 => {
            dest.fill(0.0);
            with_denormals_disabled(|| {
                for source in sources {
                    vsma(source, 0.25, dest);
                }
            });
        }
        6 => {
            dest.fill(0.0);
            let sqrt_half = 0.5f32.sqrt();
            with_denormals_disabled(|| {
                vsma(sources[0], sqrt_half, dest);
                vsma(sources[1], sqrt_half, dest);
                vadd(sources[2], dest);
                vsma(sources[4], 0.5, dest);
                vsma(sources[5], 0.5, dest);
            });
        }
        // Mono, or a layout without a speakers rule (discrete): the silent
        // destination channel copies channel 1.
        _ => dest.copy_from_slice(sources[0]),
    }
}

// ----------------------------------------------------------------------------
// vector_math as Chromium builds it.

#[cfg(target_os = "macos")]
mod vdsp {
    use std::os::raw::{c_long, c_ulong};

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn vDSP_vsma(
            a: *const f32,
            ia: c_long,
            b: *const f32,
            c: *const f32,
            ic: c_long,
            d: *mut f32,
            id: c_long,
            n: c_ulong,
        );
        pub fn vDSP_vadd(
            a: *const f32,
            ia: c_long,
            b: *const f32,
            ib: c_long,
            c: *mut f32,
            ic: c_long,
            n: c_ulong,
        );
    }
}

/// `vector_math::Vsma(source, scale, dest)`: `dest += scale * source`.
/// macOS: `vDSP_vsma` (vector_math_mac.h).
#[cfg(target_os = "macos")]
fn vsma(source: &[f32], scale: f32, dest: &mut [f32]) {
    debug_assert_eq!(source.len(), dest.len());
    let d = dest.as_mut_ptr();
    // SAFETY: both slices hold `dest.len()` floats; vDSP reads `source` and
    // reads/writes `dest` in place, as Chromium calls it.
    unsafe {
        vdsp::vDSP_vsma(source.as_ptr(), 1, &scale, d, 1, d, 1, dest.len() as _);
    }
}

/// `vector_math::Vadd(source, dest, dest)`.
/// macOS: `vDSP_vadd` (vector_math_mac.h).
#[cfg(target_os = "macos")]
fn vadd(source: &[f32], dest: &mut [f32]) {
    debug_assert_eq!(source.len(), dest.len());
    let d = dest.as_mut_ptr();
    // SAFETY: as in `vsma`.
    unsafe {
        vdsp::vDSP_vadd(source.as_ptr(), 1, d, 1, d, 1, dest.len() as _);
    }
}

/// `vector_math::Vsma` elsewhere (see [`portable`]).
#[cfg(not(target_os = "macos"))]
fn vsma(source: &[f32], scale: f32, dest: &mut [f32]) {
    portable::vsma(source, scale, dest, portable::Flush::HOST);
}

/// `vector_math::Vadd` elsewhere (see [`portable`]).
#[cfg(not(target_os = "macos"))]
fn vadd(source: &[f32], dest: &mut [f32]) {
    portable::vadd(source, dest, portable::Flush::HOST);
}

/// Chromium's `vector_math` outside macOS: NEON `vmlaq_f32`, SSE/AVX
/// `add(mul)` and the scalar `*dest += k * *source` are all unfused
/// (Chromium builds with `-ffp-contract=off`), and the audio thread runs them
/// under `DenormalDisabler` (MXCSR FTZ|DAZ on x86, FPCR.FZ on ARM; other CPUs
/// have no such mode). The flush-to-zero mode is modelled in software.
#[cfg_attr(target_os = "macos", allow(dead_code))]
mod portable {
    /// The flush-to-zero behaviour of the audio thread.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Flush {
        /// No flush-to-zero mode.
        None,
        /// FPCR.FZ: denormal inputs read as zero; results whose exact value
        /// is below the normal range become zero.
        Arm,
        /// MXCSR FTZ|DAZ: denormal inputs read as zero; results that are
        /// below the normal range after rounding with an unbounded exponent
        /// become zero.
        X86,
    }

    impl Flush {
        /// The mode of the build target.
        pub const HOST: Flush = if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
            Flush::X86
        } else if cfg!(any(target_arch = "arm", target_arch = "aarch64")) {
            Flush::Arm
        } else {
            Flush::None
        };
    }

    const MIN_NORMAL: f64 = f32::MIN_POSITIVE as f64;

    /// A denormal input reads as zero of the same sign.
    pub fn input(x: f32, flush: Flush) -> f32 {
        if flush != Flush::None && x.is_subnormal() {
            0.0f32.copysign(x)
        } else {
            x
        }
    }

    /// A tiny result becomes zero with the sign of the true result.
    fn output(exact: f64, rounded: f32, flush: Flush) -> f32 {
        if flush == Flush::None || exact == 0.0 || !exact.is_finite() {
            return rounded;
        }
        let tiny = match flush {
            // Round to 24 bits as if the exponent were unbounded (scaling by
            // 2^64 is exact and lands in the normal range).
            Flush::X86 => {
                ((exact * 2f64.powi(64)) as f32).abs() < (MIN_NORMAL * 2f64.powi(64)) as f32
            }
            _ => exact.abs() < MIN_NORMAL,
        };
        if tiny {
            0.0f32.copysign(exact as f32)
        } else {
            rounded
        }
    }

    /// `a * b` (the product of two floats is exact in f64).
    pub fn mul(a: f32, b: f32, flush: Flush) -> f32 {
        output(f64::from(a) * f64::from(b), a * b, flush)
    }

    /// `a + b` (a sum of two floats that is near the normal boundary is exact
    /// in f64).
    pub fn add(a: f32, b: f32, flush: Flush) -> f32 {
        output(f64::from(a) + f64::from(b), a + b, flush)
    }

    /// `dest += scale * source`, unfused.
    pub fn vsma(source: &[f32], scale: f32, dest: &mut [f32], flush: Flush) {
        debug_assert_eq!(source.len(), dest.len());
        for (d, &s) in dest.iter_mut().zip(source) {
            let product = mul(input(scale, flush), input(s, flush), flush);
            *d = add(input(*d, flush), product, flush);
        }
    }

    /// `dest = source + dest`.
    pub fn vadd(source: &[f32], dest: &mut [f32], flush: Flush) {
        debug_assert_eq!(source.len(), dest.len());
        for (d, &s) in dest.iter_mut().zip(source) {
            *d = add(input(s, flush), input(*d, flush), flush);
        }
    }
}

/// Runs the down-mix arithmetic in the audio thread's floating-point mode:
/// Chromium's `DenormalDisabler` sets FPCR.FZ on ARM64 and MXCSR FTZ|DAZ on
/// x86 around rendering. Only the macOS path runs foreign (vDSP) code in it;
/// elsewhere the arithmetic models the mode in software.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn with_denormals_disabled<R>(f: impl FnOnce() -> R) -> R {
    let saved: u64;
    // SAFETY: reading and writing FPCR has no memory effects; the previous
    // mode is restored before any Rust floating-point code runs.
    unsafe {
        std::arch::asm!("mrs {0}, fpcr", out(reg) saved, options(nostack));
        std::arch::asm!("msr fpcr, {0}", in(reg) saved | (1 << 24), options(nostack));
    }
    let result = f();
    unsafe {
        std::arch::asm!("msr fpcr, {0}", in(reg) saved, options(nostack));
    }
    result
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn with_denormals_disabled<R>(f: impl FnOnce() -> R) -> R {
    let mut saved: u32 = 0;
    // SAFETY: stmxcsr/ldmxcsr only move MXCSR through the given location.
    unsafe {
        std::arch::asm!("stmxcsr [{0}]", in(reg) &mut saved, options(nostack));
        let flushed = saved | 0x8040;
        std::arch::asm!("ldmxcsr [{0}]", in(reg) &flushed, options(nostack));
    }
    let result = f();
    unsafe {
        std::arch::asm!("ldmxcsr [{0}]", in(reg) &saved, options(nostack));
    }
    result
}

#[cfg(not(all(
    target_os = "macos",
    any(target_arch = "aarch64", target_arch = "x86_64")
)))]
fn with_denormals_disabled<R>(f: impl FnOnce() -> R) -> R {
    f()
}

// ----------------------------------------------------------------------------
// FFTFrame on Chromium's rustfft path (rustfft_ffi.rs).

type FftPair = (Arc<dyn Fft<f32>>, Arc<dyn Fft<f32>>);

/// `get_fft(size)`: forward and inverse plans of one size from a fresh
/// planner, cached for the process like Chromium's `CACHE`.
fn get_fft(size: usize) -> FftPair {
    static CACHE: OnceLock<RwLock<HashMap<usize, FftPair>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(pair) = cache.read().expect("fft cache").get(&size) {
        return pair.clone();
    }
    let mut planner = FftPlanner::new();
    let forward = planner.plan_fft_forward(size);
    let inverse = planner.plan_fft_inverse(size);
    let pair = (forward, inverse);
    let mut write = cache.write().expect("fft cache");
    write.entry(size).or_insert(pair).clone()
}

/// The even-size `RustFft` of `rustfft_ffi.rs` (every analyser size is a
/// power of two), with Blink's `FFTFrame` real/imaginary arrays.
#[derive(Clone)]
struct FftFrame {
    forward: Arc<dyn Fft<f32>>,
    twiddles: Vec<Complex<f32>>,
    half_size: usize,
    limit: usize,
    middle_index: Option<usize>,
    complex_data: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    size: usize,
    real: Vec<f32>,
    imag: Vec<f32>,
}

impl FftFrame {
    fn new(size: usize) -> Self {
        assert!(size.is_multiple_of(2));
        let half_size = size / 2;
        let (forward, inverse) = get_fft(half_size);
        let scratch_len = forward
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len());
        let twiddles = (0..half_size)
            .map(|k| {
                let angle = -2.0 * std::f64::consts::PI * (k as f64) / (size as f64);
                let (sin, cos) = angle.sin_cos();
                Complex::new((0.5 * cos) as f32, (0.5 * sin) as f32)
            })
            .collect();
        let limit = (half_size - 1) / 2;
        let middle_index = half_size.is_multiple_of(2).then_some(half_size / 2);
        let packed_size = size.div_ceil(2);
        FftFrame {
            forward,
            twiddles,
            half_size,
            limit,
            middle_index,
            complex_data: vec![Complex::new(0.0, 0.0); half_size],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            size,
            real: vec![0.0; packed_size],
            imag: vec![0.0; packed_size],
        }
    }

    /// `do_fft_even`: half-size complex FFT of the interleaved real input, then
    /// `reconstruct_real_fft_even`.
    fn do_fft(&mut self, data: &[f32]) {
        assert_eq!(data.len(), self.size);
        for (c, [re, im]) in self.complex_data.iter_mut().zip(data.as_chunks::<2>().0) {
            *c = Complex::new(*re, *im);
        }
        self.forward
            .process_with_scratch(&mut self.complex_data, &mut self.scratch);
        let z = &self.complex_data;
        let half_size = self.half_size;
        let real = &mut self.real;
        let imag = &mut self.imag;
        real[0] = z[0].re + z[0].im;
        imag[0] = z[0].re - z[0].im;
        for k in 1..=self.limit {
            let z_k = z[k];
            let z_mk = z[half_size - k];
            let f_even = Complex::new(0.5 * (z_k.re + z_mk.re), 0.5 * (z_k.im - z_mk.im));
            let f_odd_unscaled = Complex::new(z_k.im + z_mk.im, -(z_k.re - z_mk.re));
            let w_f_odd = self.twiddles[k] * f_odd_unscaled;
            let x_k = f_even + w_f_odd;
            let x_mk_conj = f_even - w_f_odd;
            real[k] = x_k.re;
            imag[k] = x_k.im;
            real[half_size - k] = x_mk_conj.re;
            imag[half_size - k] = -x_mk_conj.im;
        }
        if let Some(k) = self.middle_index {
            let z_k = z[k];
            real[k] = z_k.re;
            imag[k] = -z_k.im;
        }
    }
}

/// `ApplyWindow`: Blackman window (alpha 0.16) computed in double, applied
/// as a float multiply.
fn apply_window(p: &mut [f32]) {
    const ALPHA: f64 = 0.16;
    const A0: f64 = 0.5 * (1.0 - ALPHA);
    const A1: f64 = 0.5;
    const A2: f64 = 0.5 * ALPHA;
    let two_pi = std::f64::consts::PI * 2.0;
    let n = p.len();
    for (i, sample) in p.iter_mut().enumerate() {
        let x = i as f64 / n as f64;
        let window = A0 - A1 * (two_pi * x).cos() + A2 * ((two_pi * 2.0) * x).cos();
        *sample *= window as f32;
    }
}

/// `audio_utilities::LinearToDecibels`: `20 * log10f(linear)` in float.
fn linear_to_decibels(linear: f32) -> f32 {
    20.0f32 * linear.log10()
}

/// `static_cast<unsigned char>(ClampTo(value, 0, UCHAR_MAX))`.
fn clamp_to_byte(value: f64) -> u8 {
    if value >= 255.0 {
        255
    } else if value <= 0.0 {
        0
    } else {
        value as i32 as u8
    }
}

/// An `AnalyserNode` fed by the host (`RealtimeAnalyser` + `AnalyserHandler`).
#[derive(Clone)]
pub struct AudioAnalyzer {
    input_buffer: Vec<f32>,
    write_index: usize,
    assembler: QuantumAssembler,
    quanta: u64,
    last_analysis: Option<u64>,
    fft_size: u32,
    frame: Option<FftFrame>,
    magnitudes: Vec<f32>,
    smoothing_time_constant: f64,
    min_decibels: f64,
    max_decibels: f64,
}

impl fmt::Debug for AudioAnalyzer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioAnalyzer")
            .field("fft_size", &self.fft_size)
            .field("smoothing_time_constant", &self.smoothing_time_constant)
            .field("min_decibels", &self.min_decibels)
            .field("max_decibels", &self.max_decibels)
            .field("render_quanta", &self.quanta)
            .finish_non_exhaustive()
    }
}

impl Default for AudioAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioAnalyzer {
    /// An analyser with the `AnalyserNode` defaults: `fftSize` 2048,
    /// `smoothingTimeConstant` 0.8, decibel range -100..-30.
    pub fn new() -> Self {
        AudioAnalyzer {
            input_buffer: vec![0.0; INPUT_BUFFER_SIZE],
            write_index: 0,
            assembler: QuantumAssembler::new(),
            quanta: 0,
            last_analysis: None,
            fft_size: DEFAULT_FFT_SIZE,
            frame: None,
            magnitudes: vec![0.0; DEFAULT_FFT_SIZE as usize / 2],
            smoothing_time_constant: DEFAULT_SMOOTHING_TIME_CONSTANT,
            min_decibels: DEFAULT_MIN_DECIBELS,
            max_decibels: DEFAULT_MAX_DECIBELS,
        }
    }

    /// An analyser configured the way the reference's `AudioInputManager`
    /// configures its analysers: `fftSize` 256 and the given smoothing.
    pub fn for_input_manager(smoothing_time_constant: f64) -> Self {
        let mut analyser = Self::new();
        analyser.set_fft_size(256).expect("valid size");
        analyser.smoothing_time_constant = smoothing_time_constant.clamp(0.0, 1.0);
        analyser
    }

    /// `fftSize`.
    pub fn fft_size(&self) -> u32 {
        self.fft_size
    }

    /// `fftSize = size`: a power of two in 32..=32768. Changing the size
    /// clears the smoothed magnitudes.
    pub fn set_fft_size(&mut self, size: u32) -> Result<(), AnalyserError> {
        if !(MIN_FFT_SIZE..=MAX_FFT_SIZE).contains(&size) {
            return Err(AnalyserError::FftSizeOutOfRange(size));
        }
        if !size.is_power_of_two() {
            return Err(AnalyserError::FftSizeNotPowerOfTwo(size));
        }
        if self.fft_size != size {
            self.frame = None;
            self.magnitudes = vec![0.0; size as usize / 2];
            self.fft_size = size;
        }
        Ok(())
    }

    /// `frequencyBinCount`.
    pub fn frequency_bin_count(&self) -> usize {
        self.fft_size as usize / 2
    }

    /// `minDecibels`.
    pub fn min_decibels(&self) -> f64 {
        self.min_decibels
    }

    /// `minDecibels = value`: must be below `maxDecibels`.
    pub fn set_min_decibels(&mut self, value: f64) -> Result<(), AnalyserError> {
        if !value.is_finite() {
            return Err(AnalyserError::NonFinite(value));
        }
        if value < self.max_decibels {
            self.min_decibels = value;
            Ok(())
        } else {
            Err(AnalyserError::MinDecibels {
                value,
                max: self.max_decibels,
            })
        }
    }

    /// `maxDecibels`.
    pub fn max_decibels(&self) -> f64 {
        self.max_decibels
    }

    /// `maxDecibels = value`: must be above `minDecibels`.
    pub fn set_max_decibels(&mut self, value: f64) -> Result<(), AnalyserError> {
        if !value.is_finite() {
            return Err(AnalyserError::NonFinite(value));
        }
        if value > self.min_decibels {
            self.max_decibels = value;
            Ok(())
        } else {
            Err(AnalyserError::MaxDecibels {
                value,
                min: self.min_decibels,
            })
        }
    }

    /// Both decibel bounds at once (the constructor's `SetMinMaxDecibels`).
    pub fn set_decibel_range(&mut self, min: f64, max: f64) -> Result<(), AnalyserError> {
        if !min.is_finite() {
            return Err(AnalyserError::NonFinite(min));
        }
        if !max.is_finite() {
            return Err(AnalyserError::NonFinite(max));
        }
        if min >= max {
            return Err(AnalyserError::DecibelRange { min, max });
        }
        self.min_decibels = min;
        self.max_decibels = max;
        Ok(())
    }

    /// `smoothingTimeConstant`.
    pub fn smoothing_time_constant(&self) -> f64 {
        self.smoothing_time_constant
    }

    /// `smoothingTimeConstant = value`: within 0..=1.
    pub fn set_smoothing_time_constant(&mut self, value: f64) -> Result<(), AnalyserError> {
        if !value.is_finite() {
            return Err(AnalyserError::NonFinite(value));
        }
        if (0.0..=1.0).contains(&value) {
            self.smoothing_time_constant = value;
            Ok(())
        } else {
            Err(AnalyserError::Smoothing(value))
        }
    }

    /// Feeds interleaved float frames of `channels` channels. Frames join the
    /// analysis one complete 128-frame render quantum at a time; a partial
    /// quantum waits for the next write. A change of channel count mixes the
    /// frames already received with the previous layout.
    pub fn write(&mut self, interleaved: &[f32], channels: usize) {
        if channels == 0 {
            return;
        }
        for frame in interleaved.chunks_exact(channels) {
            let assembler = &mut self.assembler;
            if assembler.channels != channels {
                assembler.flush();
                assembler.channels = channels;
                if assembler.planar.len() < channels {
                    assembler.planar.resize(channels, Quantum::ZERO);
                }
            }
            let position = assembler.mixed + assembler.buffered;
            for (planar, &sample) in assembler.planar.iter_mut().zip(frame) {
                planar.0[position] = sample;
            }
            assembler.buffered += 1;
            if position + 1 == RENDER_QUANTUM_FRAMES {
                assembler.flush();
                let mono = assembler.mono;
                assembler.mixed = 0;
                self.write_input(&mono.0);
            }
        }
    }

    /// Feeds one planar render quantum: `channels[c]` holds the 128 frames of
    /// channel `c`. Any partial interleaved quantum is discarded first.
    pub fn write_planar_quantum(&mut self, channels: &[&[f32]]) {
        assert!(!channels.is_empty());
        for channel in channels {
            assert_eq!(channel.len(), RENDER_QUANTUM_FRAMES);
        }
        self.assembler.mixed = 0;
        self.assembler.buffered = 0;
        let mut planar = vec![Quantum::ZERO; channels.len()];
        for (dst, src) in planar.iter_mut().zip(channels) {
            dst.0.copy_from_slice(src);
        }
        let sources: Vec<&[f32]> = planar.iter().map(|q| &q.0[..]).collect();
        let mut mono = Quantum::ZERO;
        down_mix(&sources, &mut mono.0);
        self.write_input(&mono.0);
    }

    /// `WriteInput` after the down-mix: appends one mono quantum to the ring.
    fn write_input(&mut self, src: &[f32]) {
        let src = if src.len() > INPUT_BUFFER_SIZE {
            &src[src.len() - INPUT_BUFFER_SIZE..]
        } else {
            src
        };
        let frames_to_end = INPUT_BUFFER_SIZE - self.write_index;
        if src.len() <= frames_to_end {
            self.input_buffer[self.write_index..self.write_index + src.len()].copy_from_slice(src);
        } else {
            let (head, tail) = src.split_at(frames_to_end);
            self.input_buffer[self.write_index..].copy_from_slice(head);
            self.input_buffer[..tail.len()].copy_from_slice(tail);
        }
        self.write_index = (self.write_index + src.len()) % INPUT_BUFFER_SIZE;
        self.quanta += 1;
    }

    /// Number of complete render quanta written (the analyser's clock).
    pub fn render_quanta(&self) -> u64 {
        self.quanta
    }

    /// Frames of the current, incomplete render quantum.
    pub fn pending_frames(&self) -> usize {
        self.assembler.mixed + self.assembler.buffered
    }

    fn ring_sample(&self, index: usize) -> f32 {
        let fft_size = self.fft_size as usize;
        self.input_buffer
            [(index + self.write_index + INPUT_BUFFER_SIZE - fft_size) % INPUT_BUFFER_SIZE]
    }

    /// Runs `DoFFTAnalysis` when a render quantum has passed since the last
    /// analysis.
    fn analyse_if_time_advanced(&mut self) {
        if self.last_analysis.is_some_and(|last| self.quanta <= last) {
            return;
        }
        self.last_analysis = Some(self.quanta);
        self.do_fft_analysis();
    }

    /// `DoFFTAnalysis`.
    fn do_fft_analysis(&mut self) {
        let fft_size = self.fft_size as usize;
        let mut temporary: Vec<f32> = (0..fft_size).map(|i| self.ring_sample(i)).collect();
        apply_window(&mut temporary);
        let frame = self.frame.get_or_insert_with(|| FftFrame::new(fft_size));
        frame.do_fft(&temporary);
        // Blow away the packed nyquist component.
        frame.imag[0] = 0.0;
        // Normalize so an input sine wave at 0dBfs registers as 0dBfs.
        let magnitude_scale = 1.0 / fft_size as f64;
        let k = self.smoothing_time_constant.clamp(0.0, 1.0);
        for (i, destination) in self.magnitudes.iter_mut().enumerate() {
            let scalar_magnitude =
                f64::from(frame.real[i]).hypot(f64::from(frame.imag[i])) * magnitude_scale;
            let value = (k * f64::from(*destination) + (1.0 - k) * scalar_magnitude) as f32;
            *destination = if value.is_finite() { value } else { 0.0 };
        }
    }

    /// `getFloatFrequencyData(array)`: decibels of the smoothed magnitudes.
    pub fn get_float_frequency_data(&mut self, array: &mut [f32]) {
        self.analyse_if_time_advanced();
        for (dst, &linear) in array.iter_mut().zip(&self.magnitudes) {
            *dst = linear_to_decibels(linear);
        }
    }

    /// `getByteFrequencyData(array)`: decibels scaled from `minDecibels`
    /// (0) to `maxDecibels` (255).
    pub fn get_byte_frequency_data(&mut self, array: &mut [u8]) {
        self.analyse_if_time_advanced();
        let range_scale_factor = if self.max_decibels == self.min_decibels {
            1.0
        } else {
            1.0 / (self.max_decibels - self.min_decibels)
        };
        let min_decibels = self.min_decibels;
        for (dst, &linear) in array.iter_mut().zip(&self.magnitudes) {
            let db = f64::from(linear_to_decibels(linear));
            let scaled = 255.0 * (db - min_decibels) * range_scale_factor;
            *dst = clamp_to_byte(scaled);
        }
    }

    /// `getFloatTimeDomainData(array)`: the last `fftSize` samples.
    pub fn get_float_time_domain_data(&self, array: &mut [f32]) {
        let len = array.len().min(self.fft_size as usize);
        for (i, dst) in array[..len].iter_mut().enumerate() {
            *dst = self.ring_sample(i);
        }
    }

    /// `getByteTimeDomainData(array)`: the last `fftSize` samples scaled from
    /// -1..1 to 0..255 (128 = silence).
    pub fn get_byte_time_domain_data(&self, array: &mut [u8]) {
        let len = array.len().min(self.fft_size as usize);
        for (i, dst) in array[..len].iter_mut().enumerate() {
            let value = self.ring_sample(i);
            let scaled = f64::from(128.0f32 * (value + 1.0f32));
            *dst = clamp_to_byte(scaled);
        }
    }

    /// The smoothed linear magnitudes (`magnitude_buffer_`).
    pub fn magnitudes(&self) -> &[f32] {
        &self.magnitudes
    }

    /// Clears the input, the pending quantum, the clock and the magnitudes, as
    /// a freshly created node with the same settings.
    pub fn reset(&mut self) {
        self.input_buffer.fill(0.0);
        self.write_index = 0;
        self.assembler = QuantumAssembler::new();
        self.quanta = 0;
        self.last_analysis = None;
        self.magnitudes.fill(0.0);
    }
}

impl FrequencyDataSource for AudioAnalyzer {
    fn frequency_bin_count(&self) -> usize {
        AudioAnalyzer::frequency_bin_count(self)
    }

    fn get_byte_frequency_data(&mut self, array: &mut [u8]) {
        AudioAnalyzer::get_byte_frequency_data(self, array);
    }
}

#[cfg(test)]
mod tests;
