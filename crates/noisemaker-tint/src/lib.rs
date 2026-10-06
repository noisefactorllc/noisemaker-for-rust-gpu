//! Tint, the WGSL compiler of Dawn and Chromium, at the Dawn revision pinned in
//! `dawn.json`: WGSL to MSL exactly as Dawn's Metal backend generates it.
//!
//! The build (`build.rs`) fetches and verifies the pinned Dawn source and the
//! third-party directories its CMake build reads, builds Tint with the WGSL
//! reader and the MSL writer only, and links it with a small C++ shim
//! (`shim/nm_tint.cc`). The shim parses WGSL the way Dawn parses a shader
//! module (`ShaderModule.cpp` `ParseWGSL`), lowers it to Tint's IR and
//! generates MSL with the options of [`MslOptions`]
//! (`metal/ShaderModuleMTL.mm` `TranslateToMSL`), and prepends the heading
//! Dawn prepends. [`dawn`] holds the option values Dawn's Metal backend uses
//! for a Chromium WebGPU device.

use std::ffi::{CStr, CString, c_char, c_int};
use std::fmt;

/// The Dawn revision this Tint is built from (`dawn.json`).
pub const DAWN_COMMIT: &str = env!("NM_TINT_DAWN_COMMIT");

mod ffi {
    use std::ffi::{c_char, c_int};

    pub const BINDING_UNIFORM: u32 = 0;
    pub const BINDING_STORAGE: u32 = 1;
    pub const BINDING_TEXTURE: u32 = 2;
    pub const BINDING_STORAGE_TEXTURE: u32 = 3;
    pub const BINDING_SAMPLER: u32 = 4;

    #[repr(C)]
    pub struct Binding {
        pub group: u32,
        pub binding: u32,
        pub kind: u32,
        pub slot: u32,
    }

    #[repr(C)]
    pub struct BufferSize {
        pub group: u32,
        pub binding: u32,
        pub index: u32,
    }

    #[repr(C)]
    pub struct Options {
        pub entry_point: *const c_char,
        pub remapped_entry_point: *const c_char,
        pub strip_all_names: u8,
        pub disable_robustness: u8,
        pub disable_integer_range_analysis: u8,
        pub disable_workgroup_init: u8,
        pub emit_vertex_point_size: u8,
        pub disable_polyfill_integer_div_mod: u8,
        pub fixed_sample_mask: u32,
        pub bindings: *const Binding,
        pub binding_count: usize,
        pub buffer_sizes: *const BufferSize,
        pub buffer_size_count: usize,
        pub has_buffer_sizes_offset: u8,
        pub buffer_sizes_offset: u32,
        pub has_immediate_binding: u8,
        pub immediate_slot: u32,
        pub has_depth_range_offsets: u8,
        pub depth_range_min_offset: u32,
        pub depth_range_max_offset: u32,
        pub has_vertex_pulling: u8,
        pub vertex_pulling_group: u32,
        pub scalarize_max_min_clamp: u8,
        pub disable_module_constant_f16: u8,
        pub polyfill_subgroup_broadcast_f16: u8,
        pub polyfill_clamp_float: u8,
        pub polyfill_unpack_2x16_snorm: u8,
        pub polyfill_unpack_2x16_unorm: u8,
        pub polyfill_tanh_f16: u8,
        pub replace_workgroup_bool_with_u32: u8,
        pub collapse_subgroup_min_max: u8,
        pub fix_u32_div_mod: u8,
        pub polyfill_bool_vec_dynamic_store: u8,
        pub disable_demote_to_helper: u8,
        pub allow_unsafe_apis: u8,
        pub dawn_heading: u8,
        pub strict_math: u8,
    }

    #[repr(C)]
    pub struct Output {
        pub msl: *mut c_char,
        pub msl_len: usize,
        pub error: *mut c_char,
        pub workgroup_size: [u32; 3],
        pub has_invariant_attribute: u8,
        pub needs_storage_buffer_sizes: u8,
    }

    unsafe extern "C" {
        pub fn nm_tint_wgsl_to_msl(
            wgsl: *const c_char,
            len: usize,
            options: *const Options,
            out: *mut Output,
        ) -> c_int;
        pub fn nm_tint_output_free(out: *mut Output);
        pub fn nm_tint_math_mode_pragma_available() -> c_int;
        pub fn nm_tint_dawn_revision() -> *const c_char;
    }
}

/// A shader stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    Vertex,
    Fragment,
    Compute,
}

/// The resource class of a binding (the member of `tint::Bindings` it is
/// remapped in).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceClass {
    Uniform,
    Storage,
    Texture,
    StorageTexture,
    Sampler,
}

/// A WGSL binding point remapped to a Metal argument-table index:
/// `[[buffer(slot)]]`, `[[texture(slot)]]` or `[[sampler(slot)]]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Binding {
    pub group: u32,
    pub binding: u32,
    pub class: ResourceClass,
    pub slot: u32,
}

/// Where a storage buffer's byte size is read for `arrayLength` and the
/// robustness clamps of its runtime-sized array: element `index` of the
/// `u32` array at `buffer_sizes_offset` in the immediate block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferSize {
    pub group: u32,
    pub binding: u32,
    pub index: u32,
}

/// `tint::msl::writer::Options::Workarounds`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Workarounds {
    pub scalarize_max_min_clamp: bool,
    pub disable_module_constant_f16: bool,
    pub polyfill_subgroup_broadcast_f16: bool,
    pub polyfill_clamp_float: bool,
    pub polyfill_unpack_2x16_snorm: bool,
    pub polyfill_unpack_2x16_unorm: bool,
    pub polyfill_tanh_f16: bool,
    pub replace_workgroup_bool_with_u32: bool,
    pub collapse_subgroup_min_max: bool,
    pub fix_u32_div_mod: bool,
    pub polyfill_bool_vec_dynamic_store: bool,
}

/// The floating-point math mode of Dawn's heading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MathMode {
    /// `#pragma METAL fp math_mode(relaxed)` (WebGPU's default).
    Relaxed,
    /// `#pragma METAL fp math_mode(safe)` (Dawn's strict math).
    Safe,
}

/// The `tint::msl::writer::Options` members Dawn sets, the WGSL reader's
/// allowed features and Dawn's MSL heading.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MslOptions {
    /// The WGSL entry point to translate.
    pub entry_point: String,
    /// The name of the MSL function generated for it.
    pub remapped_entry_point: String,
    pub strip_all_names: bool,
    pub disable_robustness: bool,
    pub disable_integer_range_analysis: bool,
    pub disable_workgroup_init: bool,
    pub emit_vertex_point_size: bool,
    pub disable_polyfill_integer_div_mod: bool,
    pub fixed_sample_mask: u32,
    /// The Metal index of every binding visible to the stage.
    pub bindings: Vec<Binding>,
    /// `array_length_from_constants.bindpoint_to_size_index`.
    pub buffer_sizes: Vec<BufferSize>,
    /// `array_length_from_constants.buffer_sizes_offset` (bytes).
    pub buffer_sizes_offset: Option<u32>,
    /// `immediate_binding_point`: the Metal buffer index of the immediate
    /// block.
    pub immediate_slot: Option<u32>,
    /// `depth_range_offsets`: the byte offsets of the minimum and maximum
    /// depth in the immediate block, which clamp a written `frag_depth`.
    pub depth_range_offsets: Option<(u32, u32)>,
    /// `vertex_pulling_config` with this pulling group and no vertex buffer.
    pub vertex_pulling_group: Option<u32>,
    pub workarounds: Workarounds,
    pub disable_demote_to_helper: bool,
    /// Parse with the language features and extensions a device of an
    /// instance with Dawn's `allow_unsafe_apis` toggle admits.
    pub allow_unsafe_apis: bool,
    /// Prepend Dawn's heading: the `-Wall` suppression and, where Metal has
    /// the pragma (macOS 15, iOS 18), this math mode.
    pub heading: Option<MathMode>,
}

/// Generated MSL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Msl {
    pub source: String,
    /// `workgroup_info` of a compute entry point (zero otherwise).
    pub workgroup_size: [u32; 3],
    pub has_invariant_attribute: bool,
    pub needs_storage_buffer_sizes: bool,
}

/// A WGSL parse, IR or MSL generation error, with Tint's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

fn flag(b: bool) -> u8 {
    u8::from(b)
}

fn c_string(s: &str, what: &str) -> Result<CString, Error> {
    CString::new(s).map_err(|_| Error(format!("{what} contains a NUL byte")))
}

/// Translate `wgsl` to MSL with `options`.
pub fn wgsl_to_msl(wgsl: &str, options: &MslOptions) -> Result<Msl, Error> {
    let entry_point = c_string(&options.entry_point, "the entry point name")?;
    let remapped = c_string(
        &options.remapped_entry_point,
        "the remapped entry point name",
    )?;
    let bindings: Vec<ffi::Binding> = options
        .bindings
        .iter()
        .map(|b| ffi::Binding {
            group: b.group,
            binding: b.binding,
            kind: match b.class {
                ResourceClass::Uniform => ffi::BINDING_UNIFORM,
                ResourceClass::Storage => ffi::BINDING_STORAGE,
                ResourceClass::Texture => ffi::BINDING_TEXTURE,
                ResourceClass::StorageTexture => ffi::BINDING_STORAGE_TEXTURE,
                ResourceClass::Sampler => ffi::BINDING_SAMPLER,
            },
            slot: b.slot,
        })
        .collect();
    let sizes: Vec<ffi::BufferSize> = options
        .buffer_sizes
        .iter()
        .map(|s| ffi::BufferSize {
            group: s.group,
            binding: s.binding,
            index: s.index,
        })
        .collect();
    let w = &options.workarounds;
    let raw = ffi::Options {
        entry_point: entry_point.as_ptr(),
        remapped_entry_point: remapped.as_ptr(),
        strip_all_names: flag(options.strip_all_names),
        disable_robustness: flag(options.disable_robustness),
        disable_integer_range_analysis: flag(options.disable_integer_range_analysis),
        disable_workgroup_init: flag(options.disable_workgroup_init),
        emit_vertex_point_size: flag(options.emit_vertex_point_size),
        disable_polyfill_integer_div_mod: flag(options.disable_polyfill_integer_div_mod),
        fixed_sample_mask: options.fixed_sample_mask,
        bindings: bindings.as_ptr(),
        binding_count: bindings.len(),
        buffer_sizes: sizes.as_ptr(),
        buffer_size_count: sizes.len(),
        has_buffer_sizes_offset: flag(options.buffer_sizes_offset.is_some()),
        buffer_sizes_offset: options.buffer_sizes_offset.unwrap_or(0),
        has_immediate_binding: flag(options.immediate_slot.is_some()),
        immediate_slot: options.immediate_slot.unwrap_or(0),
        has_depth_range_offsets: flag(options.depth_range_offsets.is_some()),
        depth_range_min_offset: options.depth_range_offsets.map_or(0, |o| o.0),
        depth_range_max_offset: options.depth_range_offsets.map_or(0, |o| o.1),
        has_vertex_pulling: flag(options.vertex_pulling_group.is_some()),
        vertex_pulling_group: options.vertex_pulling_group.unwrap_or(0),
        scalarize_max_min_clamp: flag(w.scalarize_max_min_clamp),
        disable_module_constant_f16: flag(w.disable_module_constant_f16),
        polyfill_subgroup_broadcast_f16: flag(w.polyfill_subgroup_broadcast_f16),
        polyfill_clamp_float: flag(w.polyfill_clamp_float),
        polyfill_unpack_2x16_snorm: flag(w.polyfill_unpack_2x16_snorm),
        polyfill_unpack_2x16_unorm: flag(w.polyfill_unpack_2x16_unorm),
        polyfill_tanh_f16: flag(w.polyfill_tanh_f16),
        replace_workgroup_bool_with_u32: flag(w.replace_workgroup_bool_with_u32),
        collapse_subgroup_min_max: flag(w.collapse_subgroup_min_max),
        fix_u32_div_mod: flag(w.fix_u32_div_mod),
        polyfill_bool_vec_dynamic_store: flag(w.polyfill_bool_vec_dynamic_store),
        disable_demote_to_helper: flag(options.disable_demote_to_helper),
        allow_unsafe_apis: flag(options.allow_unsafe_apis),
        dawn_heading: flag(options.heading.is_some()),
        strict_math: flag(options.heading == Some(MathMode::Safe)),
    };
    let mut out = ffi::Output {
        msl: std::ptr::null_mut(),
        msl_len: 0,
        error: std::ptr::null_mut(),
        workgroup_size: [0; 3],
        has_invariant_attribute: 0,
        needs_storage_buffer_sizes: 0,
    };
    // SAFETY: every pointer in `raw` points into a buffer that outlives the
    // call (the CStrings and vectors above); `wgsl` is passed with its byte
    // length (the shim does not need a terminator); `out` is a valid,
    // writable `Output` that the shim fills and `nm_tint_output_free`
    // releases.
    let ok: c_int = unsafe {
        ffi::nm_tint_wgsl_to_msl(wgsl.as_ptr().cast::<c_char>(), wgsl.len(), &raw, &mut out)
    };
    // SAFETY: the shim sets `msl` (with its length) on success and `error` on
    // failure, both NUL-terminated and valid until `nm_tint_output_free`.
    let result = unsafe {
        if ok == 1 && !out.msl.is_null() {
            let bytes = std::slice::from_raw_parts(out.msl.cast::<u8>(), out.msl_len);
            Ok(Msl {
                source: String::from_utf8_lossy(bytes).into_owned(),
                workgroup_size: out.workgroup_size,
                has_invariant_attribute: out.has_invariant_attribute != 0,
                needs_storage_buffer_sizes: out.needs_storage_buffer_sizes != 0,
            })
        } else if !out.error.is_null() {
            Err(Error(
                CStr::from_ptr(out.error).to_string_lossy().into_owned(),
            ))
        } else {
            Err(Error("Tint produced neither MSL nor an error".into()))
        }
    };
    // SAFETY: `out` was filled by `nm_tint_wgsl_to_msl` and is freed once.
    unsafe { ffi::nm_tint_output_free(&mut out) };
    result
}

/// `true` where Dawn adds the math-mode pragma to its heading
/// (`@available(macOS 15.0, iOS 18.0, *)`).
pub fn math_mode_pragma_available() -> bool {
    // SAFETY: a pure query without arguments.
    unsafe { ffi::nm_tint_math_mode_pragma_available() != 0 }
}

/// The Dawn revision the linked shim was built from (equal to
/// [`DAWN_COMMIT`]).
pub fn linked_dawn_revision() -> &'static str {
    // SAFETY: the shim returns a pointer to a static NUL-terminated string.
    unsafe { CStr::from_ptr(ffi::nm_tint_dawn_revision()) }
        .to_str()
        .expect("the revision is ASCII")
}

/// The option values of Dawn's Metal backend at the pinned revision, for a
/// device of Chromium's WebGPU implementation (robustness on, symbol renaming
/// on, unsafe APIs allowed as `--enable-unsafe-webgpu` allows them).
pub mod dawn {
    use super::{MathMode, MslOptions, Stage, Workarounds};

    /// `DeviceBase::GetIsolatedEntryPointName()` without an isolation key.
    /// (Chromium passes the page's origin as the key, which only appends its
    /// hexadecimal bytes to this name; the name never changes the code.)
    pub const ENTRY_POINT: &str = "dawn_entry_point";

    /// `kPullingBufferBindingSet` (`kMaxBindGroups`): the bind group the
    /// vertex-pulling transform reserves.
    pub const VERTEX_PULLING_GROUP: u32 = 4;

    /// `kImmediateBlockBufferSlot`: the Metal buffer index of Dawn's
    /// immediate block (pipeline immediates, then storage-buffer sizes).
    pub const IMMEDIATE_BLOCK_SLOT: u32 = 30;

    /// The GPU vendor, as Dawn's `gpu_info` classifies it.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum Vendor {
        Apple,
        Intel,
        Amd,
        Other,
    }

    /// The properties of a Metal GPU that select Dawn's shader toggles
    /// (`metal/PhysicalDeviceMTL.mm` `SetupBackendDeviceToggles`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MetalGpu {
        pub vendor: Vendor,
        /// `[device supportsFamily:MTLGPUFamilyApple9]` (Apple M3 and later).
        pub apple9: bool,
    }

    impl MetalGpu {
        /// The `tint::msl::writer::Options::Workarounds` Dawn's toggles
        /// enable on this GPU.
        pub fn workarounds(&self) -> Workarounds {
            let intel = self.vendor == Vendor::Intel;
            let amd = self.vendor == Vendor::Amd;
            Workarounds {
                // ScalarizeMaxMinClamp: never defaulted on Metal.
                scalarize_max_min_clamp: false,
                // MetalDisableModuleConstantF16: Intel.
                disable_module_constant_f16: intel,
                // EnableSubgroupsIntelGen9: opt-in only.
                polyfill_subgroup_broadcast_f16: false,
                // MetalPolyfillClampFloat: AMD.
                polyfill_clamp_float: amd,
                // MetalPolyfillUnpack2x16snorm: AMD, Apple9 family.
                polyfill_unpack_2x16_snorm: amd || self.apple9,
                // MetalPolyfillUnpack2x16unorm: Apple9 family.
                polyfill_unpack_2x16_unorm: self.apple9,
                // MetalPolyfillTanhF16: AMD.
                polyfill_tanh_f16: amd,
                // MetalReplaceWorkgroupBoolWithU32: AMD, Intel.
                replace_workgroup_bool_with_u32: amd || intel,
                // CollapseSubgroupMinMax: AMD.
                collapse_subgroup_min_max: amd,
                // MetalFixU32DivMod: Apple.
                fix_u32_div_mod: self.vendor == Vendor::Apple,
                // MetalPolyfillBoolVecDynamicStore: Intel (macOS).
                polyfill_bool_vec_dynamic_store: intel,
            }
        }
    }

    /// The options `TranslateToMSL` passes for `entry_point` of `stage`,
    /// before the pipeline-layout dependent members (`bindings`,
    /// `buffer_sizes`, `buffer_sizes_offset`, and `depth_range_offsets` for a
    /// pipeline whose fragment stage writes `frag_depth`), which the caller
    /// sets from its own Metal argument-table assignment, as it sets
    /// `immediate_slot` when its immediate block is not at
    /// [`IMMEDIATE_BLOCK_SLOT`].
    ///
    /// * `point_list`: the render pipeline draws `point-list`
    ///   (`emit_vertex_point_size` for its vertex stage).
    /// * `sample_mask`: the render pipeline's `multisample.mask` (passed for
    ///   the fragment stage; Dawn passes `0xFFFFFFFF` for the others).
    pub fn options(
        stage: Stage,
        entry_point: &str,
        gpu: &MetalGpu,
        point_list: bool,
        sample_mask: u32,
    ) -> MslOptions {
        MslOptions {
            entry_point: entry_point.to_owned(),
            remapped_entry_point: ENTRY_POINT.to_owned(),
            // !DisableSymbolRenaming
            strip_all_names: true,
            // !IsRobustnessEnabled()
            disable_robustness: false,
            // !EnableIntegerRangeAnalysisInRobustness: the toggle follows
            // Chromium's kWebGPUEnableRangeAnalysisForRobustness, enabled in
            // Chromium 153 (its MSL elides the clamps the analysis proves
            // unnecessary).
            disable_integer_range_analysis: false,
            // DisableWorkgroupInit
            disable_workgroup_init: false,
            emit_vertex_point_size: stage == Stage::Vertex && point_list,
            // DisablePolyfillsOnIntegerDivisonAndModulo
            disable_polyfill_integer_div_mod: false,
            fixed_sample_mask: if stage == Stage::Fragment {
                sample_mask
            } else {
                0xFFFF_FFFF
            },
            bindings: Vec::new(),
            buffer_sizes: Vec::new(),
            buffer_sizes_offset: None,
            immediate_slot: Some(IMMEDIATE_BLOCK_SLOT),
            depth_range_offsets: None,
            // MetalEnableVertexPulling (on for Metal); the pipelines have no
            // vertex buffer.
            vertex_pulling_group: (stage == Stage::Vertex).then_some(VERTEX_PULLING_GROUP),
            workarounds: gpu.workarounds(),
            // DisableDemoteToHelper (on for Metal)
            disable_demote_to_helper: true,
            allow_unsafe_apis: true,
            // GetStrictMath().value_or(false): WebGPU never sets strict math.
            heading: Some(MathMode::Relaxed),
        }
    }
}
