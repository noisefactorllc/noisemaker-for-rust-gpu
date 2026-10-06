// C interface of the noisemaker-tint shim: Tint (built from the pinned Dawn
// revision) translating WGSL to MSL the way Dawn's Metal backend does.
//
// Every struct here is mirrored field for field by `#[repr(C)]` types in
// src/lib.rs; booleans are uint8_t (0 or 1).

#ifndef NOISEMAKER_TINT_SHIM_H_
#define NOISEMAKER_TINT_SHIM_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// The resource class of a binding (the member of tint::Bindings it goes to).
enum {
    NM_TINT_BINDING_UNIFORM = 0,
    NM_TINT_BINDING_STORAGE = 1,
    NM_TINT_BINDING_TEXTURE = 2,
    NM_TINT_BINDING_STORAGE_TEXTURE = 3,
    NM_TINT_BINDING_SAMPLER = 4,
};

// One WGSL binding point remapped to a Metal argument-table index.
typedef struct nm_tint_binding {
    uint32_t group;
    uint32_t binding;
    uint32_t kind;  // NM_TINT_BINDING_*
    uint32_t slot;  // [[buffer(n)]], [[texture(n)]] or [[sampler(n)]]
} nm_tint_binding;

// The index of a storage buffer's size in the immediate block's size array.
typedef struct nm_tint_buffer_size {
    uint32_t group;
    uint32_t binding;
    uint32_t index;
} nm_tint_buffer_size;

// tint::msl::writer::Options (the members Dawn sets), plus the WGSL reader's
// allowed features.
typedef struct nm_tint_options {
    const char* entry_point;
    const char* remapped_entry_point;
    uint8_t strip_all_names;
    uint8_t disable_robustness;
    uint8_t disable_integer_range_analysis;
    uint8_t disable_workgroup_init;
    uint8_t emit_vertex_point_size;
    uint8_t disable_polyfill_integer_div_mod;
    uint32_t fixed_sample_mask;

    const nm_tint_binding* bindings;
    size_t binding_count;

    // array_length_from_constants
    const nm_tint_buffer_size* buffer_sizes;
    size_t buffer_size_count;
    uint8_t has_buffer_sizes_offset;
    uint32_t buffer_sizes_offset;

    // immediate_binding_point = {0, immediate_slot}
    uint8_t has_immediate_binding;
    uint32_t immediate_slot;

    // depth_range_offsets = {min, max} (byte offsets in the immediate block)
    uint8_t has_depth_range_offsets;
    uint32_t depth_range_min_offset;
    uint32_t depth_range_max_offset;

    // vertex_pulling_config = {pulling_group, no vertex buffers}
    uint8_t has_vertex_pulling;
    uint32_t vertex_pulling_group;

    // Options::Workarounds
    uint8_t scalarize_max_min_clamp;
    uint8_t disable_module_constant_f16;
    uint8_t polyfill_subgroup_broadcast_f16;
    uint8_t polyfill_clamp_float;
    uint8_t polyfill_unpack_2x16_snorm;
    uint8_t polyfill_unpack_2x16_unorm;
    uint8_t polyfill_tanh_f16;
    uint8_t replace_workgroup_bool_with_u32;
    uint8_t collapse_subgroup_min_max;
    uint8_t fix_u32_div_mod;
    uint8_t polyfill_bool_vec_dynamic_store;

    // Options::Extensions
    uint8_t disable_demote_to_helper;

    // The WGSL reader admits what a device of an instance with Dawn's
    // AllowUnsafeAPIs toggle admits (experimental language features and the
    // chromium_disable_uniformity_analysis extension).
    uint8_t allow_unsafe_apis;

    // Prepend Dawn's MSL heading: the -Wall suppression and, when Metal
    // supports the pragma (macOS 15, iOS 18), the floating-point math mode
    // (relaxed, or safe for strict math).
    uint8_t dawn_heading;
    uint8_t strict_math;
} nm_tint_options;

// The result of a translation. Strings are NUL-terminated and owned by the
// shim; release them with nm_tint_output_free.
typedef struct nm_tint_output {
    char* msl;
    size_t msl_len;
    char* error;
    uint32_t workgroup_size[3];
    uint8_t has_invariant_attribute;
    uint8_t needs_storage_buffer_sizes;
} nm_tint_output;

// Translate `wgsl` (`len` bytes of UTF-8) to MSL. Returns 1 on success (msl
// set) and 0 on failure (error set). `out` is always filled.
int nm_tint_wgsl_to_msl(const char* wgsl,
                        size_t len,
                        const nm_tint_options* options,
                        nm_tint_output* out);

void nm_tint_output_free(nm_tint_output* out);

// 1 when this process runs where Dawn adds the math-mode pragma
// (`@available(macOS 15.0, iOS 18.0, *)`), else 0.
int nm_tint_math_mode_pragma_available(void);

// The Dawn revision the shim was built from (NUL-terminated, static).
const char* nm_tint_dawn_revision(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // NOISEMAKER_TINT_SHIM_H_
