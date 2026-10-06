// The noisemaker-tint shim: WGSL to MSL with the Tint of the pinned Dawn
// revision, following what Dawn does for a WebGPU shader module on its Metal
// backend:
//
//   * src/dawn/native/ShaderModule.cpp ParseWGSL: tint::wgsl::reader::Parse of
//     a Source::File with an empty path and the device's allowed features
//     (src/dawn/native/Device.cpp SetWGSLExtensionAllowList, Instance.cpp
//     GatherWGSLFeatures);
//   * src/dawn/native/metal/ShaderModuleMTL.mm TranslateToMSL:
//     tint::wgsl::reader::ProgramToLoweredIR, tint::msl::writer::Generate with
//     the options the caller passes (nm_tint_options mirrors the members Dawn
//     sets), and the heading Dawn prepends to the generated MSL.
//
// Nothing here changes Tint's behaviour; the options are the only inputs.

#include "nm_tint.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

#include "src/tint/api/common/binding_point.h"
#include "src/tint/api/common/bindings.h"
#include "src/tint/api/common/vertex_pulling_config.h"
#include "src/tint/lang/core/ir/module.h"
#include "src/tint/lang/msl/writer/common/options.h"
#include "src/tint/lang/msl/writer/common/output.h"
#include "src/tint/lang/msl/writer/writer.h"
#include "src/tint/lang/wgsl/allowed_features.h"
#include "src/tint/lang/wgsl/enums.h"
#include "src/tint/lang/wgsl/feature_status.h"
#include "src/tint/lang/wgsl/program/program.h"
#include "src/tint/lang/wgsl/reader/reader.h"
#include "src/tint/utils/diagnostic/source.h"

#ifndef NM_DAWN_COMMIT
#error "NM_DAWN_COMMIT must name the Dawn revision the shim is built from"
#endif

namespace {

char* CopyString(std::string_view s) {
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (out == nullptr) {
        std::abort();
    }
    std::memcpy(out, s.data(), s.size());
    out[s.size()] = '\0';
    return out;
}

int Fail(nm_tint_output* out, std::string_view message) {
    out->error = CopyString(message);
    return 0;
}

// Instance.cpp GatherWGSLFeatures (without ExposeWGSLTestingFeatures) and
// Device.cpp SetWGSLExtensionAllowList for a device with no WGSL-extension
// features enabled (the reference requests only float32-filterable).
tint::wgsl::AllowedFeatures DeviceAllowedFeatures(bool allowUnsafeAPIs) {
    tint::wgsl::AllowedFeatures allowed;
    for (auto feature : tint::wgsl::kAllLanguageFeatures) {
        switch (feature) {
            case tint::wgsl::LanguageFeature::kChromiumTestingUnimplemented:
            case tint::wgsl::LanguageFeature::kChromiumTestingUnsafeExperimental:
            case tint::wgsl::LanguageFeature::kChromiumTestingExperimental:
            case tint::wgsl::LanguageFeature::kChromiumTestingShippedWithKillswitch:
            case tint::wgsl::LanguageFeature::kChromiumTestingShipped:
                continue;
            default:
                break;
        }
        bool enable = false;
        switch (tint::wgsl::GetLanguageFeatureStatus(feature)) {
            case tint::wgsl::FeatureStatus::kUnknown:
            case tint::wgsl::FeatureStatus::kUnimplemented:
                enable = false;
                break;
            case tint::wgsl::FeatureStatus::kUnsafeExperimental:
            case tint::wgsl::FeatureStatus::kExperimental:
                enable = allowUnsafeAPIs;
                break;
            case tint::wgsl::FeatureStatus::kShippedWithKillswitch:
            case tint::wgsl::FeatureStatus::kShipped:
                enable = true;
                break;
        }
        if (enable) {
            allowed.features.insert(feature);
        }
    }
    if (allowUnsafeAPIs) {
        allowed.extensions.insert(tint::wgsl::Extension::kChromiumDisableUniformityAnalysis);
    }
    return allowed;
}

tint::msl::writer::Options MslOptions(const nm_tint_options& o) {
    tint::msl::writer::Options options;
    options.entry_point_name = o.entry_point ? o.entry_point : "";
    options.remapped_entry_point_name = o.remapped_entry_point ? o.remapped_entry_point : "";
    options.strip_all_names = o.strip_all_names != 0;
    options.disable_robustness = o.disable_robustness != 0;
    options.disable_integer_range_analysis = o.disable_integer_range_analysis != 0;
    options.disable_workgroup_init = o.disable_workgroup_init != 0;
    options.emit_vertex_point_size = o.emit_vertex_point_size != 0;
    options.disable_polyfill_integer_div_mod = o.disable_polyfill_integer_div_mod != 0;
    options.fixed_sample_mask = o.fixed_sample_mask;

    for (size_t i = 0; i < o.binding_count; ++i) {
        const nm_tint_binding& b = o.bindings[i];
        tint::BindingPoint src{.group = b.group, .binding = b.binding};
        tint::BindingPoint dst{.group = 0, .binding = b.slot};
        switch (b.kind) {
            case NM_TINT_BINDING_UNIFORM:
                options.bindings.uniform.emplace(src, dst);
                break;
            case NM_TINT_BINDING_STORAGE:
                options.bindings.storage.emplace(src, dst);
                break;
            case NM_TINT_BINDING_TEXTURE:
                options.bindings.texture.emplace(src, dst);
                break;
            case NM_TINT_BINDING_STORAGE_TEXTURE:
                options.bindings.storage_texture.emplace(src, dst);
                break;
            case NM_TINT_BINDING_SAMPLER:
                options.bindings.sampler.emplace(src, dst);
                break;
            default:
                break;
        }
    }

    for (size_t i = 0; i < o.buffer_size_count; ++i) {
        const nm_tint_buffer_size& s = o.buffer_sizes[i];
        options.array_length_from_constants.bindpoint_to_size_index.emplace(
            tint::BindingPoint{.group = s.group, .binding = s.binding}, s.index);
    }
    if (o.has_buffer_sizes_offset) {
        options.array_length_from_constants.buffer_sizes_offset = o.buffer_sizes_offset;
    }
    if (o.has_immediate_binding) {
        options.immediate_binding_point = tint::BindingPoint{.group = 0, .binding = o.immediate_slot};
    }
    if (o.has_depth_range_offsets) {
        options.depth_range_offsets = tint::msl::writer::Options::RangeOffsets{
            .min = o.depth_range_min_offset, .max = o.depth_range_max_offset};
    }
    if (o.has_vertex_pulling) {
        tint::VertexPullingConfig cfg;
        cfg.pulling_group = o.vertex_pulling_group;
        options.vertex_pulling_config = std::move(cfg);
    }

    options.workarounds.scalarize_max_min_clamp = o.scalarize_max_min_clamp != 0;
    options.workarounds.disable_module_constant_f16 = o.disable_module_constant_f16 != 0;
    options.workarounds.polyfill_subgroup_broadcast_f16 = o.polyfill_subgroup_broadcast_f16 != 0;
    options.workarounds.polyfill_clamp_float = o.polyfill_clamp_float != 0;
    options.workarounds.polyfill_unpack_2x16_snorm = o.polyfill_unpack_2x16_snorm != 0;
    options.workarounds.polyfill_unpack_2x16_unorm = o.polyfill_unpack_2x16_unorm != 0;
    options.workarounds.polyfill_tanh_f16 = o.polyfill_tanh_f16 != 0;
    options.workarounds.replace_workgroup_bool_with_u32 = o.replace_workgroup_bool_with_u32 != 0;
    options.workarounds.collapse_subgroup_min_max = o.collapse_subgroup_min_max != 0;
    options.workarounds.fix_u32_div_mod = o.fix_u32_div_mod != 0;
    options.workarounds.polyfill_bool_vec_dynamic_store = o.polyfill_bool_vec_dynamic_store != 0;
    options.extensions.disable_demote_to_helper = o.disable_demote_to_helper != 0;
    return options;
}

}  // namespace

extern "C" int nm_tint_math_mode_pragma_available(void) {
#if defined(__APPLE__)
    if (__builtin_available(macOS 15.0, iOS 18.0, *)) {
        return 1;
    }
#endif
    return 0;
}

extern "C" const char* nm_tint_dawn_revision(void) {
    return NM_DAWN_COMMIT;
}

extern "C" int nm_tint_wgsl_to_msl(const char* wgsl,
                                   size_t len,
                                   const nm_tint_options* options,
                                   nm_tint_output* out) {
    *out = nm_tint_output{};
    if (wgsl == nullptr || options == nullptr) {
        return Fail(out, "nm_tint_wgsl_to_msl: null argument");
    }

    // ShaderModule.cpp ParseWGSL.
    tint::Source::File file("", std::string_view(wgsl, len));
    tint::wgsl::reader::Options readerOptions;
    readerOptions.allowed_features = DeviceAllowedFeatures(options->allow_unsafe_apis != 0);
    tint::Program program = tint::wgsl::reader::Parse(&file, readerOptions);
    if (!program.IsValid()) {
        return Fail(out, "Error while parsing WGSL: " + program.Diagnostics().Str() + "\n");
    }

    // ShaderModuleMTL.mm TranslateToMSL.
    tint::wgsl::reader::IROptions irOptions{};
    tint::Result<tint::core::ir::Module> ir =
        tint::wgsl::reader::ProgramToLoweredIR(program, irOptions);
    if (ir != tint::Success) {
        return Fail(out, "An error occurred while generating Tint IR\n" + ir.Failure().reason);
    }
    tint::Result<tint::msl::writer::Output> result =
        tint::msl::writer::Generate(ir.Get(), MslOptions(*options));
    if (result != tint::Success) {
        return Fail(out, "An error occurred while generating MSL:\n" + result.Failure().reason);
    }

    std::string msl = std::move(result->msl);
    if (options->dawn_heading) {
        std::string math_mode_heading;
        if (nm_tint_math_mode_pragma_available()) {
            math_mode_heading = "\n#pragma METAL fp math_mode(";
            math_mode_heading += options->strict_math ? "safe" : "relaxed";
            math_mode_heading += +")\n";
        }
        msl = R"(#ifdef __clang__
#pragma clang diagnostic ignored "-Wall"
#endif
)" + math_mode_heading +
              msl;
    }

    out->msl = CopyString(msl);
    out->msl_len = msl.size();
    out->workgroup_size[0] = result->workgroup_info.x;
    out->workgroup_size[1] = result->workgroup_info.y;
    out->workgroup_size[2] = result->workgroup_info.z;
    out->has_invariant_attribute = result->has_invariant_attribute ? 1 : 0;
    out->needs_storage_buffer_sizes = result->needs_storage_buffer_sizes ? 1 : 0;
    return 1;
}

extern "C" void nm_tint_output_free(nm_tint_output* out) {
    if (out == nullptr) {
        return;
    }
    std::free(out->msl);
    std::free(out->error);
    *out = nm_tint_output{};
}
