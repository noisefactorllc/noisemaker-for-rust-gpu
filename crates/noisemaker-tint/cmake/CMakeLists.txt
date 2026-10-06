# The noisemaker-tint native library: Dawn's own CMake build configured for
# Tint alone (the WGSL reader, the IR and its transforms, the MSL writer; no
# Dawn backend, no other reader or writer, no tools or tests), plus the C shim
# (../shim) that drives it the way Dawn's Metal backend does. build.rs
# configures this project with the pinned Dawn source and its pinned
# third-party directories and builds the nm_tint_shim target.

cmake_minimum_required(VERSION 3.22)
project(noisemaker_tint LANGUAGES C CXX)

foreach(var NM_DAWN_SOURCE_DIR NM_DAWN_COMMIT DAWN_ABSEIL_DIR DAWN_SPIRV_HEADERS_DIR)
  if(NOT ${var})
    message(FATAL_ERROR "${var} must be set (build.rs sets it)")
  endif()
endforeach()

macro(nm_force var value)
  set(${var} ${value} CACHE BOOL "" FORCE)
endmacro()

# No Dawn backend: Dawn's CMakeLists then adds only third_party, src/utils and
# src/tint.
nm_force(DAWN_ENABLE_D3D11 OFF)
nm_force(DAWN_ENABLE_D3D12 OFF)
nm_force(DAWN_ENABLE_METAL OFF)
nm_force(DAWN_ENABLE_NULL OFF)
nm_force(DAWN_ENABLE_WEBGPU_ON_WEBGPU OFF)
nm_force(DAWN_ENABLE_DESKTOP_GL OFF)
nm_force(DAWN_ENABLE_OPENGLES OFF)
nm_force(DAWN_ENABLE_VULKAN OFF)
nm_force(DAWN_ENABLE_SPIRV_VALIDATION OFF)
nm_force(DAWN_USE_GLFW OFF)
nm_force(DAWN_USE_X11 OFF)
nm_force(DAWN_USE_WAYLAND OFF)
nm_force(DAWN_BUILD_SAMPLES OFF)
nm_force(DAWN_BUILD_TESTS OFF)
nm_force(DAWN_BUILD_BENCHMARKS OFF)
nm_force(DAWN_BUILD_FUZZERS OFF)
nm_force(DAWN_BUILD_PROTOBUF OFF)
nm_force(DAWN_BUILD_NODE_BINDINGS OFF)
nm_force(DAWN_ENABLE_INSTALL OFF)
nm_force(DAWN_FETCH_DEPENDENCIES OFF)
set(DAWN_BUILD_MONOLITHIC_LIBRARY "OFF" CACHE STRING "" FORCE)

# Tint: the WGSL reader and the MSL writer only.
nm_force(TINT_BUILD_WGSL_READER ON)
nm_force(TINT_BUILD_MSL_WRITER ON)
nm_force(TINT_BUILD_SPV_READER OFF)
nm_force(TINT_BUILD_SPV_WRITER OFF)
nm_force(TINT_BUILD_GLSL_WRITER OFF)
nm_force(TINT_BUILD_GLSL_VALIDATOR OFF)
nm_force(TINT_BUILD_HLSL_WRITER OFF)
nm_force(TINT_BUILD_WGSL_WRITER OFF)
nm_force(TINT_BUILD_NULL_WRITER OFF)
nm_force(TINT_BUILD_IR_BINARY OFF)
nm_force(TINT_BUILD_CMD_TOOLS OFF)
nm_force(TINT_BUILD_TESTS OFF)
nm_force(TINT_BUILD_BENCHMARKS OFF)
nm_force(TINT_BUILD_FUZZERS OFF)
nm_force(TINT_BUILD_TINTD OFF)
nm_force(TINT_BUILD_MESA OFF)
nm_force(TINT_ENABLE_INSTALL OFF)
nm_force(TINT_RANDOMIZE_HASHES OFF)

add_subdirectory("${NM_DAWN_SOURCE_DIR}" dawn EXCLUDE_FROM_ALL)

add_library(nm_tint_shim STATIC "${CMAKE_CURRENT_LIST_DIR}/../shim/nm_tint.cc")
target_include_directories(nm_tint_shim PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../shim")
target_compile_definitions(nm_tint_shim PRIVATE "NM_DAWN_COMMIT=\"${NM_DAWN_COMMIT}\"")
set_target_properties(nm_tint_shim PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
target_link_libraries(nm_tint_shim PRIVATE
  tint_api
  tint_lang_core_ir
  tint_lang_msl_writer
  tint_lang_msl_writer_common
  tint_lang_wgsl
  tint_lang_wgsl_program
  tint_lang_wgsl_reader
  tint_utils_diagnostic
)
