<sub>Open source from <a href="https://noisefactor.io">Noise Factor</a> &middot; <a href="https://github.com/noisefactorllc">more projects</a></sub>

# Noisemaker for Rust GPU

> This package supports the "Export Shader Pipeline" feature in Noisedeck.app. The feature runs shader compositions on other platforms. Noise Factor derives this package from the upstream Noisemaker Engine project and tests it for pixel-level parity.

**Noisemaker** is a procedural visual engine. Small text programs combine effects into chains, compile to a render graph, and run live as animated GPU textures. **Noisemaker for Rust GPU** is a Rust port of the reference engine ([noisefactorllc/noisemaker](https://github.com/noisefactorllc/noisemaker)) onto [wgpu](https://wgpu.rs):

- the Polymorphic DSL compiler (lexer, parser, validator, expander, resource allocation and render-graph compiler), with the reference's registries, `ProgramState` and DSL tooling;
- the full effect catalog: 210 effects and their 308 WGSL programs, byte copies of the reference's, embedded at compile time;
- the renderer: the reference's pipeline executor and WebGPU backend on wgpu, rendering the reference WGSL unmodified, and the host behavior of the reference demo page (parameter controls, media, text, meshes, traced overlays, MIDI and audio state, user-defined Portable effects).

On Metal the port compiles WGSL the way Chromium does: with Tint, the WGSL compiler of Dawn, built from the Dawn revision of Chromium 153.0.8010.12 with the options of Dawn's Metal backend. wgpu then compiles that MSL as a passthrough module. Other wgpu backends compile WGSL with naga.

## Requirements

- Rust 1.88 or newer. The workspace declares `rust-version = "1.88"`; `rust-toolchain.toml` selects the stable channel with rustfmt and clippy.
- A GPU adapter that wgpu can open.
- macOS with Metal is the qualified platform: the parity results below were measured on an Apple M4 under macOS 26.6.2. On other platforms wgpu uses its other backends and naga. That path is not parity-qualified: the workspace compiles for x86_64 Linux (`cargo check --workspace --all-targets --target x86_64-unknown-linux-gnu`), and no render on it has been compared with the reference.
- On macOS, for the Tint build: git, curl, tar and a C++17 compiler (the Xcode Command Line Tools provide them). CMake 3.22 or newer is used when it is on `PATH`; otherwise the build downloads CMake 3.23.3 (see [The first build](#the-first-build)).
- For the parity gates only: Node.js, Python 3 (numpy and Pillow; `scripts/parity-summary` creates a virtual environment with them when they are missing) and a checkout of the reference engine.

## Quick start

Run the commands from the repository root.

Build the command-line renderer:

```sh
cargo build --release -p nm-render
```

List the catalog, and one effect's parameters, types, defaults, ranges and choices:

```sh
target/release/nm-render effects
target/release/nm-render effect synth/noise
```

Render a program to a PNG. This renders the flagship program for 10 seconds of frames at 60 frames per second and writes the last frame as `hero.t10.png`:

```sh
target/release/nm-render render --dsl parity/programs/hero.dsl --out hero.png --size 512 \
    --orientation presented --run-seconds 10 --sample-every 10
```

Without `--run-seconds`, `render` draws `--frames` frames (8) at one normalized loop time (`--time`, 0.25). `--orientation presented` writes the image as the reference's canvas shows it; the default, `texture`, writes rows in texture order, the order the parity protocol compares. `--param step_N.NAME=VALUE` sets a parameter as the demo's controls do, `--media IMAGE.png` replaces the default media image and `--obj FILE.obj` loads a mesh. `target/release/nm-render render --help` lists every option.

Render one loop of a program (10 seconds at 30 frames per second) to numbered PNG frames. `--mp4 loop.mp4` also encodes them with `ffmpeg` (H.264) when it is installed:

```sh
target/release/nm-render animate --dsl parity/programs/hero.dsl --out-dir frames --fps 30
```

Show a program live in a window. Saving the file reloads it; a version that does not compile is reported on standard error while the last good one keeps running:

```sh
cargo run --release -p noisemaker-for-rust-gpu --example viewer -- parity/programs/hero.dsl
```

Media steps show the reference demo's default media image, a test card that the catalog embeds, unless `--media` names another image. Text steps draw with the bundled Nunito font; other font families resolve through the installed system fonts.

### Library use

`crates/noisemaker-gpu/examples/render_dsl.rs` renders a program to a PNG through the library API, loading it the way the reference demo page loads it:

```sh
cargo run --release -p noisemaker-for-rust-gpu --example render_dsl -- parity/programs/hero.dsl hero.png --size 512
```

Its core:

```rust
use noisemaker_gpu::demo::{DemoHost, DemoHostOptions};
use noisemaker_gpu::host::{CanvasRenderer, CanvasRendererOptions};
use noisemaker_gpu::{GpuDevice, Orientation};

let device = GpuDevice::create(&Default::default())?;
let renderer = CanvasRenderer::new(
    &device,
    CanvasRendererOptions { width: 512, height: 512, ..Default::default() },
);
let mut host = DemoHost::new(renderer, DemoHostOptions::default());
host.rebuild_pipeline_from_dsl(source, true)?; // compile, load ProgramState, apply the controls
host.settle()?; // media, text, meshes and overlays, as the page completes them
let renderer = host.renderer_mut();
renderer.sync_time(0.25);
for _ in 0..8 {
    renderer.render(0.25)?;
}
let pixels = renderer.read_output()?.oriented(Orientation::Presented);
```

`examples/animate.rs` renders a program over its loop to a PNG sequence. The crate documentation (`cargo doc -p noisemaker-for-rust-gpu --open`) describes the lower-level API: graphs, pipelines, the WebGPU backend, the presenter and the input state.

## The first build

On Apple targets, `crates/noisemaker-tint/build.rs` builds Tint (the WGSL reader and the MSL writer only) with a small C++ shim and links it statically:

- It fetches the Dawn commit pinned in `crates/noisemaker-tint/dawn.json` (`git fetch --depth 1`, the fetched commit id checked against the pin) and extracts the paths the Tint build reads.
- It fetches the two third-party directories Dawn's CMake build reads, Abseil and SPIRV-Headers, at the commits that the Dawn commit records for them and `dawn.json` pins.
- It uses CMake 3.22 or newer from `CMAKE` or `PATH`. Otherwise it downloads CMake 3.23.3, the version Dawn's DEPS pins, and checks the archive's SHA-256 against `dawn.json`.

The sources are cached in `~/Library/Caches/noisemaker-tint` (`$XDG_CACHE_HOME/noisemaker-tint` when that is set; `NM_TINT_CACHE` overrides it). `NM_DAWN_SOURCE` names an existing Dawn checkout at the pinned commit to build from instead. The native build is incremental in Cargo's `OUT_DIR`. On other targets nothing is fetched or built, and `noisemaker-tint` compiles without its compiler functions.

Measured on the Apple M4 with an empty Tint cache, CMake not installed and the crates already in Cargo's download cache: the first `cargo build --release -p nm-render` of the export kit's copy of this workspace, fetches included, took 95 seconds.

## Parity

The reference is the noisemaker repository at the commit pinned in `parity/reference.json`. The catalog in `crates/noisemaker-effects/catalog` is generated from that commit (`tools/convert-effects.mjs`), and `parity/check_effects.mjs` requires every file of it to be byte-identical to a fresh conversion.

### Pixel parity

`scripts/parity-summary` runs the canonical sweep, `parity/sweep.py`, fresh:

1. **Mint.** The reference engine renders every case in its own demo page (`demo/shaders/`) on its WebGPU backend in Chromium 153.0.8010.12 (Playwright 1.63.0), on the same machine, and the presented surface is read back as the golden. No golden from an earlier run is graded.
2. **Render.** `nm-render` renders every case from its DSL through the port's own compiler and demo host. It produces every host input natively: the media image, text canvases, traced overlays and meshes.
3. **Grade.** Each case is compared pixel by pixel (RGBA8). A case is *exact* when every pixel is identical, *strict* when the maximum difference is at most 2 levels and the global SSIM is at least 0.98, and *fail* otherwise (a *near* bucket admits cases with a recorded tolerance policy; none is recorded). A pass on a golden with no structure (one colour over more than 99% of the pixels, or a luminance standard deviation below one level) is counted as *uninformative*, never as exact or strict. Timed cases are graded on every sample of their run.

The 2729 cases are the shared fixture programs (`parity/programs`, 360), the generated coverage corpus, every effect with its defaults and with each value of its choice parameters and each flipped boolean (`parity/coverage`, 1815), user-defined Portable effects (`parity/portable`, 2), the timed tier of every effect that evolves across frames (`parity/timed`, 169) and the sibling ports' author-curated programs (`parity/curated`, 383). Every catalog effect needs at least one informative exact or strict case of its own.

Fresh sweep on 2026-10-07 (Apple M4, macOS 26.6.2):

```text
PARITY-SUMMARY {"expected":2729,"executed":2729,"exact":2692,"strict":5,"near":0,"defer":0,"skip":0,"fail":15,"missing":0,"uninformative":17,"effects":210,"effects_evidenced":207}
```

2692 cases are exact. The 5 strict cases are the `text()` canvases, which the port rasterizes on the CPU where the reference draws on Chromium's canvas. Each differs from its golden by at most 1 level (SSIM 1.0 to five decimals), and each is exact when the port renders with the host textures the reference page produced (`parity/sweep.py --captured-host-inputs`). No case is near.

The 15 failing cases are every case of the three traced overlays, `fibers`, `scratches` and `strayHair`. The reference draws them on a canvas that requests `willReadFrequently`, which pins it to Chromium's software rasterizer, so every host draws the same overlay. The port's stroke model (`crates/noisemaker-host/src/canvas.rs`) reproduces the GPU rasterizer the reference used before that change (Skia Graphite on Metal). With the overlays the reference page produced, all 15 cases are exact, so the difference lies in the overlay rasterization alone. Until the port models the software rasterizer, these three effects have no informative exact or strict evidence; the other 207 catalog effects have their own.

The 17 uninformative cases (goldens without structure) are exact but excluded from the evidence.

### The other gates

`scripts/test` runs every check that needs no GPU, against the pinned reference: the catalog freshness gate, the freshness of the generated coverage corpus and of the export kit's effect list, the Rust tests of the GPU-free crates, and the parity gates of the frontend (tokens, AST, validated plans, expanded passes and render graph of every fixture), the DSL tooling, `ProgramState`, the public API, Portable effect registration, and the MIDI, audio and automation state.

### Continuous integration

`.github/workflows/tests.yml` runs `cargo fmt --all --check` and `scripts/test` on every push to `main`, on a GitHub-hosted Linux runner. `.github/workflows/parity.yml` runs `scripts/parity-summary` every Monday and on demand, never on push, on a GitHub-hosted Apple-silicon macOS runner, and keeps its log as an artifact.

### Reproducing

```sh
node scripts/test
scripts/parity-summary
```

Both clone the reference at the pinned commit when `NM_REFERENCE_ROOT` is unset; a checkout it names must be at the pinned commit. `scripts/parity-summary` installs Playwright 1.63.0 and its Chromium into that checkout, builds `target/release/nm-render` when it is missing (`NM_RENDER` names another binary), takes case ids to run a subset, and prints one `PARITY-SUMMARY` line. It exits 0 only when no case is near, failing, skipped or missing and every catalog effect has informative exact or strict evidence. The run above took 16 minutes on the Apple M4.

## Repository layout

| Path | What it is |
| --- | --- |
| `crates/noisemaker-dsl/` | The DSL frontend, the registries, `ProgramState` and the DSL tooling. Its binaries `nm-dsl`, `nm-api` and `nm-program-state` serve the parity gates. |
| `crates/noisemaker-effects/` | The embedded effect catalog (`catalog/`, generated) and its shared files: built-in meshes, palettes, effect strings, the Nunito font and the default media image. |
| `crates/noisemaker-gpu/` | The renderer (package `noisemaker-for-rust-gpu`, library `noisemaker_gpu`): pipeline, WebGPU backend, `CanvasRenderer`, the demo host and the presenter, with the `render_dsl`, `animate` and `viewer` examples. |
| `crates/noisemaker-host/` | CPU host inputs: OBJ meshes, the traced overlays of `fibers`, `scratches` and `strayHair`, and canvas text. |
| `crates/noisemaker-input/` | MIDI and audio state, Chromium's `AnalyserNode` and automation evaluation. Optional `midir` and `cpal` features connect devices. |
| `crates/noisemaker-tint/` | The Tint build for Apple targets, with the license texts of the sources it builds. |
| `crates/nm-render/` | The command-line renderer and parity tool. |
| `parity/` | Fixture programs, corpora, the golden minter (`batch-golden.mjs`), the sweep (`sweep.py`) and the parity gates. |
| `tools/` | The catalog converter, the corpus and export kit list generators, and the reference oracles the gates call. |
| `scripts/` | `test` and `parity-summary`. |
| `export-kit/` | The definition of the `rust-gpu` export kit (see [Export kit](#export-kit)). |

## Export kit

The `rust-gpu` export kit packages this port for Noisedeck's "Export shader pipeline": the Cargo workspace as `engine/` (its manifests, lock file and crate sources, without tests), the user's program as `program.dsl`, the licenses, and a README (`export-kit/kit/README.template.md`) that builds `nm-render` and renders the program. `export-kit/kit.config.json` defines the kit; `export-kit/compat-effects.json`, the list of effects the kit renders, is generated by `tools/generate-kit-compat.mjs` and checked by `scripts/test`. A push to `main` that changes the workspace, `export-kit/`, `rust-toolchain.toml` or `LICENSE` runs `.github/workflows/export-kit.yml`, which dispatches the Noise Factor release workflow. That workflow builds and validates the kit at that commit, publishes it at `https://kits.noisedeck.app/rust-gpu/<version>/` and tags the commit `kit-rust-gpu-v<version>`.

## License and credits

The code is MIT licensed. See [LICENSE](LICENSE).

The effect definitions, WGSL programs, meshes, palettes and effect strings in `crates/noisemaker-effects/catalog` are generated from or copied unmodified from the reference engine ([noisefactorllc/noisemaker](https://github.com/noisefactorllc/noisemaker)), which is MIT licensed by the same copyright holder. Shader code adapted from third-party work keeps its license notice in the shader source; the reference's [CREDITS.md](https://github.com/noisefactorllc/noisemaker/blob/main/CREDITS.md) lists those works.

These files are not covered by the MIT license:

- `crates/noisemaker-effects/catalog/share/fonts/Nunito/Nunito-VariableFont_wght.ttf` is the Nunito font, Copyright 2014 The Nunito Project Authors, licensed under the SIL Open Font License 1.1 ([OFL.txt](crates/noisemaker-effects/catalog/share/fonts/Nunito/OFL.txt)). It is a copy of the reference demo's `demo/font/Nunito/`.
- `crates/noisemaker-effects/catalog/share/img/testcard.png`, the default media image, is the Philips PM5544 test card, by Ebnz, modified by Tucvbif (Wikimedia Commons, File:Philips_PM5544.svg), CC BY 2.5; rasterized to 768×576 for the Noisemaker demo. It is a copy of the reference demo's `demo/shaders/img/testcard.png`.

On Apple targets the build compiles these into the binaries it links (`crates/noisemaker-tint/`):

- Tint, from Dawn, at the commit in `dawn.json`: BSD 3-Clause, [LICENSE-dawn.txt](crates/noisemaker-tint/LICENSE-dawn.txt).
- Abseil (abseil-cpp), at the commit in `dawn.json`: Apache License 2.0, [LICENSE-abseil-cpp.txt](crates/noisemaker-tint/LICENSE-abseil-cpp.txt).
- SPIRV-Headers, at the commit in `dawn.json`: the Khronos license in [LICENSE-spirv-headers.txt](crates/noisemaker-tint/LICENSE-spirv-headers.txt).

The crates Cargo downloads carry their own licenses; `cargo tree` lists them, with the versions `Cargo.lock` pins.
