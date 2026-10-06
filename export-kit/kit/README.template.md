# {{NM_PROGRAM_NAME}}

This Rust project renders your Noisedeck program on the GPU. `engine/` is the Cargo workspace of [Noisemaker for Rust GPU](https://github.com/noisefactorllc/noisemaker-for-rust-gpu): the Noisemaker DSL compiler, the effect catalog with its WGSL shaders, and a renderer built on [wgpu](https://wgpu.rs). `nm-render` compiles `program.dsl` and writes PNG images. The `viewer` example shows the program live in a window.

## Requirements

- A **Rust toolchain, 1.88 or newer**. The workspace declares 1.88 as its minimum supported version, and an older `rustc` refuses to build it. <https://rustup.rs> installs a toolchain.
- A GPU that wgpu can use. On macOS, wgpu renders with Metal and the shaders compile with Tint, the WGSL compiler of Chromium's WebGPU implementation (Dawn), at the revision Chromium 153 uses. This is the platform the port is tested on: in its parity tests the renders there have the reference engine's exact pixels, except where text and canvas-drawn overlays are rasterized, which differ by at most 1 of 255 levels. On other systems wgpu selects Vulkan, DirectX 12 or OpenGL and the shaders compile with naga. The workspace compiles for Linux, but renders on those backends have not been compared with the reference.
- On macOS, git, curl and a C++ compiler for the first build: Xcode or its Command Line Tools (`xcode-select --install`) provide all three.

## Build

Unzip this folder and open a terminal in it. Build the renderer once:

```sh
cargo build --release --manifest-path engine/Cargo.toml -p nm-render
```

**The first build downloads source code.** Cargo fetches the crates that `engine/Cargo.lock` pins from crates.io. On macOS the build also compiles Tint: the build script of `engine/crates/noisemaker-tint` fetches the Dawn source at the commit pinned in its `dawn.json`, and Abseil and SPIRV-Headers at the commits that Dawn commit records, checks each commit id, and builds them with CMake. Without CMake 3.22 or newer on your `PATH` it downloads CMake 3.23.3 and checks its SHA-256. The sources stay in `~/Library/Caches/noisemaker-tint` (`$XDG_CACHE_HOME/noisemaker-tint` when that is set; set `NM_TINT_CACHE` to use another directory), so later builds do not fetch them again. Other systems do not build Tint.

Cargo writes the build to `engine/target/`. On an Apple M4 the first build, with the crates already downloaded and nothing in the Tint cache, took 95 seconds. Once it is done, everything is on your disk: rendering never uses the network.

## Render it

```sh
engine/target/release/nm-render render --dsl program.dsl --out out.png --size 512 --orientation presented
```

The command writes a 512×512 `out.png` beside your program. Paths are relative to your working directory. `--orientation presented` writes the image the way Noisedeck shows it; without it the rows are in GPU texture order, upside down.

`render` draws 8 frames at one moment of the program's 10 second loop (`--time 0.25` by default, a fraction of the loop). Programs with feedback, particles or simulations build up over time, so render them in timed mode instead. This runs 10 seconds of frames at 60 frames per second and writes the last one as `out.t10.png`:

```sh
engine/target/release/nm-render render --dsl program.dsl --out out.png --size 512 --orientation presented --run-seconds 10 --sample-every 10
```

Useful options:

- `--width W --height H` in place of `--size N` for a rectangular image.
- `--param step_N.NAME=VALUE` sets a parameter of step N, as Noisedeck's controls do. `engine/target/release/nm-render effect synth/noise` lists an effect's parameters.
- `--media IMAGE.png` gives `media()` steps an image. Without it they show the Noisemaker demo's test card.
- `--obj MESH.obj` loads a mesh for `meshLoader()`. Without it a step shows its built-in mesh.

`engine/target/release/nm-render render --help` lists the rest. The renderer exits with status 0 when it wrote the image, 1 when the program does not compile or render, and 2 for a command-line or input file error.

## Animate it

```sh
engine/target/release/nm-render animate --dsl program.dsl --out-dir frames --size 512
```

This renders one loop, 10 seconds at 30 frames per second, as `frames/frame_00000.png` to `frames/frame_00299.png`. Add `--mp4 loop.mp4` to encode them with `ffmpeg` when it is installed. `--fps`, `--loop-seconds` and `--frames` change the timing.

## Watch it live

```sh
cargo run --release --manifest-path engine/Cargo.toml -p noisemaker-for-rust-gpu --example viewer -- program.dsl
```

This builds the viewer the first time, then opens a window that runs the program in real time. Edit `program.dsl` and save it: the viewer loads the new version, and keeps the last good one running while a version does not compile (the error goes to the terminal). `--size N` sets the render size and `--media IMAGE.png` the media image. The `--` separates Cargo's arguments from the viewer's. Escape closes the window.

## Inputs

Noisedeck exports the program, not the media or devices it ran with.

- **Media.** `media()` steps show the Noisemaker demo's test card unless you pass `--media`.
- **Text.** `text()` steps draw with the bundled Nunito font. Other font families come from the fonts installed on your system, so they can differ from the font your browser used.
- **Meshes.** A `meshLoader()` step shows its built-in mesh unless you pass `--obj`.
- **MIDI and audio.** There is no live MIDI or audio input. Programs render as the Noisemaker demo renders them when the browser denies MIDI and microphone access.

## What's inside

| Path | What it is |
| --- | --- |
| `program.dsl` | Your program's source, exactly as it was in Noisedeck. |
| `engine/Cargo.toml`, `engine/Cargo.lock` | The workspace manifest and the exact dependency versions this export was built against. |
| `engine/crates/nm-render/` | The command-line renderer. |
| `engine/crates/noisemaker-gpu/` | The renderer library (package `noisemaker-for-rust-gpu`) and its examples, `viewer` among them. |
| `engine/crates/noisemaker-dsl/` | The DSL compiler. |
| `engine/crates/noisemaker-effects/` | The effect catalog: definitions, WGSL shaders, built-in meshes, the Nunito font for `text()` and the test card for `media()`. |
| `engine/crates/noisemaker-host/` | Text, mesh and overlay inputs, drawn on the CPU. |
| `engine/crates/noisemaker-input/` | MIDI and audio input state. |
| `engine/crates/noisemaker-tint/` | The Tint build for macOS, with the license texts of the sources it builds. |
| `noisedeck-export.json` | What was exported, when, and from which port commit. |
| `LICENSES/` | Licenses for everything shipped here and built from it. |

`engine/` is self-contained: copied elsewhere, it builds the same way. You can delete `engine/target/` at any time; the next build takes longer.

## Using the engine in your own code

The workspace is ordinary Rust. Add `engine/crates/noisemaker-gpu` as a path dependency (package `noisemaker-for-rust-gpu`, library `noisemaker_gpu`). `engine/crates/noisemaker-gpu/examples/render_dsl.rs` renders a program to a PNG through the library API. <https://github.com/noisefactorllc/noisemaker-for-rust-gpu> documents the port.

## Effects used by this program

{{NM_EFFECT_LIST}}

## Editing it

Replace `program.dsl` with another Noisemaker program and run the same render command. The build is already done, so only the render repeats. `engine/target/release/nm-render effects` lists every effect this engine contains.

Noisedeck exported this program against Noisemaker `{{NM_ENGINE_VERSION}}`. The engine here is a separate implementation, tested for pixel parity against the reference engine revision it pins. An effect added to Noisemaker after that revision is not in its catalog.

## License

The Noisemaker engine and this port are MIT licensed: `LICENSES/noisemaker-MIT.txt` and `LICENSES/noisemaker-for-rust-gpu-LICENSE.txt`. On macOS the build compiles third-party code into the renderer: Tint, from Dawn (BSD 3-Clause, `LICENSES/Dawn-LICENSE.txt`), Abseil (Apache 2.0, `LICENSES/abseil-cpp-Apache-2.0.txt`) and SPIRV-Headers (`LICENSES/SPIRV-Headers-LICENSE.txt`). The Nunito font is licensed under the SIL Open Font License 1.1: `LICENSES/Nunito-OFL.txt`. The test card is the Philips PM5544 test card, by Ebnz, modified by Tucvbif (Wikimedia Commons, File:Philips_PM5544.svg), CC BY 2.5, rasterized to 768×576 for the Noisemaker demo. The crates Cargo fetches carry their own licenses, which `cargo tree` and each crate's page on crates.io state. Your program and the imagery it renders are yours.
