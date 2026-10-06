//! Host API tests: the CanvasRenderer port (compile and recompile, parameter
//! application, meshes, media, sinks, frame export, cubemaps, MIDI) and the
//! demo host (ProgramState driving the renderer, text canvases, overlays).
//! They render on the GPU.

use std::cell::RefCell;
use std::rc::Rc;

use noisemaker_gpu::demo::{DemoHost, DemoHostOptions};
use noisemaker_gpu::dsl::Registry;
use noisemaker_gpu::dsl::program_state::ProgramState;
use noisemaker_gpu::frame_export::{ExportedFrame, FrameExportOptions};
use noisemaker_gpu::host::{CanvasRenderer, CanvasRendererOptions, CompileOptions, cube_export};
use noisemaker_gpu::png_io::Rgba8Image;
use noisemaker_gpu::sink::{Sink, SinkDescriptor};
use noisemaker_gpu::{
    GpuDevice, Object, Orientation, Presenter, RenderError, Value, WebGpuBackend,
};

thread_local! {
    static REGISTRY: Rc<Registry> = Rc::new(Registry::with_catalog());
}

fn device() -> GpuDevice {
    GpuDevice::create(&Default::default()).expect("a GPU adapter")
}

fn renderer(device: &GpuDevice, size: u32) -> CanvasRenderer {
    CanvasRenderer::new(
        device,
        CanvasRendererOptions {
            width: size,
            height: size,
            registry: Some(REGISTRY.with(|r| r.clone())),
            ..Default::default()
        },
    )
}

fn compile(r: &mut CanvasRenderer, dsl: &str) {
    r.compile(dsl, &CompileOptions::default())
        .unwrap_or_else(|e| panic!("compile failed: {e}\n{dsl}"));
}

fn output(r: &mut CanvasRenderer) -> Vec<u8> {
    let p = r.pipeline_mut().unwrap();
    let name = p.graph.render_surface_name().unwrap_or("o0").to_owned();
    let id = p.surfaces[&name].read.clone().unwrap();
    p.backend.read_pixels(&id).unwrap().data
}

fn uniform(r: &mut CanvasRenderer, pass_index: usize, name: &str) -> Value {
    r.pipeline().unwrap().graph.passes[pass_index]
        .get_or_undefined("uniforms")
        .get(name)
        .clone()
}

fn object(json: &str) -> Object {
    Value::from_json(json)
        .unwrap()
        .as_object()
        .cloned()
        .unwrap()
}

const SOLID: &str = "search synth\nsolid(color: #ff0000, alpha: 1).write(o0)\nrender(o0)";
const SOLID_BLUE: &str = "search synth\nsolid(color: #0000ff, alpha: 1).write(o0)\nrender(o0)";

#[test]
fn compile_creates_then_recompile_keeps_the_pipeline_and_its_surfaces() {
    let device = device();
    let mut r = renderer(&device, 16);
    assert!(r.pipeline().is_none());
    compile(&mut r, SOLID);
    r.render(0.25).unwrap();
    assert_eq!(&output(&mut r)[..4], &[255, 0, 0, 255]);
    assert_eq!(r.frame_count(), 1);
    let surface_texture = {
        let p = r.pipeline().unwrap();
        p.backend.textures["global_o0_read"].clone()
    };

    compile(&mut r, SOLID_BLUE);
    assert_eq!(r.frame_count(), 0, "a compile resets the frame count");
    {
        let p = r.pipeline().unwrap();
        assert_eq!(p.graph.source.as_str(), Some(SOLID_BLUE));
        assert!(
            Rc::ptr_eq(&p.backend.textures["global_o0_read"], &surface_texture),
            "recompile keeps the surfaces of an unchanged size"
        );
    }
    r.render(0.25).unwrap();
    assert_eq!(&output(&mut r)[..4], &[0, 0, 255, 255]);

    // A failed compile reports the error and keeps the running pipeline.
    let err = r
        .compile(
            "search synth\nnope().write(o0)\nrender(o0)",
            &CompileOptions::default(),
        )
        .unwrap_err();
    assert!(!err.to_string().is_empty());
    assert_eq!(
        r.current_dsl(),
        "search synth\nnope().write(o0)\nrender(o0)"
    );
    r.render(0.25).unwrap();
    assert_eq!(&output(&mut r)[..4], &[0, 0, 255, 255]);

    // After dispose, the next compile creates a fresh pipeline.
    r.dispose().unwrap();
    assert!(r.pipeline().is_none());
    compile(&mut r, SOLID);
    r.render(0.25).unwrap();
    assert_eq!(&output(&mut r)[..4], &[255, 0, 0, 255]);

    // resize resizes the surfaces.
    r.resize(8, 4).unwrap();
    r.render(0.25).unwrap();
    let p = r.pipeline().unwrap();
    let read = p.surfaces["o0"].read.clone().unwrap();
    assert_eq!(
        (
            p.backend.textures[&read].width,
            p.backend.textures[&read].height
        ),
        (8.0, 4.0)
    );
}

#[test]
fn step_parameter_values_reach_their_own_passes() {
    let device = device();
    let mut r = renderer(&device, 8);
    compile(
        &mut r,
        "search synth, filter\nsolid(color: #ff0000).write(o0)\nsolid(color: #00ff00).write(o1)\nrender(o0)",
    );
    let values = object(
        r##"{"step_0": {"alpha": 0.5, "color": "#0000ff", "_skip": true},
            "step_2": {"alpha": {"type": "Oscillator", "oscType": 0}, "color": [1, 1, 0]}}"##,
    );
    r.apply_step_parameter_values(&values).unwrap();
    assert_eq!(uniform(&mut r, 0, "alpha"), Value::Number(0.5));
    // Colors convert to [r, g, b].
    assert_eq!(
        uniform(&mut r, 0, "color"),
        Value::from_json("[0, 0, 1]").unwrap()
    );
    let second = r
        .pipeline()
        .unwrap()
        .graph
        .passes
        .iter()
        .position(|p| p.get_or_undefined("stepIndex") == &Value::Number(2.0))
        .unwrap();
    // Automation-controlled values are left to the automation.
    assert_eq!(uniform(&mut r, second, "alpha"), Value::Number(1.0));
    assert_eq!(
        uniform(&mut r, second, "color"),
        Value::from_json("[1, 1, 0]").unwrap()
    );
}

#[test]
fn parameter_values_follow_the_uniform_bindings() {
    let device = device();
    let mut r = renderer(&device, 8);
    compile(&mut r, SOLID);
    let registry = r.registry().clone();
    let solid = registry.get_effect("synth.solid").unwrap().clone();
    r.build_uniform_bindings(&solid);
    let bindings = r.uniform_bindings().clone();
    assert_eq!(bindings["alpha"][0].uniform_name, "alpha");
    assert_eq!(bindings["color"][0].pass_index, 0);
    r.apply_parameter_values(&solid, &object(r##"{"alpha": 0.25, "color": "#00ff00"}"##))
        .unwrap();
    assert_eq!(uniform(&mut r, 0, "alpha"), Value::Number(0.25));
    r.render(0.25).unwrap();
    let px = output(&mut r);
    // solid outputs premultiplied color.
    assert_eq!((px[0], px[1], px[2], px[3]), (0, 64, 0, 64));
}

#[test]
fn program_state_drives_the_renderer() {
    let device = device();
    let mut r = renderer(&device, 8);
    compile(&mut r, SOLID);
    let registry = r.registry().clone();
    let mut state = ProgramState::with_renderer(registry, r);
    state.from_dsl(SOLID).unwrap();
    state
        .set_value("step_0", "alpha", Value::Number(0.5))
        .unwrap();
    let r = state.renderer_mut().unwrap();
    assert_eq!(uniform(r, 0, "alpha"), Value::Number(0.5));
    r.render(0.25).unwrap();
    assert_eq!(output(r)[3], 128);
}

#[test]
fn meshes_load_cache_and_survive_new_pipelines() {
    let device = device();
    let mut r = renderer(&device, 8);
    // Without a pipeline nothing loads.
    assert!(!r.load_obj_from_string("v 0 0 0", "mesh0").success);
    compile(&mut r, "search render\nmeshLoader().write(o0)\nrender(o0)");
    let tri = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
    let result = r.load_obj_from_string(tri, "mesh0");
    assert!(result.success);
    assert_eq!(result.vertex_count, 3);
    let builtin = r.load_builtin_mesh("cube", "mesh1");
    assert_eq!(builtin.vertex_count, 36);
    assert!(!r.load_builtin_mesh("teapot", "mesh2").success);
    // Positions of mesh0: the reference's (v0, v2, v1) winding, w = 1, as
    // bytes after the reference float readback conversion.
    let read = |r: &mut CanvasRenderer, id: &str| {
        r.pipeline_mut()
            .unwrap()
            .backend
            .read_pixels(id)
            .unwrap()
            .data
    };
    let px = read(&mut r, "global_mesh0_positions");
    assert_eq!(&px[..8], &[0, 0, 0, 255, 0, 255, 0, 255]);
    // A fresh pipeline gets the cached meshes again.
    r.dispose().unwrap();
    compile(&mut r, "search render\nmeshLoader().write(o0)\nrender(o0)");
    let px = read(&mut r, "global_mesh0_positions");
    assert_eq!(&px[..8], &[0, 0, 0, 255, 0, 255, 0, 255]);
    let px = read(&mut r, "global_mesh1_positions");
    assert_eq!(px[3], 255);
}

fn image(width: u32, height: u32, rgba: [u8; 4]) -> Rgba8Image {
    Rgba8Image {
        width,
        height,
        data: rgba.repeat((width * height) as usize),
    }
}

#[test]
fn demo_host_loads_media_and_sets_its_size() {
    let device = device();
    let r = renderer(&device, 8);
    let mut host = DemoHost::new(
        r,
        DemoHostOptions {
            default_media: Some(Rc::new(image(4, 2, [0, 255, 0, 255]))),
            ..Default::default()
        },
    );
    host.rebuild_pipeline_from_dsl("search synth\nmedia().write(o0)\nrender(o0)", true)
        .unwrap();
    assert!(host.has_pending_inputs());
    host.settle().unwrap();
    assert!(!host.has_pending_inputs());
    let r = host.renderer_mut();
    assert_eq!(
        uniform(r, 0, "imageSize"),
        Value::from_json("[4, 2]").unwrap()
    );
    {
        let p = r.pipeline().unwrap();
        let tex = &p.backend.textures["imageTex_step_0"];
        assert!(tex.is_external);
        assert_eq!((tex.width, tex.height), (4.0, 2.0));
    }
    r.render(0.25).unwrap();
    let px = output(r);
    assert!(px.chunks(4).any(|p| p[1] == 255 && p[0] == 0));
    assert_eq!(r.get_media_steps()[0].texture_id, "imageTex_step_0");

    // A new image (the file input) replaces it.
    host.set_media_image(0, Rc::new(image(2, 2, [255, 0, 0, 255])))
        .unwrap();
    let r = host.renderer_mut();
    assert_eq!(
        uniform(r, 0, "imageSize"),
        Value::from_json("[2, 2]").unwrap()
    );
}

#[test]
fn demo_host_draws_text_canvases() {
    let device = device();
    let r = renderer(&device, 32);
    let mut host = DemoHost::new(r, DemoHostOptions::default());
    host.rebuild_pipeline_from_dsl(
        "search synth, filter\nsolid(color: #000000).text(text: \"Hi\", size: 0.5, color: #ffffff).write(o0)\nrender(o0)",
        true,
    )
    .unwrap();
    host.settle().unwrap();
    let input = host.text_inputs()[&1].clone();
    assert_eq!(input.texture_id, "textTex_step_1");
    assert_eq!(input.params.text, "Hi");
    assert_eq!(
        host.program_state().get_value("step_1", "textSize"),
        Value::from_json("[32, 32]").unwrap()
    );
    let r = host.renderer_mut();
    r.render(0.25).unwrap();
    let px = output(r);
    assert!(px.chunks(4).any(|p| p[0] > 128), "the text is drawn");
}

#[test]
fn demo_host_controls_write_validated_values() {
    let device = device();
    let r = renderer(&device, 8);
    let mut host = DemoHost::new(r, DemoHostOptions::default());
    // An int parameter given a fraction is coerced by the control's setValue.
    host.rebuild_pipeline_from_dsl(
        "search synth\nnoise(seed: 2.7, octaves: 3).write(o0)\nrender(o0)",
        true,
    )
    .unwrap();
    host.settle().unwrap();
    let values = host.effect_parameter_values();
    let step = values.get_or_undefined("step_0");
    assert_eq!(step.get("seed"), &Value::Number(2.0));
    assert_eq!(step.get("octaves"), &Value::Number(3.0));
}

#[test]
fn overlays_trace_on_workers_and_regenerate_on_change() {
    let device = device();
    let r = renderer(&device, 64);
    let mut host = DemoHost::new(r, DemoHostOptions::default());
    host.rebuild_pipeline_from_dsl(
        "search synth, filter\nsolid(color: #000000).fibers(density: 1).write(o0)\nrender(o0)",
        true,
    )
    .unwrap();
    {
        let p = host.renderer_mut().pipeline().unwrap();
        assert!(
            p.has_pending_async_regens(),
            "the step values schedule a regen"
        );
    }
    host.settle().unwrap();
    let overlay = |host: &mut DemoHost| {
        host.renderer_mut()
            .pipeline_mut()
            .unwrap()
            .backend
            .read_pixels("node_1_overlayTex")
            .unwrap()
            .data
    };
    let first = overlay(&mut host);
    let drawn = first.chunks(4).filter(|p| p[3] > 0).count();
    assert!(drawn > 100, "{drawn} overlay pixels drawn");

    // A changed seed regenerates the overlay; an unchanged one does not.
    host.program_state_mut()
        .set_value("step_1", "seed", Value::Number(7.0))
        .unwrap();
    assert!(
        host.renderer_mut()
            .pipeline()
            .unwrap()
            .has_pending_async_regens()
    );
    host.settle().unwrap();
    let second = overlay(&mut host);
    assert_ne!(first, second);
    host.program_state_mut()
        .set_value("step_1", "alpha", Value::Number(0.25))
        .unwrap();
    assert!(
        !host
            .renderer_mut()
            .pipeline()
            .unwrap()
            .has_pending_async_regens()
    );
}

#[derive(Default)]
struct CountingSink {
    log: Rc<RefCell<Vec<String>>>,
}

impl Sink for CountingSink {
    fn configure(&mut self, d: &SinkDescriptor) -> Result<(), String> {
        self.log
            .borrow_mut()
            .push(format!("configure {}x{}", d.width, d.height));
        Ok(())
    }
    fn submit(
        &mut self,
        _: &mut WebGpuBackend,
        texture_id: &str,
        _: f64,
    ) -> Result<Option<bool>, String> {
        self.log.borrow_mut().push(format!("submit {texture_id}"));
        Ok(Some(true))
    }
    fn close(&mut self) -> Result<(), String> {
        self.log.borrow_mut().push("close".into());
        Ok(())
    }
}

#[test]
fn sinks_receive_presented_frames() {
    let device = device();
    let mut r = renderer(&device, 8);
    assert!(r.add_sink(Box::new(CountingSink::default())).is_err());
    compile(&mut r, SOLID);
    let log = Rc::new(RefCell::new(Vec::new()));
    let id = r
        .add_sink(Box::new(CountingSink { log: log.clone() }))
        .unwrap();
    r.render(0.25).unwrap();
    r.remove_sink(id).unwrap();
    r.render(0.25).unwrap();
    assert_eq!(
        *log.borrow(),
        vec![
            "configure 8x8".to_owned(),
            "submit global_o0_write".to_owned(),
            "close".to_owned()
        ]
    );
}

#[test]
fn frame_export_reads_back_presented_frames() {
    let device = device();
    let mut r = renderer(&device, 8);
    compile(&mut r, SOLID);
    r.render(0.25).unwrap();
    let mut queue = r
        .create_frame_export_queue(FrameExportOptions::default())
        .unwrap();
    queue
        .configure(&SinkDescriptor {
            width: 8.0,
            height: 8.0,
            ..Default::default()
        })
        .unwrap();
    let presented = r.pipeline().unwrap().last_presented.clone().unwrap();
    let frames = Rc::new(RefCell::new(Vec::new()));
    let sink = frames.clone();
    assert!(queue.enqueue(
        r.backend().unwrap(),
        &presented,
        1.0,
        Box::new(move |frame: &ExportedFrame<'_>, t| {
            sink.borrow_mut()
                .push((frame.width, frame.row_stride, frame.data[..4].to_vec(), t));
            Ok(())
        }),
    ));
    for _ in 0..1000 {
        queue.poll();
        if !frames.borrow().is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(*frames.borrow(), vec![(8, 32, vec![255, 0, 0, 255], 1.0)]);
    assert_eq!(queue.stats.completed, 1);
    queue.close(false).unwrap();
}

#[test]
fn cubemaps_render_six_faces_and_assemble_a_cross() {
    let device = device();
    let mut r = renderer(&device, 8);
    assert!(r.render_cubemap(4, "o0", 0.0).unwrap().is_empty());
    compile(&mut r, SOLID);
    let faces = r.render_cubemap(4, "o0", 0.0).unwrap();
    assert_eq!(faces.len(), 6);
    let cross = cube_export::cross_layout(&faces);
    assert_eq!((cross.width, cross.height), (16, 12));
    assert_eq!(cube_export::face_file_names()[5], "nz.png");
    // The renderer is back at its own size.
    let p = r.pipeline().unwrap();
    let read = p.surfaces["o0"].read.clone().unwrap();
    assert_eq!(p.backend.textures[&read].width, 8.0);
}

#[test]
fn midi_state_feeds_automation_and_the_note_grid() {
    let device = device();
    let mut r = renderer(&device, 4);
    compile(
        &mut r,
        "search synth\nsolid(color: #ffffff, alpha: midi(channel: 1, mode: midiMode.gateVelocity)).write(o0)\nrender(o0)",
    );
    r.render(0.25).unwrap();
    assert_eq!(output(&mut r)[3], 0, "no MIDI state: the minimum");
    let midi = r.set_midi_state(None);
    midi.borrow_mut().handle_message(&[0x90, 60, 127], None);
    r.render(0.25).unwrap();
    assert_eq!(output(&mut r)[3], 255);
    let p = r.pipeline().unwrap();
    assert_eq!(
        p.global_uniforms.get_or_undefined("midiClockCount"),
        &Value::Number(0.0)
    );
    assert!(p.backend.textures.contains_key("midiNoteGrid"));
    // A new pipeline gets the same state.
    r.dispose().unwrap();
    compile(
        &mut r,
        "search synth\nsolid(color: #ffffff, alpha: midi(channel: 1, mode: midiMode.gateVelocity)).write(o0)\nrender(o0)",
    );
    r.render(0.25).unwrap();
    assert_eq!(output(&mut r)[3], 255);
}

#[test]
fn media_dimensions_reach_the_lifecycle_fallback() {
    let device = device();
    let mut r = renderer(&device, 4);
    r.set_media_dimensions(640.0, 480.0);
    let hooks = r.effects().get("synth.media").unwrap().clone();
    let lifecycle = hooks.lifecycle.unwrap();
    let globals = Object::new();
    let uniforms = lifecycle
        .borrow_mut()
        .on_update(&noisemaker_gpu::hooks::UpdateContext {
            time: 0.0,
            delta: 0.0,
            uniforms: &globals,
        })
        .unwrap();
    assert_eq!(
        uniforms.get_or_undefined("imageSize"),
        &Value::from_json("[640, 480]").unwrap()
    );
}

#[test]
fn present_blit_shows_the_texture_rows_mirrored() {
    let device = device();
    let mut r = renderer(&device, 24);
    compile(
        &mut r,
        "search synth\nnoise(seed: 4, octaves: 2).write(o0)\nrender(o0)",
    );
    r.render(0.25).unwrap();
    let pixels = r.read_output().unwrap();
    assert_eq!(pixels.data, output(&mut r));
    let presented = pixels.clone().oriented(Orientation::Presented);
    assert_eq!(presented, pixels.flip_rows());
    assert_eq!(
        presented.flip_rows(),
        pixels,
        "flipping twice restores the rows"
    );
    assert_ne!(presented, pixels, "the test image is not symmetric");

    // The reference's present(): a same-size canvas samples the nearest
    // texel, which mirrors the rows.
    let p = r.pipeline_mut().unwrap();
    let id = p.surfaces["o0"].read.clone().unwrap();
    let mut presenter = Presenter::new(&device);
    let canvas = presenter.capture(&p.backend, &id, 24, 24).unwrap().unwrap();
    assert_eq!(canvas, presented);
    // A scaled canvas filters; an unknown texture draws nothing.
    let scaled = presenter.capture(&p.backend, &id, 48, 30).unwrap().unwrap();
    assert_eq!((scaled.width, scaled.height), (48, 30));
    assert!(
        presenter
            .capture(&p.backend, "nope", 24, 24)
            .unwrap()
            .is_none()
    );
    // A bgra8unorm canvas (the preferred canvas format) takes the same blit.
    let target = device.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 24,
            height: 24,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    assert!(presenter.present(
        &p.backend,
        &id,
        &view,
        wgpu::TextureFormat::Bgra8Unorm,
        24,
        24
    ));
    assert_eq!(p.backend.device_errors(), 0);
}

#[test]
fn compile_errors_keep_the_frontend_error() {
    let device = device();
    let mut r = renderer(&device, 8);
    let source = "search synth\nnoise(\n  .write(o0)\nrender(o0)";
    let err = r.compile(source, &CompileOptions::default()).unwrap_err();
    let RenderError::Dsl(js) = &err else {
        panic!("not a DSL error: {err:?}");
    };
    assert_eq!(
        err.to_string(),
        "SyntaxError: Unexpected token DOT at line 3 col 3"
    );
    assert!(
        noisemaker_gpu::dsl::error_formatter::format_compile_error(source, js)
            .contains("^-- error here")
    );
    let err = r
        .compile(
            "search synth\nnope().write(o0)\nrender(o0)",
            &CompileOptions::default(),
        )
        .unwrap_err();
    assert_eq!(err.code().as_deref(), Some("ERR_COMPILATION_FAILED"));
    assert!(err.dsl_error().is_some());
}

#[test]
fn demo_host_control_changes_follow_the_page() {
    let device = device();
    let r = renderer(&device, 32);
    let mut host = DemoHost::new(r, DemoHostOptions::default());
    host.rebuild_pipeline_from_dsl(
        "search synth, filter\nsolid(color: #000000).text(text: \"Hi\", size: 0.5, color: #ffffff).write(o0)\nrender(o0)",
        true,
    )
    .unwrap();
    host.settle().unwrap();
    // A changed text re-draws the text canvas and rewrites the editor DSL.
    host.set_control_value("step_1", "text", Value::from("Yo"))
        .unwrap();
    assert_eq!(host.text_inputs()[&1].params.text, "Yo");
    assert_eq!(host.text_inputs()[&1].params.size, 0.5);
    assert!(host.dsl().contains("Yo"), "{}", host.dsl());
    // Values are validated and coerced as the control's setValue does.
    host.set_control_value("step_1", "size", Value::from(0.25))
        .unwrap();
    assert_eq!(host.text_inputs()[&1].params.size, 0.25);
    host.settle().unwrap();
    let r = host.renderer_mut();
    r.render(0.25).unwrap();
    assert!(output(r).chunks(4).any(|p| p[0] > 128), "the text is drawn");
}

/// A Portable generator: a solid color from its own WGSL, with a choice
/// parameter and a parameter alias.
fn portable_solid() -> Value {
    Value::from_json(
        r#"{
            "namespace": "user", "func": "portableSolid", "name": "Portable Solid",
            "globals": {
                "level": {"type": "float", "default": 1, "uniform": "level"},
                "channel": {"type": "int", "default": 0, "uniform": "channel",
                            "choices": {"red": 0, "green": 1, "blue": 2}}
            },
            "paramAliases": {"amount": "level"},
            "passes": [{"name": "main", "program": "main", "inputs": {}, "outputs": {"fragColor": "outputTex"}}],
            "shaders": {"main": {"wgsl": "@group(0) @binding(0) var<uniform> level: f32;\n@group(0) @binding(1) var<uniform> channel: i32;\n@fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {\n  var c = vec3<f32>(0.0);\n  c[channel] = level;\n  return vec4<f32>(c, 1.0);\n}\n"}}
        }"#,
    )
    .unwrap()
}

#[test]
fn portable_effects_register_compile_and_render() {
    let device = device();
    let mut r = renderer(&device, 8);
    let effect = r.register_portable_effect(&portable_solid()).unwrap();
    assert_eq!(effect.id(), "user/portableSolid");
    assert_eq!(
        r.enums()
            .get("user")
            .get("portableSolid")
            .get("channel")
            .get("blue")
            .get("value"),
        &Value::Number(2.0)
    );
    // The shared catalog registry is untouched (copy on write).
    assert!(REGISTRY.with(|reg| reg.get_effect("user.portableSolid").is_none()));
    compile(
        &mut r,
        "search user\nportableSolid(amount: 0.5, channel: blue).write(o0)\nrender(o0)",
    );
    r.render(0.25).unwrap();
    assert_eq!(&output(&mut r)[..4], &[0, 0, 128, 255]);
    // A second registration of the name is rejected and changes nothing.
    let err = r.register_portable_effect(&portable_solid()).unwrap_err();
    assert!(
        err.to_string()
            .contains("user.portableSolid is already registered")
    );
}

#[test]
fn demo_host_resolves_registered_portable_effects() {
    let device = device();
    let mut host = DemoHost::new(renderer(&device, 8), DemoHostOptions::default());
    host.register_portable_effect(&portable_solid()).unwrap();
    host.rebuild_pipeline_from_dsl(
        "search user\nportableSolid(level: 1, channel: green).write(o0)\nrender(o0)",
        true,
    )
    .unwrap();
    host.settle().unwrap();
    let step = host.effect_parameter_values();
    assert_eq!(
        step.get_or_undefined("step_0").get("channel"),
        &Value::Number(1.0)
    );
    host.set_control_value("step_0", "level", Value::Number(0.25))
        .unwrap();
    host.settle().unwrap();
    let r = host.renderer_mut();
    r.render(0.25).unwrap();
    assert_eq!(&output(r)[..4], &[0, 64, 0, 255]);
}
