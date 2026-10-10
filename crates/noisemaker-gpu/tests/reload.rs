use std::rc::Rc;

use noisemaker_gpu::demo::{DemoHost, DemoHostOptions};
use noisemaker_gpu::host::{CanvasRenderer, CanvasRendererOptions, CompileOptions};
use noisemaker_gpu::{GpuDevice, Value};

const RED: &str = "search synth\nsolid(color: #ff0000).write(o0)\nrender(o0)";
const BLUE: &str = "search synth\nsolid(color: #0000ff).write(o0)\nrender(o0)";
const TEXT: &str = "search synth, filter\nsolid(color: #000000).text(text: \"Hi\", font: \"Nunito\", size: 0.5, color: #ffffff).write(o0)\nrender(o0)";

fn renderer(device: &GpuDevice) -> CanvasRenderer {
    CanvasRenderer::new(
        device,
        CanvasRendererOptions {
            width: 64,
            height: 48,
            ..Default::default()
        },
    )
}

fn pixels(renderer: &mut CanvasRenderer) -> Vec<u8> {
    renderer.render(0.25).unwrap();
    renderer.read_output().unwrap().data
}

#[test]
fn same_structure_reload_refreshes_text_canvas() {
    let device =
        GpuDevice::create(&Default::default()).expect("GPU required for reload regressions");
    for (old, new) in [
        ("Hi", "WWWW"),
        ("Nunito", "monospace"),
        ("0.5", "0.25"),
        ("#ffffff", "#ff0000"),
    ] {
        let mut host = DemoHost::new(renderer(&device), DemoHostOptions::default());
        host.rebuild_pipeline_from_dsl(TEXT, true).unwrap();
        host.settle().unwrap();
        let before = pixels(host.renderer_mut());
        let new_dsl = TEXT.replace(old, new);
        assert!(!host.program_state().would_change_structure(&new_dsl));
        host.rebuild_pipeline_from_dsl(&new_dsl, true).unwrap();
        host.settle().unwrap();
        let actual = pixels(host.renderer_mut());
        let mut fresh = DemoHost::new(renderer(&device), DemoHostOptions::default());
        fresh.rebuild_pipeline_from_dsl(&new_dsl, true).unwrap();
        fresh.settle().unwrap();
        let expected = pixels(fresh.renderer_mut());
        assert!(before != expected, "fixture must visibly change {old}");
        assert!(
            actual == expected,
            "reloaded {old} must match a fresh canvas"
        );
        assert_eq!(
            host.text_inputs()[&1].params,
            fresh.text_inputs()[&1].params
        );
        assert_eq!(
            host.program_state().get_value("step_1", "textSize"),
            Value::from_json("[64,64]").unwrap()
        );
    }
}

#[test]
fn invalid_shader_reload_preserves_last_good_render_and_recovers() {
    let device =
        GpuDevice::create(&Default::default()).expect("GPU required for reload regressions");
    let mut renderer = renderer(&device);
    renderer.compile(RED, &CompileOptions::default()).unwrap();
    let before = pixels(&mut renderer);
    assert_eq!(&before[..4], &[255, 0, 0, 255]);
    let texture = renderer.pipeline().unwrap().backend.textures["global_o0_read"].clone();
    let programs = renderer.pipeline().unwrap().backend.programs.clone();
    let overrides = Value::from_json(r#"{"0":{"solid":{"wgsl":"this is invalid WGSL"}}}"#)
        .unwrap()
        .as_object()
        .unwrap()
        .clone();
    assert!(
        renderer
            .compile(
                BLUE,
                &CompileOptions {
                    shader_overrides: overrides
                }
            )
            .is_err()
    );
    assert_eq!(
        renderer.pipeline().unwrap().graph.source.as_str(),
        Some(RED)
    );
    assert!(Rc::ptr_eq(
        &texture,
        &renderer.pipeline().unwrap().backend.textures["global_o0_read"]
    ));
    assert_eq!(pixels(&mut renderer), before);
    for (id, program) in &programs {
        assert!(Rc::ptr_eq(
            program,
            &renderer.pipeline().unwrap().backend.programs[id]
        ));
    }
    renderer.compile(BLUE, &CompileOptions::default()).unwrap();
    assert_eq!(&pixels(&mut renderer)[..4], &[0, 0, 255, 255]);
}

#[test]
fn invalid_resource_reload_preserves_last_good_render_and_recovers() {
    let device =
        GpuDevice::create(&Default::default()).expect("GPU required for reload regressions");
    let definition = r#"{
        "namespace": "user", "func": "badResource", "name": "Bad Resource",
        "globals": {},
        "passes": [
            {"name":"scratch", "program":"scratch", "inputs":{}, "outputs":{"color":"_scratch"}},
            {"name":"main", "program":"main", "inputs":{"inputTex":"_scratch"}, "outputs":{"color":"outputTex"}}
        ],
        "textures": {"_scratch":{"width":32,"height":16,"format":"invalid-texture-format"}},
        "shaders":{"scratch":{"wgsl":"@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 0.0, 1.0, 1.0); }"}, "main":{"wgsl":"@group(0) @binding(0) var inputTex: texture_2d<f32>; @fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> { return textureLoad(inputTex, vec2<i32>(p.xy) % vec2<i32>(16), 0); }"}}
    }"#;
    for definition in [
        definition.to_owned(),
        definition
            .replace("invalid-texture-format", "rgba8unorm")
            .replace("\"width\":32", "\"width\":1048576"),
    ] {
        let mut renderer = renderer(&device);
        let effect = Value::from_json(&definition).unwrap();
        renderer.register_portable_effect(&effect).unwrap();
        renderer.compile(RED, &CompileOptions::default()).unwrap();
        let scratch_pixels = [90, 30, 60, 255].repeat(16 * 16);
        let spec = Value::from_json(r#"{"width":16,"height":16,"format":"rgba8unorm"}"#).unwrap();
        renderer
            .pipeline_mut()
            .unwrap()
            .backend
            .create_texture("node_0__scratch", spec.as_object().unwrap())
            .unwrap();
        renderer
            .update_texture_from_source(
                "node_0__scratch",
                16,
                16,
                &scratch_pixels,
                Default::default(),
            )
            .unwrap();
        let before = pixels(&mut renderer);
        assert_eq!(&before[..4], &[255, 0, 0, 255]);
        assert!(
            renderer
                .pipeline_mut()
                .unwrap()
                .backend
                .read_pixels("node_0__scratch")
                .unwrap()
                .data
                == scratch_pixels,
            "fixture upload must be readable"
        );
        let textures = renderer.pipeline().unwrap().backend.textures.clone();
        let texture = textures["global_o0_read"].clone();
        let error = renderer
            .compile(
                "search user\nbadResource().write(o0)\nrender(o0)",
                &CompileOptions::default(),
            )
            .unwrap_err();
        if definition.contains("1048576") {
            assert_eq!(error.code().as_deref(), Some("ERR_DEVICE_VALIDATION"));
        } else {
            assert!(
                error.to_string().contains("format"),
                "expected invalid format: {error}"
            );
        }
        assert_eq!(
            renderer.pipeline().unwrap().graph.source.as_str(),
            Some(RED)
        );
        assert!(Rc::ptr_eq(
            &texture,
            &renderer.pipeline().unwrap().backend.textures["global_o0_read"]
        ));
        for (id, texture) in &textures {
            assert!(Rc::ptr_eq(
                texture,
                &renderer.pipeline().unwrap().backend.textures[id]
            ));
        }
        assert_eq!(
            renderer
                .pipeline_mut()
                .unwrap()
                .backend
                .read_pixels("node_0__scratch")
                .unwrap()
                .data,
            scratch_pixels
        );
        assert_eq!(pixels(&mut renderer), before);
        renderer.compile(BLUE, &CompileOptions::default()).unwrap();
        assert_eq!(&pixels(&mut renderer)[..4], &[0, 0, 255, 255]);
    }
}

#[test]
fn text_reload_preserves_uploaded_media_dimensions() {
    let device =
        GpuDevice::create(&Default::default()).expect("GPU required for reload regressions");
    let source = Rc::new(noisemaker_gpu::png_io::Rgba8Image {
        width: 7,
        height: 3,
        data: [20, 40, 80, 255].repeat(21),
    });
    let options = DemoHostOptions {
        default_media: Some(source.clone()),
        ..Default::default()
    };
    let mut host = DemoHost::new(renderer(&device), options.clone());
    let old_dsl = TEXT.replace("solid(color: #000000)", "media()");
    host.rebuild_pipeline_from_dsl(&old_dsl, true).unwrap();
    host.settle().unwrap();
    let new_dsl = old_dsl.replace("Hi", "WWWW");
    host.rebuild_pipeline_from_dsl(&new_dsl, true).unwrap();
    host.settle().unwrap();
    assert_eq!(
        host.program_state().get_value("step_0", "imageSize"),
        Value::from_json("[7,3]").unwrap()
    );
    assert_eq!(
        host.program_state().get_value("step_1", "textSize"),
        Value::from_json("[64,64]").unwrap()
    );
    assert!(Rc::ptr_eq(
        host.media_inputs()[&0].source.as_ref().unwrap(),
        &source
    ));
    let mut fresh = DemoHost::new(renderer(&device), options);
    fresh.rebuild_pipeline_from_dsl(&new_dsl, true).unwrap();
    fresh.settle().unwrap();
    assert_eq!(pixels(host.renderer_mut()), pixels(fresh.renderer_mut()));
}

#[test]
fn same_structure_numeric_reload_updates_uniforms() {
    let device =
        GpuDevice::create(&Default::default()).expect("GPU required for reload regressions");
    let mut host = DemoHost::new(renderer(&device), DemoHostOptions::default());
    let original = "search synth\nsolid(color: #ff0000, alpha: 1).write(o0)\nrender(o0)";
    host.rebuild_pipeline_from_dsl(original, true).unwrap();
    host.settle().unwrap();
    let before = pixels(host.renderer_mut());
    assert_eq!(&before[..4], &[255, 0, 0, 255]);
    let changed = original.replace("alpha: 1", "alpha: 0.5");
    assert!(!host.program_state().would_change_structure(&changed));
    host.rebuild_pipeline_from_dsl(&changed, true).unwrap();
    host.settle().unwrap();
    let actual = pixels(host.renderer_mut());
    let mut fresh = DemoHost::new(renderer(&device), DemoHostOptions::default());
    fresh.rebuild_pipeline_from_dsl(&changed, true).unwrap();
    fresh.settle().unwrap();
    assert_ne!(actual, before);
    assert_eq!(actual, pixels(fresh.renderer_mut()));
}
