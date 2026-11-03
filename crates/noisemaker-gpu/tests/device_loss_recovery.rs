//! Recovery after a lost device: the fresh-device rebuild the viewer's
//! recovery performs, at the library level (a headless run has no window for
//! the example itself).
//!
//! The viewer's recovery disposes the dead device's renderer, builds a fresh
//! renderer on a fresh device, carries the render loop's clock (loop start and
//! loop duration) and the current program over, compiles and renders again.
//! The reference's rebuild replaces the pipeline on the renderer's running
//! clock, not a restarted one.

use std::time::{Duration, Instant};

use noisemaker_gpu::GpuDevice;
use noisemaker_gpu::demo::{DemoHost, DemoHostOptions};
use noisemaker_gpu::host::{CanvasRenderer, CanvasRendererOptions};

const PROGRAM: &str = "noise(seed: 1, scaleX: 50, scaleY: 50).adjust().write(o0)\nrender(o0)\n";

fn options() -> CanvasRendererOptions {
    CanvasRendererOptions {
        width: 64,
        height: 64,
        ..Default::default()
    }
}

/// The recovery rebuilds on a fresh device and resumes rendering on the old
/// loop's clock: the normalized time continues where the lost device's loop
/// left it, instead of restarting.
#[test]
fn recovery_rebuilds_on_a_fresh_device_and_keeps_the_loop_clock() {
    let device = GpuDevice::create(&Default::default()).expect("a GPU adapter");
    let mut host = DemoHost::new(
        CanvasRenderer::new(&device, options()),
        DemoHostOptions::default(),
    );
    host.load_dsl_and_create_controls(PROGRAM)
        .expect("the program loads");
    host.settle().expect("the program settles");

    // Run the loop for 6 of the default 10 loop seconds.
    let mut now = Instant::now();
    for _ in 0..60 {
        now += Duration::from_millis(100);
        host.renderer_mut().tick(now).expect("the loop renders");
    }
    let loop_start = host.renderer().loop_start();
    let loop_duration = host.renderer().loop_duration();

    // The device is lost; the recovery rebuilds on a fresh device and carries
    // the loop's clock and the current program over.
    let _ = host.renderer_mut().dispose();
    drop(host);
    let fresh_device = GpuDevice::create(&Default::default()).expect("a fresh adapter");
    let mut recovered = DemoHost::new(
        CanvasRenderer::new(&fresh_device, options()),
        DemoHostOptions::default(),
    );
    recovered.renderer_mut().set_loop_duration(loop_duration);
    recovered.renderer_mut().set_loop_start(loop_start);
    recovered.renderer_mut().set_current_dsl(PROGRAM);
    recovered
        .load_dsl_and_create_controls(PROGRAM)
        .expect("the program loads on the fresh device");
    recovered.settle().expect("the program settles");
    now += Duration::from_millis(100);
    recovered
        .renderer_mut()
        .tick(now)
        .expect("the rebuilt renderer renders");

    // The normalized time continued on the carried clock (6.1 of 10 loop
    // seconds); a restarted clock would show a fraction of it.
    let elapsed = now.saturating_duration_since(loop_start).as_secs_f64();
    let expected = (elapsed % loop_duration) / loop_duration;
    let actual = recovered.renderer().normalized_time_at(now);
    assert!(
        (actual - expected).abs() < 1e-6,
        "the rebuilt loop's normalized time is {actual}, expected {expected}"
    );
    assert!(expected > 0.5, "the loop clock must not have restarted");
    assert_eq!(
        recovered.renderer().current_dsl(),
        PROGRAM,
        "the current program carries over"
    );
}
