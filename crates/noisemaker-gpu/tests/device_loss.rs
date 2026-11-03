//! `device.lost` of the pipeline's WebGPU device: the watch the viewer's
//! recovery waits on (`CanvasRenderer._watchDeviceLoss`).
//!
//! The reference watches `pipeline.backend.device.lost` and rebuilds the
//! program on a fresh canvas when it resolves (test_webgpu_device_loss_recovery.mjs
//! covers `GPUDevice.destroy()`, a retired device and a GPU process exit).
//! These tests cover the watch itself: a live device reports no loss, and a
//! destroyed device reports its loss with the WebGPU reason. The recovery the
//! watch drives is covered in device_loss_recovery.rs.

use std::time::{Duration, Instant};

use noisemaker_gpu::GpuDevice;

/// A live device reports no loss.
#[test]
fn a_live_device_is_not_lost() {
    let device = GpuDevice::create(&Default::default()).expect("a GPU adapter");
    let mut watch = device.watch_loss();
    let _ = device.device.poll(wgpu::PollType::Poll);
    assert!(watch.lost().is_none(), "a live device is not lost");
}

/// `GPUDevice.destroy()`: the watch yields the loss with the WebGPU destroy
/// reason, once the device is maintained and its queues are empty.
#[test]
fn a_destroyed_device_is_lost() {
    let device = GpuDevice::create(&Default::default()).expect("a GPU adapter");
    let mut watch = device.watch_loss();
    device.device.destroy();
    // wgpu fires the loss callback while maintaining the device (the browser
    // resolves the lost promise on its own; the viewer pumps the device every
    // iteration).
    let deadline = Instant::now() + Duration::from_secs(30);
    let loss = loop {
        if let Some(loss) = watch.lost() {
            break loss;
        }
        let _ = device.device.poll(wgpu::PollType::Poll);
        assert!(
            Instant::now() < deadline,
            "the destroyed device's loss never arrived"
        );
        std::thread::sleep(Duration::from_millis(10));
    };
    assert_eq!(format!("{:?}", loss.reason), "Destroyed");
    assert_eq!(loss.message, "", "destroy() reports no message");
}
