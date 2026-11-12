//! Output sinks (port of `runtime/sink.js`): every frame the pipeline submits the
//! texture id of its render surface to each registered sink.

use crate::backend::WebGpuBackend;

/// A sink descriptor (`_sinkDescriptor`).
#[derive(Debug, Clone, PartialEq)]
pub struct SinkDescriptor {
    pub width: f64,
    pub height: f64,
    pub format: String,
    pub color_space: String,
    pub alpha_mode: String,
    pub fps: f64,
}

impl Default for SinkDescriptor {
    fn default() -> Self {
        SinkDescriptor {
            width: 0.0,
            height: 0.0,
            format: "rgba8unorm".into(),
            color_space: "srgb".into(),
            alpha_mode: "premultiplied".into(),
            fps: 60.0,
        }
    }
}

/// An output sink.
pub trait Sink {
    fn configure(&mut self, descriptor: &SinkDescriptor) -> Result<(), String>;
    /// Submit the presented texture. `Ok(Some(true))` counts as accepted,
    /// `Ok(Some(false))` as dropped.
    fn submit(
        &mut self,
        backend: &mut WebGpuBackend,
        texture_id: &str,
        timestamp: f64,
    ) -> Result<Option<bool>, String>;
    fn close(&mut self) -> Result<(), String>;
    /// `deferRender()`: ask the renderer to skip the next frame. A sink whose
    /// deferRender fails is counted as failed, reported, and does not defer
    /// rendering.
    fn defer_render(&mut self) -> Result<Option<bool>, String> {
        Ok(None)
    }
    /// `onError(error)`: an optional hook that receives each error this sink
    /// threw from `configure`, `deferRender` or `submit`, for example to end
    /// an output whose new size it cannot send. A failing report stays
    /// isolated from rendering and from the other sinks.
    fn on_error(&mut self, _error: &str) -> Result<(), String> {
        Ok(())
    }
}

/// `CanvasSink`: presents through `backend.present` (a no-op without a canvas).
#[derive(Default)]
pub struct CanvasSink {
    closed: bool,
}

impl Sink for CanvasSink {
    fn configure(&mut self, _descriptor: &SinkDescriptor) -> Result<(), String> {
        Ok(())
    }

    fn submit(
        &mut self,
        backend: &mut WebGpuBackend,
        texture_id: &str,
        _timestamp: f64,
    ) -> Result<Option<bool>, String> {
        backend.present(texture_id);
        Ok(Some(true))
    }

    fn close(&mut self) -> Result<(), String> {
        self.closed = true;
        Ok(())
    }
}

/// Per-sink counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SinkStats {
    pub accepted: u64,
    pub dropped: u64,
    pub failed: u64,
}

/// A registration handle (`add` returns the reference's removal function;
/// [`SinkManager::remove`] with this id is that function).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SinkId(u64);

/// `reportSinkError` (runtime/pipeline.js): pass an isolated sink failure to
/// the sink that threw it, through its optional `onError` hook. Rendering and
/// the other sinks continue either way.
fn report_sink_error(sink: &mut dyn Sink, error: &str) {
    let _ = sink.on_error(error);
}

/// `SinkManager`. A failure a sink throws in `configure`, `deferRender` or
/// `submit` is counted and reported to that sink's own `on_error`
/// (`reportSinkError`); the pipeline gives its manager that reporter.
#[derive(Default)]
pub struct SinkManager {
    sinks: Vec<(SinkId, Box<dyn Sink>, SinkStats)>,
    descriptor: Option<SinkDescriptor>,
    closed: bool,
    next_id: u64,
}

impl SinkManager {
    /// `add(sink)`: register a sink, configuring it first when the manager is
    /// configured (a sink whose configure fails is not registered).
    pub fn add(&mut self, mut sink: Box<dyn Sink>) -> Result<SinkId, String> {
        if self.closed {
            return Err("Error: SinkManager is closed".into());
        }
        if let Some(d) = &self.descriptor {
            sink.configure(d)?;
        }
        let id = SinkId(self.next_id);
        self.next_id += 1;
        self.sinks.push((id, sink, SinkStats::default()));
        Ok(id)
    }

    /// `remove(sink)` / the removal function: unregister and close the sink
    /// (removing twice is a no-op).
    pub fn remove(&mut self, id: SinkId) -> Result<(), String> {
        let Some(index) = self.sinks.iter().position(|(sid, _, _)| *sid == id) else {
            return Ok(());
        };
        let (_, mut sink, _) = self.sinks.remove(index);
        sink.close()
    }

    /// `configure(descriptor)`.
    pub fn configure(&mut self, descriptor: &SinkDescriptor) {
        if self.closed {
            return;
        }
        self.descriptor = Some(descriptor.clone());
        for (_, sink, stats) in &mut self.sinks {
            if let Err(e) = sink.configure(descriptor) {
                stats.failed += 1;
                report_sink_error(sink.as_mut(), &e);
            }
        }
    }

    /// `shouldDeferRender()`. A sink whose deferRender fails is counted as
    /// failed, reported, and does not defer rendering.
    pub fn should_defer_render(&mut self) -> bool {
        if self.closed {
            return false;
        }
        for (_, sink, stats) in &mut self.sinks {
            match sink.defer_render() {
                Ok(Some(true)) => return true,
                Ok(_) => {}
                Err(e) => {
                    stats.failed += 1;
                    report_sink_error(sink.as_mut(), &e);
                }
            }
        }
        false
    }

    /// `submit(textureId, timestamp)`.
    pub fn submit(&mut self, backend: &mut WebGpuBackend, texture_id: &str, timestamp: f64) {
        if self.closed {
            return;
        }
        for (_, sink, stats) in &mut self.sinks {
            match sink.submit(backend, texture_id, timestamp) {
                Ok(Some(true)) => stats.accepted += 1,
                Ok(Some(false)) => stats.dropped += 1,
                Ok(None) => {}
                Err(e) => {
                    stats.failed += 1;
                    report_sink_error(sink.as_mut(), &e);
                }
            }
        }
    }

    /// `close()`: close every sink; the first error is returned.
    pub fn close(&mut self) -> Result<(), String> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        let mut first = Ok(());
        for (_, mut sink, _) in self.sinks.drain(..) {
            if let Err(e) = sink.close()
                && first.is_ok()
            {
                first = Err(e);
            }
        }
        first
    }

    /// Per-sink stats in registration order.
    pub fn stats(&self) -> Vec<SinkStats> {
        self.sinks.iter().map(|(_, _, s)| *s).collect()
    }

    /// The stats of one sink.
    pub fn stats_of(&self, id: SinkId) -> Option<SinkStats> {
        self.sinks
            .iter()
            .find(|(sid, _, _)| *sid == id)
            .map(|(_, _, s)| *s)
    }
}

/// `CUBE_FACE_BASES` (`renderer/cubeCamera.js`): column-major `[right | up |
/// forward]` mat3 per face, in GL cubemap order +X, -X, +Y, -Y, +Z, -Z.
pub const CUBE_FACE_BASES: [[f64; 9]; 6] = {
    // forward and up per face; right = cross(up, forward).
    const FACES: [([f64; 3], [f64; 3]); 6] = [
        ([1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),
        ([0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),
    ];
    let mut out = [[0.0; 9]; 6];
    let mut i = 0;
    while i < 6 {
        let (f, u) = FACES[i];
        let r = [
            u[1] * f[2] - u[2] * f[1],
            u[2] * f[0] - u[0] * f[2],
            u[0] * f[1] - u[1] * f[0],
        ];
        out[i] = [r[0], r[1], r[2], u[0], u[1], u[2], f[0], f[1], f[2]];
        i += 1;
    }
    out
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    type Log = Rc<RefCell<Vec<String>>>;
    type Flag = Rc<RefCell<bool>>;

    /// A sink that fails in `configure` when its flag is set. (Its submit
    /// failures are covered by the GPU-gated host test, which can call
    /// `submit`.)
    #[derive(Default)]
    struct FailingSink {
        fail: Flag,
        reports: Log,
    }

    impl Sink for FailingSink {
        fn configure(&mut self, _: &SinkDescriptor) -> Result<(), String> {
            if *self.fail.borrow() {
                return Err("configure failed".into());
            }
            Ok(())
        }

        fn submit(
            &mut self,
            _: &mut WebGpuBackend,
            _: &str,
            _: f64,
        ) -> Result<Option<bool>, String> {
            Ok(Some(true))
        }

        fn close(&mut self) -> Result<(), String> {
            Ok(())
        }

        fn on_error(&mut self, error: &str) -> Result<(), String> {
            self.reports.borrow_mut().push(error.to_owned());
            Ok(())
        }
    }

    /// A sink without `on_error` behavior: default hook, no report.
    #[derive(Default)]
    struct SilentSink {
        fail: Flag,
    }

    impl Sink for SilentSink {
        fn configure(&mut self, _: &SinkDescriptor) -> Result<(), String> {
            if *self.fail.borrow() {
                return Err("silent configure failed".into());
            }
            Ok(())
        }

        fn submit(
            &mut self,
            _: &mut WebGpuBackend,
            _: &str,
            _: f64,
        ) -> Result<Option<bool>, String> {
            Ok(Some(true))
        }

        fn close(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    /// A sink whose reporter fails: the report stays isolated.
    #[derive(Default)]
    struct ThrowingReporter {
        fail: Flag,
        calls: Rc<RefCell<usize>>,
    }

    impl Sink for ThrowingReporter {
        fn configure(&mut self, _: &SinkDescriptor) -> Result<(), String> {
            if *self.fail.borrow() {
                return Err("configure failed again".into());
            }
            Ok(())
        }

        fn submit(
            &mut self,
            _: &mut WebGpuBackend,
            _: &str,
            _: f64,
        ) -> Result<Option<bool>, String> {
            Ok(Some(true))
        }

        fn close(&mut self) -> Result<(), String> {
            Ok(())
        }

        fn on_error(&mut self, _: &str) -> Result<(), String> {
            *self.calls.borrow_mut() += 1;
            Err("reporter failed".into())
        }
    }

    /// A sink that does not fail records the calls it sees.
    struct HealthySink {
        calls: Log,
    }

    impl Sink for HealthySink {
        fn configure(&mut self, d: &SinkDescriptor) -> Result<(), String> {
            self.calls
                .borrow_mut()
                .push(format!("configure {}x{}", d.width, d.height));
            Ok(())
        }

        fn submit(
            &mut self,
            _: &mut WebGpuBackend,
            _: &str,
            _: f64,
        ) -> Result<Option<bool>, String> {
            Ok(Some(true))
        }

        fn close(&mut self) -> Result<(), String> {
            Ok(())
        }

        fn on_error(&mut self, error: &str) -> Result<(), String> {
            self.calls.borrow_mut().push(format!("onError {error}"));
            Ok(())
        }
    }

    fn descriptor(width: f64, height: f64) -> SinkDescriptor {
        SinkDescriptor {
            width,
            height,
            ..Default::default()
        }
    }

    #[test]
    fn a_configure_failure_reaches_only_the_sink_that_threw_it() {
        let failed_reports: Log = Default::default();
        let reporter_calls: Rc<RefCell<usize>> = Default::default();
        let healthy_calls: Log = Default::default();
        let fail: Flag = Default::default();
        let mut manager = SinkManager::default();
        manager
            .add(Box::new(FailingSink {
                fail: fail.clone(),
                reports: failed_reports.clone(),
            }))
            .unwrap();
        manager
            .add(Box::new(SilentSink {
                fail: fail.clone(),
            }))
            .unwrap();
        manager
            .add(Box::new(ThrowingReporter {
                fail: fail.clone(),
                calls: reporter_calls.clone(),
            }))
            .unwrap();
        manager
            .add(Box::new(HealthySink {
                calls: healthy_calls.clone(),
            }))
            .unwrap();
        *fail.borrow_mut() = true;

        manager.configure(&descriptor(640.0, 480.0));

        assert_eq!(*failed_reports.borrow(), vec!["configure failed"]);
        assert_eq!(*reporter_calls.borrow(), 1);
        assert_eq!(
            *healthy_calls.borrow(),
            vec!["configure 640x480".to_owned()]
        );
        let stats = manager.stats();
        assert_eq!(
            stats,
            vec![
                SinkStats {
                    accepted: 0,
                    dropped: 0,
                    failed: 1
                },
                SinkStats {
                    accepted: 0,
                    dropped: 0,
                    failed: 1
                },
                SinkStats {
                    accepted: 0,
                    dropped: 0,
                    failed: 1
                },
                SinkStats::default(),
            ]
        );
    }

    #[test]
    fn a_failing_defer_render_is_reported_and_does_not_defer() {
        let reports: Log = Default::default();
        let mut manager = SinkManager::default();
        struct FailingDefer {
            reports: Log,
        }
        impl Sink for FailingDefer {
            fn configure(&mut self, _: &SinkDescriptor) -> Result<(), String> {
                Ok(())
            }
            fn submit(
                &mut self,
                _: &mut WebGpuBackend,
                _: &str,
                _: f64,
            ) -> Result<Option<bool>, String> {
                Ok(Some(true))
            }
            fn close(&mut self) -> Result<(), String> {
                Ok(())
            }
            fn defer_render(&mut self) -> Result<Option<bool>, String> {
                Err("defer probe failed".into())
            }
            fn on_error(&mut self, error: &str) -> Result<(), String> {
                self.reports.borrow_mut().push(error.to_owned());
                Ok(())
            }
        }
        manager
            .add(Box::new(FailingDefer {
                reports: reports.clone(),
            }))
            .unwrap();

        assert!(!manager.should_defer_render());
        assert_eq!(*reports.borrow(), vec!["defer probe failed"]);
        assert_eq!(
            manager.stats_of(SinkId(0)).map(|s| s.failed),
            Some(1),
            "the defer failure counts, the sink that deferred does not defer"
        );

        // A sink that defers still defers the frame.
        struct DeferringSink;
        impl Sink for DeferringSink {
            fn configure(&mut self, _: &SinkDescriptor) -> Result<(), String> {
                Ok(())
            }
            fn submit(
                &mut self,
                _: &mut WebGpuBackend,
                _: &str,
                _: f64,
            ) -> Result<Option<bool>, String> {
                Ok(Some(true))
            }
            fn close(&mut self) -> Result<(), String> {
                Ok(())
            }
            fn defer_render(&mut self) -> Result<Option<bool>, String> {
                Ok(Some(true))
            }
        }
        manager.add(Box::new(DeferringSink)).unwrap();
        assert!(manager.should_defer_render());
    }

    #[test]
    fn cube_bases_are_right_handed() {
        // +X face: forward (1,0,0), up (0,-1,0), right = cross(up, forward) = (0,0,1).
        assert_eq!(
            CUBE_FACE_BASES[0],
            [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0]
        );
        // +Y face: up (0,0,1), forward (0,1,0) → right = (-1,0,0).
        assert_eq!(&CUBE_FACE_BASES[2][..3], &[-1.0, 0.0, 0.0]);
    }
}
