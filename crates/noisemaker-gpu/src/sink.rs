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
    /// `deferRender()`: ask the renderer to skip the next frame.
    fn defer_render(&mut self) -> Option<bool> {
        None
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

/// `SinkManager`.
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
            if sink.configure(descriptor).is_err() {
                stats.failed += 1;
            }
        }
    }

    /// `shouldDeferRender()`.
    pub fn should_defer_render(&mut self) -> bool {
        if self.closed {
            return false;
        }
        self.sinks
            .iter_mut()
            .any(|(_, sink, _)| sink.defer_render() == Some(true))
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
                Err(_) => stats.failed += 1,
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
