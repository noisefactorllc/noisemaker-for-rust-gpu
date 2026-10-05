//! Asynchronous frame export: ports of `runtime/frame-export.js`
//! (`FrameExportQueue`) and `runtime/backends/webgpu-frame-export.js`
//! (`WebGPUFrameExportAdapter`).
//!
//! A queue owns a fixed ring of slots (2 to 8). `enqueue` starts a slot's
//! readback of a texture; `poll` hands every slot whose readback finished to
//! its callback, as packed RGBA8 rows. The WebGPU adapter resolves the source
//! texture into an rgba8unorm texture (rows flipped to the presented
//! orientation, alpha converted to the descriptor's `alphaMode`), copies it
//! into a mappable buffer and maps it asynchronously; `poll` drives the
//! device (`wgpu::PollType::Poll`) where a browser resolves the `mapAsync`
//! promise by itself.

use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::sync::{Arc, Mutex};

use crate::backend::TextureRecord;
use crate::backend::WebGpuBackend;
use crate::sink::SinkDescriptor;

/// An error a queue or adapter reports (the reference throws `TypeError`s,
/// `RangeError`s and `Error`s; the message carries the name).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameExportError(pub String);

impl std::fmt::Display for FrameExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FrameExportError {}

fn error(message: impl Into<String>) -> FrameExportError {
    FrameExportError(message.into())
}

/// One exported frame: `height` rows of `row_stride` bytes, RGBA8.
#[derive(Debug, Clone, Copy)]
pub struct ExportedFrame<'a> {
    pub width: u32,
    pub height: u32,
    pub row_stride: usize,
    pub data: &'a [u8],
}

/// What a frame export queue drives (`createSlot`, `begin`, `poll`, `read`,
/// `destroySlot`). `Source` is what `begin` reads the texture from.
pub trait FrameExportAdapter {
    type Slot;
    type Source: ?Sized;
    fn create_slot(
        &mut self,
        index: usize,
        descriptor: &SinkDescriptor,
    ) -> Result<Self::Slot, FrameExportError>;
    fn begin(
        &mut self,
        source: &Self::Source,
        slot: &mut Self::Slot,
        texture_id: &str,
        timestamp: f64,
    ) -> Result<(), FrameExportError>;
    /// `Ok(true)` when the slot's frame is ready to read.
    fn poll(&mut self, slot: &mut Self::Slot) -> Result<bool, FrameExportError>;
    fn read<'a>(&mut self, slot: &'a mut Self::Slot)
    -> Result<ExportedFrame<'a>, FrameExportError>;
    fn destroy_slot(&mut self, slot: Self::Slot) -> Result<(), FrameExportError>;
}

/// `onFrame(frame, timestamp, context)`: the context is the closure's own
/// capture.
pub type FrameCallback = Box<dyn FnOnce(&ExportedFrame<'_>, f64) -> Result<(), FrameExportError>>;

/// `onError(error)`.
pub type ErrorCallback = Box<dyn FnMut(&FrameExportError)>;

/// `queue.stats`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameExportStats {
    pub accepted: u64,
    pub dropped: u64,
    pub completed: u64,
    pub failed: u64,
}

/// Options of [`FrameExportQueue::new`].
pub struct FrameExportOptions {
    /// Slot count, 2 to 8 (default 3).
    pub slots: usize,
    pub on_error: Option<ErrorCallback>,
}

impl Default for FrameExportOptions {
    fn default() -> Self {
        FrameExportOptions {
            slots: 3,
            on_error: None,
        }
    }
}

struct SlotRecord<S> {
    adapter_slot: Option<S>,
    pending: bool,
    texture_id: Option<String>,
    timestamp: f64,
    on_frame: Option<FrameCallback>,
}

impl<S> SlotRecord<S> {
    fn release(&mut self) {
        self.pending = false;
        self.texture_id = None;
        self.timestamp = f64::NAN;
        self.on_frame = None;
    }
}

/// `FrameExportQueue`.
pub struct FrameExportQueue<A: FrameExportAdapter> {
    adapter: Option<A>,
    on_error: Option<ErrorCallback>,
    slots: Vec<SlotRecord<A::Slot>>,
    configured: bool,
    closed: bool,
    pub stats: FrameExportStats,
}

impl<A: FrameExportAdapter> FrameExportQueue<A> {
    /// `new FrameExportQueue(adapter, {slots, onError})`.
    pub fn new(adapter: A, options: FrameExportOptions) -> Result<Self, FrameExportError> {
        if !(2..=8).contains(&options.slots) {
            return Err(error(
                "RangeError: Frame export slots must be an integer from 2 through 8",
            ));
        }
        Ok(FrameExportQueue {
            adapter: Some(adapter),
            on_error: options.on_error,
            slots: (0..options.slots)
                .map(|_| SlotRecord {
                    adapter_slot: None,
                    pending: false,
                    texture_id: None,
                    timestamp: f64::NAN,
                    on_frame: None,
                })
                .collect(),
            configured: false,
            closed: false,
            stats: FrameExportStats::default(),
        })
    }

    /// The adapter (until the queue is closed).
    pub fn adapter(&self) -> Option<&A> {
        self.adapter.as_ref()
    }

    /// `available`: a free slot exists on a configured, open queue.
    pub fn available(&self) -> bool {
        self.configured && !self.closed && self.slots.iter().any(|s| !s.pending)
    }

    /// The texture ids of the pending slots, in slot order.
    pub fn pending_textures(&self) -> Vec<&str> {
        self.slots
            .iter()
            .filter(|s| s.pending)
            .filter_map(|s| s.texture_id.as_deref())
            .collect()
    }

    /// `configure(descriptor)`: destroy the slots and create them for the
    /// descriptor.
    pub fn configure(&mut self, descriptor: &SinkDescriptor) -> Result<(), FrameExportError> {
        if self.closed {
            return Ok(());
        }
        let destroy_error = self.destroy_slots();
        self.configured = false;
        if let Some(e) = destroy_error {
            return Err(e);
        }
        let adapter = self
            .adapter
            .as_mut()
            .expect("an open queue has its adapter");
        for i in 0..self.slots.len() {
            match adapter.create_slot(i, descriptor) {
                Ok(slot) => self.slots[i].adapter_slot = Some(slot),
                Err(e) => {
                    if let Some(cleanup) = self.destroy_slots() {
                        self.report(&cleanup);
                    }
                    return Err(e);
                }
            }
        }
        self.configured = true;
        Ok(())
    }

    /// `enqueue(textureId, timestamp, onFrame, context)`: start reading the
    /// texture into a free slot. `false` (counted as dropped) when the queue is
    /// not configured, closed or full; `false` (counted as failed, reported)
    /// when the adapter cannot begin.
    pub fn enqueue(
        &mut self,
        source: &A::Source,
        texture_id: &str,
        timestamp: f64,
        on_frame: FrameCallback,
    ) -> bool {
        if !self.configured || self.closed {
            self.stats.dropped += 1;
            return false;
        }
        let Some(index) = self.slots.iter().position(|s| !s.pending) else {
            self.stats.dropped += 1;
            return false;
        };
        let record = &mut self.slots[index];
        record.pending = true;
        record.texture_id = Some(texture_id.to_owned());
        record.timestamp = timestamp;
        record.on_frame = Some(on_frame);
        let adapter = self
            .adapter
            .as_mut()
            .expect("an open queue has its adapter");
        let slot = record
            .adapter_slot
            .as_mut()
            .expect("a configured queue has its slots");
        if let Err(e) = adapter.begin(source, slot, texture_id, timestamp) {
            self.slots[index].release();
            self.stats.failed += 1;
            self.report(&e);
            return false;
        }
        self.stats.accepted += 1;
        true
    }

    /// `poll()`: hand every finished slot's frame to its callback.
    pub fn poll(&mut self) {
        if !self.configured || self.closed {
            return;
        }
        for i in 0..self.slots.len() {
            if !self.slots[i].pending {
                continue;
            }
            let adapter = self
                .adapter
                .as_mut()
                .expect("an open queue has its adapter");
            let record = &mut self.slots[i];
            let slot = record
                .adapter_slot
                .as_mut()
                .expect("a configured queue has its slots");
            match adapter.poll(slot) {
                Ok(false) => continue,
                Ok(true) => {}
                Err(e) => {
                    record.release();
                    self.stats.failed += 1;
                    self.report(&e);
                    continue;
                }
            }
            let timestamp = record.timestamp;
            let on_frame = record.on_frame.take();
            let result = match adapter.read(slot) {
                Ok(frame) => Ok(on_frame.map(|f| f(&frame, timestamp))),
                Err(e) => Err(e),
            };
            self.slots[i].release();
            match result {
                Err(e) => {
                    self.stats.failed += 1;
                    self.report(&e);
                }
                Ok(Some(Err(e))) => {
                    self.stats.failed += 1;
                    self.report(&e);
                }
                Ok(_) => self.stats.completed += 1,
            }
        }
    }

    /// `close({backendLost})`: destroy the slots (or abandon them when the
    /// backend is gone); pending frames count as dropped.
    pub fn close(&mut self, backend_lost: bool) -> Result<(), FrameExportError> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        self.configured = false;
        let destroy_error = if backend_lost {
            for record in &mut self.slots {
                record.adapter_slot = None;
                if record.pending {
                    self.stats.dropped += 1;
                }
                record.release();
            }
            None
        } else {
            self.destroy_slots()
        };
        self.adapter = None;
        match destroy_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn destroy_slots(&mut self) -> Option<FrameExportError> {
        let mut first = None;
        for i in 0..self.slots.len() {
            let Some(slot) = self.slots[i].adapter_slot.take() else {
                continue;
            };
            if self.slots[i].pending {
                self.stats.dropped += 1;
            }
            self.slots[i].release();
            if let Some(adapter) = self.adapter.as_mut()
                && let Err(e) = adapter.destroy_slot(slot)
            {
                first.get_or_insert(e);
            }
        }
        first
    }

    fn report(&mut self, error: &FrameExportError) {
        if let Some(on_error) = self.on_error.as_mut() {
            on_error(error);
        }
    }
}

const ALPHA_MODES: [&str; 3] = ["straight", "opaque", "premultiplied"];

const RESOLVE_SHADER: &str = r#"
@group(0) @binding(0) var sourceTexture: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertexIndex], 0.0, 1.0);
    return output;
}

fn loadColor(position: vec4<f32>) -> vec4<f32> {
    // Match the row orientation used when presenting the surface to the canvas.
    let sourceSize = textureDimensions(sourceTexture);
    let sourceCoord = vec2<i32>(i32(position.x), i32(sourceSize.y) - 1 - i32(position.y));
    return textureLoad(sourceTexture, sourceCoord, 0);
}

@fragment
fn straight_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return loadColor(input.position);
}

@fragment
fn opaque_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let color = loadColor(input.position);
    return vec4<f32>(color.rgb, 1.0);
}

@fragment
fn premultiplied_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let color = loadColor(input.position);
    return vec4<f32>(color.rgb * color.a, color.a);
}
"#;

/// `validateDescriptor(descriptor)`: the slot layout.
struct SlotLayout {
    row_stride: usize,
    bytes_per_row: usize,
    buffer_size: usize,
    packed_size: usize,
}

const MAX_SAFE_INTEGER: f64 = 9007199254740991.0;

fn is_safe_integer(x: f64) -> bool {
    x.is_finite() && x.trunc() == x && x.abs() <= MAX_SAFE_INTEGER
}

fn checked_product(left: f64, right: f64, message: &str) -> Result<f64, FrameExportError> {
    let value = left * right;
    if !is_safe_integer(value) {
        return Err(error(format!("RangeError: {message}")));
    }
    Ok(value)
}

fn validate_descriptor(d: &SinkDescriptor) -> Result<SlotLayout, FrameExportError> {
    if !is_safe_integer(d.width) || d.width <= 0.0 {
        return Err(error(
            "RangeError: Frame export width must be a positive integer",
        ));
    }
    if !is_safe_integer(d.height) || d.height <= 0.0 {
        return Err(error(
            "RangeError: Frame export height must be a positive integer",
        ));
    }
    if d.format != "rgba8unorm" {
        return Err(error(
            "TypeError: WebGPU frame export format must be 'rgba8unorm'",
        ));
    }
    if d.color_space != "srgb" && d.color_space != "display-p3" {
        return Err(error(
            "TypeError: WebGPU frame export colorSpace must be 'srgb' or 'display-p3'",
        ));
    }
    if !ALPHA_MODES.contains(&d.alpha_mode.as_str()) {
        return Err(error(
            "TypeError: WebGPU frame export alphaMode must be 'opaque', 'straight', or 'premultiplied'",
        ));
    }
    if !d.fps.is_finite() || d.fps <= 0.0 {
        return Err(error(
            "RangeError: Frame export fps must be finite and positive",
        ));
    }
    let row_stride = checked_product(d.width, 4.0, "Frame export dimensions are too large")?;
    let aligned_rows = (row_stride / 256.0).ceil();
    let bytes_per_row = checked_product(
        aligned_rows,
        256.0,
        "Frame export aligned row size is too large",
    )?;
    let buffer_size = checked_product(
        bytes_per_row,
        d.height,
        "Frame export dimensions are too large",
    )?;
    let packed_size = checked_product(
        row_stride,
        d.height,
        "Frame export dimensions are too large",
    )?;
    Ok(SlotLayout {
        row_stride: row_stride as usize,
        bytes_per_row: bytes_per_row as usize,
        buffer_size: buffer_size as usize,
        packed_size: packed_size as usize,
    })
}

/// The map state of a slot (`slot.state`): the `mapAsync` callback writes it.
#[derive(Debug, Clone, PartialEq)]
enum MapState {
    Idle,
    Pending,
    Ready,
    Failed(String),
    Destroyed,
}

/// A WebGPU frame export slot.
pub struct WebGpuExportSlot {
    pub index: usize,
    pub width: u32,
    pub height: u32,
    alpha_mode: String,
    row_stride: usize,
    bytes_per_row: usize,
    data: Vec<u8>,
    resolve_texture: wgpu::Texture,
    resolve_view: wgpu::TextureView,
    buffer: wgpu::Buffer,
    state: Arc<Mutex<(u64, MapState)>>,
    registered: bool,
}

impl WebGpuExportSlot {
    fn state(&self) -> MapState {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .1
            .clone()
    }

    fn set_state(&self, state: MapState) {
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        guard.0 += 1;
        guard.1 = state;
    }
}

struct SharedResolve {
    bind_group_layout: wgpu::BindGroupLayout,
    pipelines: HashMap<&'static str, wgpu::RenderPipeline>,
}

/// `WebGPUFrameExportAdapter`.
pub struct WebGpuFrameExportAdapter {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shared: Option<SharedResolve>,
    /// Bind groups by source texture (the reference's `WeakMap` keyed by the
    /// texture handle).
    bind_groups: HashMap<usize, (Weak<TextureRecord>, wgpu::BindGroup)>,
    slot_count: usize,
}

impl WebGpuFrameExportAdapter {
    /// `new WebGPUFrameExportAdapter(backend)`.
    pub fn new(backend: &WebGpuBackend) -> Self {
        WebGpuFrameExportAdapter {
            device: backend.device.clone(),
            queue: backend.queue.clone(),
            shared: None,
            bind_groups: HashMap::new(),
            slot_count: 0,
        }
    }

    fn ensure_shared(&mut self) {
        if self.shared.is_some() {
            return;
        }
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("frame export"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    }],
                });
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("frame export resolve"),
                source: wgpu::ShaderSource::Wgsl(RESOLVE_SHADER.into()),
            });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("frame export"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
        let mut pipelines = HashMap::new();
        for alpha_mode in ALPHA_MODES {
            let entry = format!("{alpha_mode}_main");
            let pipeline = self
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("frame export resolve"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some(&entry),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                });
            pipelines.insert(alpha_mode, pipeline);
        }
        self.shared = Some(SharedResolve {
            bind_group_layout,
            pipelines,
        });
    }

    fn bind_group(&mut self, source: &Rc<TextureRecord>) -> wgpu::BindGroup {
        let key = Rc::as_ptr(source) as usize;
        if let Some((weak, group)) = self.bind_groups.get(&key)
            && weak.upgrade().is_some_and(|r| Rc::ptr_eq(&r, source))
        {
            return group.clone();
        }
        let shared = self.shared.as_ref().expect("shared resolve resources");
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame export source"),
            layout: &shared.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&source.view),
            }],
        });
        self.bind_groups
            .retain(|_, (weak, _)| weak.strong_count() > 0);
        self.bind_groups
            .insert(key, (Rc::downgrade(source), group.clone()));
        group
    }

    fn assert_usable(slot: &WebGpuExportSlot) -> Result<(), FrameExportError> {
        if !slot.registered || slot.state() == MapState::Destroyed {
            return Err(error("Error: WebGPU frame export slot is not usable"));
        }
        Ok(())
    }
}

impl FrameExportAdapter for WebGpuFrameExportAdapter {
    type Slot = WebGpuExportSlot;
    type Source = WebGpuBackend;

    fn create_slot(
        &mut self,
        index: usize,
        descriptor: &SinkDescriptor,
    ) -> Result<WebGpuExportSlot, FrameExportError> {
        let layout = validate_descriptor(descriptor)?;
        self.ensure_shared();
        let (width, height) = (descriptor.width as u32, descriptor.height as u32);
        let resolve_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frame export resolve"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let resolve_view = resolve_texture.create_view(&Default::default());
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame export staging"),
            size: layout.buffer_size as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.slot_count += 1;
        Ok(WebGpuExportSlot {
            index,
            width,
            height,
            alpha_mode: descriptor.alpha_mode.clone(),
            row_stride: layout.row_stride,
            bytes_per_row: layout.bytes_per_row,
            data: vec![0; layout.packed_size],
            resolve_texture,
            resolve_view,
            buffer,
            state: Arc::new(Mutex::new((0, MapState::Idle))),
            registered: true,
        })
    }

    fn begin(
        &mut self,
        backend: &WebGpuBackend,
        slot: &mut WebGpuExportSlot,
        texture_id: &str,
        _timestamp: f64,
    ) -> Result<(), FrameExportError> {
        Self::assert_usable(slot)?;
        if slot.state() != MapState::Idle {
            return Err(error(
                "Error: WebGPU frame export slot already has a pending map",
            ));
        }
        let Some(source) = backend.textures.get(texture_id).cloned() else {
            return Err(error(format!(
                "Error: WebGPU frame export texture {texture_id} not found"
            )));
        };
        if source.width != slot.width as f64 || source.height != slot.height as f64 {
            return Err(error(format!(
                "Error: WebGPU frame export source extent {}x{} does not match configured extent {}x{}",
                crate::jsv::interpolate(&noisemaker_dsl::Value::Number(source.width)),
                crate::jsv::interpolate(&noisemaker_dsl::Value::Number(source.height)),
                slot.width,
                slot.height
            )));
        }
        let bind_group = self.bind_group(&source);
        let shared = self.shared.as_ref().expect("shared resolve resources");
        let pipeline = &shared.pipelines[slot.alpha_mode.as_str()];
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frame export resolve"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &slot.resolve_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            slot.resolve_texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &slot.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(slot.bytes_per_row as u32),
                    rows_per_image: Some(slot.height),
                },
            },
            wgpu::Extent3d {
                width: slot.width,
                height: slot.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        slot.set_state(MapState::Pending);
        let token = slot.state.lock().unwrap_or_else(|e| e.into_inner()).0;
        let state = slot.state.clone();
        slot.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let mut guard = state.lock().unwrap_or_else(|e| e.into_inner());
                if guard.0 != token || guard.1 != MapState::Pending {
                    return;
                }
                guard.1 = match result {
                    Ok(()) => MapState::Ready,
                    Err(e) => MapState::Failed(e.to_string()),
                };
            });
        Ok(())
    }

    fn poll(&mut self, slot: &mut WebGpuExportSlot) -> Result<bool, FrameExportError> {
        Self::assert_usable(slot)?;
        // A browser resolves mapAsync on its own; wgpu runs the callback when
        // the device is polled.
        let _ = self.device.poll(wgpu::PollType::Poll);
        match slot.state() {
            MapState::Pending => Ok(false),
            MapState::Ready => Ok(true),
            MapState::Failed(message) => {
                slot.set_state(MapState::Idle);
                Err(error(if message.is_empty() {
                    "Error: WebGPU frame export mapping failed".to_owned()
                } else {
                    format!("OperationError: {message}")
                }))
            }
            _ => Err(error("Error: WebGPU frame export slot has no pending map")),
        }
    }

    fn read<'a>(
        &mut self,
        slot: &'a mut WebGpuExportSlot,
    ) -> Result<ExportedFrame<'a>, FrameExportError> {
        Self::assert_usable(slot)?;
        if slot.state() != MapState::Ready {
            return Err(error(
                "Error: WebGPU frame export slot is not ready after a completed map",
            ));
        }
        let copied = {
            match slot.buffer.slice(..).get_mapped_range() {
                Ok(range) => {
                    let source: &[u8] = &range;
                    for row in 0..slot.height as usize {
                        let from = row * slot.bytes_per_row;
                        let to = row * slot.row_stride;
                        slot.data[to..to + slot.row_stride]
                            .copy_from_slice(&source[from..from + slot.row_stride]);
                    }
                    Ok(())
                }
                Err(e) => Err(error(format!("OperationError: getMappedRange: {e:?}"))),
            }
        };
        slot.buffer.unmap();
        if let Err(e) = copied {
            slot.set_state(MapState::Idle);
            return Err(e);
        }
        slot.set_state(MapState::Idle);
        Ok(ExportedFrame {
            width: slot.width,
            height: slot.height,
            row_stride: slot.row_stride,
            data: &slot.data,
        })
    }

    fn destroy_slot(&mut self, slot: WebGpuExportSlot) -> Result<(), FrameExportError> {
        let previous = slot.state();
        if previous == MapState::Destroyed {
            return Ok(());
        }
        slot.set_state(MapState::Destroyed);
        if matches!(previous, MapState::Ready) {
            slot.buffer.unmap();
        }
        slot.resolve_texture.destroy();
        slot.buffer.destroy();
        if slot.registered {
            self.slot_count -= 1;
        }
        if self.slot_count == 0 {
            self.shared = None;
            self.bind_groups.clear();
        }
        Ok(())
    }
}

/// `backend.createFrameExportQueue(options)`.
pub fn create_frame_export_queue(
    backend: &WebGpuBackend,
    options: FrameExportOptions,
) -> Result<FrameExportQueue<WebGpuFrameExportAdapter>, FrameExportError> {
    FrameExportQueue::new(WebGpuFrameExportAdapter::new(backend), options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// A scripted adapter: slots become ready after `delay` polls.
    #[derive(Default)]
    struct Mock {
        log: Rc<RefCell<Vec<String>>>,
        fail_begin: bool,
        delay: u32,
    }

    struct MockSlot {
        index: usize,
        polls: u32,
        data: Vec<u8>,
    }

    impl FrameExportAdapter for Mock {
        type Slot = MockSlot;
        type Source = ();
        fn create_slot(
            &mut self,
            index: usize,
            d: &SinkDescriptor,
        ) -> Result<MockSlot, FrameExportError> {
            self.log.borrow_mut().push(format!("create {index}"));
            Ok(MockSlot {
                index,
                polls: 0,
                data: vec![index as u8; (d.width * d.height * 4.0) as usize],
            })
        }
        fn begin(
            &mut self,
            _: &(),
            slot: &mut MockSlot,
            id: &str,
            _: f64,
        ) -> Result<(), FrameExportError> {
            if self.fail_begin {
                return Err(error("Error: begin failed"));
            }
            slot.polls = 0;
            self.log
                .borrow_mut()
                .push(format!("begin {} {id}", slot.index));
            Ok(())
        }
        fn poll(&mut self, slot: &mut MockSlot) -> Result<bool, FrameExportError> {
            slot.polls += 1;
            Ok(slot.polls > self.delay)
        }
        fn read<'a>(
            &mut self,
            slot: &'a mut MockSlot,
        ) -> Result<ExportedFrame<'a>, FrameExportError> {
            Ok(ExportedFrame {
                width: 1,
                height: 1,
                row_stride: 4,
                data: &slot.data,
            })
        }
        fn destroy_slot(&mut self, slot: MockSlot) -> Result<(), FrameExportError> {
            self.log
                .borrow_mut()
                .push(format!("destroy {}", slot.index));
            Ok(())
        }
    }

    fn descriptor() -> SinkDescriptor {
        SinkDescriptor {
            width: 1.0,
            height: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn queue_lifecycle_matches_the_reference() {
        assert!(
            FrameExportQueue::new(
                Mock::default(),
                FrameExportOptions {
                    slots: 1,
                    on_error: None
                }
            )
            .is_err()
        );
        let mut queue = FrameExportQueue::new(
            Mock {
                delay: 1,
                ..Default::default()
            },
            FrameExportOptions {
                slots: 2,
                on_error: None,
            },
        )
        .unwrap();
        let frames = Rc::new(RefCell::new(Vec::new()));
        let cb = |frames: &Rc<RefCell<Vec<(u8, f64)>>>| -> FrameCallback {
            let frames = frames.clone();
            Box::new(move |frame, t| {
                frames.borrow_mut().push((frame.data[0], t));
                Ok(())
            })
        };
        // Not configured: dropped.
        assert!(!queue.enqueue(&(), "a", 1.0, cb(&frames)));
        assert_eq!(queue.stats.dropped, 1);
        queue.configure(&descriptor()).unwrap();
        assert!(queue.available());
        assert!(queue.enqueue(&(), "a", 1.0, cb(&frames)));
        assert!(queue.enqueue(&(), "b", 2.0, cb(&frames)));
        assert!(!queue.available());
        assert!(!queue.enqueue(&(), "c", 3.0, cb(&frames)));
        assert_eq!(queue.stats.dropped, 2);
        queue.poll();
        assert!(frames.borrow().is_empty());
        queue.poll();
        assert_eq!(*frames.borrow(), vec![(0, 1.0), (1, 2.0)]);
        assert_eq!(queue.stats.completed, 2);
        assert!(queue.enqueue(&(), "d", 4.0, cb(&frames)));
        queue.close(false).unwrap();
        assert_eq!(queue.stats.dropped, 3);
        assert!(!queue.enqueue(&(), "e", 5.0, cb(&frames)));
    }

    #[test]
    fn begin_failures_are_reported() {
        let errors = Rc::new(RefCell::new(Vec::new()));
        let sink = errors.clone();
        let mut queue = FrameExportQueue::new(
            Mock {
                fail_begin: true,
                ..Default::default()
            },
            FrameExportOptions {
                slots: 3,
                on_error: Some(Box::new(move |e| sink.borrow_mut().push(e.0.clone()))),
            },
        )
        .unwrap();
        queue.configure(&descriptor()).unwrap();
        assert!(!queue.enqueue(&(), "a", 0.0, Box::new(|_, _| Ok(()))));
        assert_eq!(queue.stats.failed, 1);
        assert_eq!(*errors.borrow(), vec!["Error: begin failed".to_owned()]);
        assert!(queue.available());
    }

    #[test]
    fn descriptors_are_validated() {
        let mut d = descriptor();
        d.width = 1.5;
        assert!(validate_descriptor(&d).is_err());
        let mut d = descriptor();
        d.alpha_mode = "bogus".into();
        assert!(validate_descriptor(&d).is_err());
        let mut d = descriptor();
        d.width = 65.0;
        d.height = 2.0;
        let layout = validate_descriptor(&d).unwrap();
        assert_eq!(
            (
                layout.row_stride,
                layout.bytes_per_row,
                layout.buffer_size,
                layout.packed_size
            ),
            (260, 512, 1024, 520)
        );
    }
}
