//! Presentation: what the reference shows on a canvas.
//!
//! The reference presents the render surface's frame texture with
//! `WebGPUBackend.present(textureId)`: a full-screen triangle samples the
//! texture into the canvas's current texture (the preferred canvas format,
//! `bgra8unorm` where Chromium runs it), cleared to transparent black first,
//! through the `nearest` sampler when the texture and the canvas have the same
//! size and the `default` (linear) sampler when the canvas scales it.
//!
//! The blit maps texture row 0 to the bottom of the canvas, so a canvas shows
//! the texture upside down relative to its row order: `readPixels` (and the
//! parity protocol's PNGs) list texture row 0 first, the canvas shows it last.
//! [`Presenter`] is the blit itself, for any wgpu target (a window surface, an
//! offscreen texture); [`Orientation`] and [`PixelData::flip_rows`] give
//! read-back pixels the orientation a same-size canvas shows.

use std::collections::HashMap;

use crate::backend::{GpuDevice, PixelData, WebGpuBackend};
use crate::error::RenderError;

/// The blit program of `getBlitPipeline()`, as the reference writes it.
pub const BLIT_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    // Standard UVs - internal blits now handle Y-flip
    var uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(0.0, 2.0)
    );
    var output: VertexOutput;
    output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
    output.uv = uv[vertexIndex];
    return output;
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(srcTex, srcSampler, input.uv);
}
"#;

/// The row order of read-back pixels.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Orientation {
    /// Texture row 0 first: the order of `readPixels` and of the parity
    /// protocol's PNGs (an image a canvas shows upside down).
    #[default]
    Texture,
    /// The rows as `present()` shows the texture on a canvas of its size:
    /// the last texture row first.
    Presented,
}

impl PixelData {
    /// The same pixels with the row order reversed.
    pub fn flip_rows(&self) -> PixelData {
        let row = self.width as usize * 4;
        let mut data = Vec::with_capacity(self.data.len());
        if row > 0 {
            for chunk in self.data.chunks_exact(row).rev() {
                data.extend_from_slice(chunk);
            }
        }
        PixelData {
            width: self.width,
            height: self.height,
            data,
        }
    }

    /// Read-back pixels (texture row order) in `orientation`. With
    /// [`Orientation::Presented`] these are the pixels `present()` draws on a
    /// canvas of the texture's size: the nearest-sampled blit of an rgba8unorm
    /// texture copies each texel to the mirrored row.
    pub fn oriented(self, orientation: Orientation) -> PixelData {
        match orientation {
            Orientation::Texture => self,
            Orientation::Presented => self.flip_rows(),
        }
    }
}

/// The reference's canvas blit (`present()`, `getBlitPipeline()`,
/// `createBlitBindGroup()`) for wgpu targets.
pub struct Presenter {
    device: wgpu::Device,
    queue: wgpu::Queue,
    module: wgpu::ShaderModule,
    /// The blit pipeline per target format (the reference builds one, for the
    /// canvas format).
    pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
}

impl Presenter {
    /// A presenter on `device` (the device the pipelines render on).
    pub fn new(device: &GpuDevice) -> Presenter {
        let module = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("present blit"),
                source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
            });
        Presenter {
            device: device.device.clone(),
            queue: device.queue.clone(),
            module,
            pipelines: HashMap::new(),
        }
    }

    /// `getBlitPipeline()` for targets of `format` (`layout: 'auto'`).
    fn pipeline(&mut self, format: wgpu::TextureFormat) -> &wgpu::RenderPipeline {
        let (device, module) = (&self.device, &self.module);
        self.pipelines.entry(format).or_insert_with(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("present blit"),
                layout: None,
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        })
    }

    /// `present(textureId)` onto `target`, a view of a `width x height`
    /// texture of `format` (a surface's current texture, or any render
    /// attachment): clear it to transparent black and draw the blit. Returns
    /// `false`, drawing nothing, when the backend has no texture `texture_id`
    /// (the reference returns silently) or has not been initialized.
    pub fn present(
        &mut self,
        backend: &WebGpuBackend,
        texture_id: &str,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> bool {
        let Some(tex) = backend.textures.get(texture_id) else {
            return false;
        };
        let exact_pixels = tex.width == f64::from(width) && tex.height == f64::from(height);
        // Interpolated UVs can drift from texel centers: a same-size copy
        // samples the nearest texel, a scaled one filters.
        let Some(sampler) = backend.sampler_for(if exact_pixels { "nearest" } else { "default" })
        else {
            return false;
        };
        let pipeline = self.pipeline(format).clone();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("present blit"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("present"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("present"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
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
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        true
    }

    /// What `present(textureId)` draws on a `width x height` rgba8unorm canvas,
    /// read back (canvas row 0, the top row, first). `None` when
    /// [`Presenter::present`] would draw nothing.
    pub fn capture(
        &mut self,
        backend: &WebGpuBackend,
        texture_id: &str,
        width: u32,
        height: u32,
    ) -> Result<Option<PixelData>, RenderError> {
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let target = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("present capture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        if !self.present(backend, texture_id, &view, format, width, height) {
            return Ok(None);
        }
        let bytes_per_row = (width * 4).div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("present capture"),
            size: u64::from(bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("present capture"),
            });
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| RenderError::Js(format!("Error: present capture: {e}")))?;
        rx.recv()
            .map_err(|e| RenderError::Js(format!("Error: present capture: {e}")))?
            .map_err(|e| RenderError::Js(format!("Error: present capture: {e}")))?;
        let row = width as usize * 4;
        let mut data = Vec::with_capacity(row * height as usize);
        {
            let mapped = slice
                .get_mapped_range()
                .map_err(|e| RenderError::Js(format!("Error: present capture: {e:?}")))?;
            for y in 0..height as usize {
                let start = y * bytes_per_row as usize;
                data.extend_from_slice(&mapped[start..start + row]);
            }
        }
        buffer.unmap();
        target.destroy();
        Ok(Some(PixelData {
            width,
            height,
            data,
        }))
    }
}
