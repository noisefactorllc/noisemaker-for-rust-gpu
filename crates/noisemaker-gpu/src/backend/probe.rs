//! Exact texture probes: raw float read-back of backend textures and
//! mid-frame snapshots, for state-divergence diagnosis against the reference
//! (`parity/batch-golden.mjs --dump-texture`, `nm-render --dump-texture`).
//!
//! Both sides decode a texture's texels from its own format on the CPU into
//! RGBA float32 the same way: float formats exactly (half floats widened),
//! 8-bit unorm channels as `byte / 255` (rounded once to float32), missing
//! channels as 0 (alpha 1). The bytes leave as raw little-endian float32,
//! never as text.

use half::f16;
use noisemaker_dsl::Value;

use super::WebGpuBackend;
use crate::error::RenderError;
use crate::jsv::enforce_range_u32;

/// A texture read back as RGBA float32, row 0 first.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatPixels {
    pub width: u32,
    pub height: u32,
    /// The texture's format (WebGPU name).
    pub format: &'static str,
    /// `width * height * 4` values.
    pub data: Vec<f32>,
}

impl FloatPixels {
    /// The values as little-endian float32 bytes.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.data.len() * 4);
        for v in &self.data {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
}

/// Bytes per texel and channel layout of the formats a probe decodes.
fn texel_layout(format: wgpu::TextureFormat) -> Option<(u32, Channel, usize)> {
    use wgpu::TextureFormat as F;
    Some(match format {
        F::R8Unorm => (1, Channel::Unorm8, 1),
        F::Rg8Unorm => (2, Channel::Unorm8, 2),
        F::Rgba8Unorm | F::Rgba8UnormSrgb => (4, Channel::Unorm8, 4),
        F::Bgra8Unorm | F::Bgra8UnormSrgb => (4, Channel::Bgra8, 4),
        F::R16Float => (2, Channel::Half, 1),
        F::Rg16Float => (4, Channel::Half, 2),
        F::Rgba16Float => (8, Channel::Half, 4),
        F::R32Float => (4, Channel::Float, 1),
        F::Rg32Float => (8, Channel::Float, 2),
        F::Rgba32Float => (16, Channel::Float, 4),
        F::R32Uint => (4, Channel::Uint32, 1),
        F::R32Sint => (4, Channel::Sint32, 1),
        _ => return None,
    })
}

#[derive(Clone, Copy)]
enum Channel {
    Unorm8,
    Bgra8,
    Half,
    Float,
    Uint32,
    Sint32,
}

/// Decode one row of texels into RGBA float32.
fn decode_row(row: &[u8], width: usize, layout: (u32, Channel, usize), out: &mut Vec<f32>) {
    let (bpp, channel, count) = layout;
    let bpp = bpp as usize;
    for x in 0..width {
        let texel = &row[x * bpp..(x + 1) * bpp];
        let mut rgba = [0.0f32, 0.0, 0.0, 1.0];
        for (c, slot) in rgba.iter_mut().enumerate().take(count) {
            *slot = match channel {
                Channel::Unorm8 => (texel[c] as f64 / 255.0) as f32,
                Channel::Bgra8 => {
                    let src = [2, 1, 0, 3][c];
                    (texel[src] as f64 / 255.0) as f32
                }
                Channel::Half => {
                    f16::from_bits(u16::from_le_bytes([texel[c * 2], texel[c * 2 + 1]])).to_f32()
                }
                Channel::Float => f32::from_le_bytes(texel[c * 4..c * 4 + 4].try_into().unwrap()),
                Channel::Uint32 => {
                    u32::from_le_bytes(texel[c * 4..c * 4 + 4].try_into().unwrap()) as f32
                }
                Channel::Sint32 => {
                    i32::from_le_bytes(texel[c * 4..c * 4 + 4].try_into().unwrap()) as f32
                }
            };
        }
        out.extend_from_slice(&rgba);
    }
}

/// A mid-frame copy of a texture, taken in the frame's command encoder.
pub struct TextureSnapshot {
    pub texture: wgpu::Texture,
    pub width: u32,
    pub height: u32,
}

impl WebGpuBackend {
    /// Read texture `texture_id` (mip level 0, first layer) back as RGBA
    /// float32. The texture needs COPY_SRC usage.
    pub fn read_texture_f32(&mut self, texture_id: &str) -> Result<FloatPixels, RenderError> {
        let tex = self
            .textures
            .get(texture_id)
            .cloned()
            .ok_or_else(|| RenderError::Js(format!("Error: Texture {texture_id} not found")))?;
        let width =
            enforce_range_u32(&Value::Number(tex.width)).map_err(RenderError::type_error)?;
        let height =
            enforce_range_u32(&Value::Number(tex.height)).map_err(RenderError::type_error)?;
        if !tex.handle.usage().contains(wgpu::TextureUsages::COPY_SRC) {
            return Err(RenderError::Js(format!(
                "Error: texture {texture_id} has no COPY_SRC usage"
            )));
        }
        self.read_handle_f32(&tex.handle, width, height)
    }

    /// Read a texture handle back as RGBA float32.
    pub fn read_handle_f32(
        &mut self,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Result<FloatPixels, RenderError> {
        let format = texture.format();
        let layout = texel_layout(format).ok_or_else(|| {
            RenderError::Js(format!(
                "Error: cannot probe a {} texture",
                super::format_name(format)
            ))
        })?;
        let bytes_per_row = (width * layout.0).div_ceil(256) * 256;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe staging"),
            size: bytes_per_row as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
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
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|e| RenderError::Js(format!("mapAsync: {e}")))?
            .map_err(|e| RenderError::Js(format!("OperationError: mapAsync failed: {e}")))?;
        let data = {
            let range = slice
                .get_mapped_range()
                .map_err(|e| RenderError::Js(format!("getMappedRange: {e:?}")))?;
            let mut out = Vec::with_capacity((width * height * 4) as usize);
            for row in 0..height as usize {
                let start = row * bytes_per_row as usize;
                decode_row(&range[start..], width as usize, layout, &mut out);
            }
            out
        };
        staging.unmap();
        staging.destroy();
        self.collect_device_errors();
        Ok(FloatPixels {
            width,
            height,
            format: super::format_name(format),
            data,
        })
    }

    /// Copy texture `texture_id` (mip level 0) into a new texture in the
    /// frame's command encoder, so the copy sees exactly the passes encoded
    /// before it (outside a frame the copy is submitted immediately). `None`
    /// when the texture does not exist or cannot be copied.
    pub fn snapshot_texture(&mut self, texture_id: &str) -> Option<TextureSnapshot> {
        let tex = self.textures.get(texture_id).cloned()?;
        if !tex.handle.usage().contains(wgpu::TextureUsages::COPY_SRC) || tex.is_3d || tex.cube {
            return None;
        }
        let width = tex.handle.width();
        let height = tex.handle.height();
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("probe snapshot"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: tex.handle.format(),
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let copy = |encoder: &mut wgpu::CommandEncoder| {
            encoder.copy_texture_to_texture(
                tex.handle.as_image_copy(),
                texture.as_image_copy(),
                size,
            );
        };
        match self.command_encoder.as_mut() {
            Some(encoder) => copy(encoder),
            None => {
                let mut encoder = self.device.create_command_encoder(&Default::default());
                copy(&mut encoder);
                self.queue.submit([encoder.finish()]);
            }
        }
        Some(TextureSnapshot {
            texture,
            width,
            height,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_decode_to_rgba_float() {
        let mut out = Vec::new();
        decode_row(
            &[0, 51, 255, 128],
            1,
            texel_layout(wgpu::TextureFormat::Rgba8Unorm).unwrap(),
            &mut out,
        );
        assert_eq!(
            out,
            vec![
                0.0,
                (51.0f64 / 255.0) as f32,
                1.0,
                (128.0f64 / 255.0) as f32
            ]
        );
        out.clear();
        decode_row(
            &[1, 2, 3, 4],
            1,
            texel_layout(wgpu::TextureFormat::Bgra8Unorm).unwrap(),
            &mut out,
        );
        assert_eq!(out[0], (3.0f64 / 255.0) as f32);
        assert_eq!(out[2], (1.0f64 / 255.0) as f32);
        out.clear();
        let half = f16::from_f32(0.5).to_bits().to_le_bytes();
        decode_row(
            &[half[0], half[1]],
            1,
            texel_layout(wgpu::TextureFormat::R16Float).unwrap(),
            &mut out,
        );
        assert_eq!(out, vec![0.5, 0.0, 0.0, 1.0]);
        out.clear();
        let mut texel = Vec::new();
        for v in [1.5f32, -2.0, 3.25, 0.125] {
            texel.extend_from_slice(&v.to_le_bytes());
        }
        decode_row(
            &texel,
            1,
            texel_layout(wgpu::TextureFormat::Rgba32Float).unwrap(),
            &mut out,
        );
        assert_eq!(out, vec![1.5, -2.0, 3.25, 0.125]);
    }
}
