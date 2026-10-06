//! Texture management of the WebGPU backend.

use std::cell::RefCell;
use std::rc::Rc;

use half::f16;
use noisemaker_dsl::js::math_round;
use noisemaker_dsl::{Object, Value};

use super::{WebGpuBackend, gpu_format_of, resolve_format, resolve_usage};
use crate::error::RenderError;
use crate::jsv::{enforce_range_u32, to_number};

/// One `this.textures` record. Optional members mirror the record shapes of the
/// different creation paths (`createTexture`, `createTexture3D`,
/// `createCubeTexture`, `uploadMeshData`, `uploadDataTexture`,
/// `updateTextureFromSource`): the reference reads members such as `gpuFormat`,
/// `mipmaps` or `isExternal` and falls back when a path did not set them.
pub struct TextureRecord {
    pub handle: wgpu::Texture,
    pub view: wgpu::TextureView,
    /// `renderView`: the level-0 single-mip view (`createTexture` only).
    pub render_view: Option<wgpu::TextureView>,
    /// `mipViews` (`createTexture` only).
    pub mip_views: Option<Vec<wgpu::TextureView>>,
    /// `width`/`height` as the creating call received them.
    pub width: f64,
    pub height: f64,
    pub depth: Option<f64>,
    /// `format`: the spec's format (short or WebGPU name), possibly undefined.
    pub format: Value,
    /// `gpuFormat`, when the creating path recorded it.
    pub gpu_format: Option<String>,
    pub usage: Option<wgpu::TextureUsages>,
    pub mipmaps: Option<bool>,
    pub mip_levels: Option<u32>,
    pub persistent: Option<bool>,
    pub is_3d: bool,
    pub filter: Value,
    pub cube: bool,
    pub is_external: bool,
    /// `mipBindGroups`, cached by `generateMipmaps`.
    pub mip_bind_groups: RefCell<Vec<Option<wgpu::BindGroup>>>,
    /// The actual texture format.
    pub texture_format: wgpu::TextureFormat,
}

impl TextureRecord {
    /// `tex.renderView || tex.view`.
    pub fn render_or_view(&self) -> &wgpu::TextureView {
        self.render_view.as_ref().unwrap_or(&self.view)
    }

    /// `tex.width && tex.height` (both truthy).
    pub fn has_size(&self) -> bool {
        self.width != 0.0 && !self.width.is_nan() && self.height != 0.0 && !self.height.is_nan()
    }

    /// `tex.mipLevels > 1`.
    pub fn has_mip_chain(&self) -> bool {
        self.mip_levels.is_some_and(|l| l > 1)
    }

    /// `tex.gpuFormat || this.resolveFormat(tex.format || 'rgba16float')`.
    pub fn resolved_gpu_format(&self) -> Value {
        match &self.gpu_format {
            Some(f) if !f.is_empty() => Value::from(f.as_str()),
            _ => {
                if self.format.is_truthy() {
                    resolve_format(&self.format)
                } else {
                    resolve_format(&Value::from("rgba16float"))
                }
            }
        }
    }
}

/// `readPixels` output: RGBA8 rows, top-down.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PixelData {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// `float16ToFloat32`.
pub fn float16_to_float32(h: u16) -> f64 {
    let sign = (h >> 15) & 0x1;
    let exponent = (h >> 10) & 0x1f;
    let mantissa = h & 0x3ff;
    let s = if sign != 0 { -1.0 } else { 1.0 };
    if exponent == 0 {
        if mantissa == 0 {
            return if sign != 0 { -0.0 } else { 0.0 };
        }
        return s * (mantissa as f64 / 1024.0) * 2f64.powi(-14);
    }
    if exponent == 31 {
        if mantissa == 0 {
            return s * f64::INFINITY;
        }
        return f64::NAN;
    }
    s * (1.0 + mantissa as f64 / 1024.0) * 2f64.powi(exponent as i32 - 15)
}

/// `Math.max(0, Math.min(255, Math.round(f * 255)))` stored into a `Uint8Array`
/// (NaN stores 0).
pub fn unit_to_byte(f: f64) -> u8 {
    let v = math_round(f * 255.0);
    let v = if v.is_nan() {
        f64::NAN
    } else {
        v.clamp(0.0, 255.0)
    };
    if v.is_nan() { 0 } else { v as u8 }
}

/// `mipLevelCount(width, height)`.
fn mip_level_count(width: f64, height: f64) -> u32 {
    let max_dim = 1f64.max(width.floor()).max(height.floor());
    if max_dim.is_nan() {
        return 1;
    }
    let max_dim = max_dim.min(u32::MAX as f64) as u32;
    (max_dim.ilog2() + 1).max(1)
}

/// `mipLevelSize(dim, level)`.
fn mip_level_size(dim: f64, level: u32) -> f32 {
    1f64.max((dim / 2f64.powi(level as i32)).floor()) as f32
}

fn bytes_per_pixel_for_upload(format: wgpu::TextureFormat) -> Option<u32> {
    use wgpu::TextureFormat as F;
    Some(match format {
        F::R8Unorm => 1,
        F::Rg8Unorm | F::R16Float => 2,
        F::Rgba8Unorm
        | F::Rgba8UnormSrgb
        | F::Bgra8Unorm
        | F::Bgra8UnormSrgb
        | F::R32Float
        | F::Rg16Float
        | F::Rgb10a2Unorm => 4,
        F::Rgba16Float | F::Rg32Float => 8,
        F::Rgba32Float => 16,
        _ => return None,
    })
}

/// Convert one RGBA8 pixel into a destination format the way
/// `copyExternalImageToTexture` does (unorm8 → `v / 255` → the destination
/// encoding).
fn encode_rgba8_pixel(format: wgpu::TextureFormat, px: &[u8], out: &mut Vec<u8>) {
    use wgpu::TextureFormat as F;
    let unit = |c: u8| c as f32 / 255.0;
    let h = |c: u8| f16::from_f32(unit(c)).to_le_bytes();
    match format {
        F::Rgba8Unorm | F::Rgba8UnormSrgb => out.extend_from_slice(&px[..4]),
        F::Bgra8Unorm | F::Bgra8UnormSrgb => out.extend_from_slice(&[px[2], px[1], px[0], px[3]]),
        F::R8Unorm => out.push(px[0]),
        F::Rg8Unorm => out.extend_from_slice(&px[..2]),
        F::R16Float => out.extend_from_slice(&h(px[0])),
        F::Rg16Float => {
            out.extend_from_slice(&h(px[0]));
            out.extend_from_slice(&h(px[1]));
        }
        F::Rgba16Float => {
            for &c in &px[..4] {
                out.extend_from_slice(&h(c));
            }
        }
        F::R32Float => out.extend_from_slice(&unit(px[0]).to_le_bytes()),
        F::Rg32Float => {
            out.extend_from_slice(&unit(px[0]).to_le_bytes());
            out.extend_from_slice(&unit(px[1]).to_le_bytes());
        }
        F::Rgba32Float => {
            for &c in &px[..4] {
                out.extend_from_slice(&unit(c).to_le_bytes());
            }
        }
        F::Rgb10a2Unorm => {
            let ten = |c: u8| ((c as f32 / 255.0) * 1023.0).round() as u32;
            let two = ((px[3] as f32 / 255.0) * 3.0).round() as u32;
            let v = ten(px[0]) | (ten(px[1]) << 10) | (ten(px[2]) << 20) | (two << 30);
            out.extend_from_slice(&v.to_le_bytes());
        }
        _ => unreachable!("checked by bytes_per_pixel_for_upload"),
    }
}

impl WebGpuBackend {
    fn texture_size(spec: &Object, depth: u32) -> Result<wgpu::Extent3d, RenderError> {
        let width = enforce_range_u32(spec.get_or_undefined("width")).map_err(|e| {
            RenderError::type_error(format!(
                "Failed to read the 'width' property from 'GPUExtent3DDict': {e}"
            ))
        })?;
        let height = enforce_range_u32(spec.get_or_undefined("height")).map_err(|e| {
            RenderError::type_error(format!(
                "Failed to execute 'createTexture' on 'GPUDevice': Failed to read the 'size' property from 'GPUTextureDescriptor': Failed to read the 'height' property from 'GPUExtent3DDict': {e}"
            ))
        })?;
        Ok(wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: depth,
        })
    }

    /// `createTexture(id, spec)`: a 2D texture, with a full mip chain when
    /// `spec.mipmaps` is truthy.
    pub fn create_texture(&mut self, id: &str, spec: &Object) -> Result<(), RenderError> {
        let resolved = resolve_format(spec.get_or_undefined("format"));
        let usage_value = spec.get_or_undefined("usage");
        let usage = if usage_value.is_truthy() {
            resolve_usage(usage_value)
        } else {
            resolve_usage(&Value::from_json(r#"["render","sample","copySrc","copyDst"]"#).unwrap())
        };
        let width = to_number(spec.get_or_undefined("width"));
        let height = to_number(spec.get_or_undefined("height"));
        let mip_levels = if spec.get_or_undefined("mipmaps").is_truthy() {
            mip_level_count(width, height)
        } else {
            1
        };
        let format = gpu_format_of(&resolved)?;
        let size = Self::texture_size(spec, 1)?;
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(id),
            size,
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let mip_views: Vec<wgpu::TextureView> = (0..mip_levels)
            .map(|level| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let record = TextureRecord {
            handle: texture,
            view,
            render_view: mip_views.first().cloned(),
            mip_views: Some(mip_views),
            width,
            height,
            depth: None,
            format: spec.get_or_undefined("format").clone(),
            gpu_format: resolved.as_str().map(str::to_owned),
            usage: Some(usage),
            mipmaps: Some(mip_levels > 1),
            mip_levels: Some(mip_levels),
            persistent: Some(spec.get_or_undefined("persistent").is_truthy()),
            is_3d: false,
            filter: Value::Undefined,
            cube: false,
            is_external: false,
            mip_bind_groups: RefCell::new(Vec::new()),
            texture_format: format,
        };
        self.textures.insert(id.to_owned(), Rc::new(record));
        Ok(())
    }

    /// `createTexture3D(id, spec)`.
    pub fn create_texture_3d(&mut self, id: &str, spec: &Object) -> Result<(), RenderError> {
        let resolved = resolve_format(spec.get_or_undefined("format"));
        let usage_value = spec.get_or_undefined("usage");
        let usage = if usage_value.is_truthy() {
            resolve_usage(usage_value)
        } else {
            resolve_usage(&Value::from_json(r#"["storage","sample","copySrc","copyDst"]"#).unwrap())
        };
        let format = gpu_format_of(&resolved)?;
        let depth = enforce_range_u32(spec.get_or_undefined("depth")).map_err(|e| {
            RenderError::type_error(format!(
                "Failed to read the 'depthOrArrayLayers' property from 'GPUExtent3DDict': {e}"
            ))
        })?;
        let size = Self::texture_size(spec, depth)?;
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(id),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        let record = TextureRecord {
            handle: texture,
            view,
            render_view: None,
            mip_views: None,
            width: to_number(spec.get_or_undefined("width")),
            height: to_number(spec.get_or_undefined("height")),
            depth: Some(to_number(spec.get_or_undefined("depth"))),
            format: spec.get_or_undefined("format").clone(),
            gpu_format: resolved.as_str().map(str::to_owned),
            usage: Some(usage),
            mipmaps: None,
            mip_levels: None,
            persistent: None,
            is_3d: true,
            filter: spec.get_or_undefined("filter").clone(),
            cube: false,
            is_external: false,
            mip_bind_groups: RefCell::new(Vec::new()),
            texture_format: format,
        };
        self.textures.insert(id.to_owned(), Rc::new(record));
        Ok(())
    }

    /// `destroyTexture(id)`.
    pub fn destroy_texture(&mut self, id: &str) {
        if let Some(record) = self.textures.shift_remove(id) {
            record.handle.destroy();
        }
    }

    /// `createCubeTexture(id, {size})`: a six-face rgba8unorm cube map.
    pub fn create_cube_texture(&mut self, id: &str, size: u32) -> Rc<TextureRecord> {
        let handle = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(id),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = handle.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let record = Rc::new(TextureRecord {
            handle,
            view,
            render_view: None,
            mip_views: None,
            width: size as f64,
            height: size as f64,
            depth: None,
            format: Value::Undefined,
            gpu_format: None,
            usage: None,
            mipmaps: None,
            mip_levels: None,
            persistent: None,
            is_3d: false,
            filter: Value::Undefined,
            cube: true,
            is_external: false,
            mip_bind_groups: RefCell::new(Vec::new()),
            texture_format: wgpu::TextureFormat::Rgba8Unorm,
        });
        self.textures.insert(id.to_owned(), record.clone());
        record
    }

    /// `uploadCubeFace(id, face, {width, height, data})`.
    pub fn upload_cube_face(&mut self, id: &str, face: u32, width: u32, height: u32, data: &[u8]) {
        let Some(tex) = self.textures.get(id).cloned() else {
            return;
        };
        let unaligned = width * 4;
        let bytes_per_row = unaligned.div_ceil(256) * 256;
        let mut upload = data.to_vec();
        if bytes_per_row != unaligned {
            upload = vec![0u8; (bytes_per_row * height) as usize];
            for y in 0..height as usize {
                let src = &data[y * unaligned as usize..(y + 1) * unaligned as usize];
                upload[y * bytes_per_row as usize..y * bytes_per_row as usize + unaligned as usize]
                    .copy_from_slice(src);
            }
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex.handle,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: face,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &upload,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn padded_rows_f32(data: &[f32], width: u32, height: u32, bytes_per_row: u32) -> Vec<u8> {
        let src_row = (width * 4) as usize;
        let dst_row = (bytes_per_row / 4) as usize;
        let mut padded = vec![0f32; dst_row * height as usize];
        for y in 0..height as usize {
            let end = ((y + 1) * src_row).min(data.len());
            let start = (y * src_row).min(end);
            padded[y * dst_row..y * dst_row + (end - start)].copy_from_slice(&data[start..end]);
        }
        padded.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// `uploadMeshData(meshId, positions, normals, uvs, width, height, vertexCount)`:
    /// three rgba32float textures `global_<meshId>_{positions,normals,uvs}`.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_mesh_data(
        &mut self,
        mesh_id: &str,
        position_data: &[f32],
        normal_data: &[f32],
        uv_data: &[f32],
        width: u32,
        height: u32,
        vertex_count: u32,
    ) -> (bool, u32) {
        let ids = [
            (format!("global_{mesh_id}_positions"), position_data),
            (format!("global_{mesh_id}_normals"), normal_data),
            (format!("global_{mesh_id}_uvs"), uv_data),
        ];
        for (id, data) in ids {
            let existing = self.textures.get(&id).cloned();
            let tex = match existing {
                Some(t) if t.width == width as f64 && t.height == height as f64 => t,
                other => {
                    if let Some(t) = other {
                        t.handle.destroy();
                    }
                    let handle = self.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some(&id),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba32Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::COPY_DST
                            | wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::STORAGE_BINDING,
                        view_formats: &[],
                    });
                    let view = handle.create_view(&Default::default());
                    let record = Rc::new(TextureRecord {
                        handle,
                        view,
                        render_view: None,
                        mip_views: None,
                        width: width as f64,
                        height: height as f64,
                        depth: None,
                        format: Value::from("rgba32f"),
                        gpu_format: Some("rgba32float".into()),
                        usage: None,
                        mipmaps: None,
                        mip_levels: None,
                        persistent: None,
                        is_3d: false,
                        filter: Value::Undefined,
                        cube: false,
                        is_external: false,
                        mip_bind_groups: RefCell::new(Vec::new()),
                        texture_format: wgpu::TextureFormat::Rgba32Float,
                    });
                    self.textures.insert(id.clone(), record.clone());
                    record
                }
            };
            let unaligned = width * 16;
            let bytes_per_row = unaligned.div_ceil(256) * 256;
            let upload = Self::padded_rows_f32(data, width, height, bytes_per_row);
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex.handle,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &upload,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
        (true, vertex_count)
    }

    /// `uploadDataTexture(id, data, width, height)`: an rgba32float data texture.
    pub fn upload_data_texture(&mut self, id: &str, data: &[f32], width: u32, height: u32) {
        let existing = self.textures.get(id).cloned();
        let tex = match existing {
            Some(t) if t.width == width as f64 && t.height == height as f64 => t,
            other => {
                if let Some(t) = other {
                    t.handle.destroy();
                }
                let handle = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(id),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view = handle.create_view(&Default::default());
                let record = Rc::new(TextureRecord {
                    handle,
                    view,
                    render_view: None,
                    mip_views: None,
                    width: width as f64,
                    height: height as f64,
                    depth: None,
                    // The reference records the WebGPU name here and no gpuFormat.
                    format: Value::from("rgba32float"),
                    gpu_format: None,
                    usage: None,
                    mipmaps: None,
                    mip_levels: None,
                    persistent: None,
                    is_3d: false,
                    filter: Value::Undefined,
                    cube: false,
                    is_external: false,
                    mip_bind_groups: RefCell::new(Vec::new()),
                    texture_format: wgpu::TextureFormat::Rgba32Float,
                });
                self.textures.insert(id.to_owned(), record.clone());
                record
            }
        };
        let bytes_per_row = width * 16;
        let aligned = bytes_per_row.div_ceil(256) * 256;
        let upload = Self::padded_rows_f32(data, width, height, aligned);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex.handle,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &upload,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// `updateTextureFromSource(id, source, {flipY})` for an RGBA8 image source
    /// (`width * height * 4` bytes, top row first): creates an rgba8unorm external
    /// texture when the id is new or its size changed, then copies the image into
    /// the texture the way `copyExternalImageToTexture` does. Returns the source
    /// dimensions (`{0, 0}` for an empty source).
    pub fn update_texture_from_rgba8(
        &mut self,
        id: &str,
        width: u32,
        height: u32,
        data: &[u8],
        flip_y: bool,
    ) -> Result<(u32, u32), RenderError> {
        if width == 0 || height == 0 {
            return Ok((0, 0));
        }
        if data.len() < (width * height * 4) as usize {
            return Err(RenderError::type_error(format!(
                "image source for {id} has {} bytes, expected {}",
                data.len(),
                width * height * 4
            )));
        }
        let existing = self.textures.get(id).cloned();
        let tex = match existing {
            Some(t) if t.width == width as f64 && t.height == height as f64 => t,
            other => {
                if let Some(t) = other {
                    t.handle.destroy();
                }
                let handle = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(id),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                let view = handle.create_view(&Default::default());
                let record = Rc::new(TextureRecord {
                    handle,
                    view,
                    render_view: None,
                    mip_views: None,
                    width: width as f64,
                    height: height as f64,
                    depth: None,
                    format: Value::from("rgba8"),
                    gpu_format: Some("rgba8unorm".into()),
                    usage: None,
                    mipmaps: None,
                    mip_levels: None,
                    persistent: None,
                    is_3d: false,
                    filter: Value::Undefined,
                    cube: false,
                    is_external: true,
                    mip_bind_groups: RefCell::new(Vec::new()),
                    texture_format: wgpu::TextureFormat::Rgba8Unorm,
                });
                self.textures.insert(id.to_owned(), record.clone());
                record
            }
        };
        let format = tex.texture_format;
        let Some(bpp) = bytes_per_pixel_for_upload(format) else {
            return Err(RenderError::type_error(format!(
                "copyExternalImageToTexture: unsupported destination format for {id}"
            )));
        };
        let unaligned = width * bpp;
        let bytes_per_row = unaligned.div_ceil(256) * 256;
        let mut upload = vec![0u8; (bytes_per_row * height) as usize];
        for y in 0..height as usize {
            let src_y = if flip_y { height as usize - 1 - y } else { y };
            let mut row = Vec::with_capacity(unaligned as usize);
            for x in 0..width as usize {
                let i = (src_y * width as usize + x) * 4;
                encode_rgba8_pixel(format, &data[i..i + 4], &mut row);
            }
            let start = y * bytes_per_row as usize;
            upload[start..start + row.len()].copy_from_slice(&row);
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex.handle,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &upload,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        Ok((width, height))
    }

    /// `copyTexture(srcId, dstId)`: a texture-to-texture copy submitted immediately,
    /// or a resampling copy when sizes differ or either side has a mip chain.
    pub fn copy_texture(&mut self, src_id: &str, dst_id: &str) {
        let (Some(src), Some(dst)) = (
            self.textures.get(src_id).cloned(),
            self.textures.get(dst_id).cloned(),
        ) else {
            return;
        };
        if src.width != dst.width
            || src.height != dst.height
            || src.has_mip_chain()
            || dst.has_mip_chain()
        {
            self.copy_texture_scaled(src_id, dst_id);
            return;
        }
        let size = wgpu::Extent3d {
            width: crate::jsv::enforce_range_u32(&Value::Number(src.width)).unwrap_or(0),
            height: crate::jsv::enforce_range_u32(&Value::Number(src.height)).unwrap_or(0),
            depth_or_array_layers: 1,
        };
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_texture(
            src.handle.as_image_copy(),
            dst.handle.as_image_copy(),
            size,
        );
        self.queue.submit([encoder.finish()]);
        self.collect_device_errors();
    }

    /// `getResamplePipeline(gpuFormat, entryPoint)`.
    fn get_resample_pipeline(
        &mut self,
        gpu_format: wgpu::TextureFormat,
        entry_point: &str,
    ) -> wgpu::RenderPipeline {
        let key = format!("{}|{entry_point}", super::format_name(gpu_format));
        if let Some(p) = self.resample_pipelines.get(&key) {
            return p.clone();
        }
        if self.resample_module.is_none() {
            self.resample_module = Some(self.builtin_shader("resample", RESAMPLE_WGSL));
        }
        let module = self.resample_module.clone().unwrap();
        // The resample shader reads no storage buffer and writes no depth: its
        // pipelines have no immediates.
        let pipeline = self.create_render_pipeline_from(
            "resample",
            (&module, "vs"),
            Some((&module, entry_point)),
            &[Some(wgpu::ColorTargetState {
                format: gpu_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            wgpu::PrimitiveState::default(),
            None,
        );
        self.resample_pipelines.insert(key, pipeline.clone());
        pipeline
    }

    /// `copyTextureScaled(srcId, dstId)`: resample level 0 of the source into the
    /// destination (nearest texel), submitted immediately.
    pub fn copy_texture_scaled(&mut self, src_id: &str, dst_id: &str) {
        let (Some(src), Some(dst)) = (
            self.textures.get(src_id).cloned(),
            self.textures.get(dst_id).cloned(),
        ) else {
            return;
        };
        if dst.is_3d {
            return;
        }
        let Ok(dst_format) = gpu_format_of(&dst.resolved_gpu_format()) else {
            return;
        };
        let pipeline = self.get_resample_pipeline(dst_format, "fsScale");
        let dims = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resample dims"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let values = [
            src.width as f32,
            src.height as f32,
            dst.width as f32,
            dst.height as f32,
        ];
        let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.queue.write_buffer(&dims, 0, &bytes);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src.render_or_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dims.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("copyTextureScaled"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dst.render_or_view(),
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
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        dims.destroy();
        self.collect_device_errors();
    }

    /// `generateMipmaps(ids)`: rebuild each mipmapped texture's chain from level 0
    /// with a 2x2 box filter, one immediately submitted render pass per level.
    pub fn generate_mipmaps(&mut self, ids: &[String]) {
        for id in ids {
            let Some(tex) = self.textures.get(id).cloned() else {
                continue;
            };
            if !tex.mipmaps.unwrap_or(false) || tex.is_3d || tex.mip_views.is_none() {
                continue;
            }
            let Ok(dst_format) = gpu_format_of(&tex.resolved_gpu_format()) else {
                continue;
            };
            let pipeline = self.get_resample_pipeline(dst_format, "fsMip");
            let levels = tex.mip_levels.unwrap_or(1);
            let mip_views = tex.mip_views.as_ref().unwrap();
            for level in 1..levels {
                let bind_group = {
                    let mut cache = tex.mip_bind_groups.borrow_mut();
                    if cache.len() < levels as usize {
                        cache.resize(levels as usize, None);
                    }
                    if cache[level as usize].is_none() {
                        cache[level as usize] =
                            Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: None,
                                layout: &pipeline.get_bind_group_layout(0),
                                entries: &[wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        &mip_views[level as usize - 1],
                                    ),
                                }],
                            }));
                    }
                    cache[level as usize].clone().unwrap()
                };
                let mut encoder = self.device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("generateMipmaps"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &mip_views[level as usize],
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
                    pass.set_pipeline(&pipeline);
                    pass.set_viewport(
                        0.0,
                        0.0,
                        mip_level_size(tex.width, level),
                        mip_level_size(tex.height, level),
                        0.0,
                        1.0,
                    );
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.draw(0..3, 0..1);
                }
                self.queue.submit([encoder.finish()]);
            }
        }
        self.collect_device_errors();
    }

    /// `clearTexture(id)`: clear the texture's default view to transparent black,
    /// submitted immediately.
    pub fn clear_texture(&mut self, id: &str) {
        let Some(tex) = self.textures.get(id).cloned() else {
            return;
        };
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clearTexture"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &tex.view,
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
        }
        self.queue.submit([encoder.finish()]);
        self.collect_device_errors();
    }

    /// `readPixels(textureId)`: copy the texture (first layer) into a staging buffer
    /// and convert it to RGBA8 the reference way (rgba16float and rgba32float:
    /// `round(v * 255)` clamped; anything else: the first `width * 4` bytes of each
    /// row).
    pub fn read_pixels(&mut self, texture_id: &str) -> Result<PixelData, RenderError> {
        let tex = self
            .textures
            .get(texture_id)
            .cloned()
            .ok_or_else(|| RenderError::Js(format!("Error: Texture {texture_id} not found")))?;
        let gpu_format = tex.gpu_format.as_deref();
        let bytes_per_pixel: u32 = match gpu_format {
            Some("rgba16float") => 8,
            Some("rgba32float") => 16,
            _ => 4,
        };
        let width =
            enforce_range_u32(&Value::Number(tex.width)).map_err(RenderError::type_error)?;
        let height =
            enforce_range_u32(&Value::Number(tex.height)).map_err(RenderError::type_error)?;
        let bytes_per_row = (width * bytes_per_pixel).div_ceil(256) * 256;
        let buffer_size = bytes_per_row as u64 * height as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readPixels staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            tex.handle.as_image_copy(),
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
        let mapped = rx
            .recv()
            .map_err(|e| RenderError::Js(format!("mapAsync: {e}")))?;
        mapped.map_err(|e| RenderError::Js(format!("OperationError: mapAsync failed: {e}")))?;
        let data = {
            let range = slice
                .get_mapped_range()
                .map_err(|e| RenderError::Js(format!("getMappedRange: {e:?}")))?;
            let src: &[u8] = &range;
            let mut out = vec![0u8; (width * height * 4) as usize];
            let w = width as usize;
            for row in 0..height as usize {
                let row_bytes = &src[row * bytes_per_row as usize..];
                match gpu_format {
                    Some("rgba16float") => {
                        for col in 0..w {
                            for c in 0..4 {
                                let i = (col * 4 + c) * 2;
                                let h = u16::from_le_bytes([row_bytes[i], row_bytes[i + 1]]);
                                out[(row * w + col) * 4 + c] = unit_to_byte(float16_to_float32(h));
                            }
                        }
                    }
                    Some("rgba32float") => {
                        for col in 0..w {
                            for c in 0..4 {
                                let i = (col * 4 + c) * 4;
                                let f = f32::from_le_bytes(row_bytes[i..i + 4].try_into().unwrap());
                                out[(row * w + col) * 4 + c] = unit_to_byte(f as f64);
                            }
                        }
                    }
                    _ => {
                        out[row * w * 4..(row + 1) * w * 4].copy_from_slice(&row_bytes[..w * 4]);
                    }
                }
            }
            out
        };
        staging.unmap();
        staging.destroy();
        self.collect_device_errors();
        Ok(PixelData {
            width,
            height,
            data,
        })
    }
}

/// `RESAMPLE_WGSL`: the shared fullscreen resample shader of `copyTextureScaled`
/// (`fsScale`) and `generateMipmaps` (`fsMip`).
pub const RESAMPLE_WGSL: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var points = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    var out: VsOut;
    out.pos = vec4f(points[vi], 0.0, 1.0);
    return out;
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> dims: vec4f;

struct FsParams {
    @builtin(position) frag: vec4f,
}

@fragment
fn fsMip(f: FsParams) -> @location(0) vec4f {
    let d = vec2u(f.frag.xy);
    let base = d * 2u;
    let a = textureLoad(src, base, 0u);
    let b = textureLoad(src, base + vec2u(1u, 0u), 0u);
    let c = textureLoad(src, base + vec2u(0u, 1u), 0u);
    let e = textureLoad(src, base + vec2u(1u, 1u), 0u);
    return (a + b + c + e) * 0.25;
}

@fragment
fn fsScale(f: FsParams) -> @location(0) vec4f {
    let d = vec2u(f.frag.xy);
    let ratio = vec2f(dims.x, dims.y) / vec2f(dims.z, dims.w);
    let scaled = vec2f(d) * ratio;
    let maxCoord = vec2u(max(dims.x - 1.0, 0.0), max(dims.y - 1.0, 0.0));
    let coord = min(vec2u(scaled), maxCoord);
    return textureLoad(src, coord, 0u);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_conversion_matches_reference() {
        assert_eq!(float16_to_float32(0x3c00), 1.0);
        assert_eq!(float16_to_float32(0x3800), 0.5);
        assert_eq!(float16_to_float32(0x0001), 2f64.powi(-24));
        assert!(float16_to_float32(0x7e00).is_nan());
        assert_eq!(float16_to_float32(0xfc00), f64::NEG_INFINITY);
        for bits in 0..=u16::MAX {
            let ours = float16_to_float32(bits);
            let theirs = f16::from_bits(bits).to_f64();
            assert!(
                ours == theirs || (ours.is_nan() && theirs.is_nan()),
                "{bits:#x}"
            );
        }
    }

    #[test]
    fn byte_conversion_rounds_half_up_and_clamps() {
        assert_eq!(unit_to_byte(0.5 / 255.0), 1);
        assert_eq!(unit_to_byte(-0.5), 0);
        assert_eq!(unit_to_byte(2.0), 255);
        assert_eq!(unit_to_byte(f64::NAN), 0);
        assert_eq!(unit_to_byte(f64::INFINITY), 255);
    }

    #[test]
    fn mip_counts() {
        assert_eq!(mip_level_count(256.0, 256.0), 9);
        assert_eq!(mip_level_count(255.0, 1.0), 8);
        assert_eq!(mip_level_size(256.0, 3), 32.0);
        assert_eq!(mip_level_size(5.0, 4), 1.0);
    }
}
