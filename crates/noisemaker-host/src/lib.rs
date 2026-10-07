//! CPU-side host inputs of the Noisemaker engine: the pieces of the
//! reference that run on the CPU and produce texture data the GPU effects
//! consume.
//!
//! - [`obj`]: Wavefront OBJ meshes parsed and packed into the three
//!   256x256 RGBA32F textures `global_<meshId>_{positions,normals,uvs}`
//!   (`obj::parse_obj`, `obj::pack_mesh`, `obj::builtin_mesh`). The f32
//!   arrays are bit-identical to the reference's Float32Arrays.
//! - [`worm`], [`overlay`], [`canvas`] and [`raster`]: the CPU worm tracer
//!   and the asyncInit overlays of filter/fibers, filter/scratches and
//!   filter/strayHair (`<nodeId>_overlayTex`, rgba8unorm, the render size;
//!   `overlay::render_async_overlay`). The traced canvas operations are
//!   identical to the reference's, and [`raster`] rasterizes them as
//!   Chromium's software canvas does.
//! - [`text`]: the 2D-canvas text of filter/text (`textTex_step_<N>`,
//!   rgba8unorm, `text::demo_canvas_size`; `text::render_text_canvas`),
//!   with the fonts and glyph rasterization documented there.
//!
//! Outputs are plain data for the GPU runtime to upload: [`Rgba8Image`]
//! holds the exact bytes of the reference's rgba8unorm textures, row 0
//! first (the canvas rows already flipped and unpremultiplied as the
//! reference's `copyExternalImageToTexture({flipY: true})` stores them),
//! and [`obj::PackedMesh`] the RGBA32F texels, row 0 first. Write them to
//! the textures as they are. The [`js`] module holds the JavaScript number
//! semantics every port here follows.

pub mod canvas;
pub mod js;
pub mod obj;
pub mod overlay;
pub mod raster;
pub mod text;
pub mod worm;

/// An RGBA8 image as the reference's rgba8unorm host textures hold it:
/// `width * height * 4` bytes, straight (non-premultiplied) alpha, row 0
/// first — texture row 0, the row a WebGPU readback returns first.
#[derive(Clone, PartialEq, Eq)]
pub struct Rgba8Image {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel bytes, RGBA, row-major, row 0 first.
    pub data: Vec<u8>,
}

impl std::fmt::Debug for Rgba8Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rgba8Image")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bytes", &self.data.len())
            .finish()
    }
}

impl Rgba8Image {
    /// A transparent image.
    pub fn new(width: u32, height: u32) -> Rgba8Image {
        Rgba8Image {
            width,
            height,
            data: vec![0; width as usize * height as usize * 4],
        }
    }

    /// The pixel at (x, y), row 0 first.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = (y as usize * self.width as usize + x as usize) * 4;
        [
            self.data[i],
            self.data[i + 1],
            self.data[i + 2],
            self.data[i + 3],
        ]
    }
}

/// The reference's upload of a 2D canvas: `updateTextureFromSource(id,
/// canvas, { flipY: true })`, i.e. WebGPU `copyExternalImageToTexture({
/// source: canvas, flipY: true }, { texture })` into an rgba8unorm texture
/// with the default `premultipliedAlpha: false`.
///
/// `premultiplied` is the canvas backing store (premultiplied RGBA8, row 0
/// at the top of the canvas). The texture receives the canvas rows bottom
/// to top (flipY: texture row 0 is the canvas's bottom row) with alpha
/// divided out as [`unpremultiply_texel`] does.
pub fn upload_canvas_rgba8(premultiplied: &[u8], width: u32, height: u32) -> Rgba8Image {
    let (w, h) = (width as usize, height as usize);
    assert_eq!(premultiplied.len(), w * h * 4, "canvas size");
    let mut out = Rgba8Image::new(width, height);
    for row in 0..h {
        let src = &premultiplied[(h - 1 - row) * w * 4..(h - row) * w * 4];
        let dst = &mut out.data[row * w * 4..(row + 1) * w * 4];
        for (s, d) in src
            .as_chunks::<4>()
            .0
            .iter()
            .zip(dst.as_chunks_mut::<4>().0)
        {
            *d = unpremultiply_texel(*s);
        }
    }
    out
}

/// One texel of [`upload_canvas_rgba8`]: premultiplied RGBA8 to straight
/// RGBA8 as Chromium's copy computes it on the GPU: each channel read as
/// c / 255 (f32), multiplied by the f32 reciprocal of a / 255, and stored
/// to unorm8 rounded to nearest. Fitted to the reference's uploaded
/// overlays: the formula reproduces every (c, a) pair the canvas produced
/// (21072 pairs in the 256x256 fibers overlay); dividing instead of
/// multiplying by the reciprocal, or converting with c * (1 / 255.0f),
/// mismatches 100 to 200 of them.
pub fn unpremultiply_texel(p: [u8; 4]) -> [u8; 4] {
    let a = p[3];
    if a == 0 {
        return [p[0], p[1], p[2], 0];
    }
    let reciprocal = 1.0f32 / (a as f32 / 255.0);
    let unorm = |c: u8| {
        let v = (c as f32 / 255.0) * reciprocal;
        // v * 255 is exact in f64; its only possible tie (127.5) rounds up
        // under every rounding rule.
        ((v as f64).clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8
    };
    [unorm(p[0]), unorm(p[1]), unorm(p[2]), a]
}
