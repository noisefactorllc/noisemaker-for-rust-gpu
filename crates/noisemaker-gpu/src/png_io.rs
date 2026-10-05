//! RGBA8 PNG reading and writing for goldens, candidates and host textures.

use std::io::BufWriter;
use std::path::Path;

/// An RGBA8 image, top row first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rgba8Image {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// Read a PNG as RGBA8 (gray, gray+alpha, RGB, palette and 16-bit images are
/// converted the way `PIL.Image.convert('RGBA')` converts them).
pub fn read_png_rgba8(path: &Path) -> Result<Rgba8Image, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut decoder = png::Decoder::new(std::io::BufReader::new(file));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| format!("{}: image too large", path.display()))?;
    let mut buf = vec![0u8; size];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let (width, height) = (info.width, info.height);
    let pixels = (width * height) as usize;
    let src = &buf[..info.buffer_size()];
    let data = match info.color_type {
        png::ColorType::Rgba => src.to_vec(),
        png::ColorType::Rgb => src
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        png::ColorType::GrayscaleAlpha => src
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[0], p[0], p[1]])
            .collect(),
        png::ColorType::Grayscale => src.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        png::ColorType::Indexed => {
            return Err(format!("{}: unexpanded palette image", path.display()));
        }
    };
    if data.len() != pixels * 4 {
        return Err(format!("{}: unexpected pixel data size", path.display()));
    }
    Ok(Rgba8Image {
        width,
        height,
        data,
    })
}

/// Write RGBA8 rows (top row first) as an 8-bit RGBA PNG.
pub fn write_png_rgba8(path: &Path, width: u32, height: u32, data: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    writer
        .write_image_data(data)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    writer
        .finish()
        .map_err(|e| format!("{}: {e}", path.display()))
}
