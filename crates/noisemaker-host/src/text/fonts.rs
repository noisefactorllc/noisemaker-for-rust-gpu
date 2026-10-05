//! Fonts for the canvas text: the bundled Nunito (the demo's default
//! family), fonts a host registers (the demo page's `@font-face` rules),
//! and the platform's installed fonts, resolved the way Chromium resolves
//! a canvas `font-family` list.
//!
//! # Resolution
//!
//! For each family of the list Blink looks for the family and moves on when
//! it is missing:
//!
//! - a registered family (an `@font-face` family; the bundled Nunito is one)
//!   shadows installed fonts of the same name;
//! - a generic family resolves to the family Chromium's default font
//!   preferences name for it on the platform (chrome/app/resources/
//!   locale_settings_{mac,win,linux}.grd): on macOS serif Times, sans-serif
//!   Helvetica, monospace Courier, cursive Apple Chancery, fantasy Papyrus
//!   (measured in Chromium 153: same widths and glyphs);
//!   on Windows Times New Roman, Arial, Consolas, Comic Sans MS, Impact; on
//!   Linux Times New Roman, Arial, Monospace, Comic Sans MS, Impact.
//!   system-ui is the platform UI font (macOS: SF, `.AppleSystemUIFont`).
//! - any other name matches an installed family name (case-insensitive),
//!   then a PostScript name.
//!
//! When no family of the list exists, Blink takes the standard family
//! (Times on macOS, Times New Roman elsewhere), each lookup also tries its
//! alias (Times / Times New Roman, Helvetica / Arial, Courier / Courier
//! New), and the last resort is Times, then Lucida Grande on macOS.
//!
//! Within a family the face is chosen by the CSS font matching algorithm
//! for the requested weight (400), style (normal) and stretch (100 %); a
//! variable face whose `wght` axis covers the weight is instanced at it.
//!
//! A character the chosen fonts do not map falls back to the next family
//! of the list, then to the platform's fallback list at weight 400: on
//! macOS PingFang SC, Lucida Grande, Hiragino Sans, Apple Symbols, ...
//! (Chromium 153 on macOS 26 draws kana, U+2605, U+2713 and U+2192 with
//! PingFang SC Regular, measured by rendering each candidate family in the
//! browser; CoreText's `CTFontCreateForString` on the primary font would
//! answer Lucida Grande and Hiragino Sans). PingFang's outlines are in
//! Apple's `hvgl` format, which no outline reader here supports, so those
//! characters take the next family that can draw them; elsewhere a fixed
//! list of common coverage fonts.

use super::css::{FontFamily, GenericFamily};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// The weight the demo's text requests (`font-weight: normal`).
pub const NORMAL_WEIGHT: f32 = 400.0;

/// Font bytes: embedded or loaded.
#[derive(Clone)]
pub enum FontBytes {
    /// Bytes embedded in the binary.
    Static(&'static [u8]),
    /// Bytes loaded at run time.
    Shared(Arc<[u8]>),
}

impl FontBytes {
    /// The bytes.
    pub fn as_slice(&self) -> &[u8] {
        match self {
            FontBytes::Static(b) => b,
            FontBytes::Shared(b) => b,
        }
    }
}

impl std::fmt::Debug for FontBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FontBytes({} bytes)", self.as_slice().len())
    }
}

/// Where a face's bytes come from.
#[derive(Clone, Debug)]
enum FaceSource {
    Bytes(FontBytes),
    File(PathBuf),
}

/// Style of a face.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceStyle {
    /// Upright.
    Normal,
    /// Italic.
    Italic,
    /// Oblique.
    Oblique,
}

/// Matching metadata of one face (a font file and a collection index).
#[derive(Clone, Debug)]
struct FaceRecord {
    source: FaceSource,
    index: u32,
    /// Family names, lower case (ASCII), for matching.
    families: Vec<String>,
    /// The first family name as written.
    display_family: String,
    post_script: String,
    weight: f32,
    style: FaceStyle,
    /// Width in percent (100 = normal).
    stretch: f32,
}

/// A face chosen for drawing: its bytes, collection index and the axis
/// values to instance a variable face at (empty for a static face).
#[derive(Clone, Debug)]
pub struct ResolvedFace {
    /// The font file bytes.
    pub bytes: FontBytes,
    /// The face index in a collection.
    pub index: u32,
    /// Variation axis values (tag, value), within the axes' ranges.
    pub variations: Vec<([u8; 4], f32)>,
    /// The font's family name (the first family name of the face, as
    /// written in the font).
    pub family: String,
    /// Whether the face is the platform UI font (`system-ui`), whose
    /// platform tracking applies.
    pub system_ui: bool,
    /// A label for diagnostics (family and PostScript name).
    pub label: String,
}

impl ResolvedFace {
    /// Parses the face (with its variations applied) for shaping.
    pub fn shaping_face(&self) -> Option<rustybuzz::Face<'_>> {
        let mut face = rustybuzz::Face::from_slice(self.bytes.as_slice(), self.index)?;
        if !self.variations.is_empty() {
            let variations: Vec<rustybuzz::Variation> = self
                .variations
                .iter()
                .map(|(tag, value)| rustybuzz::Variation {
                    tag: rustybuzz::ttf_parser::Tag::from_bytes(tag),
                    value: *value,
                })
                .collect();
            face.set_variations(&variations);
        }
        Some(face)
    }

    /// The face with `font-optical-sizing: auto` applied: a face with an
    /// `opsz` axis is instanced at the font size in CSS pixels (Blink), within
    /// the axis range.
    pub fn with_optical_size(mut self, size: f32) -> ResolvedFace {
        let Ok(face) = rustybuzz::ttf_parser::Face::parse(self.bytes.as_slice(), self.index) else {
            return self;
        };
        if let Some(axis) = face
            .variation_axes()
            .into_iter()
            .find(|a| a.tag == rustybuzz::ttf_parser::Tag::from_bytes(b"opsz"))
        {
            let value = size.clamp(axis.min_value, axis.max_value);
            match self.variations.iter_mut().find(|(t, _)| t == b"opsz") {
                Some(slot) => slot.1 = value,
                None => self.variations.push((*b"opsz", value)),
            }
        }
        self
    }

    /// The face with `font-variation-settings` applied: each setting for an
    /// axis the face has replaces that axis's value (clamped to its range);
    /// settings for other axes are ignored.
    pub fn with_variation_settings(mut self, settings: &[([u8; 4], f32)]) -> ResolvedFace {
        if settings.is_empty() {
            return self;
        }
        let Ok(face) = rustybuzz::ttf_parser::Face::parse(self.bytes.as_slice(), self.index) else {
            return self;
        };
        for (tag, value) in settings {
            let Some(axis) = face
                .variation_axes()
                .into_iter()
                .find(|a| a.tag == rustybuzz::ttf_parser::Tag::from_bytes(tag))
            else {
                continue;
            };
            let value = value.clamp(axis.min_value, axis.max_value);
            match self.variations.iter_mut().find(|(t, _)| t == tag) {
                Some(slot) => slot.1 = value,
                None => self.variations.push((*tag, value)),
            }
        }
        self
    }

    /// Whether the face has outlines this crate can draw (TrueType or CFF
    /// glyphs; not bitmap-only or Apple `hvgl` faces such as PingFang).
    pub fn is_drawable(&self) -> bool {
        rustybuzz::ttf_parser::Face::parse(self.bytes.as_slice(), self.index).is_ok_and(|f| {
            let t = f.tables();
            t.glyf.is_some() || t.cff.is_some() || t.cff2.is_some()
        })
    }

    /// Whether the face maps `c` to a glyph.
    pub fn has_char(&self, c: char) -> bool {
        rustybuzz::ttf_parser::Face::parse(self.bytes.as_slice(), self.index)
            .ok()
            .and_then(|f| f.glyph_index(c))
            .is_some()
    }
}

/// The bundled fonts: (family, catalog path).
const BUNDLED: &[(&str, &str)] = &[("Nunito", "share/fonts/Nunito/Nunito-VariableFont_wght.ttf")];

/// Generic families as Chromium's default preferences name them.
fn generic_preference(generic: GenericFamily) -> &'static [&'static str] {
    use GenericFamily::*;
    if cfg!(target_os = "macos") {
        match generic {
            Serif | Math => &["Times"],
            SansSerif => &["Helvetica"],
            Monospace => &["Courier"],
            Cursive => &["Apple Chancery"],
            Fantasy => &["Papyrus"],
            SystemUi => &[".AppleSystemUIFont", "System Font", ".SF NS"],
        }
    } else if cfg!(target_os = "windows") {
        match generic {
            Serif | Math => &["Times New Roman"],
            SansSerif => &["Arial"],
            Monospace => &["Consolas", "Courier New"],
            Cursive => &["Comic Sans MS"],
            Fantasy => &["Impact"],
            SystemUi => &["Segoe UI"],
        }
    } else {
        match generic {
            Serif | Math => &["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            SansSerif => &["Arial", "Liberation Sans", "DejaVu Sans"],
            Monospace => &["Monospace", "Liberation Mono", "DejaVu Sans Mono"],
            Cursive => &["Comic Sans MS"],
            Fantasy => &["Impact"],
            SystemUi => &["Ubuntu", "Cantarell", "DejaVu Sans"],
        }
    }
}

fn standard_family() -> &'static str {
    if cfg!(target_os = "macos") {
        "Times"
    } else {
        "Times New Roman"
    }
}

/// Blink `AlternateFamilyName`.
fn alternate_family(name: &str) -> Option<&'static str> {
    let lower = name.to_ascii_lowercase();
    Some(match lower.as_str() {
        "courier" => "Courier New",
        "courier new" if !cfg!(target_os = "windows") => "Courier",
        "times" => "Times New Roman",
        "times new roman" => "Times",
        "arial" => "Helvetica",
        "helvetica" => "Arial",
        _ => return None,
    })
}

/// Blink's last-resort families.
fn last_resort() -> &'static [&'static str] {
    if cfg!(target_os = "macos") {
        &["Times", "Lucida Grande"]
    } else if cfg!(target_os = "windows") {
        &[
            "Arial",
            "MS UI Gothic",
            "Microsoft Sans Serif",
            "Segoe UI",
            "Calibri",
            "Times New Roman",
            "Courier New",
        ]
    } else {
        &["Sans", "Arial", "DejaVu Sans", "Liberation Sans"]
    }
}

/// Platform fallback families for characters no font of the list maps, in
/// order. `han` selects the order for Han ideographs.
fn system_fallback(han: bool) -> &'static [&'static str] {
    if cfg!(target_os = "macos") {
        if han {
            &[
                "PingFang SC",
                "Hiragino Sans",
                "Hiragino Sans GB",
                "Arial Unicode MS",
                "Apple Symbols",
                "LastResort",
            ]
        } else {
            &[
                "PingFang SC",
                "Lucida Grande",
                "Hiragino Sans",
                "Apple Symbols",
                "Apple SD Gothic Neo",
                "Geeza Pro",
                "Arial Hebrew",
                "Thonburi",
                "Kohinoor Devanagari",
                "Menlo",
                "STIXGeneral",
                "Arial Unicode MS",
                "LastResort",
            ]
        }
    } else if cfg!(target_os = "windows") {
        &[
            "Segoe UI",
            "Segoe UI Symbol",
            "Microsoft YaHei",
            "Yu Gothic",
            "Malgun Gothic",
            "Arial Unicode MS",
            "Segoe UI Emoji",
        ]
    } else {
        &[
            "DejaVu Sans",
            "Noto Sans",
            "Noto Sans CJK SC",
            "Noto Sans Symbols",
            "Noto Sans Symbols2",
            "Droid Sans Fallback",
            "Arial Unicode MS",
        ]
    }
}

fn is_han(c: char) -> bool {
    matches!(c as u32,
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x3134F)
}

/// Directories scanned besides fontdb's system directories: on macOS the
/// reserved system fonts CoreText falls back to (PingFang).
fn extra_font_dirs() -> Vec<PathBuf> {
    if cfg!(target_os = "macos") {
        vec![PathBuf::from(
            "/System/Library/PrivateFrameworks/FontServices.framework/Resources/Reserved",
        )]
    } else {
        Vec::new()
    }
}

/// Loaded font files, shared by every [`TextFonts`].
fn file_cache() -> &'static Mutex<HashMap<PathBuf, Arc<[u8]>>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, Arc<[u8]>>>> = OnceLock::new();
    CACHE.get_or_init(Default::default)
}

fn load_file(path: &Path) -> Option<Arc<[u8]>> {
    let mut cache = file_cache().lock().ok()?;
    if let Some(bytes) = cache.get(path) {
        return Some(bytes.clone());
    }
    let bytes: Arc<[u8]> = std::fs::read(path).ok()?.into();
    cache.insert(path.to_path_buf(), bytes.clone());
    Some(bytes)
}

/// The installed fonts, discovered once per process.
fn system_records() -> &'static [FaceRecord] {
    static SYSTEM: OnceLock<Vec<FaceRecord>> = OnceLock::new();
    SYSTEM.get_or_init(|| {
        let mut db = fontdb::Database::new();
        db.load_system_fonts();
        for dir in extra_font_dirs() {
            if dir.is_dir() {
                db.load_fonts_dir(dir);
            }
        }
        let mut records: Vec<FaceRecord> = db
            .faces()
            .filter_map(|info| {
                let source = match &info.source {
                    fontdb::Source::File(path) => FaceSource::File(path.clone()),
                    fontdb::Source::SharedFile(path, _) => FaceSource::File(path.clone()),
                    fontdb::Source::Binary(_) => return None,
                };
                Some(FaceRecord {
                    source,
                    index: info.index,
                    families: info
                        .families
                        .iter()
                        .map(|(n, _)| n.to_ascii_lowercase())
                        .collect(),
                    display_family: info
                        .families
                        .first()
                        .map(|(n, _)| n.clone())
                        .unwrap_or_default(),
                    post_script: info.post_script_name.clone(),
                    weight: info.weight.0 as f32,
                    style: match info.style {
                        fontdb::Style::Normal => FaceStyle::Normal,
                        fontdb::Style::Italic => FaceStyle::Italic,
                        fontdb::Style::Oblique => FaceStyle::Oblique,
                    },
                    stretch: stretch_percent(info.stretch),
                })
            })
            .collect();
        // A deterministic order (fontdb follows directory listing order).
        records.sort_by(|a, b| {
            let key = |r: &FaceRecord| match &r.source {
                FaceSource::File(p) => p.clone(),
                FaceSource::Bytes(_) => PathBuf::new(),
            };
            key(a).cmp(&key(b)).then(a.index.cmp(&b.index))
        });
        records
    })
}

fn stretch_percent(stretch: fontdb::Stretch) -> f32 {
    use fontdb::Stretch::*;
    match stretch {
        UltraCondensed => 50.0,
        ExtraCondensed => 62.5,
        Condensed => 75.0,
        SemiCondensed => 87.5,
        Normal => 100.0,
        SemiExpanded => 112.5,
        Expanded => 125.0,
        ExtraExpanded => 150.0,
        UltraExpanded => 200.0,
    }
}

fn record_from_bytes(bytes: FontBytes, index: u32, family: Option<&str>) -> Option<FaceRecord> {
    let face = rustybuzz::ttf_parser::Face::parse(bytes.as_slice(), index).ok()?;
    let mut families: Vec<String> = Vec::new();
    let display_family = family.map(str::to_owned).unwrap_or_default();
    if let Some(f) = family {
        families.push(f.to_ascii_lowercase());
    } else {
        for id in [16u16, 1] {
            for name in face.names() {
                if name.name_id == id
                    && let Some(s) = name.to_string()
                {
                    let s = s.to_ascii_lowercase();
                    if !families.contains(&s) {
                        families.push(s);
                    }
                }
            }
        }
    }
    let post_script = face
        .names()
        .into_iter()
        .find(|n| n.name_id == 6)
        .and_then(|n| n.to_string())
        .unwrap_or_default();
    let weight = face.weight().to_number() as f32;
    let style = if face.is_italic() {
        FaceStyle::Italic
    } else if face.is_oblique() {
        FaceStyle::Oblique
    } else {
        FaceStyle::Normal
    };
    let display_family = if display_family.is_empty() {
        families.first().cloned().unwrap_or_default()
    } else {
        display_family
    };
    Some(FaceRecord {
        source: FaceSource::Bytes(bytes),
        index,
        families,
        display_family,
        post_script,
        weight,
        style,
        stretch: 100.0,
    })
}

/// The fonts canvas text can use.
#[derive(Clone)]
pub struct TextFonts {
    registered: Vec<FaceRecord>,
    use_system: bool,
}

impl Default for TextFonts {
    /// The bundled fonts and the platform's installed fonts.
    fn default() -> TextFonts {
        let mut fonts = TextFonts {
            registered: Vec::new(),
            use_system: true,
        };
        for (family, path) in BUNDLED {
            if let Some(bytes) = noisemaker_effects::share_file(path) {
                fonts.register_static(family, bytes);
            }
        }
        fonts
    }
}

impl TextFonts {
    /// Only the bundled fonts: no installed font is consulted, so results
    /// do not depend on the machine.
    pub fn bundled_only() -> TextFonts {
        TextFonts {
            use_system: false,
            ..TextFonts::default()
        }
    }

    /// Registers font bytes under `family`, as an `@font-face` rule does:
    /// every face of the file (each collection member) joins the family and
    /// shadows installed fonts of that name.
    pub fn register(&mut self, family: &str, bytes: Vec<u8>) {
        self.register_bytes(family, FontBytes::Shared(bytes.into()));
    }

    fn register_static(&mut self, family: &str, bytes: &'static [u8]) {
        self.register_bytes(family, FontBytes::Static(bytes));
    }

    fn register_bytes(&mut self, family: &str, bytes: FontBytes) {
        let count = rustybuzz::ttf_parser::fonts_in_collection(bytes.as_slice()).unwrap_or(1);
        for index in 0..count {
            if let Some(record) = record_from_bytes(bytes.clone(), index, Some(family)) {
                self.registered.push(record);
            }
        }
    }

    fn system(&self) -> &[FaceRecord] {
        if self.use_system {
            system_records()
        } else {
            &[]
        }
    }

    /// The face for `family` (a registered family first, then an installed
    /// family, then a PostScript name), or None when it does not exist.
    fn lookup_named(&self, name: &str) -> Option<ResolvedFace> {
        let lower = name.to_ascii_lowercase();
        let registered: Vec<&FaceRecord> = self
            .registered
            .iter()
            .filter(|r| r.families.contains(&lower))
            .collect();
        if !registered.is_empty() {
            return best_face(&registered);
        }
        let system: Vec<&FaceRecord> = self
            .system()
            .iter()
            .filter(|r| r.families.contains(&lower))
            .collect();
        if !system.is_empty() {
            return best_face(&system);
        }
        let by_post_script: Vec<&FaceRecord> = self
            .system()
            .iter()
            .filter(|r| r.post_script == name)
            .collect();
        by_post_script.first().and_then(|r| resolve_record(r))
    }

    fn lookup_with_alternate(&self, name: &str) -> Option<ResolvedFace> {
        self.lookup_named(name)
            .or_else(|| alternate_family(name).and_then(|alt| self.lookup_named(alt)))
    }

    fn lookup_generic(&self, generic: GenericFamily) -> Option<ResolvedFace> {
        let face = generic_preference(generic)
            .iter()
            .find_map(|name| self.lookup_with_alternate(name))?;
        Some(ResolvedFace {
            system_ui: generic == GenericFamily::SystemUi,
            ..face
        })
    }

    fn lookup(&self, family: &FontFamily) -> Option<ResolvedFace> {
        match family {
            FontFamily::Named(name) => self.lookup_with_alternate(name),
            FontFamily::Generic(generic) => self.lookup_generic(*generic),
        }
    }

    /// The faces of a family list in order, each family that exists once:
    /// the primary font first. When none exists, the standard family and
    /// the last resort.
    pub fn resolve(&self, families: &[FontFamily]) -> Vec<ResolvedFace> {
        let mut faces: Vec<ResolvedFace> = families.iter().filter_map(|f| self.lookup(f)).collect();
        if faces.is_empty() {
            if let Some(face) = self.lookup_with_alternate(standard_family()) {
                faces.push(face);
            } else if let Some(face) = last_resort().iter().find_map(|n| self.lookup_named(n)) {
                faces.push(face);
            } else if let Some(face) = self.registered.first().and_then(resolve_record) {
                // Without installed fonts the bundled font is the last resort.
                faces.push(face);
            }
        }
        faces
    }

    /// The platform fallback face for a character no font of the list
    /// maps: the first family of the platform's fallback list that has a
    /// drawable face mapping `c`, in the face CSS matching picks for
    /// weight 400.
    pub fn fallback_for(&self, c: char) -> Option<ResolvedFace> {
        system_fallback(is_han(c)).iter().find_map(|name| {
            let lower = name.to_ascii_lowercase();
            let faces: Vec<&FaceRecord> = self
                .registered
                .iter()
                .chain(self.system())
                .filter(|r| r.families.contains(&lower))
                .collect();
            if faces.is_empty() {
                return None;
            }
            best_face(&faces).filter(|f| f.is_drawable() && f.has_char(c))
        })
    }
}

fn resolve_record(record: &FaceRecord) -> Option<ResolvedFace> {
    let bytes = match &record.source {
        FaceSource::Bytes(b) => b.clone(),
        FaceSource::File(path) => FontBytes::Shared(load_file(path)?),
    };
    let face = rustybuzz::ttf_parser::Face::parse(bytes.as_slice(), record.index).ok()?;
    // font-weight drives the wght axis of a variable face
    let variations: Vec<([u8; 4], f32)> = face
        .variation_axes()
        .into_iter()
        .find(|a| a.tag == rustybuzz::ttf_parser::Tag::from_bytes(b"wght"))
        .map(|a| (*b"wght", NORMAL_WEIGHT.clamp(a.min_value, a.max_value)))
        .into_iter()
        .collect();
    Some(ResolvedFace {
        bytes,
        index: record.index,
        variations,
        family: record.display_family.clone(),
        system_ui: false,
        label: format!("{} ({})", record.display_family, record.post_script),
    })
}

/// The weight range a face covers (its `wght` axis, or its weight).
fn weight_range(record: &FaceRecord) -> (f32, f32) {
    let bytes = match &record.source {
        FaceSource::Bytes(b) => Some(b.clone()),
        FaceSource::File(path) => load_file(path).map(FontBytes::Shared),
    };
    let axis = bytes.and_then(|bytes| {
        let face = rustybuzz::ttf_parser::Face::parse(bytes.as_slice(), record.index).ok()?;
        face.variation_axes()
            .into_iter()
            .find(|a| a.tag == rustybuzz::ttf_parser::Tag::from_bytes(b"wght"))
            .map(|a| (a.min_value, a.max_value))
    });
    axis.unwrap_or((record.weight, record.weight))
}

/// CSS font matching (CSS Fonts 4, 5.2) for stretch 100 %, style normal and
/// weight 400 among the faces of one family.
fn best_face(candidates: &[&FaceRecord]) -> Option<ResolvedFace> {
    // stretch: normal, then narrower (closest first), then wider
    let stretch_rank = |s: f32| -> (u8, f32) {
        if s == 100.0 {
            (0, 0.0)
        } else if s < 100.0 {
            (1, 100.0 - s)
        } else {
            (2, s - 100.0)
        }
    };
    let best_stretch = candidates.iter().map(|r| stretch_rank(r.stretch)).fold(
        None,
        |acc: Option<(u8, f32)>, k| {
            Some(match acc {
                Some(a) if (a.0, a.1) <= (k.0, k.1) => a,
                _ => k,
            })
        },
    )?;
    let candidates: Vec<&FaceRecord> = candidates
        .iter()
        .copied()
        .filter(|r| stretch_rank(r.stretch) == best_stretch)
        .collect();
    // style: normal, then oblique, then italic
    let style_rank = |s: FaceStyle| match s {
        FaceStyle::Normal => 0,
        FaceStyle::Oblique => 1,
        FaceStyle::Italic => 2,
    };
    let best_style = candidates.iter().map(|r| style_rank(r.style)).min()?;
    let candidates: Vec<&FaceRecord> = candidates
        .into_iter()
        .filter(|r| style_rank(r.style) == best_style)
        .collect();
    // weight 400: a face covering 400, then 400..=500 ascending, then below
    // 400 descending, then above 500 ascending
    let rank = |r: &FaceRecord| -> (u8, f32) {
        let (lo, hi) = weight_range(r);
        let w = NORMAL_WEIGHT;
        if lo <= w && w <= hi {
            (0, 0.0)
        } else if lo > w && lo <= 500.0 {
            (1, lo - w)
        } else if hi < w {
            (2, w - hi)
        } else {
            (3, lo - w)
        }
    };
    let best = candidates.iter().min_by(|a, b| {
        rank(a)
            .partial_cmp(&rank(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;
    resolve_record(best)
}
