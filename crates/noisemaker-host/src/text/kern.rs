//! Fractional kerning variation deltas.
//!
//! In a variable font, GPOS pair adjustments carry VariationIndex device
//! tables whose deltas are fractional font units. HarfBuzz (Blink shapes
//! with it) scales them at the font's 16.16 scale (`em_scalef`), so the
//! fractions survive; rustybuzz works in font units and rounds each delta
//! to a whole unit (`Device::get_x_delta`). At 51 px a rounded delta moves
//! a glyph by up to 0.026 px, enough to change its quarter-pixel position.
//!
//! [`kern_residuals`] re-applies the `kern` feature's pair lookups the way
//! rustybuzz applied them and returns, per glyph, the part of its x-advance
//! adjustment the rounding dropped: the sum of `delta - round(delta)` over
//! the device tables applied to it. Lookups with mark attachment classes
//! or mark filtering sets contribute nothing (Latin kerning does not use
//! them).

use rustybuzz::ttf_parser::{self, GlyphId, Tag};
use ttf_parser::gdef::GlyphClass;
use ttf_parser::gpos::{Device, PairAdjustment, PositioningSubtable, ValueRecord};
use ttf_parser::opentype_layout::LookupFlags;

/// The rounded-away part of a value record's x-advance variation delta.
fn residual(record: &ValueRecord<'_>, face: &ttf_parser::Face<'_>) -> f64 {
    let Some(Device::Variation(device)) = record.x_advance_device else {
        return 0.0;
    };
    let Some(gdef) = face.tables().gdef else {
        return 0.0;
    };
    match gdef.glyph_variation_delta(
        device.outer_index,
        device.inner_index,
        face.variation_coordinates(),
    ) {
        Some(delta) => delta as f64 - delta.round() as f64,
        None => 0.0,
    }
}

/// Whether a lookup with `flags` skips `glyph` (GDEF glyph classes).
fn ignored(face: &ttf_parser::Face<'_>, flags: LookupFlags, glyph: GlyphId) -> bool {
    if !flags.ignore_flags() {
        return false;
    }
    let class = face.tables().gdef.and_then(|g| g.glyph_class(glyph));
    (flags.ignore_base_glyphs() && class == Some(GlyphClass::Base))
        || (flags.ignore_ligatures() && class == Some(GlyphClass::Ligature))
        || (flags.ignore_marks() && class == Some(GlyphClass::Mark))
}

/// The value records a pair subtable gives (first, second), if it matches.
fn pair_records<'a>(
    subtable: &PairAdjustment<'a>,
    first: GlyphId,
    second: GlyphId,
) -> Option<(ValueRecord<'a>, ValueRecord<'a>)> {
    match subtable {
        PairAdjustment::Format1 { coverage, sets } => {
            let index = coverage.get(first)?;
            sets.get(index)?.get(second)
        }
        PairAdjustment::Format2 {
            coverage,
            classes,
            matrix,
        } => {
            coverage.get(first)?;
            matrix.get((classes.0.get(first), classes.1.get(second)))
        }
    }
}

fn record_is_empty(record: &ValueRecord<'_>) -> bool {
    record.x_placement == 0
        && record.y_placement == 0
        && record.x_advance == 0
        && record.y_advance == 0
        && record.x_placement_device.is_none()
        && record.y_placement_device.is_none()
        && record.x_advance_device.is_none()
        && record.y_advance_device.is_none()
}

/// Per glyph of a shaped horizontal run (glyph ids in buffer order), the
/// fractional x-advance kerning that rounding the variation deltas to
/// whole font units dropped. Zeros for a face without variations or GPOS.
pub fn kern_residuals(face: &ttf_parser::Face<'_>, glyphs: &[u16]) -> Vec<f64> {
    let mut out = vec![0.0; glyphs.len()];
    if glyphs.len() < 2 || face.variation_coordinates().iter().all(|c| c.get() == 0) {
        return out;
    }
    let Some(gpos) = face.tables().gpos else {
        return out;
    };
    if face.tables().gdef.is_none() {
        return out;
    }
    // the lookups of every `kern` feature, in lookup-list order
    let kern = Tag::from_bytes(b"kern");
    let mut lookups: Vec<u16> = Vec::new();
    for feature in gpos.features {
        if feature.tag == kern {
            for index in feature.lookup_indices {
                if !lookups.contains(&index) {
                    lookups.push(index);
                }
            }
        }
    }
    lookups.sort_unstable();
    for index in lookups {
        let Some(lookup) = gpos.lookups.get(index) else {
            continue;
        };
        if lookup.flags.use_mark_filtering_set() || lookup.flags.mark_attachment_type() != 0 {
            continue;
        }
        let subtables: Vec<PairAdjustment<'_>> = (0..lookup.subtables.len())
            .filter_map(|i| match lookup.subtables.get::<PositioningSubtable>(i) {
                Some(PositioningSubtable::Pair(pair)) => Some(pair),
                _ => None,
            })
            .collect();
        if subtables.is_empty() {
            continue;
        }
        let skip = |g: u16| ignored(face, lookup.flags, GlyphId(g));
        // HarfBuzz PairPos: at each glyph the lookup does not skip, pair it
        // with the next such glyph; the first subtable covering the first
        // glyph applies; continue at the second glyph, or after it when
        // the second value record is not empty.
        let mut i = 0;
        while i < glyphs.len() {
            if skip(glyphs[i]) {
                i += 1;
                continue;
            }
            let mut j = i + 1;
            while j < glyphs.len() && skip(glyphs[j]) {
                j += 1;
            }
            if j >= glyphs.len() {
                break;
            }
            let first = GlyphId(glyphs[i]);
            let second = GlyphId(glyphs[j]);
            // the first subtable that has a record for the pair applies
            // (rustybuzz PairAdjustment::apply: a subtable without one
            // passes the glyph on to the next)
            let mut next = i + 1;
            for subtable in &subtables {
                if let Some((r1, r2)) = pair_records(subtable, first, second) {
                    if !record_is_empty(&r1) {
                        out[i] += residual(&r1, face);
                    }
                    if !record_is_empty(&r2) {
                        out[j] += residual(&r2, face);
                    }
                    next = if record_is_empty(&r2) { j } else { j + 1 };
                    break;
                }
            }
            i = next;
        }
    }
    out
}
