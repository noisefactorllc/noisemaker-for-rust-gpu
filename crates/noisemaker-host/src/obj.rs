//! Wavefront OBJ meshes: port of the reference parser and texture packer
//! (shaders/src/runtime/obj-parser.js: `parseOBJ`, `computeFaceNormals`,
//! `loadOBJ`, `packMeshDataForTextures`) and of the host's mesh upload
//! (shaders/src/renderer/canvas.js `_packCacheAndUploadMesh`: three 256x256
//! RGBA32F textures `global_<meshId>_{positions,normals,uvs}`).
//!
//! The reference evaluates in JavaScript doubles and stores Float32Arrays.
//! This port follows the same operations in the same order, with
//! JavaScript's `String.prototype.trim`, `split(/\s+/)`, `parseFloat`,
//! `parseInt` and `value || 0` semantics, so a given text yields
//! bit-identical f32 arrays (tests/obj_differential.rs compares them with
//! the reference's own output).

use crate::js;
use std::collections::HashMap;

/// Width and height of the mesh textures the reference host uploads.
pub const MESH_TEXTURE_SIZE: usize = 256;

/// The reference `parseOBJ` result: de-indexed triangle-list vertex data,
/// three vertices per triangle in the reference's (v0, v2, v1) order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ObjMesh {
    /// xyz per vertex.
    pub positions: Vec<f32>,
    /// xyz per vertex (from `vn`, or smooth normals when the file has none).
    pub normals: Vec<f32>,
    /// uv per vertex.
    pub uvs: Vec<f32>,
    /// Number of vertices (`positions.len() / 3`).
    pub vertex_count: usize,
}

/// The reference `packMeshDataForTextures` result: one RGBA32F texel per
/// vertex, row-major, `width * height` texels each. `position_data` has
/// w = 1 for a stored vertex and 0 for every other texel.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PackedMesh {
    /// RGBA: xyz, w = 1 for a valid vertex.
    pub position_data: Vec<f32>,
    /// RGBA: xyz, w = 0.
    pub normal_data: Vec<f32>,
    /// RGBA: uv, zw = 0.
    pub uv_data: Vec<f32>,
    /// Texture width in texels.
    pub width: usize,
    /// Texture height in texels.
    pub height: usize,
    /// Vertices stored (truncated to `width * height`).
    pub vertex_count: usize,
}

impl PackedMesh {
    /// The texture ids the reference WebGPU backend uploads the three
    /// arrays under (`uploadMeshData`), in the order position, normal, uv.
    pub fn texture_ids(mesh_id: &str) -> [String; 3] {
        [
            format!("global_{mesh_id}_positions"),
            format!("global_{mesh_id}_normals"),
            format!("global_{mesh_id}_uvs"),
        ]
    }

    /// Little-endian bytes of a data array, as `writeTexture` receives the
    /// Float32Array (rows are tightly packed: 256 * 16 bytes is already a
    /// multiple of the 256-byte row alignment).
    pub fn bytes(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

/// Errors of [`load_obj`].
#[derive(Debug)]
pub enum ObjError {
    /// The file could not be read (the reference's fetch failure).
    Io(std::io::Error),
}

impl std::fmt::Display for ObjError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ObjError::Io(e) => write!(f, "Failed to load OBJ: {e}"),
        }
    }
}

impl std::error::Error for ObjError {}

#[derive(Clone, Copy)]
struct FaceVertex {
    v: f64,
    vt: f64,
    vn: f64,
}

/// `idx >= 0 && idx < array.length` with a double index (false for NaN).
fn in_range(index: f64, len: usize) -> bool {
    index >= 0.0 && index < len as f64
}

/// Port of `parseOBJ(objText)`.
///
/// Never fails: malformed numbers read as 0, and unresolved indices give
/// the reference's placeholders ((0, 0, 0) positions, (0, 0, 1) normals,
/// (0, 0) uvs). Relative (negative) indices are not resolved, as in the
/// reference.
pub fn parse_obj(obj_text: &str) -> ObjMesh {
    let mut raw_positions: Vec<[f64; 3]> = Vec::new();
    let mut raw_normals: Vec<[f64; 3]> = Vec::new();
    let mut raw_uvs: Vec<[f64; 2]> = Vec::new();
    let mut positions: Vec<f64> = Vec::new();
    let mut normals: Vec<f64> = Vec::new();
    let mut uvs: Vec<f64> = Vec::new();

    let mut parts: Vec<&str> = Vec::new();
    let mut face: Vec<FaceVertex> = Vec::new();
    // objText.split('\n'): only LF separates lines.
    for raw_line in obj_text.split('\n') {
        let line = js::trim(raw_line);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        parts.clear();
        parts.extend(js::split_whitespace(line));
        // `parseFloat(parts[i]) || 0`; a missing part is undefined -> NaN -> 0.
        let number = |i: usize| {
            parts
                .get(i)
                .map_or(0.0, |p| js::or_zero(js::parse_float(p)))
        };
        match parts[0] {
            "v" => raw_positions.push([number(1), number(2), number(3)]),
            "vn" => raw_normals.push([number(1), number(2), number(3)]),
            "vt" => raw_uvs.push([number(1), number(2)]),
            "f" => {
                face.clear();
                for part in &parts[1..] {
                    let mut fields = part.split('/');
                    // OBJ indices are 1-based. An absent or empty uv/normal
                    // field is -1 (`indices[n] ? parseInt(...) - 1 : -1`).
                    let v = js::parse_int10(fields.next().unwrap_or("")) - 1.0;
                    let index = |field: Option<&str>| match field {
                        Some(f) if !f.is_empty() => js::parse_int10(f) - 1.0,
                        _ => -1.0,
                    };
                    let vt = index(fields.next());
                    let vn = index(fields.next());
                    face.push(FaceVertex { v, vt, vn });
                }
                // Fan triangulation, reversed winding (OBJ CW to OpenGL CCW).
                for i in 1..face.len().saturating_sub(1) {
                    for v in [face[0], face[i + 1], face[i]] {
                        // addVertex(v): placeholders for unresolved indices
                        if in_range(v.v, raw_positions.len()) {
                            positions.extend_from_slice(&raw_positions[v.v as usize]);
                        } else {
                            positions.extend_from_slice(&[0.0, 0.0, 0.0]);
                        }
                        if in_range(v.vn, raw_normals.len()) {
                            normals.extend_from_slice(&raw_normals[v.vn as usize]);
                        } else {
                            normals.extend_from_slice(&[0.0, 0.0, 1.0]);
                        }
                        if in_range(v.vt, raw_uvs.len()) {
                            uvs.extend_from_slice(&raw_uvs[v.vt as usize]);
                        } else {
                            uvs.extend_from_slice(&[0.0, 0.0]);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let vertex_count = positions.len() / 3;
    if raw_normals.is_empty() && vertex_count > 0 {
        compute_face_normals(&positions, &mut normals);
    }

    ObjMesh {
        positions: to_f32(&positions),
        normals: to_f32(&normals),
        uvs: to_f32(&uvs),
        vertex_count,
    }
}

fn to_f32(values: &[f64]) -> Vec<f32> {
    // new Float32Array(numbers): IEEE round-to-nearest-even, as `as f32`.
    values.iter().map(|&v| v as f32).collect()
}

/// `Math.round(v * 10000) / 10000` as a hashable key component. Number to
/// string is injective on doubles except that -0 and +0 both print "0", and
/// no component can be NaN, so the bits with -0 folded into +0 identify the
/// reference's string key.
fn key_component(v: f64) -> u64 {
    let rounded = js::math_round(v * 10000.0) / 10000.0;
    if rounded == 0.0 {
        0.0f64.to_bits()
    } else {
        rounded.to_bits()
    }
}

/// Port of `computeFaceNormals(positions, normals)`: the face normal
/// (b - a) x (c - a) of each stored (v0, v2, v1) triangle, stored as f32
/// (the reference's Float32Array), then averaged in doubles over the
/// vertices that share a rounded position.
fn compute_face_normals(positions: &[f64], normals: &mut [f64]) {
    let vertex_count = positions.len() / 3;
    let triangle_count = vertex_count / 3;

    let mut face_normals = vec![0.0f32; triangle_count * 3];
    for tri in 0..triangle_count {
        let i0 = tri * 9;
        let i1 = i0 + 3;
        let i2 = i0 + 6;
        let (ax, ay, az) = (positions[i0], positions[i0 + 1], positions[i0 + 2]);
        let (bx, by, bz) = (positions[i1], positions[i1 + 1], positions[i1 + 2]);
        let (cx, cy, cz) = (positions[i2], positions[i2 + 1], positions[i2 + 2]);
        let (e1x, e1y, e1z) = (bx - ax, by - ay, bz - az);
        let (e2x, e2y, e2z) = (cx - ax, cy - ay, cz - az);
        let mut nx = e1y * e2z - e1z * e2y;
        let mut ny = e1z * e2x - e1x * e2z;
        let mut nz = e1x * e2y - e1y * e2x;
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        if len > 0.0001 {
            nx /= len;
            ny /= len;
            nz /= len;
        } else {
            nx = 0.0;
            ny = 0.0;
            nz = 1.0;
        }
        face_normals[tri * 3] = nx as f32;
        face_normals[tri * 3 + 1] = ny as f32;
        face_normals[tri * 3 + 2] = nz as f32;
    }

    // Map insertion order does not matter: each accumulator is independent.
    let mut key_index: HashMap<[u64; 3], usize> = HashMap::new();
    let mut accumulators: Vec<[f64; 3]> = Vec::new();
    let mut vertex_key = vec![0usize; vertex_count];
    for v in 0..vertex_count {
        let key = [
            key_component(positions[v * 3]),
            key_component(positions[v * 3 + 1]),
            key_component(positions[v * 3 + 2]),
        ];
        let index = *key_index.entry(key).or_insert_with(|| {
            accumulators.push([0.0; 3]);
            accumulators.len() - 1
        });
        vertex_key[v] = index;
        let tri = v / 3;
        let acc = &mut accumulators[index];
        acc[0] += face_normals[tri * 3] as f64;
        acc[1] += face_normals[tri * 3 + 1] as f64;
        acc[2] += face_normals[tri * 3 + 2] as f64;
    }
    for acc in &mut accumulators {
        let len = (acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]).sqrt();
        if len > 0.0001 {
            acc[0] /= len;
            acc[1] /= len;
            acc[2] /= len;
        } else {
            *acc = [0.0, 0.0, 1.0];
        }
    }
    for v in 0..vertex_count {
        let acc = accumulators[vertex_key[v]];
        normals[v * 3..v * 3 + 3].copy_from_slice(&acc);
    }
}

/// Decodes OBJ file bytes as the reference's `fetch(url).then(r => r.text())`
/// does: UTF-8, an initial byte order mark dropped, invalid sequences
/// replaced by U+FFFD.
pub fn decode_obj_text(bytes: &[u8]) -> String {
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    String::from_utf8_lossy(bytes).into_owned()
}

/// Port of `loadOBJ(url)` for a file path: reads, decodes
/// ([`decode_obj_text`]) and parses.
pub fn load_obj(path: impl AsRef<std::path::Path>) -> Result<ObjMesh, ObjError> {
    let bytes = std::fs::read(path).map_err(ObjError::Io)?;
    Ok(parse_obj(&decode_obj_text(&bytes)))
}

/// Port of `packMeshDataForTextures(positions, normals, uvs, texWidth,
/// texHeight)`. Truncates (with a warning on stderr, as the reference's
/// `console.warn`) when there are more vertices than texels.
///
/// The arrays must hold whole vertices (`positions` and `normals` three
/// floats per vertex, `uvs` two), as [`parse_obj`] returns them.
pub fn pack_mesh_data_for_textures(
    positions: &[f32],
    normals: &[f32],
    uvs: &[f32],
    tex_width: usize,
    tex_height: usize,
) -> PackedMesh {
    let max_vertices = tex_width * tex_height;
    let vertex_count = positions.len() / 3;
    if vertex_count > max_vertices {
        eprintln!(
            "[OBJ] Mesh has {vertex_count} vertices, but texture can only hold {max_vertices}. Truncating."
        );
    }
    let used = vertex_count.min(max_vertices);
    let pixel_count = tex_width * tex_height;
    let mut position_data = vec![0.0f32; pixel_count * 4];
    let mut normal_data = vec![0.0f32; pixel_count * 4];
    let mut uv_data = vec![0.0f32; pixel_count * 4];
    // A Float32Array read past its end yields undefined, stored as NaN.
    let at = |data: &[f32], i: usize| data.get(i).copied().unwrap_or(f32::NAN);
    for i in 0..used {
        let pi = i * 4;
        let vi3 = i * 3;
        let vi2 = i * 2;
        position_data[pi] = at(positions, vi3);
        position_data[pi + 1] = at(positions, vi3 + 1);
        position_data[pi + 2] = at(positions, vi3 + 2);
        position_data[pi + 3] = 1.0;
        normal_data[pi] = at(normals, vi3);
        normal_data[pi + 1] = at(normals, vi3 + 1);
        normal_data[pi + 2] = at(normals, vi3 + 2);
        uv_data[pi] = at(uvs, vi2);
        uv_data[pi + 1] = at(uvs, vi2 + 1);
    }
    PackedMesh {
        position_data,
        normal_data,
        uv_data,
        width: tex_width,
        height: tex_height,
        vertex_count: used,
    }
}

/// The host's mesh upload (`_packCacheAndUploadMesh`): `mesh` packed into
/// the 256x256 textures.
pub fn pack_mesh(mesh: &ObjMesh) -> PackedMesh {
    pack_mesh_data_for_textures(
        &mesh.positions,
        &mesh.normals,
        &mesh.uvs,
        MESH_TEXTURE_SIZE,
        MESH_TEXTURE_SIZE,
    )
}

/// The host's `loadOBJFromString(objText)`: parse and pack.
pub fn load_obj_from_string(obj_text: &str) -> PackedMesh {
    pack_mesh(&parse_obj(obj_text))
}

/// Names of the catalog's built-in meshes (render/meshLoader
/// `builtinMeshes`, in definition order: the reference host loads the
/// first one as an effect's default mesh).
pub fn builtin_mesh_names() -> Vec<&'static str> {
    builtin_mesh_table()
        .into_iter()
        .map(|(name, _)| name)
        .collect()
}

/// The catalog path (`share/meshes/<name>.obj`) of a built-in mesh.
pub fn builtin_mesh_path(name: &str) -> Option<&'static str> {
    builtin_mesh_table()
        .into_iter()
        .find(|(n, _)| *n == name)
        .map(|(_, path)| path)
}

/// A built-in mesh by name (`"sphere"`, `"cube"`, ...) or by catalog path
/// (`"share/meshes/cube.obj"`), parsed as the host's `loadOBJFromURL` does.
pub fn builtin_mesh(name: &str) -> Option<ObjMesh> {
    let path = builtin_mesh_path(name).or_else(|| {
        builtin_mesh_table()
            .into_iter()
            .find(|(_, p)| *p == name)
            .map(|(_, p)| p)
    })?;
    let bytes = noisemaker_effects::share_file(path)?;
    Some(parse_obj(&decode_obj_text(bytes)))
}

/// `(name, path)` pairs of every `builtinMeshes` map in the catalog, in
/// catalog and definition order, without duplicates.
fn builtin_mesh_table() -> Vec<(&'static str, &'static str)> {
    use std::sync::OnceLock;
    static TABLE: OnceLock<Vec<(&'static str, &'static str)>> = OnceLock::new();
    TABLE
        .get_or_init(|| {
            let mut table: Vec<(&'static str, &'static str)> = Vec::new();
            for effect in noisemaker_effects::EFFECTS {
                for (name, path) in builtin_meshes_of(effect.definition_json) {
                    if !table.iter().any(|(n, _)| *n == name) {
                        table.push((name, path));
                    }
                }
            }
            table
        })
        .clone()
}

/// The `"builtinMeshes": { "name": "path", ... }` entries of a definition,
/// in order. The catalog writes definitions with `JSON.stringify(def, null,
/// 2)`, so the object is a flat map of plain strings.
fn builtin_meshes_of(json: &'static str) -> Vec<(&'static str, &'static str)> {
    let mut out = Vec::new();
    let Some(start) = json.find("\"builtinMeshes\"") else {
        return out;
    };
    let rest = &json[start..];
    let (Some(open), Some(close)) = (rest.find('{'), rest.find('}')) else {
        return out;
    };
    for entry in rest[open + 1..close].split(',') {
        let mut strings = entry.split('"').skip(1).step_by(2);
        if let (Some(name), Some(path)) = (strings.next(), strings.next()) {
            out.push((name, path));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_meshes_come_from_the_catalog() {
        let names = builtin_mesh_names();
        assert_eq!(
            names,
            [
                "sphere",
                "cube",
                "torus",
                "cylinder",
                "cone",
                "capsule",
                "icosphere"
            ]
        );
        let cube = builtin_mesh("cube").unwrap();
        assert_eq!(cube.vertex_count, 36);
        assert_eq!(builtin_mesh("share/meshes/cube.obj").unwrap(), cube);
        assert!(builtin_mesh("teapot").is_none());
    }

    #[test]
    fn parses_like_the_reference() {
        let mesh = parse_obj("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n");
        assert_eq!(mesh.vertex_count, 6);
        // fan (v0, v2, v1), (v0, v3, v2)
        assert_eq!(&mesh.positions[..9], &[0., 0., 0., 1., 1., 0., 1., 0., 0.]);
        // face normal (b - a) x (c - a) of (v0, v2, v1) is -z
        assert_eq!(&mesh.normals[..3], &[0., 0., -1.]);
        let empty = parse_obj("");
        assert_eq!(empty.vertex_count, 0);
        let packed = pack_mesh(&mesh);
        assert_eq!(packed.position_data.len(), 256 * 256 * 4);
        assert_eq!(packed.position_data[3], 1.0);
        assert_eq!(packed.position_data[6 * 4 + 3], 0.0);
    }
}
