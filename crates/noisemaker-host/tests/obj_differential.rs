//! OBJ differential test: the port's parse_obj + pack_mesh against the
//! reference's parseOBJ + packMeshDataForTextures(..., 256, 256), run by
//! `tools/reference-host.mjs obj` on every catalog mesh, every parity OBJ
//! and the tool's edge-case texts. Every Float32Array must be bit-identical.
//!
//! Needs NM_REFERENCE_ROOT (a reference checkout) and node; without them
//! the test explains why and passes.

mod common;

use noisemaker_host::obj;

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Index of the first element whose bits differ, if any.
fn first_difference(a: &[f32], b: &[f32]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter()
        .zip(b)
        .position(|(x, y)| x.to_bits() != y.to_bits())
}

/// Minimal reader for the tool's manifest: an array of flat objects with
/// string and integer values.
fn manifest_entries(json: &str) -> Vec<(String, usize, usize)> {
    let mut out = Vec::new();
    for object in json.split('{').skip(1) {
        let field = |key: &str| -> &str {
            let start = object.find(&format!("\"{key}\"")).expect(key) + key.len() + 2;
            let rest = object[start..].trim_start_matches([':', ' ']);
            let end = rest.find([',', '\n', '}']).unwrap();
            rest[..end].trim().trim_matches('"')
        };
        out.push((
            field("name").to_owned(),
            field("vertexCount").parse().unwrap(),
            field("packedVertexCount").parse().unwrap(),
        ));
    }
    out
}

#[test]
fn obj_parse_and_pack_match_the_reference_bit_for_bit() {
    let Some(reference) = common::reference_env("obj_differential") else {
        return;
    };
    let dir = common::scratch_dir("obj");
    common::reference_host(&reference, &["obj", "--out", dir.to_str().unwrap()]);
    let manifest = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    let entries = manifest_entries(&manifest);
    assert!(
        entries.len() >= 30,
        "expected the catalog, parity and edge cases"
    );

    let mut failures = Vec::new();
    let mut floats = 0usize;
    for (name, vertex_count, packed_vertex_count) in &entries {
        let bytes = std::fs::read(dir.join(format!("{name}.obj"))).unwrap();
        let mesh = obj::parse_obj(&obj::decode_obj_text(&bytes));
        let indices = std::fs::read(dir.join(format!("{name}.indices.u32"))).unwrap();
        let expected_indices: Vec<u32> = indices
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| u32::from_le_bytes(*bytes))
            .collect();
        assert_eq!(mesh.indices, expected_indices, "{name} indices");
        let packed = obj::pack_mesh(&mesh);
        if mesh.vertex_count != *vertex_count || packed.vertex_count != *packed_vertex_count {
            failures.push(format!(
                "{name}: vertex counts {}/{} vs reference {vertex_count}/{packed_vertex_count}",
                mesh.vertex_count, packed.vertex_count
            ));
        }
        for (key, ours) in [
            ("positions", &mesh.positions),
            ("normals", &mesh.normals),
            ("uvs", &mesh.uvs),
            ("positionData", &packed.position_data),
            ("normalData", &packed.normal_data),
            ("uvData", &packed.uv_data),
        ] {
            let theirs = read_f32(&dir.join(format!("{name}.{key}.f32")));
            floats += theirs.len();
            if let Some(i) = first_difference(ours, &theirs) {
                failures.push(format!(
                    "{name}.{key}: first difference at {i}: ours {:?} reference {:?} (lengths {} / {})",
                    ours.get(i),
                    theirs.get(i),
                    ours.len(),
                    theirs.len()
                ));
            }
        }
    }
    eprintln!(
        "obj_differential: {} cases, {floats} floats compared, {} mismatching arrays",
        entries.len(),
        failures.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn indexed_geometry_matches_reference_packing() {
    let reference = common::reference_env("geometry_differential")
        .expect("geometry differential requires the pinned reference and Node");
    let dir = common::scratch_dir("geometry");
    common::reference_host(&reference, &["geometry", "--out", dir.to_str().unwrap()]);
    let cases: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("geometry.json")).unwrap()).unwrap();
    for tc in cases.as_array().unwrap() {
        let name = tc["name"].as_str().unwrap();
        let floats = |key: &str| {
            tc[key]
                .as_array()
                .map(|values| values.iter().map(|v| v.as_f64().unwrap() as f32).collect())
                .unwrap_or_default()
        };
        let indices = tc["indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let mesh = obj::ObjMesh::new(
            floats("positions"),
            floats("normals"),
            floats("uvs"),
            indices,
        );
        assert_eq!(
            mesh.vertex_count,
            tc["vertexCount"].as_u64().unwrap() as usize,
            "{name}"
        );
        assert_eq!(
            mesh.triangle_count(),
            tc["triangleCount"].as_f64().unwrap(),
            "{name}"
        );
        let flat = mesh.deindex();
        assert_eq!(
            flat.vertex_count,
            tc["flatCount"].as_u64().unwrap() as usize,
            "{name}"
        );
        assert_eq!(
            obj::mesh_texture_size(flat.vertex_count),
            (
                tc["size"]["texWidth"].as_u64().unwrap() as usize,
                tc["size"]["texHeight"].as_u64().unwrap() as usize
            )
        );
        let packed = mesh.to_packed_textures(
            tc["width"].as_u64().unwrap() as usize,
            tc["height"].as_u64().unwrap() as usize,
        );
        assert_eq!(
            packed.vertex_count,
            tc["packedCount"].as_u64().unwrap() as usize
        );
        for (key, actual) in [
            ("positions", &flat.positions),
            ("normals", &flat.normals),
            ("uvs", &flat.uvs),
            ("positionData", &packed.position_data),
            ("normalData", &packed.normal_data),
            ("uvData", &packed.uv_data),
        ] {
            let expected = read_f32(&dir.join(format!("{name}.{key}.f32")));
            assert_eq!(actual.len(), expected.len(), "{name}.{key}");
            for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan(),
                    "{name}.{key}[{i}]: {a} vs {b}"
                );
            }
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}
