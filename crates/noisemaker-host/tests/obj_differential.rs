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
