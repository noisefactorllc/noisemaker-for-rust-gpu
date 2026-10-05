//! Generates the static catalog table from `catalog/` (written by
//! tools/convert-effects.mjs). Effects are listed in namespace/effect order and
//! WGSL programs in file-name order, the order the converter writes them.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

fn sorted_entries(dir: &Path) -> Vec<PathBuf> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    entries
}

fn literal(path: &Path) -> String {
    format!("{:?}", path.to_str().expect("catalog paths are UTF-8"))
}

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("catalog");
    println!("cargo:rerun-if-changed={}", root.display());
    let mut out = String::new();
    writeln!(out, "pub static EFFECTS: &[EffectSource] = &[").unwrap();
    for ns_dir in sorted_entries(&root).into_iter().filter(|p| p.is_dir()) {
        let namespace = ns_dir.file_name().unwrap().to_str().unwrap().to_owned();
        if namespace == "share" {
            continue;
        }
        println!("cargo:rerun-if-changed={}", ns_dir.display());
        for effect_dir in sorted_entries(&ns_dir).into_iter().filter(|p| p.is_dir()) {
            let name = effect_dir.file_name().unwrap().to_str().unwrap().to_owned();
            let def = effect_dir.join("definition.json");
            if !def.exists() {
                continue;
            }
            println!("cargo:rerun-if-changed={}", effect_dir.display());
            writeln!(out, "    EffectSource {{").unwrap();
            writeln!(out, "        namespace: {namespace:?},").unwrap();
            writeln!(out, "        name: {name:?},").unwrap();
            writeln!(
                out,
                "        definition_json: include_str!({}),",
                literal(&def)
            )
            .unwrap();
            writeln!(out, "        wgsl: &[").unwrap();
            let wgsl_dir = effect_dir.join("wgsl");
            if wgsl_dir.is_dir() {
                println!("cargo:rerun-if-changed={}", wgsl_dir.display());
                for file in sorted_entries(&wgsl_dir) {
                    if file.extension().and_then(|e| e.to_str()) != Some("wgsl") {
                        continue;
                    }
                    let program = file.file_stem().unwrap().to_str().unwrap();
                    writeln!(
                        out,
                        "            ({program:?}, include_str!({})),",
                        literal(&file)
                    )
                    .unwrap();
                }
            }
            writeln!(out, "        ],").unwrap();
            writeln!(out, "    }},").unwrap();
        }
    }
    writeln!(out, "];").unwrap();

    writeln!(out, "pub static SHARE_FILES: &[(&str, &[u8])] = &[").unwrap();
    let share = root.join("share");
    let mut stack = vec![share.clone()];
    let mut files = Vec::new();
    while let Some(dir) = stack.pop() {
        if !dir.is_dir() {
            continue;
        }
        println!("cargo:rerun-if-changed={}", dir.display());
        for entry in sorted_entries(&dir) {
            if entry.is_dir() {
                stack.push(entry);
            } else {
                files.push(entry);
            }
        }
    }
    files.sort();
    for file in files {
        let rel = file
            .strip_prefix(&root)
            .unwrap()
            .to_str()
            .unwrap()
            .replace('\\', "/");
        writeln!(out, "    ({rel:?}, include_bytes!({})),", literal(&file)).unwrap();
    }
    writeln!(out, "];").unwrap();
    writeln!(
        out,
        "pub static MANIFEST_JSON: &str = include_str!({});",
        literal(&root.join("manifest.json"))
    )
    .unwrap();

    let dest = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("catalog.rs");
    std::fs::write(dest, out).unwrap();
}
