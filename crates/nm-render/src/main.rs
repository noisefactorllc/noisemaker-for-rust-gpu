//! nm-render — render Noisemaker DSL programs on the GPU and dump the frontend's
//! stages for the parity gates.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use noisemaker_dsl::{Registry, Stage, Value};

#[derive(Parser)]
#[command(
    name = "nm-render",
    version,
    about = "Render Noisemaker DSL programs on the GPU"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Dump one frontend stage for each program as JSON lines
    /// ({"program", "stage", "result"|"error"}), the format of
    /// tools/reference-oracle.mjs.
    Dump {
        /// tokens | ast | validated | expanded | graph
        stage: String,
        /// Output file (default: standard output)
        #[arg(long)]
        out: Option<PathBuf>,
        /// Run only this stage, on the previous stage's output read from a
        /// reference JSON-lines dump (checks one stage in isolation)
        #[arg(long)]
        isolated: Option<PathBuf>,
        /// DSL program files
        #[arg(required = true)]
        programs: Vec<PathBuf>,
    },
}

fn program_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.strip_suffix(".dsl").unwrap_or(n).to_owned())
        .unwrap_or_default()
}

fn read_jsonl(path: &Path) -> Result<std::collections::HashMap<String, Value>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = std::collections::HashMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let rec = Value::from_json(line).map_err(|e| format!("{}: {e}", path.display()))?;
        let name = rec.get("program").as_str().unwrap_or_default().to_owned();
        out.insert(name, rec);
    }
    Ok(out)
}

fn dump(
    stage: &str,
    out: Option<PathBuf>,
    isolated: Option<PathBuf>,
    programs: Vec<PathBuf>,
) -> Result<(), String> {
    let stage = Stage::from_name(stage).ok_or_else(|| format!("unknown stage '{stage}'"))?;
    let registry = Registry::with_catalog();
    let previous = match &isolated {
        Some(path) => {
            let prev = stage.previous().ok_or_else(|| {
                format!("stage {} has no previous stage to start from", stage.name())
            })?;
            Some((prev, read_jsonl(path)?))
        }
        None => None,
    };
    let mut lines = String::new();
    for path in &programs {
        let name = program_name(path);
        let result = match &previous {
            Some((prev, records)) => match records.get(&name) {
                Some(rec) if rec.as_object().is_some_and(|o| o.contains_key("error")) => {
                    Err(noisemaker_dsl::JsError::Thrown(rec.get("error").clone()))
                }
                Some(rec) => noisemaker_dsl::run_stage_from(
                    stage,
                    *prev,
                    rec.get("result").clone(),
                    &registry,
                ),
                None => Err(noisemaker_dsl::JsError::error(format!(
                    "no {} record for {name}",
                    prev.name()
                ))),
            },
            None => {
                let src = std::fs::read_to_string(path)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                noisemaker_dsl::run_stage(stage, &src, &registry)
            }
        };
        let mut rec = noisemaker_dsl::Object::new();
        rec.insert("program", Value::from(name));
        rec.insert("stage", Value::from(stage.name()));
        match result {
            Ok(v) => rec.insert("result", v),
            Err(e) => rec.insert("error", e.to_value()),
        };
        lines.push_str(&Value::Object(rec).to_json().unwrap());
        lines.push('\n');
    }
    match out {
        Some(path) => std::fs::write(&path, lines).map_err(|e| format!("{}: {e}", path.display())),
        None => std::io::stdout()
            .write_all(lines.as_bytes())
            .map_err(|e| e.to_string()),
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Dump {
            stage,
            out,
            isolated,
            programs,
        } => dump(&stage, out, isolated, programs),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("nm-render: {e}");
            ExitCode::from(2)
        }
    }
}
