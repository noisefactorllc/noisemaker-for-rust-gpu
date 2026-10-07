//! nm-input-dump — runs parity scenarios through noisemaker-input and prints
//! the results as JSON, for the gates under `parity/`.
//!
//! Usage: `nm-input-dump <midi|audio|analyser|automation|math> <scenarios.json>`
//!
//! `analyser` prints one JSON array; the other commands print one JSON record
//! per line (per step, evaluation or requirement set), like the reference
//! runner.
//!
//! The scenario formats, and the reference-side runners that produce the
//! expected output, are in `tools/reference-input.mjs` (MIDI, audio state,
//! automation, math) and `tools/reference-input-analyser.mjs` (Chromium's
//! AnalyserNode).

use std::io::{BufWriter, Write};
use std::process::ExitCode;

use serde_json::Value;

mod dump;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: nm-input-dump <midi|audio|analyser|automation|math> <scenarios.json>");
        return ExitCode::from(2);
    }
    let text = match std::fs::read_to_string(&args[2]) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("nm-input-dump: cannot read {}: {error}", args[2]);
            return ExitCode::from(2);
        }
    };
    let scenarios: Value = match serde_json::from_str(&text) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("nm-input-dump: invalid JSON in {}: {error}", args[2]);
            return ExitCode::from(2);
        }
    };
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let mut emit = |record: Value| {
        serde_json::to_writer(&mut out, &record).expect("serializable record");
        out.write_all(b"\n").expect("stdout");
    };
    let result = match args[1].as_str() {
        "midi" => dump::midi::run(&scenarios, &mut emit),
        "audio" => dump::audio::run(&scenarios, &mut emit),
        "automation" => dump::automation::run(&scenarios, &mut emit),
        "math" => dump::math::run(&scenarios, &mut emit),
        "analyser" => dump::analyser::run(&scenarios).map(&mut emit),
        other => {
            eprintln!("nm-input-dump: unknown command {other}");
            return ExitCode::from(2);
        }
    };
    if let Err(error) = out.flush() {
        eprintln!("nm-input-dump: {error}");
        return ExitCode::FAILURE;
    }
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("nm-input-dump: {error}");
            ExitCode::FAILURE
        }
    }
}
