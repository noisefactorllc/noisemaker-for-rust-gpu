//! The Polymorphic DSL of the Noisemaker rendering engine.
//!
//! A stage-by-stage port of the reference frontend (`shaders/src/lang/` and the
//! graph-building half of `shaders/src/runtime/`):
//!
//! `lex → parse → validate → expand → allocate_resources → compile_graph`
//!
//! Every stage produces the same data the reference produces (see [`value::Value`]),
//! so each one is checked against the reference's own output over the parity corpus
//! (`parity/check_frontend.mjs`).
//!
//! The rest of the reference's public entry point (`shaders/src/index.js`) that
//! needs no GPU is here too: the registries ([`Registry`]: effects, ops,
//! starter ops, enums, namespaces with [`tags`]' registration rules, and
//! user-defined Portable effects, [`portable`]), the `Effect` constructor and
//! parameter categories ([`effect`]), the `renderer/canvas.js` helpers
//! ([`canvas`]), the cosine palettes ([`palettes`]), the block categories
//! ([`constants`]), effect strings and manifest queries ([`strings`]),
//! [`compile`], [`VERSION`] and [`PHASE`] (`parity/check_api.mjs` and
//! `parity/check_portable.mjs` gate them against the reference).

pub mod canvas;
pub mod compiler;
pub mod constants;
pub mod diagnostics;
pub mod effect;
pub mod effect_validator;
pub mod error;
pub mod error_formatter;
pub mod expander;
pub mod js;
pub mod jsmath;
pub mod lexer;
pub mod palette;
pub mod palettes;
pub mod parser;
pub mod portable;
pub mod program_state;
pub mod registry;
pub mod resources;
pub mod strings;
pub mod tagged;
pub mod tags;
pub mod transform;
pub mod unparser;
pub mod validator;
pub mod value;

pub use error::JsError;
pub use registry::Registry;
pub use value::{Object, Value};

/// `VERSION` of the reference engine's public entry point
/// (`shaders/src/index.js`).
pub const VERSION: &str = "0.1.0";

/// `PHASE` of the reference engine's public entry point.
pub const PHASE: u32 = 4;

/// `compile(src)` of `lang/index.js`: `validate(parse(lex(src)))`, the
/// validated program (`{plans, diagnostics, render, ...}`), with `search`
/// directives checked against `registry`'s namespaces.
pub fn compile(src: &str, registry: &Registry) -> Result<Value, JsError> {
    compile_with_options(src, &parser::ParseOptions::default(), registry)
}

/// `compile(src, options)`: [`compile`] with the parser options the
/// reference passes through to `parse(tokens, options)`.
pub fn compile_with_options(
    src: &str,
    options: &parser::ParseOptions,
    registry: &Registry,
) -> Result<Value, JsError> {
    let tokens = lexer::lex(src)?;
    let ast = parser::parse_with_options(&tokens, registry, options).map_err(JsError::from)?;
    validator::validate(&ast, registry)
}

/// The frontend stages that `nm-render dump` and the parity gate compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Tokens,
    Ast,
    Validated,
    Expanded,
    Graph,
}

impl Stage {
    pub const ALL: [Stage; 5] = [
        Stage::Tokens,
        Stage::Ast,
        Stage::Validated,
        Stage::Expanded,
        Stage::Graph,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Stage::Tokens => "tokens",
            Stage::Ast => "ast",
            Stage::Validated => "validated",
            Stage::Expanded => "expanded",
            Stage::Graph => "graph",
        }
    }

    pub fn from_name(name: &str) -> Option<Stage> {
        Stage::ALL.into_iter().find(|s| s.name() == name)
    }

    /// The stage whose reference output `--isolated` feeds to this stage.
    pub fn previous(self) -> Option<Stage> {
        match self {
            Stage::Tokens => None,
            Stage::Ast => None,
            Stage::Validated => Some(Stage::Ast),
            Stage::Expanded => Some(Stage::Validated),
            Stage::Graph => Some(Stage::Validated),
        }
    }
}

/// Run the frontend from source through `stage` and return that stage's output as
/// the reference oracle serializes it (`tools/reference-oracle.mjs`).
pub fn run_stage(stage: Stage, src: &str, registry: &Registry) -> Result<Value, JsError> {
    if stage == Stage::Graph {
        return compiler::dump_graph(src, registry);
    }
    let tokens = lexer::lex(src)?;
    if stage == Stage::Tokens {
        return Ok(lexer::tokens_to_value(&tokens));
    }
    let ast = parser::parse_with_registry(&tokens, registry)?;
    if stage == Stage::Ast {
        return Ok(ast);
    }
    run_stage_from(stage, Stage::Ast, ast, src, registry)
}

/// Run `stage` on `input`, the output of stage `from`, for the program `src`. This
/// checks one stage in isolation against the reference's output of the stage
/// before it (`nm-render dump --isolated`).
pub fn run_stage_from(
    stage: Stage,
    from: Stage,
    input: Value,
    src: &str,
    registry: &Registry,
) -> Result<Value, JsError> {
    match (from, stage) {
        (Stage::Ast, Stage::Validated) => validator::validate(&input, registry),
        (Stage::Ast, Stage::Expanded) => {
            let validated = validator::validate(&input, registry)?;
            expander::expand_to_value(&validated, registry)
        }
        (Stage::Validated, Stage::Expanded) => expander::expand_to_value(&input, registry),
        (Stage::Validated, Stage::Graph) => {
            compiler::dump_graph_from_validated(src, &input, registry)
        }
        (from, to) => Err(JsError::error(format!(
            "cannot run stage {} from stage {}",
            to.name(),
            from.name()
        ))),
    }
}
