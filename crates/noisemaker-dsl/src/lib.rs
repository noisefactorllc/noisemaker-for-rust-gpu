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

pub mod compiler;
pub mod diagnostics;
pub mod error;
pub mod expander;
pub mod js;
pub mod lexer;
pub mod parser;
pub mod registry;
pub mod resources;
pub mod validator;
pub mod value;

pub use error::JsError;
pub use registry::Registry;
pub use value::{Object, Value};

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
    let ast = parser::parse(&tokens)?;
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
