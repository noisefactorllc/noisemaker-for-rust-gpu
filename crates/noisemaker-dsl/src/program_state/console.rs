//! The `console` the ProgramState layer reports to.
//!
//! The reference logs through the global `console`: `Emitter.emit` reports a
//! throwing listener with `console.error`, ProgramState warns about unknown
//! serialization versions and DSL it cannot regenerate, and
//! `extractEffectsFromDsl` warns about programs that do not compile. Those
//! messages are part of the observable behavior, so the port routes them
//! through a [`Console`] the embedder can replace (per thread, like the
//! process-wide JavaScript global). The default writes to standard error.

use std::cell::RefCell;
use std::rc::Rc;

use crate::JsError;
use crate::value::Value;

/// One argument of a `console.warn`/`console.error` call.
#[derive(Debug, Clone, PartialEq)]
pub enum ConsoleArg {
    /// A string or other plain value.
    Value(Value),
    /// A thrown value (an `Error` instance or any other thrown value).
    Error(JsError),
}

impl From<&str> for ConsoleArg {
    fn from(s: &str) -> Self {
        ConsoleArg::Value(Value::from(s))
    }
}

impl From<String> for ConsoleArg {
    fn from(s: String) -> Self {
        ConsoleArg::Value(Value::from(s))
    }
}

impl From<Value> for ConsoleArg {
    fn from(v: Value) -> Self {
        ConsoleArg::Value(v)
    }
}

impl From<JsError> for ConsoleArg {
    fn from(e: JsError) -> Self {
        ConsoleArg::Error(e)
    }
}

/// A `console` sink.
pub trait Console {
    /// `console.warn(...args)`.
    fn warn(&self, args: &[ConsoleArg]);
    /// `console.error(...args)`.
    fn error(&self, args: &[ConsoleArg]);
}

/// The default console: one line per call on standard error, arguments joined
/// by spaces (strings as they are, errors as `Name: message`, other values as
/// JSON).
#[derive(Debug, Default, Clone, Copy)]
pub struct StderrConsole;

fn format_args(args: &[ConsoleArg]) -> String {
    args.iter()
        .map(|a| match a {
            ConsoleArg::Value(Value::String(s)) => s.clone(),
            ConsoleArg::Value(Value::Number(n)) => crate::js::number_to_string(*n),
            ConsoleArg::Value(Value::Undefined) => "undefined".into(),
            ConsoleArg::Value(v) => v.to_json().unwrap_or_else(|| "undefined".into()),
            ConsoleArg::Error(e) => e.to_string(),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

impl Console for StderrConsole {
    fn warn(&self, args: &[ConsoleArg]) {
        eprintln!("{}", format_args(args));
    }

    fn error(&self, args: &[ConsoleArg]) {
        eprintln!("{}", format_args(args));
    }
}

thread_local! {
    static CONSOLE: RefCell<Rc<dyn Console>> = RefCell::new(Rc::new(StderrConsole));
}

/// Replace this thread's console; returns the previous one.
pub fn set_console(console: Rc<dyn Console>) -> Rc<dyn Console> {
    CONSOLE.with(|c| std::mem::replace(&mut *c.borrow_mut(), console))
}

/// This thread's console.
pub fn console() -> Rc<dyn Console> {
    CONSOLE.with(|c| c.borrow().clone())
}

/// `console.warn(...args)`.
pub(crate) fn warn(args: &[ConsoleArg]) {
    console().warn(args);
}

/// `console.error(...args)`.
pub(crate) fn error(args: &[ConsoleArg]) {
    console().error(args);
}
