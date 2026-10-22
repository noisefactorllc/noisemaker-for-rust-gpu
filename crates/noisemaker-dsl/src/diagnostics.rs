//! Diagnostic codes and default messages (`lang/diagnostics.js`).

/// One diagnostic code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiagnosticInfo {
    pub code: &'static str,
    pub stage: &'static str,
    pub severity: &'static str,
    pub message: &'static str,
}

/// The reference table, in its declaration order.
pub const DIAGNOSTICS: &[DiagnosticInfo] = &[
    DiagnosticInfo {
        code: "L001",
        stage: "lexer",
        severity: "error",
        message: "Unexpected character",
    },
    DiagnosticInfo {
        code: "L002",
        stage: "lexer",
        severity: "error",
        message: "Unterminated string literal",
    },
    DiagnosticInfo {
        code: "L003",
        stage: "lexer",
        severity: "error",
        message: "Unterminated comment",
    },
    DiagnosticInfo {
        code: "L004",
        stage: "lexer",
        severity: "error",
        message: "Output surface reference out of range",
    },
    DiagnosticInfo {
        code: "P001",
        stage: "parser",
        severity: "error",
        message: "Unexpected token",
    },
    DiagnosticInfo {
        code: "P002",
        stage: "parser",
        severity: "error",
        message: "Expected closing parenthesis",
    },
    DiagnosticInfo {
        code: "P003",
        stage: "parser",
        severity: "error",
        message: "Invalid automation arguments",
    },
    DiagnosticInfo {
        code: "P004",
        stage: "parser",
        severity: "error",
        message: "Invalid search directive",
    },
    DiagnosticInfo {
        code: "P005",
        stage: "parser",
        severity: "error",
        message: "Invalid output operation",
    },
    DiagnosticInfo {
        code: "P006",
        stage: "parser",
        severity: "error",
        message: "Invalid subchain",
    },
    DiagnosticInfo {
        code: "P007",
        stage: "parser",
        severity: "error",
        message: "Invalid call expression",
    },
    DiagnosticInfo {
        code: "P008",
        stage: "parser",
        severity: "warning",
        message: "Unknown subchain argument key",
    },
    DiagnosticInfo {
        code: "P009",
        stage: "parser",
        severity: "warning",
        message: "Duplicate subchain argument key",
    },
    DiagnosticInfo {
        code: "P010",
        stage: "parser",
        severity: "warning",
        message: "Missing ',' between subchain arguments",
    },
    DiagnosticInfo {
        code: "S001",
        stage: "semantic",
        severity: "error",
        message: "Unknown identifier",
    },
    DiagnosticInfo {
        code: "S002",
        stage: "semantic",
        severity: "warning",
        message: "Argument out of range",
    },
    DiagnosticInfo {
        code: "S003",
        stage: "semantic",
        severity: "error",
        message: "Variable used before assignment",
    },
    DiagnosticInfo {
        code: "S004",
        stage: "semantic",
        severity: "error",
        message: "Cannot assign null or undefined",
    },
    DiagnosticInfo {
        code: "S005",
        stage: "semantic",
        severity: "error",
        message: "Illegal chain structure",
    },
    DiagnosticInfo {
        code: "S006",
        stage: "semantic",
        severity: "error",
        message: "Starter chain missing write() call",
    },
    DiagnosticInfo {
        code: "S007",
        stage: "semantic",
        severity: "warning",
        message: "Deprecated parameter alias",
    },
    DiagnosticInfo {
        code: "S008",
        stage: "semantic",
        severity: "warning",
        message: "Deprecated effect",
    },
    DiagnosticInfo {
        code: "R001",
        stage: "runtime",
        severity: "error",
        message: "Runtime error",
    },
];

/// `diagnostics[code]`.
pub fn lookup(code: &str) -> Option<&'static DiagnosticInfo> {
    DIAGNOSTICS.iter().find(|d| d.code == code)
}
