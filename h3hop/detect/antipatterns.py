"""Multi-language error-handling / quality anti-pattern detection (regex fallback).

Categories: error-swallowing | panic-prone | lint-suppression | discarded-result.
Mirrors real lints (clippy unwrap_used/expect_used/panic, ruff BLE001/S110/TRY, go errcheck,
eslint no-empty). semgrep rule files in eval/icd_rules/*-quality.yml provide the heavier version;
this regex layer runs when semgrep is absent or the language is unsupported.
"""
from __future__ import annotations

import re

# (category, lang, rule_id, regex, severity)
_RULES: list[tuple[str, str, str, str, str]] = [
    # Rust
    ("panic-prone", "rust", "rust-unwrap", r"\.unwrap\(\)", "WARNING"),
    ("panic-prone", "rust", "rust-expect", r"\.expect\s*\(", "WARNING"),
    ("panic-prone", "rust", "rust-panic-macro", r"\b(panic|unreachable|todo|unimplemented)\s*!\s*\(", "WARNING"),
    ("discarded-result", "rust", "rust-let-underscore", r"\blet\s+_\s*=", "WARNING"),
    ("error-swallowing", "rust", "rust-ok-swallow", r"\.ok\(\)\s*;", "INFO"),
    ("lint-suppression", "rust", "rust-allow", r"#\[allow\(", "INFO"),
    # Python
    ("error-swallowing", "python", "py-except-pass", r"except[^\n:]*:\s*(?:\n\s*)?(?:pass|\.\.\.)\b", "WARNING"),
    ("error-swallowing", "python", "py-bare-except", r"except\s*:", "WARNING"),
    ("lint-suppression", "python", "py-noqa", r"#\s*noqa", "INFO"),
    ("lint-suppression", "python", "py-type-ignore", r"#\s*type:\s*ignore", "INFO"),
    # Go
    ("error-swallowing", "go", "go-ignore-err", r"\b_\s*=\s*\w*[Ee]rr\w*", "WARNING"),
    ("error-swallowing", "go", "go-ignore-both", r"\b_,\s*_\s*=", "WARNING"),
    ("panic-prone", "go", "go-panic", r"\bpanic\s*\(", "WARNING"),
    # JS / TS
    ("error-swallowing", "js", "js-empty-catch", r"catch\s*\([^)]*\)\s*\{\s*\}", "WARNING"),
    ("lint-suppression", "js", "js-ts-ignore", r"@ts-ignore", "INFO"),
    ("lint-suppression", "js", "js-as-any", r"\bas\s+any\b", "INFO"),
    ("lint-suppression", "js", "js-eslint-disable", r"eslint-disable", "INFO"),
    # Java / C# / C++
    ("error-swallowing", "clike", "clike-empty-catch", r"catch\s*\([^)]*\)\s*\{\s*\}", "WARNING"),
]

_FAMILIES = {
    "rust": ("rust", "rs"),
    "python": ("python", "py"),
    "go": ("go", "golang"),
    "js": ("js", "javascript", "typescript", "ts", "jsx", "tsx", "node"),
    "clike": ("java", "csharp", "cs", "cpp", "c++", "c"),
}


def _lang_match(language: str | None, rlang: str) -> bool:
    if not language:
        return True
    lang = language.lower()
    return any(lang.startswith(x) or x.startswith(lang) for x in _FAMILIES.get(rlang, (rlang,)))


def scan_antipatterns(code: str, language: str | None = None) -> list[dict]:
    findings = []
    for category, rlang, rule_id, pattern, severity in _RULES:
        if not _lang_match(language, rlang):
            continue
        for m in re.finditer(pattern, code, re.MULTILINE):
            findings.append({
                "category": category,
                "rule": rule_id,
                "lang": rlang,
                "severity": severity,
                "line": code[: m.start()].count("\n") + 1,
                "match": m.group(0).strip()[:60],
            })
    return findings
