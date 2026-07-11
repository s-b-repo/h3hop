"""Integrity detection: code anti-patterns, cheating (static + agent traces), and hidden content.

Stdlib + regex first (runs offline); semgrep rule files under eval/icd_rules/ augment when present.
The single entry point is `aggregate.scan_all`.
"""
