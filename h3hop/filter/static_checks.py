"""Stage B (2/3): ruff (style) + bandit & semgrep (security/CWE) on each candidate.

Missing tools are skipped gracefully (recorded in meta), never fatal. Insecure findings
do NOT delete the record here — build_dataset routes insecure code to the BAD set.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from ..io import ensure_dir, read_records, write_jsonl

# bandit test id -> CWE (subset)
BANDIT_CWE = {
    "B602": "CWE-78", "B603": "CWE-78", "B605": "CWE-78", "B607": "CWE-78",
    "B608": "CWE-89", "B301": "CWE-502", "B506": "CWE-502",
    "B105": "CWE-259", "B106": "CWE-259", "B107": "CWE-259",
    "B303": "CWE-327", "B324": "CWE-327", "B311": "CWE-330",
}


def _run(cmd: list[str], stdin: str | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, input=stdin, timeout=60)
        return p.returncode, p.stdout + p.stderr
    except FileNotFoundError:
        return -1, "__missing__"
    except subprocess.TimeoutExpired:
        return -2, "__timeout__"


def _ruff(fp: Path) -> tuple[int | None, str]:
    rc, out = _run(["ruff", "check", "--output-format", "json", str(fp)])
    if out == "__missing__":
        return None, "not-installed"
    try:
        return len(json.loads(out or "[]")), "ok"
    except Exception:
        return None, "parse-error"


def _bandit(fp: Path) -> tuple[list[dict], str]:
    rc, out = _run(["bandit", "-f", "json", "-q", str(fp)])
    if out == "__missing__":
        return [], "not-installed"
    try:
        data = json.loads(out)
    except Exception:
        return [], "parse-error"
    findings = []
    for r in data.get("results", []):
        tid = r.get("test_id", "")
        findings.append({
            "rule": tid,
            "severity": (r.get("issue_severity") or "").upper(),
            "line": r.get("line_number"),
            "cwe": BANDIT_CWE.get(tid) or (r.get("issue_cwe") or {}).get("id"),
            "tool": "bandit",
        })
    return findings, "ok"


def _semgrep(fp: Path, rules: str) -> tuple[list[dict], str]:
    if not Path(rules).exists():
        return [], "no-rules"
    rc, out = _run(["semgrep", "--config", rules, "--json", "--quiet", str(fp)])
    if out == "__missing__":
        return [], "not-installed"
    try:
        data = json.loads(out)
    except Exception:
        return [], "parse-error"
    findings = []
    for r in data.get("results", []):
        meta = r.get("extra", {}).get("metadata", {})
        findings.append({
            "rule": r.get("check_id"),
            "severity": (r.get("extra", {}).get("severity") or "").upper(),
            "line": r.get("start", {}).get("line"),
            "cwe": (meta.get("cwe") or [None])[0] if isinstance(meta.get("cwe"), list) else meta.get("cwe"),
            "tool": "semgrep",
        })
    return findings, "ok"


def scan_code(code: str, rules: str = "eval/icd_rules") -> dict:
    """Run ruff+bandit+semgrep on a code string. Reused by filtering, the RLVR reward, and eval."""
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "c.py"
        fp.write_text(code, encoding="utf-8")
        style, s1 = _ruff(fp)
        bfind, s2 = _bandit(fp)
        sfind, s3 = _semgrep(fp, rules)
    return {
        "style_violations": style,
        "cwe_findings": bfind + sfind,
        "scanners": {"ruff": s1, "bandit": s2, "semgrep": s3},
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Static + security checks.")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--rules", default="eval/icd_rules")
    args = ap.parse_args(argv)

    recs = read_records(args.inp)
    tool_status: dict[str, str] = {}
    n_insecure = 0
    for rec in recs:
        if not (rec.get("language") or "").lower().startswith("py"):
            rec["style_violations"] = None
            continue
        res = scan_code(rec["code"], args.rules)
        rec["style_violations"] = res["style_violations"]
        rec["cwe_findings"] = (rec.get("cwe_findings") or []) + res["cwe_findings"]
        rec.setdefault("meta", {})["scanners"] = res["scanners"]
        tool_status = res["scanners"]
        if any((f.get("severity") in {"HIGH", "ERROR", "CRITICAL"}) for f in rec["cwe_findings"]):
            n_insecure += 1

    out = ensure_dir(args.out) / "records.jsonl"
    write_jsonl(out, recs)
    print(f"[static_checks] {len(recs)} records -> {out}  insecure(hi-sev)={n_insecure}  tools={tool_status}")


if __name__ == "__main__":
    main()
