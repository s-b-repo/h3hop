"""Execute candidate code + its tests in an isolated subprocess.

Two backends:
  * subprocess (default): rlimits (CPU/memory/file) via `resource`, no inherited env secrets.
  * docker (--docker): `python:3.12-slim` with `--network none` for stronger isolation.

A test passes iff the combined `code + tests` script exits 0. Tests are expected to use
plain `assert`s (no pytest dependency required).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

DEFAULT_TIMEOUT = 10
# Address-space cap: generous enough not to break the CPython interpreter's virtual mappings,
# tight enough to bound runaway allocations. Tune per workload.
_MEM_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


def _limits():  # pragma: no cover - only meaningful on POSIX
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (DEFAULT_TIMEOUT, DEFAULT_TIMEOUT + 2))
        resource.setrlimit(resource.RLIMIT_AS, (_MEM_BYTES, _MEM_BYTES))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass


def run_python(code: str, tests: str | None, timeout: int = DEFAULT_TIMEOUT,
               use_docker: bool = False) -> dict:
    script = code if not tests else f"{code}\n\n# --- tests ---\n{tests}\n"
    if not tests:
        # No tests: at least require it to parse.
        try:
            compile(code, "<candidate>", "exec")
            return {"passed": None, "log": "no tests; syntax OK"}
        except SyntaxError as e:
            return {"passed": False, "log": f"SyntaxError: {e}"}

    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "candidate.py"
        fp.write_text(script, encoding="utf-8")
        if use_docker:
            return _run_docker(td, timeout)
        return _run_subprocess(fp, timeout)


def _run_subprocess(fp: Path, timeout: int) -> dict:
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        proc = subprocess.run(
            ["python3", str(fp)],
            capture_output=True, text=True, timeout=timeout,
            preexec_fn=_limits, env=env, cwd=fp.parent,
        )
        ok = proc.returncode == 0
        log = (proc.stdout + proc.stderr)[-4000:]
        return {"passed": ok, "log": log}
    except subprocess.TimeoutExpired:
        return {"passed": False, "log": f"timeout after {timeout}s"}
    except Exception as e:  # pragma: no cover
        return {"passed": False, "log": f"runner error: {e}"}


def _run_docker(td: str, timeout: int) -> dict:  # pragma: no cover - needs docker
    try:
        proc = subprocess.run(
            [
                "docker", "run", "--rm", "--network", "none",
                "--memory", "512m", "--cpus", "1", "--pids-limit", "128",
                "-v", f"{td}:/w:ro", "-w", "/w", "python:3.12-slim",
                "timeout", str(timeout), "python", "candidate.py",
            ],
            capture_output=True, text=True, timeout=timeout + 30,
        )
        return {"passed": proc.returncode == 0, "log": (proc.stdout + proc.stderr)[-4000:]}
    except FileNotFoundError:
        return {"passed": False, "log": "docker not available"}
    except subprocess.TimeoutExpired:
        return {"passed": False, "log": f"timeout after {timeout}s"}
