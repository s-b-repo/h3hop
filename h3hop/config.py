"""Config + environment loading."""
from __future__ import annotations

import os
from pathlib import Path


def load_yaml(path: str | Path) -> dict:
    try:
        import yaml  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "pyyaml is required to read configs: pip install pyyaml (or pip install -e .)"
        ) from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env() -> None:
    """Load .env if python-dotenv is available; otherwise rely on the real environment."""
    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()
    except Exception:
        pass


def require_env(name: str) -> str:
    load_env()
    val = os.environ.get(name)
    if not val:
        raise SystemExit(f"missing required environment variable: {name} (see .env.example)")
    return val
