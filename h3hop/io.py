"""JSONL / filesystem helpers (stdlib only)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


def read_records(src: str | Path) -> list[dict]:
    """Read a .jsonl file, or every *.jsonl under a directory."""
    src = Path(src)
    if src.is_file():
        return read_jsonl(src)
    if src.is_dir():
        recs: list[dict] = []
        for fp in sorted(src.glob("*.jsonl")):
            recs.extend(iter_jsonl(fp))
        return recs
    raise FileNotFoundError(f"no such file or directory: {src}")


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    path = Path(path)
    ensure_dir(path.parent)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n
