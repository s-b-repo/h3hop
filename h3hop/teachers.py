"""OpenRouter teacher client with a hard budget cap and retries.

Heavy/optional imports (openai) are lazy so the package imports without them; only the
actual API path needs them.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .config import load_yaml, require_env


@dataclass
class CostMeter:
    cap_usd: float
    warn_usd: float = 0.0
    spent_usd: float = 0.0
    calls: int = 0
    _warned: bool = field(default=False, repr=False)

    def check(self) -> None:
        if self.spent_usd >= self.cap_usd:
            raise SystemExit(
                f"budget cap reached: ${self.spent_usd:.2f} >= ${self.cap_usd:.2f} — stopping."
            )

    def add(self, in_tok: int, out_tok: int, price_in: float, price_out: float) -> None:
        self.spent_usd += in_tok / 1e6 * price_in + out_tok / 1e6 * price_out
        self.calls += 1
        if self.warn_usd and not self._warned and self.spent_usd >= self.warn_usd:
            self._warned = True
            print(f"[cost] warning: ${self.spent_usd:.2f} of ${self.cap_usd:.2f} spent")


def _extract_json(text: str) -> dict:
    """Best-effort parse of a JSON object possibly wrapped in prose or a code fence."""
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                text = p
                break
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


class TeacherPool:
    def __init__(self, config_path: str):
        cfg = load_yaml(config_path)
        self.cfg = cfg
        self.defaults = cfg["defaults"]
        self.teachers = cfg["teachers"]
        self.meter = CostMeter(
            cap_usd=cfg["budget"]["total_usd_cap"],
            warn_usd=cfg["budget"].get("warn_at_usd", 0.0),
        )
        self._client = None
        self._base_url = cfg["openrouter"]["base_url"]
        self._api_key_env = cfg["openrouter"]["api_key_env"]

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # noqa: PLC0415
            except ImportError as e:  # pragma: no cover
                raise SystemExit("pip install openai to use the API path") from e
            self._client = OpenAI(
                base_url=self._base_url, api_key=require_env(self._api_key_env)
            )
        return self._client

    def pick(self, sample_idx: int) -> dict:
        """Round-robin weighted by teacher 'weight'."""
        expanded = []
        for t in self.teachers:
            expanded += [t] * max(1, int(round(t.get("weight", 1.0) * 10)))
        return expanded[sample_idx % len(expanded)]

    def generate(self, teacher: dict, messages: list[dict]) -> tuple[str, dict]:
        d = self.defaults
        last = None
        for attempt in range(d.get("max_retries", 5)):
            self.meter.check()
            try:
                resp = self.client.chat.completions.create(
                    model=teacher["model"],
                    messages=messages,
                    temperature=d["temperature"],
                    top_p=d.get("top_p", 1.0),
                    max_tokens=d["max_tokens"],
                    timeout=d.get("request_timeout_s", 180),
                )
                usage = resp.usage
                self.meter.add(
                    getattr(usage, "prompt_tokens", 0),
                    getattr(usage, "completion_tokens", 0),
                    teacher["price_per_mtok_in"],
                    teacher["price_per_mtok_out"],
                )
                return resp.choices[0].message.content, {"usage": usage.model_dump() if usage else {}}
            except SystemExit:
                raise
            except Exception as e:  # transient API error -> backoff
                last = e
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"teacher {teacher['name']} failed after retries: {last}")

    def generate_json(self, teacher: dict, messages: list[dict]) -> tuple[dict, dict]:
        text, meta = self.generate(teacher, messages)
        return _extract_json(text), meta
