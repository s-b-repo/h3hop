"""Single entry point for integrity detection, reused by reward, filtering, and eval."""
from __future__ import annotations

from .antipatterns import scan_antipatterns
from .cheating import scan_code_cheating, scan_trace
from .stego import scan_text_stego

DEFAULT_STEGO_THRESHOLD = 0.5
# Flag cheating only above this score, so weak context signals (a bare subprocess / network import)
# don't trip it alone — genuine single-signal cheats (git/curl/read-answers/fake-tool/hardcode) do.
CHEAT_THRESHOLD = 0.5


def scan_all(text: str, kind: str = "code", language: str | None = None,
             stego_threshold: float = DEFAULT_STEGO_THRESHOLD) -> dict:
    ap = scan_antipatterns(text, language) if kind == "code" else []
    ch = scan_code_cheating(text) if kind == "code" else {"score": 0.0, "events": []}
    st = scan_text_stego(text)
    return {
        "antipatterns": ap,
        "antipattern_count": len(ap),
        "cheating": ch,
        "cheating_flagged": ch["score"] >= CHEAT_THRESHOLD,
        "stego": st,
        "hidden_flagged": st["score"] >= stego_threshold,
    }


def annotate_record(rec: dict, stego_threshold: float = DEFAULT_STEGO_THRESHOLD) -> dict:
    """Add integrity verdicts to a data record (used by the filter step; read by schema.is_good)."""
    res = scan_all(rec.get("code", ""), "code", rec.get("language"), stego_threshold)
    rec["antipattern_count"] = res["antipattern_count"]
    rec["antipatterns"] = res["antipatterns"]
    rec["cheating_flagged"] = res["cheating_flagged"]
    rec["cheating"] = res["cheating"]
    rec["hidden_flagged"] = res["hidden_flagged"]
    rec["stego"] = res["stego"]
    return rec


# convenience re-export
scan_trace = scan_trace
