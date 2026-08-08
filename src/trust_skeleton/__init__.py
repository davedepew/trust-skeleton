"""Trust Skeleton — a fixed-order epistemic contract for high-stakes AI reports.

Seven bones, FIXED ORDER. The order IS the enforcement: the rational spine
(observed / inferred / options / unknowns) and the statement of authority
render before the emotional channel ever speaks, and that ordering cannot
be inverted by any caller.

Fail-closed: a render missing WHO_DECIDES, or with inference language
contaminating OBSERVED, refuses to ship as high-stakes and falls back to
an ordinary reply with the refusal DISCLOSED. WEATHER (estimated emotional
or framing influence) ships only as estimates with quoted cues, or it is
withheld — it is never a diagnosis of a person.

Design rule carried from the system this was extracted from: perceived
trust is a diagnostic, never an objective. Nothing in this module or its
intended callers optimizes "the reader feels trust"; the skeleton earns
trust by construction or refuses.
"""
from __future__ import annotations

import re
from typing import Any

SKELETON_VERSION = "trust_skeleton.v1"
ORDER = ("OBSERVED", "INFERRED", "OPTIONS", "UNKNOWNS", "WEATHER",
         "WHO_DECIDES", "RECEIPT")

_INFERENCE_MARKERS = re.compile(
    r"(?i)\b(probably|likely|I (think|believe|suspect)|seems?|appears?|"
    r"suggests?|might|presumably|my guess)\b")
_DIAGNOSTIC = re.compile(
    r"(?i)\b(you (are|were|sound|seem)( \w+){0,2} (afraid|scared|angry|furious|"
    r"panick\w+|anxious|ashamed|desperate)|"
    r"(he|she|they) (is|are)( \w+){0,2} (afraid|angry|panick\w+)|"
    r"clinically|diagnos\w+|disorder)\b")
_DECISION_SHAPED = re.compile(
    r"(?i)\b(should (we|i)|decide|decision|choose|which option|go or no.?go|"
    r"renew or|approve|sign (the|this)|commit to)\b")
_STAKES_ADJACENT = re.compile(
    r"(?i)\b(money|\$\d|payment|invest|lease|contract|public(ly)?|launch|"
    r"press|hard.?line|irreversible|fire|hire)\b")


def is_high_stakes(text: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Should this exchange get the full skeleton ceremony?

    Fires on: decision-shaped text | manipulation_level >= 1 from your
    governance layer | an emotional_bias estimate with a decision_frame
    family or influence >= 0.35 | money/public/irreversibility adjacency.
    Mild lone affect cues do NOT force ceremony — chitchat stays chitchat.

    context keys (all optional):
      manipulation_level: int — from a manipulation-governance layer
      emotional_bias: {"influences": [{"family", "kind", "estimated_influence"}]}
        — compatible with the cognitive-governance traits package
    """
    ctx = context or {}
    t = text or ""
    reasons = []
    if _DECISION_SHAPED.search(t):
        reasons.append("decision_shaped")
    if int(ctx.get("manipulation_level") or 0) >= 1:
        reasons.append("manipulation_level")
    eb = ctx.get("emotional_bias") or {}
    infl = eb.get("influences") or []
    if any(i.get("kind") == "decision_frame" for i in infl) or \
       any(float(i.get("estimated_influence") or 0) >= 0.35 for i in infl):
        reasons.append("emotional_bias")
    if _STAKES_ADJACENT.search(t):
        reasons.append("stakes_adjacent")
    return {"high_stakes": bool(reasons), "reasons": reasons}


def _weather_gate(weather: Any) -> tuple[list, str | None]:
    """Estimates with quoted cues or nothing. Diagnostic language withholds."""
    if not weather:
        return [], None
    items = weather if isinstance(weather, list) else [weather]
    clean = []
    for w in items:
        if not isinstance(w, dict):
            return [], "weather withheld: non-schema weather entry"
        txt = f"{w.get('family', '')} {w.get('note', '')}"
        if _DIAGNOSTIC.search(txt):
            return [], "weather withheld: diagnostic language is not an estimate"
        cues = [c for c in (w.get("cues") or [])
                if (c.get("span") if isinstance(c, dict) else str(c)).strip()]
        if not cues:
            return [], f"weather withheld: family {w.get('family')!r} had no quoted cue"
        clean.append({"family": w.get("family"), "kind": w.get("kind", "affect"),
                      "estimated_influence": w.get("estimated_influence"),
                      "cues": cues})
    return clean, None


def compose(sections: dict[str, Any]) -> dict[str, Any]:
    """Render the skeleton. Returns {ok, render, skeleton_version, mode,
    refusals[]}. mode: high_stakes | refused_high_stakes_fallback."""
    s = {k.upper(): v for k, v in (sections or {}).items()}
    refusals: list[str] = []

    observed = [str(x) for x in (s.get("OBSERVED") or [])]
    contaminated = [o for o in observed if _INFERENCE_MARKERS.search(o)]
    if contaminated:
        refusals.append(f"OBSERVED contaminated with inference language: "
                        f"{contaminated[0][:80]!r}")
    if not observed:
        refusals.append("OBSERVED empty — a high-stakes render needs receipts")

    who = str(s.get("WHO_DECIDES") or "").strip()
    if not who:
        refusals.append("WHO_DECIDES missing — authority must be stated")

    if refusals:
        return {"ok": False, "mode": "refused_high_stakes_fallback",
                "skeleton_version": SKELETON_VERSION, "refusals": refusals,
                "note": "refused as high-stakes; caller falls back to ordinary "
                        "reply and DISCLOSES the refusal"}

    weather, withheld = _weather_gate(s.get("WEATHER"))

    lines = ["[OBSERVED]"] + [f"  - {o}" for o in observed]
    inferred = s.get("INFERRED") or []
    lines.append("[INFERRED]")
    if inferred:
        for i in inferred:
            if isinstance(i, dict):
                lines.append(f"  - {i.get('text')} (confidence "
                             f"{i.get('confidence', '?')})")
            else:
                lines.append(f"  - {i} (confidence unstated)")
    else:
        lines.append("  - none")
    lines.append("[OPTIONS]")
    for o in (s.get("OPTIONS") or []):
        if isinstance(o, dict):
            row = f"  - {o.get('label')}"
            if o.get("constraints"):
                row += f"  | constraints: {'; '.join(map(str, o['constraints']))}"
            lines.append(row)
        else:
            lines.append(f"  - {o}")
    lines.append("[UNKNOWNS]")
    unknowns = s.get("UNKNOWNS") or []
    if unknowns:
        for u in unknowns:
            if isinstance(u, dict):
                lines.append(f"  - {u.get('question')}"
                             + (f"  (io: {u['io_id']})" if u.get("io_id") else ""))
            else:
                lines.append(f"  - {u}")
    else:
        lines.append("  - none stated")
    lines.append("[WEATHER]")
    if withheld:
        lines.append(f"  - {withheld}")
    elif weather:
        for w in weather:
            cue = w["cues"][0]
            span = cue.get("span") if isinstance(cue, dict) else str(cue)
            lines.append(f"  - possible {w['family']} ({w['kind']}) "
                         f"~{w.get('estimated_influence')}: \"{span}\" "
                         f"(estimate, not a diagnosis)")
    else:
        lines.append("  - no affect/frame cues estimated")
    lines.append(f"[WHO DECIDES]\n  - {who}")
    receipt = s.get("RECEIPT")
    if receipt:
        lines.append(f"[RECEIPT]\n  - {receipt}")

    return {"ok": True, "mode": "high_stakes",
            "skeleton_version": SKELETON_VERSION,
            "render": "\n".join(lines),
            "weather_withheld": bool(withheld), "refusals": []}


def status() -> dict[str, Any]:
    return {"ok": True, "module": "trust_skeleton", "version": SKELETON_VERSION,
            "order": list(ORDER),
            "doctrine": "rational spine first, weather second, never inverted; "
                        "perceived trust is a diagnostic, never an objective"}


__all__ = ["compose", "is_high_stakes", "status", "ORDER", "SKELETON_VERSION"]
