"""Wire trust_skeleton to an LLM in ~30 lines.

The division of labor: the system prompt (system_prompt.md) asks the model
to emit seven skeleton sections as JSON. This script is the enforcement
side: compose() either renders the fixed-order report or refuses, and the
refusal is disclosed instead of silently degraded.

Runs with no API key: a canned model response stands in for the LLM call.
Swap `fake_llm` for your real client.
"""
from __future__ import annotations

import json

from trust_skeleton import compose, is_high_stakes

USER_REQUEST = "Should we renew the office lease this week? The window closes Friday."


def fake_llm(request: str) -> str:
    """Stand-in for your model call (Anthropic, OpenAI, local, any).

    Real version: send system_prompt.md as the system prompt and `request`
    as the user turn; return the text of the reply.
    """
    return json.dumps({
        "observed": ["Lease expires 2026-11-15 per amended agreement (receipt dr_x1)"],
        "inferred": [{"text": "landlord expects an early answer", "confidence": 0.6}],
        "options": [
            {"label": "renew this week", "constraints": ["locks rate for 24 months"]},
            {"label": "wait and negotiate", "constraints": ["risk of losing current rate"]},
        ],
        "unknowns": ["whether the Friday deadline is real or a pressure tactic"],
        "weather": [{"family": "urgency", "kind": "decision_frame",
                     "estimated_influence": 0.5,
                     "cues": [{"span": "The window closes Friday"}]}],
        "who_decides": "The owner decides; this is a recommendation",
        "receipt": "demo_001",
    })


def answer(request: str) -> str:
    trigger = is_high_stakes(request)
    if not trigger["high_stakes"]:
        return "(ordinary reply — no ceremony needed)"

    sections = json.loads(fake_llm(request))
    result = compose(sections)

    if result["ok"]:
        return result["render"]
    # Fail-closed, disclosed: the model's output did not meet the contract.
    return ("A full high-stakes report was refused: "
            + "; ".join(result["refusals"])
            + "\n(Falling back to an ordinary reply, with that disclosed.)")


if __name__ == "__main__":
    print(f"request: {USER_REQUEST}\n")
    print(answer(USER_REQUEST))
