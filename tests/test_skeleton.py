"""Attack suite — every test hands the skeleton a dishonesty shape and
passes only on refusal or containment. Tests that do not attack are
decoration; this suite ships with the package on purpose."""
from __future__ import annotations

import time

from trust_skeleton import ORDER, SKELETON_VERSION, compose, is_high_stakes

GOOD = {
    "observed": [
        "Lease expires 2026-11-15 per amended agreement (receipt dr_x1)",
        "Waitlist for the premium tier is 14 people (system export)",
    ],
    "inferred": [{"text": "capacity is the binding constraint", "confidence": 0.7}],
    "options": [
        {"label": "expand coaching capacity",
         "constraints": ["no new monthly spend without the owner"]},
        {"label": "do nothing this quarter"},
    ],
    "unknowns": [{"question": "does the landlord intend to renegotiate",
                  "io_id": "inv_abc123"}],
    "weather": [{"family": "urgency", "kind": "decision_frame",
                 "estimated_influence": 0.45,
                 "cues": [{"span": "the window closes Friday"}]}],
    "who_decides": "The owner decides; this is a recommendation",
    "receipt": "dr_x1",
}


def test_seven_bones_render_in_fixed_order():
    r = compose(GOOD)
    assert r["ok"] and r["mode"] == "high_stakes"
    idx = [r["render"].find(f"[{k.replace('_', ' ')}]")
           for k in ("OBSERVED", "INFERRED", "OPTIONS", "UNKNOWNS", "WEATHER",
                     "WHO DECIDES", "RECEIPT")]
    assert all(i >= 0 for i in idx)
    assert idx == sorted(idx), "order inverted — never-invert broken"


def test_constraints_ride_inside_options_no_eighth_bone():
    r = compose(GOOD)
    assert "constraints: no new monthly spend" in r["render"]
    assert len(ORDER) == 7


def test_weather_is_estimate_with_quoted_cue():
    r = compose(GOOD)
    assert '"the window closes Friday"' in r["render"]
    assert "not a diagnosis" in r["render"]


def test_missing_who_decides_refuses():
    bad = dict(GOOD, who_decides="")
    r = compose(bad)
    assert r["ok"] is False
    assert r["mode"] == "refused_high_stakes_fallback"
    assert any("WHO_DECIDES" in x for x in r["refusals"])


def test_inference_smuggled_into_observed_refuses():
    bad = dict(GOOD, observed=["The landlord probably wants to renegotiate soon"])
    r = compose(bad)
    assert r["ok"] is False
    assert any("contaminated" in x for x in r["refusals"])


def test_diagnostic_weather_withheld_spine_still_ships():
    bad = dict(GOOD, weather=[{"family": "fear",
                               "note": "you are clearly afraid of losing it",
                               "cues": [{"span": "x"}]}])
    r = compose(bad)
    assert r["ok"] and r["weather_withheld"]
    assert "weather withheld" in r["render"]


def test_adjacency_diagnosis_still_caught():
    # adversarial: adverb between verb and emotion word must not slip through
    bad = dict(GOOD, weather=[{"family": "fear",
                               "note": "they are quite obviously panicking",
                               "cues": [{"span": "x"}]}])
    r = compose(bad)
    assert r["ok"] and r["weather_withheld"]


def test_weather_without_quoted_cue_withheld():
    bad = dict(GOOD, weather=[{"family": "urgency", "cues": []}])
    r = compose(bad)
    assert r["ok"] and r["weather_withheld"]


def test_trigger_fires_on_decision_text():
    t = is_high_stakes("Should we sign the lease contract this week?", {})
    assert t["high_stakes"] and "decision_shaped" in t["reasons"]


def test_chitchat_stays_quiet():
    t = is_high_stakes("Good morning, how are things running?", {})
    assert t["high_stakes"] is False


def test_mild_lone_affect_cue_stays_quiet():
    t = is_high_stakes("interesting thought", {"emotional_bias": {"influences": [
        {"family": "excitement", "kind": "affect", "estimated_influence": 0.3}]}})
    assert t["high_stakes"] is False


def test_decision_frame_family_fires_regardless_of_magnitude():
    t = is_high_stakes("hm", {"emotional_bias": {"influences": [
        {"family": "urgency", "kind": "decision_frame",
         "estimated_influence": 0.3}]}})
    assert t["high_stakes"]


def test_manipulation_level_forces_ceremony():
    t = is_high_stakes("nice weather today", {"manipulation_level": 1})
    assert t["high_stakes"] and "manipulation_level" in t["reasons"]


def test_deterministic_and_fast():
    t0 = time.perf_counter()
    for _ in range(50):
        compose(GOOD)
    ms = (time.perf_counter() - t0) * 1000 / 50
    assert compose(GOOD)["render"] == compose(GOOD)["render"]
    assert ms < 5.0
    assert compose(GOOD)["skeleton_version"] == SKELETON_VERSION
