# Trust Skeleton

A trust skeleton is a fixed-order epistemic contract for high-stakes AI
reports: what was observed, what was inferred, what the options are, what
remains unknown, what emotional weather may be influencing the exchange,
and who actually decides. The order is fixed and the renderer fails
closed, so a report cannot bury its facts under its feelings or ship
without naming its authority.

Extracted from a production system where every high-stakes recommendation
renders through it. Published with its own attack suite: every test hands
the skeleton a dishonesty shape and passes only when the skeleton refuses.

## The seven bones, in the only order they render

```
[OBSERVED]      facts with receipts. Inference language here REFUSES the render
[INFERRED]      interpretations, each with a stated confidence
[OPTIONS]       the real choices, constraints attached
[UNKNOWNS]      questions that would change the answer
[WEATHER]       estimated emotional/framing influence, quoted cues only,
                never a diagnosis of a person
[WHO DECIDES]   the human authority, stated plainly. Missing = REFUSED
[RECEIPT]       pointer to the decision record
```

The order is the enforcement. The rational spine and the statement of
authority always render before the emotional channel speaks, and no
caller can invert that.

## Sixty-second demo

```python
from trust_skeleton import compose, is_high_stakes

trigger = is_high_stakes("Should we sign the lease this week?")
# {'high_stakes': True, 'reasons': ['decision_shaped', 'stakes_adjacent']}

report = compose({
    "observed": ["Lease expires 2026-11-15 per amended agreement (receipt dr_x1)"],
    "inferred": [{"text": "capacity is the binding constraint", "confidence": 0.7}],
    "options": [{"label": "renew early", "constraints": ["no new spend without the owner"]},
                {"label": "wait for the landlord to move"}],
    "unknowns": ["does the landlord intend to renegotiate"],
    "weather": [{"family": "urgency", "kind": "decision_frame",
                 "estimated_influence": 0.45,
                 "cues": [{"span": "the window closes Friday"}]}],
    "who_decides": "The owner decides; this is a recommendation",
    "receipt": "dr_x1",
})
print(report["render"])
```

And the part that matters, what it refuses:

```python
compose({"observed": ["The landlord probably wants to renegotiate"],
         "who_decides": "The owner"})
# ok=False, mode='refused_high_stakes_fallback'
# refusals: ["OBSERVED contaminated with inference language: ..."]

compose({"observed": ["Lease expires 2026-11-15 (receipt dr_x1)"],
         "who_decides": ""})
# ok=False: "WHO_DECIDES missing - authority must be stated"
```

Weather with diagnostic language ("you are clearly afraid") is withheld
and the withholding is disclosed in the render. Estimates about influence
are allowed; diagnoses of people are not.

## Design rules this encodes

1. Facts and interpretations never share a section. Inference language in
   OBSERVED refuses the whole render rather than shipping a blend.
2. Authority is always named. A recommendation that does not say who
   decides is not a recommendation, it is a decision wearing a costume.
3. The emotional channel is real but subordinate. It renders as quoted,
   estimated influence after the spine, or not at all.
4. Refusals are disclosed, not silent. A refused high-stakes render falls
   back to an ordinary reply that says a full render was refused and why.
5. Perceived trust is a diagnostic, never an objective. Nothing here
   optimizes for the reader feeling reassured.

## Install

```
pip install trust-skeleton
```

Zero dependencies. Two functions and a status probe. Deterministic,
under a millisecond per render.

## Fits with

- [evidence-binding-compiler](https://github.com/davedepew/evidence-binding-compiler):
  claims bind to evidence or fail closed. Feeds OBSERVED.
- [cognitive-governance](https://github.com/davedepew/cognitive-governance):
  governed reasoning dispositions. Its emotional bias estimates plug into
  WEATHER via the `emotional_bias` context key, and a manipulation
  governance level plugs into the trigger via `manipulation_level`.

All three are extractions from the same production system. Each ships
with the attack suite that gates its releases.

## License

Apache 2.0. Copyright 2026 Dave DePew Enterprises, Inc.
