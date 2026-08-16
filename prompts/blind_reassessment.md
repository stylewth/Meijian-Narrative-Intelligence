<!-- prompt-version: v1 -->

# Blind reassessment

Use only the human-confirmed blind-test manuscript supplied in this request.
The manuscript is the only blind-test evidence that may be read.

Do not read, quote, reconstruct, or rely on the original blind-test files,
original binary artifacts, or any original candidate text. Do not output
candidate text, rewritten candidate text, editing instructions, or revision
suggestions. This is an evaluation pass only; it must not change a narrative.

Evaluate every supplied admitted candidate ID (`candidate_id`) exactly once. Do not add an ID,
omit an ID, merge IDs, or invent an ID. Return exactly one JSON object matching
the supplied `BlindReassessmentResult` schema and no prose outside that JSON.

For every candidate, provide the five scores and rationales for exactly these
dimensions: `evidence_strength`, `emotional_tension`,
`meijian_fit_and_exclusivity`, `competitor_difference`, and `scene_conversion`.
Set `weighted_score` to the exact rounded weighted sum defined by the schema
contract. Return unique contiguous ranks `1..N`, a business status, and a
short impact summary grounded only in the confirmed manuscript.
