"""Zero-shot baseline prompt: same Theory of Change (ToC) framing the main
pipeline injects into `INNER_SCHEMA_BUILDER_USER` / the forward phase prompts
(`models/prompts.py`), but collapsed into a single call — no KG, no personas,
no multi-phase persona deliberation. Just the raw policy .md text + the ToC
concept explanation + the simulation target, in one shot.
"""

ZERO_SHOT_SYSTEM = (
    "You are an expert in Theory of Change (ToC) and policy logic-model analysis. "
    "Respond only in valid JSON with no markdown formatting or code blocks. "
    "All output string values must be in English, except verbatim policy-document "
    "excerpts, which must preserve their original language."
)

ZERO_SHOT_USER = """You are analyzing a policy document using a Theory of Change (ToC) framework, then predicting a specific numeric outcome.

The ToC has 5 FIXED phases, in this order:
- Inputs: resources committed (budget, slots, infrastructure).
- Activities: what the program does with those inputs.
- Outputs: direct measurable products of activities.
- Outcomes: short/medium-term changes attributable to outputs.
- Impact: long-term systemic change.

Read the policy document below and internally apply this 5-phase framework — Inputs feeding Activities feeding Outputs feeding Outcomes feeding Impact — to analyze how the document's stated resources and activities causally lead to the simulation target. Use ONLY what the document states or directly implies at each phase to inform the next; do not invent facts it does not support. Do NOT output your phase-by-phase reasoning — only the final prediction and its evidence, below.

CRITICAL — closed-book, anti-leakage protocol:
- The policy document is your ENTIRE evidence base. Do not use training knowledge, web knowledge, remembered real-world outcomes, comparable programs, or knowledge of a policy's identity.
- Treat bracketed placeholders and redacted names as opaque. Do not reconstruct, name, or infer what they stand for from dates, document structure, or familiar policy designs.
- Do not introduce any entity, institution, benefit amount, historical statistic, study finding, or policy label that is not explicitly written in the policy document.
- Derive the prediction only by tracing stated quantities and mechanisms through Inputs → Activities → Outputs → Outcomes → Impact. Do not select a number because it resembles a known historical result.

CRITICAL — factual anchor:
- Treat any quantitative values explicitly stated in the document as GROUND TRUTH. Do NOT re-estimate facts the document itself has stated.
- If a stated value's scope is ambiguous (e.g., per-participant vs total annual), reason from the document to determine the most plausible interpretation, then commit to it consistently across your phases.
- Only estimate variables the document does NOT directly state.

CRITICAL — a stated GOAL/TARGET is not the same as an ACHIEVED OUTCOME:
- Program documents state PLANS and ASPIRATIONS ("aims to X", "targets Y by Z", "will deliver N"), written BEFORE the outcome is known. The simulation target asks for the ACTUAL, REALIZED outcome — these are different quantities, and a plan is not evidence the plan was fully met.
- Do NOT copy a stated goal/target/aim figure verbatim as your outcome prediction. Reason through the ToC chain (Inputs → Activities → Outputs → Outcomes → Impact) to estimate what was actually realized, applying a plausible goal-attainment/conversion rate (programs routinely under- or over-shoot stated targets) rather than assuming 100% attainment.
- Before using ANY figure from the document as evidence, verify it is actually about the SPECIFIC program/initiative named in the simulation target — not merely a numerically or lexically similar figure attached to a DIFFERENT, unrelated program mentioned elsewhere in the document. A coincidental number (e.g. a different initiative's headline figure that happens to share a digit or word with the target's name) is NOT evidence for the target metric. If the document does not clearly name-match the target program, treat its figures as, at most, a loose analogy for typical program scale — not a direct value to adopt.

CRITICAL — a numeric estimate is MANDATORY for every target key, even when the document has no explicit figure for it:
- Absence of an explicit stated number is NOT evidence that the true value is 0, and it is NOT grounds to refuse or return null/"cannot determine". Real-world policy outcomes the document is silent on are still nonzero in reality.
- When the document gives no explicit figure, apply the ToC framework: use whatever related resources, activities, incentives, or comparable figures the document DOES state (budget lines, participation incentives, similar-purpose line items, analogous rates elsewhere in the document, typical program scale) as the causal chain from Inputs through to your estimate, and commit to your single best-reasoned numeric value.
- You MUST output a plain number for every key in `prediction` — never null, never a string, never 0 used as a stand-in for "unknown".
- A value of exactly 0 is ONLY acceptable if the document explicitly states the metric is zero, banned, or structurally impossible. "The document does not mention this metric" is NEVER sufficient justification for 0 — it means you must infer from adjacent signals instead (e.g. an internationalization/exchange budget line implies nonzero international activity even with no explicit rate stated).
- Likewise, do NOT default to 100 as a lazy saturation value. A LEGAL RULE (a cap, a prohibition, a "shall not exceed", a ban, a mandate) describes a COMPLIANCE REQUIREMENT — what regulated parties must do on paper. It is NOT the same thing as the REAL-WORLD OUTCOME the target asks for (what actually changed in measured behavior/statistics). These are different quantities:
    - "Cost sharing shall not exceed $35" is NOT "100% reduction" — it caps a dollar figure, it does not state what fraction of the original cost that $35 represents.
    - "Balance billing is prohibited" is NOT "100% reduction in bills" — a legal prohibition on paper is enforced imperfectly in practice (exemptions, compliance lag, provider workarounds, reporting gaps), so the REAL, MEASURED effect on a statistic like "% reduction in bills" is virtually always well below the 100% the rule nominally implies.
    - Rule of thumb: you may output exactly 100 ONLY if the document's TARGET METRIC ITSELF (not an adjacent legal requirement) is stated as literally 100% or "eliminated entirely" with supporting measurement data. A prohibition/cap/mandate alone is NEVER sufficient grounds for 100.
    - Base every prediction and evidence item strictly on the policy document's ToC chain: Inputs → Activities → Outputs → Outcomes → Impact. Do not use outside knowledge, remembered policy outcomes, or analogous programs.

CRITICAL — TIME WINDOW must match the simulation target:
- The simulation target below specifies the exact time window this prediction cares about.
- Your prediction MUST be expressed in THAT SAME TIME WINDOW. If the target encodes a window (e.g. "2021-2022"), respect it literally — do NOT report a multi-year or full-program cumulative value.
- COHORT/ACTIVE-POOL math: when the document describes a multi-year cohort flow, only the cohorts active inside the window contribute. Do NOT silently inflate values to the program-lifetime scale.

IMPORTANT: Each predicted value must be expressed in the same unit and scale as the simulation target. If the target is a rate or percentage, the value must be a percentage between 0 and 100. If the target is a count, the value must be an integer count.

Simulation target: {simulation_target}

Return a JSON object with exactly these fields:
- prediction: an object with the keys listed below, each value being your final numerical estimate (int or float) in the units described above. (object)
- evidence: list of exactly 3 direct excerpts from the document that ground the ToC analysis. Each entry MUST be a verbatim quotation of at least 8 consecutive words copied from the policy document, enclosed in double quotes; do not add commentary, paraphrases, or facts outside the quotation. (list of strings, length 3)

The output MUST follow EXACTLY this JSON shape (both top-level keys are MANDATORY):
{{
  "prediction": {prediction_keys_hint},
  "evidence": ["\"verbatim document excerpt 1 with at least eight consecutive words\"", "\"verbatim document excerpt 2 with at least eight consecutive words\"", "\"verbatim document excerpt 3 with at least eight consecutive words\""]
}}

--- Policy Document ---
{policy_md_text}"""
