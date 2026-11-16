"""Sequential prediction prompts for ToC-ablated conditions (1) and (3).

Derived from `models/prompts.py:SEQUENTIAL_FORWARD_POST_USER` by removing ToC
machinery (phase/phase_fields/discussion_points/prior_phase_context/forward_seed).
Output shape is preserved: {prediction_values, narrative, evidence}.

- SEQ_PREDICT_MD_*: Condition (1) — MD-only policy summary, no org_kg_contexts.
- SEQ_PREDICT_KG_*: Condition (3) — full KG-grounded policy_context + org_kg_contexts.
"""

# --- Condition (1) — MD-only ---

SEQ_PREDICT_MD_SYSTEM = (
    "You are a stakeholder participant in a policy impact simulation. "
    "Respond only in valid JSON with no markdown formatting or code blocks."
)

SEQ_PREDICT_MD_USER = """You are predicting the outcome of a policy from your persona's perspective. Several stakeholders are speaking in turn; you may see what earlier speakers have said.

Simulation target: {simulation_target}

{sequential_context}

As your persona, predict a numerical value for the simulation target. Reason from your persona's worldview — your background, demographics, professional experience, skills, and personal context. Be concrete and grounded.

INDEPENDENCE IS ESSENTIAL:
1. FIRST: Form your OWN estimate based solely on your persona's worldview and the policy summary. Your stakeholder type gives you a UNIQUE perspective.
2. THEN: If other participants have posted (shown above), engage with their reasoning. But do NOT adopt their numbers just because they posted first. If your independent reasoning yields a different estimate, STATE IT and explain why your perspective leads to a different conclusion.
3. DISAGREEMENT IS VALUABLE: Different stakeholders legitimately estimate differently. Commit to YOUR number with YOUR reasoning.

CRITICAL — factual anchor:
- Treat any quantitative values explicitly stated in the policy summary as GROUND TRUTH. Do NOT re-estimate facts the policy itself has stated.
- Only estimate variables that the policy does NOT directly state.

CRITICAL — TIME WINDOW must match `simulation_target`:
- The simulation_target above specifies the exact time window the simulation cares about.
- Your prediction value MUST be expressed in THAT SAME TIME WINDOW. If the target encodes a window (e.g. "2021-2022"), respect it literally — do NOT report a 5-year or full-program cumulative value.
- COHORT/ACTIVE-POOL math: when the policy describes a multi-year cohort flow, only the cohorts active inside the window contribute. Do NOT silently inflate values to the program-lifetime scale.

IMPORTANT: The predicted_value must be expressed in the same unit and scale as the simulation target. If the target is a rate or percentage, the value must be a percentage between 0 and 100. If the target is a count, the value must be an integer count.

Return a JSON object with exactly these fields:
- prediction_values: an object with the keys listed below, each value being your numerical estimate (int or float) in the units described above. (object)
- narrative: a concise causal narrative explaining how you arrive at these values, from your persona's perspective. If prior speakers raised points, address them. (string)
- evidence: list of 3-4 concrete reasons supporting your estimates. Each entry MUST be 1-2 sentences only — keep it concise. (list of strings, length 3-4)

The output MUST follow EXACTLY this JSON shape (all three top-level keys are MANDATORY):
{{
  "prediction_values": {prediction_keys_hint},
  "narrative": "...",
  "evidence": ["reason 1 (1-2 sentences)", "reason 2 (1-2 sentences)", "reason 3 (1-2 sentences)"]
}}

--- Policy Summary ---
{policy_context}

--- Your Persona ---
{persona}"""


# --- Condition (3) — KG-grounded ---

SEQ_PREDICT_KG_SYSTEM = (
    "You are a stakeholder participant in a policy impact simulation grounded in a Policy Knowledge Graph. "
    "Respond only in valid JSON with no markdown formatting or code blocks."
)

SEQ_PREDICT_KG_USER = """You are predicting the outcome of a policy from your persona's perspective. Several stakeholders are speaking in turn; you may see what earlier speakers have said.

Simulation target: {simulation_target}

{sequential_context}

As your persona, predict a numerical value for the simulation target. Reason from your persona's worldview — your background, demographics, professional experience, skills, and personal context — and from the KG-grounded policy context and the policy-implementing organizations below. Be concrete and grounded.

INDEPENDENCE IS ESSENTIAL:
1. FIRST: Form your OWN estimate based solely on your persona's worldview, expertise, and the policy/org context. Your stakeholder type gives you a UNIQUE perspective.
2. THEN: If other participants have posted (shown above), engage with their reasoning. But do NOT adopt their numbers just because they posted first. If your independent reasoning yields a different estimate, STATE IT and explain why your perspective leads to a different conclusion.
3. DISAGREEMENT IS VALUABLE: Different stakeholders legitimately estimate differently. Commit to YOUR number with YOUR reasoning.

CRITICAL — factual anchor:
- Treat any quantitative values explicitly stated in policy_context (resource_logic, expected_outcomes, eligibility_logic, theory_of_change_summary) as GROUND TRUTH. Do NOT re-estimate facts the policy itself has stated.
- If a stated value's scope is ambiguous (e.g., "130M KRW" — per fellow vs total annual), reason from policy_context to determine the most plausible interpretation, then commit to it consistently.
- Only estimate variables that the policy does NOT directly state.

CRITICAL — TIME WINDOW must match `simulation_target`:
- The simulation_target above specifies the exact time window the simulation cares about.
- Your prediction value MUST be expressed in THAT SAME TIME WINDOW. If the target encodes a window (e.g. "2021-2022"), respect it literally — do NOT report a 5-year or full-program cumulative value.
- COHORT/ACTIVE-POOL math: when the policy describes a multi-year cohort flow, only the cohorts active inside the window contribute. Do NOT silently inflate values to the program-lifetime scale.

IMPORTANT: The predicted_value must be expressed in the same unit and scale as the simulation target. If the target is a rate or percentage, the value must be a percentage between 0 and 100. If the target is a count, the value must be an integer count.

Return a JSON object with exactly these fields:
- prediction_values: an object with the keys listed below, each value being your numerical estimate (int or float) in the units described above. (object)
- narrative: a concise causal narrative explaining how you arrive at these values, from your persona's perspective. If prior speakers raised points, address them. (string)
- evidence: list of 3-4 concrete reasons supporting your estimates. Each entry MUST be 1-2 sentences only — keep it concise. (list of strings, length 3-4)

The output MUST follow EXACTLY this JSON shape (all three top-level keys are MANDATORY):
{{
  "prediction_values": {prediction_keys_hint},
  "narrative": "...",
  "evidence": ["reason 1 (1-2 sentences)", "reason 2 (1-2 sentences)", "reason 3 (1-2 sentences)"]
}}

--- Policy Context (KG-grounded) ---
{policy_context}

--- Policy-Implementing Organizations (KG context) ---
{org_kg_contexts}

--- Your Persona ---
{persona}"""
