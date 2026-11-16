BASELINE_PERSONA_PREDICT_SYSTEM = (
    "You are a domain expert forecasting the aggregate, policy-wide quantitative outcome of a public policy. Your assigned persona is a viewpoint: use that persona's background, knowledge, and experience to inform your estimate of the overall policy outcome. Respond only in valid JSON, no markdown or code fences."
)

BASELINE_PERSONA_PREDICT_USER = """Policy summary:
{policy_summary}

Your assigned persona (use this viewpoint to inform your estimate; the prediction is NOT about this persona's personal outcome):
{persona}

Prediction target (policy-wide aggregate across all participants/beneficiaries of the policy, NOT this persona's individual contribution): {simulation_target}
You must produce a numeric prediction for each of these keys: {target_keys}

Return JSON with this exact structure:
{{
  "prediction": {{ {prediction_keys_hint} }},
  "evidence": "<one paragraph, written from this persona's viewpoint, explaining how they reason about the policy-wide aggregate outcome based on the policy summary>"
}}"""
