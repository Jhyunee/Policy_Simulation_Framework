"""Prompts specific to Condition (1) Baseline — MD-driven Nemotron sampling.

`STAKEHOLDER_TO_DEMOGRAPHIC_HINTS` produces an (age_range, education_level
keywords, occupation keywords) hint set per stakeholder_type. These hints drive
a DuckDB filter on the country's Nemotron parquet. No KG, no scope_definer.
"""

STAKEHOLDER_TO_DEMOGRAPHIC_HINTS_SYSTEM = (
    "You are an expert policy analyst inferring realistic demographic envelopes "
    "for stakeholder types of a policy. Respond only in valid JSON with no "
    "markdown formatting or code blocks. All output string values must be in English."
)

STAKEHOLDER_TO_DEMOGRAPHIC_HINTS_USER = """Given a stakeholder group involved in a policy, infer the plausible demographic envelope (age range, education level keywords, occupation keywords) that real individuals of this stakeholder type are likely to have.

These hints will be used to filter rows from a national population dataset (Nemotron-Personas). Be neither too narrow (excludes valid individuals) nor too broad (loses signal).

Policy summary:
{policy_summary}

Stakeholder group:
- stakeholder_type: {stakeholder_type}
- representative_type: {representative_type}
- core_objectives: {core_objectives}
- core_constraints: {core_constraints}

Return a JSON object with EXACTLY this shape (no extra keys):
{{
  "age_range": [<int>, <int>],
  "education_level_keywords": ["<keyword1>", "<keyword2>", ...],
  "occupation_keywords": ["<keyword1>", "<keyword2>", ...]
}}

Rules:
- age_range: [min_inclusive, max_inclusive], realistic 10-20 year window for adult roles (e.g. researchers: [25, 60]; senior administrators: [40, 65]).
- education_level_keywords: 1-4 short keywords that may appear in the dataset's education_level column (e.g. "graduate school", "university", "PhD", "master"). For roles requiring tertiary education, include "university" and "graduate school".
- occupation_keywords: 2-6 short keywords that may appear in the occupation column. Use plain nouns (e.g. "researcher", "professor", "engineer", "administrator", "manager", "officer", "scientist"). Do NOT use the literal stakeholder_type string.
"""
