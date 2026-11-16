from __future__ import annotations

from typing import Any, Mapping


SIMULATION_PERSONA_FIELDS = (
    "persona_id",
    "name",
    "stakeholder_type",
    "role_prototype_id",
    "role_anchor_entity_id",
    "panel_position",
    "agent_modality",
    "age",
    "sex",
    "marital_status",
    "education_level",
    "occupation",
    "province",
    "district",
    "professional_persona",
    "skills_and_expertise",
    "career_goals_and_ambitions",
    "policy_role_context",
)

POLICY_ROLE_CONTEXT_FIELDS = (
    "role_label",
    "mechanism_functions",
    "objectives",
    "constraints",
    "actions",
    "causal_leverage_points",
    "toc_phase_relevance",
    "constraint_scope",
)


def compact_policy_role_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    source = context or {}
    return {
        key: source[key]
        for key in POLICY_ROLE_CONTEXT_FIELDS
        if source.get(key) not in (None, "", [], {})
    }


def sanitize_simulation_persona(record: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the fixed Stage 1-to-Stage 2 simulation persona contract."""
    compact = {
        key: record[key]
        for key in SIMULATION_PERSONA_FIELDS
        if key != "policy_role_context" and record.get(key) not in (None, "", [], {})
    }
    role_context = compact_policy_role_context(record.get("policy_role_context"))
    if role_context:
        compact["policy_role_context"] = role_context
    return compact
