"""Normalization helpers for local policy KG extraction."""

import re
from copy import deepcopy
from typing import Any


_QUOTE_CHARS = "\"'`()[]{}"


# Default node_family -> IAD 7 Rules-in-Use (Position/Boundary/Choice/Aggregation/Information/Payoff/Scope).
# Used as a fallback when ontology entity types or extracted node instances omit iad_rule_type.
DEFAULT_IAD_RULE_BY_FAMILY = {
    "ActorParticipant": "position",
    "PositionRole": "position",
    "EligibilityCondition": "boundary",
    "RestrictionConstraint": "boundary",
    "PolicyActionProcess": "choice",
    "InformationEvidence": "information",
    "ResourceBenefitCost": "payoff",
    "AuthorityControl": "aggregation",
    "TemporalMilestone": "scope",
    "EvaluationMetricOutcome": "scope",
}


def normalize_name(name: str) -> str:
    """Normalize an extracted node name while preserving readable text."""
    cleaned = (name or "").strip()
    cleaned = cleaned.strip(_QUOTE_CHARS)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def make_node_id(node_type: str, name: str) -> str:
    """Create a stable local node id from ontology type and normalized name."""
    return f"{node_type}:{normalize_name(name)}"


def canonical_node_name(node_type: str, name: str) -> str:
    """Collapse type-label aliases such as 'Postdoctoral Researcher' into the ontology type label."""
    cleaned = normalize_name(name)
    compact_name = re.sub(r"[^a-z0-9]", "", cleaned.lower())
    compact_type = re.sub(r"[^a-z0-9]", "", str(node_type or "").lower())
    if compact_name and compact_name in {compact_type, f"{compact_type}s"}:
        return str(node_type)
    return cleaned


def normalize_node(raw_node: dict[str, Any], type_meta: dict[str, Any], chunk_id: str | None = None) -> dict[str, Any]:
    node_type = raw_node.get("type") or type_meta.get("name")
    name = canonical_node_name(node_type, raw_node.get("name", ""))
    source_text = raw_node.get("source_text") or raw_node.get("source") or ""
    attributes = raw_node.get("attributes") if isinstance(raw_node.get("attributes"), dict) else {}

    return {
        "id": make_node_id(node_type, name),
        "name": name,
        "type": node_type,
        "iad_category": raw_node.get("iad_category") or type_meta.get("iad_category", "Participants"),
        "node_family": raw_node.get("node_family") or type_meta.get("node_family", ""),
        "iad_rule_type": raw_node.get("iad_rule_type") or type_meta.get("iad_rule_type"),
        "is_actor": bool(type_meta.get("is_actor", True)),
        "persona_eligible": bool(type_meta.get("persona_eligible", type_meta.get("is_actor", True))),
        "summary": raw_node.get("summary", ""),
        "attributes": attributes,
        "rationale": type_meta.get("rationale", ""),
        "source_spans": [{"chunk_id": chunk_id, "text": source_text}] if source_text else [],
    }


def merge_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate nodes by stable id and merge summaries, attributes, and source spans."""
    merged: dict[str, dict[str, Any]] = {}
    for node in nodes:
        node_id = node["id"]
        if node_id not in merged:
            merged[node_id] = deepcopy(node)
            continue

        existing = merged[node_id]
        if node.get("summary") and node["summary"] not in existing.get("summary", ""):
            if existing.get("summary"):
                existing["summary"] += " " + node["summary"]
            else:
                existing["summary"] = node["summary"]
        existing.setdefault("attributes", {}).update(node.get("attributes", {}))

        seen_spans = {(s.get("chunk_id"), s.get("text")) for s in existing.get("source_spans", [])}
        for span in node.get("source_spans", []):
            key = (span.get("chunk_id"), span.get("text"))
            if key not in seen_spans:
                existing.setdefault("source_spans", []).append(span)
                seen_spans.add(key)

    return list(merged.values())
