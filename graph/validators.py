"""Validation and light recovery for local policy KG artifacts."""

import json
import re
from typing import Any

from graph.normalizer import canonical_node_name, make_node_id, normalize_name, DEFAULT_IAD_RULE_BY_FAMILY


def validate_policy_graph(
    graph: dict[str, Any],
    ontology: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a local policy graph while preserving inspectable LLM output.

    `config` is the same config dict consumed by the rest of the pipeline. The validator reads
    ``config["kg"]["edge_evidence_keywords"]`` (mapping of edge_type name -> list of cue tokens)
    and ``config["kg"]["negative_restrictive_cues"]`` (list of additional cue tokens) so that
    domain-specific cues live in config, not in this module.
    """
    initial_node_count = len(graph.get("nodes", []))
    initial_edge_count = len(graph.get("edges", []))
    warnings = list(graph.get("validation_warnings", []))
    warning_count_before = len(warnings)
    entity_types = {e["name"]: e for e in ontology.get("entity_types", [])}
    edge_types = {e["name"]: e for e in ontology.get("edge_types", [])}
    kg_config = (config or {}).get("kg", {}) if isinstance(config, dict) else {}
    edge_evidence_keywords = _build_edge_evidence_keywords(
        ontology, kg_config.get("edge_evidence_keywords")
    )
    negative_cues = _build_negative_restrictive_cues(
        ontology, kg_config.get("negative_restrictive_cues")
    )

    valid_nodes = []
    merged_nodes: list[dict[str, Any]] = []
    dropped_nodes: list[dict[str, Any]] = []
    for node in graph.get("nodes", []):
        if not node.get("id") and node.get("name"):
            node["id"] = make_node_id(node.get("type") or "UnclassifiedPolicyObject", node["name"])
        if not node.get("id") or not node.get("name"):
            warnings.append(f"Dropped node with no usable id/name: {node}")
            continue
        node_type = node.get("type")
        if node_type not in entity_types:
            warnings.append(f"Kept node with non-ontology type: {node.get('id') or node.get('name')} ({node_type})")
        meta = entity_types.get(node_type, {})
        node.setdefault("iad_category", meta.get("iad_category", "Participants"))
        node.setdefault("node_family", meta.get("node_family", ""))
        if not node.get("iad_rule_type"):
            node["iad_rule_type"] = meta.get("iad_rule_type") or DEFAULT_IAD_RULE_BY_FAMILY.get(node.get("node_family"))
        node.setdefault("is_actor", meta.get("is_actor", True))
        node.setdefault("persona_eligible", meta.get("persona_eligible", False))
        node.setdefault("attributes", {})
        node.setdefault("source_spans", [])
        _repair_non_actor_policy_object_type(node, warnings)
        if _looks_like_non_actor_policy_object(node):
            node["is_actor"] = False
            node["persona_eligible"] = False
            if node.get("iad_category") in {"Participants", "Positions"}:
                node["iad_category"] = "Costs_and_Benefits"
            if node.get("node_family") in {"ActorParticipant", "PositionRole", ""}:
                node["node_family"] = "ResourceBenefitCost"
            warnings.append(f"Demoted support/policy-object node from persona anchor: {node.get('id')}")
        _repair_iad_category(node, warnings)
        _ensure_actor_methodology_attributes(node)
        valid_nodes.append(node)

    node_ids = {n["id"] for n in valid_nodes}
    valid_edges = []
    dropped_edges: list[dict[str, Any]] = []
    for index, edge in enumerate(graph.get("edges", []), 1):
        edge_type = edge.get("type") or "UNSPECIFIED_RELATION"
        edge["type"] = edge_type
        if edge_type not in edge_types:
            warnings.append(f"Kept edge with non-ontology type: {edge_type}")
        if edge.get("source") not in node_ids or edge.get("target") not in node_ids:
            for side in ("source", "target"):
                endpoint = edge.get(side)
                if endpoint in node_ids:
                    continue
                placeholder = _placeholder_node(endpoint, side)
                edge[side] = placeholder["id"]
                if placeholder["id"] not in node_ids:
                    valid_nodes.append(placeholder)
                    node_ids.add(placeholder["id"])
                    warnings.append(f"Recovered unresolved edge endpoint as placeholder: {placeholder['id']}")
                else:
                    warnings.append(f"Canonicalized unresolved edge endpoint to existing node: {placeholder['id']}")

        meta = edge_types.get(edge_type, {})
        if not edge.get("id"):
            edge["id"] = f"edge_{index:04d}"
        edge.setdefault("edge_category", meta.get("edge_category", "structural"))
        edge.setdefault("edge_family", meta.get("edge_family", ""))
        edge.setdefault("attributes", {})
        edge.setdefault("fact", "")
        _migrate_deprecated_condition_fields(edge)
        if edge["edge_category"] == "regulative":
            edge["deontic"] = edge.get("deontic") or meta.get("ig_deontic")
            if edge["deontic"] not in {"must", "can", "cannot"}:
                _drop_edge(
                    edge,
                    "invalid_regulative_deontic",
                    "Regulative edge is missing a valid deontic value.",
                    dropped_edges,
                    warnings,
                )
                continue
        else:
            edge["deontic"] = None

        evidence_drop_reason = _edge_evidence_drop_reason(edge, edge_evidence_keywords, negative_cues)
        if evidence_drop_reason:
            _drop_edge(
                edge,
                "unsupported_source_evidence",
                evidence_drop_reason,
                dropped_edges,
                warnings,
            )
            continue

        role_drop_reason = _role_edge_assignment_drop_reason(
            edge,
            {node["id"]: node for node in valid_nodes},
        )
        if role_drop_reason:
            _drop_edge(
                edge,
                "invalid_role_scope",
                role_drop_reason,
                dropped_edges,
                warnings,
            )
            continue

        _mark_edge_supported(edge)
        valid_edges.append(edge)

    pre_merge_node_count = len(valid_nodes)
    valid_nodes, valid_edges, merged_nodes = _merge_alias_nodes(valid_nodes, valid_edges, warnings)
    pre_dedupe_edge_count = len(valid_edges)
    valid_edges = _dedupe_semantic_edges(valid_edges, warnings)
    valid_nodes = _drop_empty_unreferenced_placeholders(valid_nodes, valid_edges, warnings)
    valid_nodes, dropped_nodes = _drop_unreferenced_nodes(valid_nodes, valid_edges, warnings)
    _promote_supported_edge_source_actors(valid_nodes, valid_edges, warnings)
    _enforce_persona_anchor_scope(valid_nodes, valid_edges, warnings)
    _renumber_edges(valid_edges)

    graph["nodes"] = valid_nodes
    graph["edges"] = valid_edges
    graph.setdefault("metadata", {})["validation"] = {
        "initial_node_count": initial_node_count,
        "validated_node_count": len(valid_nodes),
        "pre_merge_node_count": pre_merge_node_count,
        "initial_edge_count": initial_edge_count,
        "validated_edge_count": len(valid_edges),
        "node_merge_count": len(merged_nodes),
        "orphan_node_drop_count": len(dropped_nodes),
        "deduplicated_edge_count": pre_dedupe_edge_count - len(valid_edges),
        "evidence_alignment_warning_count": 0,
        "role_assignment_warning_count": 0,
        "dropped_edge_count": len(dropped_edges),
        "dropped_unsupported_evidence_edge_count": sum(
            1 for item in dropped_edges
            if item.get("reason_code") == "unsupported_source_evidence"
        ),
        "dropped_role_scope_edge_count": sum(
            1 for item in dropped_edges
            if item.get("reason_code") == "invalid_role_scope"
        ),
        "warnings_added": len(warnings) - warning_count_before,
        "policy": (
            "Conservative validation: preserve source-backed nodes, repair schema/category mismatches, "
            "merge alias nodes, deduplicate semantically repeated edges, prune unreferenced non-authoritative nodes, "
            "and exclude unsupported or role-misaligned edges from the authoritative KG while retaining rejected "
            "candidates in the validation audit."
        ),
    }
    graph.setdefault("metadata", {})["validation_audit"] = {
        "dropped_edges": dropped_edges,
        "dropped_nodes": dropped_nodes,
        "merged_nodes": merged_nodes,
        "audit_policy": (
            "Dropped and merged candidates are not part of the final authoritative KG. They are retained only "
            "for inspection and prompt/ontology improvement."
        ),
    }
    graph["validation_warnings"] = warnings
    return graph


def _migrate_deprecated_condition_fields(edge: dict[str, Any]) -> None:
    """Accept legacy condition_text while normalizing authoritative edges to Cac/Cex."""
    attributes = edge.setdefault("attributes", {})
    activation = _normalize_nullable(
        edge.get("activation_condition")
        or edge.get("activationCondition")
        or attributes.get("activation_condition")
    )
    execution = _normalize_nullable(
        edge.get("execution_constraint")
        or edge.get("executionConstraint")
        or attributes.get("execution_constraint")
    )
    legacy_condition = _normalize_nullable(_legacy_condition_text(edge, attributes))
    if legacy_condition:
        attributes.setdefault("deprecated_condition_text", legacy_condition)
        if not activation and not execution:
            activation, execution = _split_legacy_condition_text(
                str(legacy_condition),
                edge_type=str(edge.get("type") or ""),
            )
    edge["activation_condition"] = activation
    edge["execution_constraint"] = execution
    edge.pop("condition_text", None)


def _legacy_condition_text(edge: dict[str, Any], attributes: dict[str, Any]) -> Any:
    for container in (edge, attributes):
        for key in ("condition_text", "condition", "conditions"):
            value = container.get(key)
            if value in (None, "", [], {}):
                continue
            if isinstance(value, list):
                return "; ".join(str(item) for item in value if item not in (None, ""))
            if isinstance(value, dict):
                return json.dumps(value, ensure_ascii=False)
            return value
    return None


def _normalize_nullable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "null", "none", "n/a"}:
        return None
    return value


def _split_legacy_condition_text(condition_text: str, edge_type: str = "") -> tuple[str | None, str | None]:
    clauses = _condition_clauses(condition_text)
    activation_parts: list[str] = []
    execution_parts: list[str] = []
    for clause in clauses:
        if _looks_like_execution_constraint(clause, edge_type) and not _looks_like_activation_condition(clause, edge_type):
            execution_parts.append(clause)
        else:
            activation_parts.append(clause)
    return _join_condition_parts(activation_parts), _join_condition_parts(execution_parts)


def _condition_clauses(condition_text: str) -> list[str]:
    clauses = [
        part.strip()
        for part in re.split(r"\s*;\s*|\s+\|\s+|\n+", condition_text)
        if part.strip()
    ]
    return clauses or [condition_text.strip()]


def _join_condition_parts(parts: list[str]) -> str | None:
    joined = "; ".join(part for part in parts if part)
    return joined or None


_ACTIVATION_EDGE_TYPES = {
    "CAN_APPLY",
    "CANNOT_APPLY",
    "ELIGIBLE_FOR",
    "INELIGIBLE_FOR",
    "EXCLUDES",
    "TARGETS",
    "RECEIVES",
}

_EXECUTION_EDGE_TYPES = {
    "LIMITED_TO",
    "MUST_SUBMIT",
    "MUST_REPORT",
    "REQUIRES_APPROVAL",
    "REQUIRES_DOCUMENT",
}

_ACTIVATION_CONDITION_CUES = (
    "after selection",
    "applicant",
    "as of",
    "candidate",
    "degree",
    "eligible",
    "eligibility",
    "if ",
    "ineligible",
    "non-tenure",
    "phd",
    "qualification",
    "selected",
    "target",
    "when ",
    "within seven years",
    "within 7 years",
    "공고일",
    "대상",
    "박사",
    "비전임",
    "선정된",
    "이내인",
    "자격",
)

_EXECUTION_CONSTRAINT_CUES = (
    "annually",
    "approval",
    "approved",
    "deadline",
    "document",
    "every year",
    "format",
    "online",
    "per year",
    "prior approval",
    "quarterly",
    "submit",
    "submitted",
    "within 30",
    "within one month",
    "written",
    "매년",
    "분기",
    "서면",
    "승인",
    "온라인",
    "이내에",
    "제출",
)


def _looks_like_activation_condition(clause: str, edge_type: str) -> bool:
    edge_type = edge_type.upper()
    lowered = clause.lower()
    return edge_type in _ACTIVATION_EDGE_TYPES or any(cue in lowered for cue in _ACTIVATION_CONDITION_CUES)


def _looks_like_execution_constraint(clause: str, edge_type: str) -> bool:
    edge_type = edge_type.upper()
    lowered = clause.lower()
    return edge_type in _EXECUTION_EDGE_TYPES or any(cue in lowered for cue in _EXECUTION_CONSTRAINT_CUES)


def _repair_iad_category(node: dict[str, Any], warnings: list[str]) -> None:
    """Repair common IAD category / node family mismatches without changing raw evidence."""
    original = node.get("iad_category")
    family = node.get("node_family")
    is_actor = bool(node.get("is_actor") or node.get("persona_eligible"))

    if family == "PositionRole":
        node["iad_category"] = "Positions"
    elif family == "ActorParticipant" or is_actor:
        node["iad_category"] = "Participants"
        node["node_family"] = family or "ActorParticipant"
    elif family == "PolicyActionProcess":
        node["iad_category"] = "Actions"
    elif family in {"InformationEvidence", "EligibilityCondition", "EvaluationMetricOutcome"}:
        node["iad_category"] = "Information"
    elif family in {"ResourceBenefitCost", "RestrictionConstraint", "TemporalMilestone"}:
        node["iad_category"] = "Costs_and_Benefits"

    if original != node.get("iad_category"):
        warnings.append(
            f"Repaired IAD category for {node.get('id')}: {original} -> {node.get('iad_category')}"
        )


_NON_ACTOR_TYPE_REPAIRS = {
    "ResearchGrant": ("Costs_and_Benefits", "ResourceBenefitCost"),
    "FundingLimit": ("Costs_and_Benefits", "ResourceBenefitCost"),
    "FundingDisbursement": ("Costs_and_Benefits", "ResourceBenefitCost"),
    "EligibilityCriteria": ("Costs_and_Benefits", "EligibilityCondition"),
    "ResearchCompliance": ("Costs_and_Benefits", "RestrictionConstraint"),
    "ResearchProposal": ("Information", "InformationEvidence"),
    "ApplicationNotification": ("Information", "InformationEvidence"),
    "PolicyInformation": ("Information", "InformationEvidence"),
    "EvaluationCriteria": ("Information", "EvaluationMetricOutcome"),
    "ResearchOutcome": ("Information", "EvaluationMetricOutcome"),
    "ApplicationProcess": ("Actions", "PolicyActionProcess"),
    "ResearchMonitoring": ("Actions", "PolicyActionProcess"),
}


def _repair_non_actor_policy_object_type(node: dict[str, Any], warnings: list[str]) -> None:
    """Correct ontology-generated non-actor types that were mislabeled as participants."""
    node_type = str(node.get("type") or "")
    repair = _NON_ACTOR_TYPE_REPAIRS.get(node_type)
    if not repair:
        return
    if node.get("is_actor") or node.get("persona_eligible"):
        return
    original = (node.get("iad_category"), node.get("node_family"))
    node["iad_category"], node["node_family"] = repair
    if original != repair:
        warnings.append(
            f"Repaired non-actor node family for {node.get('id')}: "
            f"{original[0]}/{original[1]} -> {repair[0]}/{repair[1]}"
        )


def _ensure_actor_methodology_attributes(node: dict[str, Any]) -> None:
    """Populate Stage 2 handoff hints for persona-eligible actors."""
    if not (node.get("is_actor") or node.get("persona_eligible")):
        return
    attributes = node.setdefault("attributes", {})
    direct_beneficiary_actor = _looks_like_direct_beneficiary_actor(node)
    if direct_beneficiary_actor:
        if _looks_like_program_label(node.get("name")):
            node["name"] = _humanize_actor_type(node.get("type"))
            node["summary"] = _repair_program_actor_summary(node.get("summary"))
        attributes["agent_modality"] = "individual"
        attributes["role_family"] = "direct_beneficiary"
        return

    inferred_agent_modality = _infer_agent_modality(node)
    configured_agent_modality = str(attributes.get("agent_modality") or "").strip()
    if (
        configured_agent_modality == "individual"
        and inferred_agent_modality in {"organization", "official_position_holder", "committee_member"}
    ):
        attributes["agent_modality"] = inferred_agent_modality
    else:
        attributes.setdefault("agent_modality", inferred_agent_modality)
    inferred_role_family = _infer_role_family(node)
    configured_role_family = _normalize_role_family(attributes.get("role_family"))
    if configured_role_family == "direct_beneficiary" and inferred_role_family != "direct_beneficiary":
        attributes["role_family"] = inferred_role_family
    elif configured_role_family == "affected_actor" and inferred_role_family != "affected_actor":
        attributes["role_family"] = inferred_role_family
    else:
        attributes["role_family"] = configured_role_family or inferred_role_family
    if node.get("is_actor") and not node.get("persona_eligible") and _looks_like_persona_anchor_actor(node):
        node["persona_eligible"] = True


def _normalize_role_family(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"direct_beneficiary", "beneficiary", "applicant", "fellow"}:
        return "direct_beneficiary"
    if text in {"evaluator", "evaluation", "reviewer", "committee_member", "panel_member"}:
        return "evaluator"
    if text in {"host_organization", "host", "institution", "research_institution", "research_organization"}:
        return "host_organization"
    if text in {"policy_administrator", "administrator", "program_officer", "funding_organization", "funder", "funding_agency"}:
        return "policy_administrator"
    if text in {"mentor", "mentoring"}:
        return "mentor"
    if text in {"pipeline_actor", "pipeline", "student", "trainee"}:
        return "pipeline_actor"
    if text in {"affected_actor", "affected", "stakeholder"}:
        return "affected_actor"
    return text or "affected_actor"


def _looks_like_direct_beneficiary_actor(node: dict[str, Any]) -> bool:
    text = _node_text(node)
    entity_type = str(node.get("type") or "").lower()
    if any(
        term in text
        for term in (
            "researchinstitution",
            "research institution",
            "host organization",
            "institution",
            "university",
            "institute",
            "foundation",
            "agency",
            "ministry",
            "committee",
            "panel",
            "authority",
        )
    ):
        return False
    if any(term in text for term in ("postdoctoral", "postdoc", "beneficiary", "applicant", "candidate")):
        return True
    if "fellow" in entity_type and not any(
        term in text
        for term in ("foundation", "agency", "institution", "ministry", "committee", "panel", "program officer")
    ):
        return True
    if "fellow" in text and any(term in text for term in ("selected", "young scientist", "researcher", "recipient")):
        return True
    return False


def _looks_like_program_label(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return any(term in text for term in ("program", "project", "scheme", "fellowship"))


def _humanize_actor_type(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "Direct Beneficiary"
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", text).strip()
    if words.endswith("Fellow"):
        return f"{words}s"
    return words


def _repair_program_actor_summary(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "Direct beneficiary actor role repaired from a program-label extraction."
    lowered = text.lower()
    if "program" in lowered or "fellowship" in lowered:
        return "Direct beneficiary actor role in the policy; repaired from a program-label extraction."
    return text


def _infer_agent_modality(node: dict[str, Any]) -> str:
    text = _node_text(node)
    if any(term in text for term in ("panel", "committee", "review board", " rb")):
        return "committee_member"
    if any(term in text for term in ("institution", "university", "institute", "host", "ministry", "foundation", "agency")):
        return "organization"
    if any(term in text for term in ("authority", "approvalauthority", "officer", "reviewer", "administrator")):
        return "official_position_holder"
    if any(term in text for term in ("group", "population", "students", "applicants", "candidates")):
        return "stakeholder_group_member"
    return "individual"


def _infer_role_family(node: dict[str, Any]) -> str:
    text = _node_text(node)
    if any(term in text for term in ("evaluation", "panel", "reviewer", "committee", "review board", " rb")):
        return "evaluator"
    if any(term in text for term in ("institution", "university", "institute", "host")):
        return "host_organization"
    if any(term in text for term in ("ministry", "foundation", "agency", "program officer", "administrator", "authority", "approval")):
        return "policy_administrator"
    if any(term in text for term in ("mentor", "mentoring")):
        return "mentor"
    if any(term in text for term in ("student", "pipeline", "trainee")):
        return "pipeline_actor"
    if any(term in text for term in ("postdoctoral", "researcher", "fellow", "beneficiary", "applicant", "candidate")):
        return "direct_beneficiary"
    return "affected_actor"


def _looks_like_persona_anchor_actor(node: dict[str, Any]) -> bool:
    text = _node_text(node)
    if any(term in text for term in ("notification", "applicationnotification")):
        return False
    return any(
        term in text
        for term in (
            "researcher",
            "institution",
            "foundation",
            "ministry",
            "agency",
            "authority",
            "committee",
            "panel",
            "officer",
            "mentor",
        )
    )


def _node_text(node: dict[str, Any]) -> str:
    return " ".join(
        str(value or "")
        for value in (
            node.get("id"),
            node.get("name"),
            node.get("type"),
            node.get("summary"),
        )
    ).lower()


def _merge_alias_nodes(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge obvious alias nodes created by generic type labels or repeated funding labels."""
    merged: dict[str, dict[str, Any]] = {}
    id_map: dict[str, str] = {}
    audit_records: list[dict[str, Any]] = []

    for node in nodes:
        original_id = node.get("id")
        canonical_name = _canonical_validation_name(node)
        canonical_id = make_node_id(str(node.get("type") or "UnclassifiedPolicyObject"), canonical_name)
        id_map[str(original_id)] = canonical_id
        normalized_node = dict(node)
        normalized_node["name"] = canonical_name
        normalized_node["id"] = canonical_id

        if canonical_id not in merged:
            merged[canonical_id] = normalized_node
            if original_id != canonical_id:
                audit_records.append(_node_merge_record(node, normalized_node, "canonical_name_repair"))
                warnings.append(f"Canonicalized node {original_id} -> {canonical_id}")
            continue

        kept = merged[canonical_id]
        _merge_node_payload(kept, normalized_node)
        audit_records.append(_node_merge_record(node, kept, "alias_duplicate"))
        warnings.append(f"Merged alias node {original_id} into {canonical_id}")

    for edge in edges:
        for side in ("source", "target"):
            endpoint = edge.get(side)
            if endpoint in id_map:
                edge[side] = id_map[endpoint]

    return list(merged.values()), edges, audit_records


def _canonical_validation_name(node: dict[str, Any]) -> str:
    node_type = str(node.get("type") or "")
    name = normalize_name(str(node.get("name") or node_type))
    type_label = canonical_node_name(node_type, name)
    if type_label != name:
        return type_label
    compact = re.sub(r"[^a-z0-9]+", "", name.lower())
    if node_type == "ResearchGrant" and any(marker in compact for marker in ("researchgrant", "researchfunding", "funding")):
        return "Research Grant"
    if node_type == "ResearchProposal" and compact in {"researchproposal", "proposal"}:
        return "Research Proposal"
    if node_type == "EligibilityCriteria" and compact in {"eligibilitycriteria", "eligibility"}:
        return "Eligibility Criteria"
    if node_type == "FundingLimit" and compact in {"fundinglimit", "limit"}:
        return "Funding Limit"
    if node_type == "ResearchCompliance" and compact in {"researchcompliance", "compliance"}:
        return "Research Compliance"
    if node_type == "EvaluationCriteria" and compact in {"evaluationcriteria", "criteria"}:
        return "Evaluation Criteria"
    if node_type == "ResearchMonitoring" and compact in {"researchmonitoring", "monitoring"}:
        return "Research Monitoring"
    if node_type == "EvaluationPanel" and compact in {"evaluationpanel", "dedicatedevaluationpanel"}:
        return "Evaluation Panel"
    return name


def _merge_node_payload(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    if incoming.get("summary") and incoming["summary"] not in str(target.get("summary") or ""):
        target["summary"] = " ".join(value for value in (target.get("summary"), incoming["summary"]) if value)
    target_attributes = target.setdefault("attributes", {})
    incoming_attributes = incoming.get("attributes") or {}
    target_role_family = _normalize_role_family(target_attributes.get("role_family"))
    incoming_role_family = _normalize_role_family(incoming_attributes.get("role_family"))
    target_attributes.update(incoming_attributes)
    preferred_role_family = _preferred_role_family(target_role_family, incoming_role_family)
    if preferred_role_family:
        target_attributes["role_family"] = preferred_role_family
    target["is_actor"] = bool(target.get("is_actor") or incoming.get("is_actor"))
    target["persona_eligible"] = bool(target.get("persona_eligible") or incoming.get("persona_eligible"))
    target.setdefault("source_spans", [])
    seen_spans = {(span.get("chunk_id"), span.get("text")) for span in target.get("source_spans", [])}
    for span in incoming.get("source_spans", []):
        key = (span.get("chunk_id"), span.get("text"))
        if key not in seen_spans:
            target["source_spans"].append(span)
            seen_spans.add(key)


_ROLE_FAMILY_PRIORITY = {
    "policy_administrator": 7,
    "evaluator": 7,
    "host_organization": 6,
    "direct_beneficiary": 6,
    "mentor": 5,
    "pipeline_actor": 4,
    "affected_actor": 1,
}


def _preferred_role_family(left: str | None, right: str | None) -> str | None:
    if not left:
        return right
    if not right:
        return left
    if left == right:
        return left
    return left if _ROLE_FAMILY_PRIORITY.get(left, 0) >= _ROLE_FAMILY_PRIORITY.get(right, 0) else right


def _node_merge_record(original: dict[str, Any], kept: dict[str, Any], reason_code: str) -> dict[str, Any]:
    return {
        "node_id": original.get("id"),
        "name": original.get("name"),
        "type": original.get("type"),
        "merged_into": kept.get("id"),
        "reason_code": reason_code,
    }


def _dedupe_semantic_edges(edges: list[dict[str, Any]], warnings: list[str]) -> list[dict[str, Any]]:
    """Merge repeated edges when they express the same relation between the same endpoints."""
    deduped: list[dict[str, Any]] = []
    buckets: dict[tuple[str, str, str, Any], list[int]] = {}

    for edge in edges:
        bucket_key = (
            str(edge.get("type") or ""),
            str(edge.get("source") or ""),
            str(edge.get("target") or ""),
            edge.get("deontic"),
        )
        duplicate_index = None
        for candidate_index in buckets.get(bucket_key, []):
            if _edges_are_semantic_duplicates(edge, deduped[candidate_index]):
                duplicate_index = candidate_index
                break

        if duplicate_index is None:
            buckets.setdefault(bucket_key, []).append(len(deduped))
            deduped.append(edge)
            continue

        kept = _prefer_edge(deduped[duplicate_index], edge)
        dropped = edge if kept is deduped[duplicate_index] else deduped[duplicate_index]
        kept.setdefault("attributes", {})
        duplicate_ids = list(kept["attributes"].get("duplicate_edge_ids", []))
        duplicate_ids.append(dropped.get("id"))
        kept["attributes"]["duplicate_edge_ids"] = [value for value in duplicate_ids if value]
        kept["attributes"]["merged_duplicate_edge_count"] = len(kept["attributes"]["duplicate_edge_ids"])
        deduped[duplicate_index] = kept
        warnings.append(
            f"Merged duplicate edge {dropped.get('id')} into {kept.get('id')}: "
            f"{kept.get('type')} {kept.get('source')} -> {kept.get('target')}"
        )

    return deduped


def _edges_are_semantic_duplicates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_fact = _normalized_sentence(left.get("fact"))
    right_fact = _normalized_sentence(right.get("fact"))
    if left_fact and left_fact == right_fact:
        return True

    left_text = _edge_semantic_text(left)
    right_text = _edge_semantic_text(right)

    if not left_text and not right_text:
        return True
    if not left_text or not right_text:
        return False

    return _token_similarity(left_text, right_text) >= 0.72


def _normalized_sentence(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return text.rstrip(".")


def _edge_semantic_text(edge: dict[str, Any]) -> str:
    activation = edge.get("activation_condition") or ""
    execution = edge.get("execution_constraint") or ""
    fact = edge.get("fact") or ""
    source_text = edge.get("source_text") or ""
    return f"{activation} {execution} {fact} {source_text}".strip()


# Language-agnostic English seed keywords by edge type. Used as a baseline that gets augmented at
# validate time with (a) tokens automatically derived from the ontology's edge_types descriptions
# and (b) per-policy overrides supplied via config["kg"]["edge_evidence_keywords"]. Korean / domain
# specific cues live in the ontology + config, NOT in this module-level table, so the validator
# stays general across policies.
_DEFAULT_EDGE_EVIDENCE_KEYWORDS = {
    "APPLIES_TO": ("apply", "application"),
    "APPROVES": ("approve", "approval", "endorse"),
    "CAN_APPLY": ("apply", "eligible", "may apply", "is allowed to apply"),
    "CANNOT_APPLY": ("cannot", "ineligible", "excluded", "not eligible", "not allowed", "prohibited", "restricted", "limited to"),
    "CANNOT_CHANGE": ("cannot change", "approval required", "must obtain approval"),
    "CAN_CHANGE": ("may change", "is allowed to change", "can change"),
    "CAN_CHOOSE": ("choose", "select", "may select"),
    "CAN_MOVE": ("move", "transfer", "relocate"),
    "COVERS_COST": ("cost", "fund", "covers", "covered by"),
    "COUNTS_AS": ("counts as", "treated as", "considered", "constitutes"),
    "CONSTITUTES": ("constitutes", "counts as", "is considered", "is treated as"),
    "DEFINED_AS": ("defined as", "means", "refers to", "is a", "is an"),
    "ELIGIBLE_FOR": ("eligible", "qualifies", "eligibility", "may participate"),
    "EVALUATES": ("evaluate", "review", "assess"),
    "FUNDS": ("fund", "funding", "budget", "grant"),
    "IMPLEMENTS": ("implement", "operate", "carry out"),
    "INFLUENCES": ("influence", "affect", "contribute to"),
    "INELIGIBLE_FOR": ("ineligible", "excluded", "not eligible", "prohibited"),
    "LIMITED_TO": ("limited", "limited to", "cap", "capped", "maximum", "up to", "about"),
    "MUST_EVALUATE": ("must evaluate", "shall evaluate", "is required to review"),
    "MUST_REPORT": ("must report", "shall report", "is required to report"),
    "MUST_SUBMIT": ("must submit", "shall submit", "is required to submit"),
    "NOTIFIES": ("notify", "notification", "announce", "inform"),
    "REQUIRES": ("require", "necessary", "needed"),
    "REQUIRES_APPROVAL": ("requires approval", "must be approved", "subject to approval"),
    "REQUIRES_DOCUMENT": ("document", "report", "submit"),
    "REPORTS_TO": ("report to", "reports to"),
    "QUALIFIES_AS": ("qualifies as", "eligible as", "meets the definition", "is considered"),
    "SELECTS": ("select", "selection", "choose"),
    "SUBMITS": ("submit", "submission"),
    "TARGETS": ("target", "aimed at", "for the purpose of"),
    "TRIGGERS": ("trigger", "after", "upon", "following"),
    "USES_INFORMATION": ("based on", "criteria", "information", "uses"),
}


# Language-agnostic English cues that signal a negative or restrictive statement. Augmented at
# validate time with tokens derived from regulative edges in the ontology.
_DEFAULT_NEGATIVE_OR_RESTRICTIVE_CUES = (
    "cannot",
    "not",
    "ineligible",
    "excluded",
    "exclude",
    "prohibited",
    "prohibit",
    "restricted",
    "restrict",
    "approval",
    "limited",
    "limited to",
    "may not",
    "must not",
    "shall not",
    "ban",
    "banned",
)


# Tokens treated as noise when deriving keywords from ontology text — generic English/Korean filler
# that does not help discriminate one edge type from another.
_KEYWORD_NOISE_TOKENS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "of", "on", "or", "policy", "process", "relation", "relationship",
    "the", "this", "that", "their", "to", "with", "when", "while", "via",
    "actor", "actors", "entity", "entities", "object", "objects", "thing", "things",
    "true", "false", "null", "none", "ig", "iad",
}


_TOKEN_RE = re.compile(r"[A-Za-z가-힯]{2,}")


def _ontology_keyword_tokens(edge_def: dict[str, Any]) -> set[str]:
    """Extract candidate evidence keywords from an ontology edge type definition.

    Pulls tokens from the edge name (split on _), description, rationale, and the names of any
    declared attributes. Strips noise/stopwords and tokens shorter than 2 chars. Korean tokens
    (in the AC00–D7AF Hangul range) are kept as-is for substring matching against source_text.
    """
    fragments = [
        str(edge_def.get("description") or ""),
        str(edge_def.get("rationale") or ""),
        " ".join(str(item.get("name") or "") for item in edge_def.get("attributes", []) if isinstance(item, dict)),
    ]
    name = str(edge_def.get("name") or "")
    if name:
        fragments.append(name.replace("_", " "))

    tokens: set[str] = set()
    for fragment in fragments:
        for raw_token in _TOKEN_RE.findall(fragment):
            token = raw_token.lower()
            if len(token) <= 2 or token in _KEYWORD_NOISE_TOKENS:
                continue
            tokens.add(token)
    return tokens


def _build_edge_evidence_keywords(
    ontology: dict[str, Any],
    config_override: dict[str, Any] | None,
) -> dict[str, tuple[str, ...]]:
    """Merge default seed + ontology-derived + config-supplied keywords per edge type.

    The returned dict maps EDGE_TYPE_NAME -> tuple of cue tokens. An edge type is dropped from
    the dict (i.e. validator skips evidence checking) only when no source contributes any tokens.
    """
    keywords: dict[str, set[str]] = {
        edge_type: {token.lower() for token in tokens}
        for edge_type, tokens in _DEFAULT_EDGE_EVIDENCE_KEYWORDS.items()
    }

    for edge_def in ontology.get("edge_types", []):
        if not isinstance(edge_def, dict):
            continue
        name = str(edge_def.get("name") or "").upper()
        if not name:
            continue
        derived = _ontology_keyword_tokens(edge_def)
        keywords.setdefault(name, set()).update(derived)

    if isinstance(config_override, dict):
        for edge_type, extras in config_override.items():
            if not edge_type or not extras:
                continue
            key = str(edge_type).upper()
            normalized = (
                {str(item).lower() for item in extras}
                if isinstance(extras, (list, tuple, set))
                else {str(extras).lower()}
            )
            keywords.setdefault(key, set()).update(normalized)

    return {key: tuple(sorted(value)) for key, value in keywords.items() if value}


def _build_negative_restrictive_cues(
    ontology: dict[str, Any],
    config_override: list[Any] | tuple[Any, ...] | None,
) -> tuple[str, ...]:
    """Compose the negative/restrictive cue list from defaults + ontology + config override.

    Adds tokens from CANNOT_/INELIGIBLE_/EXCLUDES-style edges' descriptions and rationales so the
    validator picks up domain-specific prohibition wording without hard-coding it.
    """
    cues: set[str] = {token.lower() for token in _DEFAULT_NEGATIVE_OR_RESTRICTIVE_CUES}

    for edge_def in ontology.get("edge_types", []):
        if not isinstance(edge_def, dict):
            continue
        name = str(edge_def.get("name") or "").upper()
        deontic = str(edge_def.get("ig_deontic") or "").lower()
        if deontic == "cannot" or name.startswith("CANNOT") or name in {"INELIGIBLE_FOR", "EXCLUDES", "REQUIRES_APPROVAL"}:
            cues.update(_ontology_keyword_tokens(edge_def))

    if config_override:
        if isinstance(config_override, (list, tuple, set)):
            cues.update(str(item).lower() for item in config_override if item)
        else:
            cues.add(str(config_override).lower())

    return tuple(sorted(cues))


def _drop_edge(
    edge: dict[str, Any],
    reason_code: str,
    reason: str,
    dropped_edges: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    audit_record = {
        "edge_id": edge.get("id"),
        "type": edge.get("type"),
        "source": edge.get("source"),
        "target": edge.get("target"),
        "deontic": edge.get("deontic"),
        "fact": edge.get("fact"),
        "activation_condition": edge.get("activation_condition"),
        "execution_constraint": edge.get("execution_constraint"),
        "or_else_text": edge.get("or_else_text"),
        "source_text": edge.get("source_text"),
        "reason_code": reason_code,
        "reason": reason,
    }
    dropped_edges.append({key: value for key, value in audit_record.items() if value not in (None, "", [])})
    warnings.append(f"Dropped edge {edge.get('id')} {edge.get('type')}: {reason}")


def _edge_evidence_drop_reason(
    edge: dict[str, Any],
    edge_evidence_keywords: dict[str, tuple[str, ...]],
    negative_cues: tuple[str, ...],
) -> str | None:
    """Return a drop reason when source_text does not support an authoritative KG edge."""
    source_text = str(edge.get("source_text") or "").strip()
    if not source_text:
        return "Edge has no verbatim source_text."
    edge_type = str(edge.get("type") or "").upper()
    keywords = edge_evidence_keywords.get(edge_type)
    if not keywords:
        return None
    lowered = source_text.lower()
    if not any(str(keyword).lower() in lowered for keyword in keywords):
        return (
            "source_text does not contain direct cues for this edge type; "
            "the relation is retained only in the validation audit."
        )
    if _is_negative_or_restrictive_edge(edge) and not any(
        cue.lower() in lowered for cue in negative_cues
    ):
        return (
            "negative/restrictive edge lacks explicit prohibition, exclusion, restriction, "
            "or approval-required language in source_text."
        )
    return None


def _is_negative_or_restrictive_edge(edge: dict[str, Any]) -> bool:
    edge_type = str(edge.get("type") or "").upper()
    return (
        edge.get("deontic") == "cannot"
        or edge_type.startswith("CANNOT")
        or edge_type in {"INELIGIBLE_FOR", "EXCLUDES", "REQUIRES_APPROVAL"}
    )


def _mark_edge_supported(edge: dict[str, Any]) -> None:
    attributes = edge.setdefault("attributes", {})
    attributes.pop("evidence_alignment_reason", None)
    attributes.pop("role_assignment_warning", None)
    attributes["evidence_alignment"] = "supported"


def _role_edge_assignment_drop_reason(
    edge: dict[str, Any],
    node_map: dict[str, dict[str, Any]],
) -> str | None:
    """Return a drop reason when an edge assigns authority or rules to the wrong actor."""
    source = node_map.get(edge.get("source"), {})
    target = node_map.get(edge.get("target"), {})
    source_is_actor = bool(source.get("is_actor") or source.get("persona_eligible"))
    target_is_actor = bool(target.get("is_actor") or target.get("persona_eligible"))
    edge_type = str(edge.get("type") or "").upper()
    source_role_family = _normalize_role_family((source.get("attributes") or {}).get("role_family"))

    if edge_type in {"ALLOCATES", "FUNDS", "WITHHOLDS"} and source_role_family == "direct_beneficiary":
        return "Direct beneficiaries cannot be the source of funding allocation unless the source text explicitly assigns that authority."
    if edge_type in {"MUST_EVALUATE", "EVALUATES", "APPROVES", "SELECTS", "MONITORS"} and source_role_family == "direct_beneficiary":
        return "Direct beneficiaries cannot hold evaluator or approval authority for their own policy benefit."

    if edge.get("deontic") not in {"must", "can", "cannot"}:
        return None
    if not source_is_actor:
        return "Regulative edge source is not an actor, so the IG rule-holder cannot be assigned reliably."
    if not target_is_actor:
        return None
    return None


_EDGE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "based",
    "be",
    "by",
    "for",
    "from",
    "have",
    "is",
    "of",
    "or",
    "the",
    "their",
    "to",
    "who",
    "with",
}


def _token_similarity(left: str, right: str) -> float:
    left_tokens = _semantic_tokens(left)
    right_tokens = _semantic_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _semantic_tokens(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[A-Za-z0-9가-힣]+", str(text).lower()):
        if token in _EDGE_STOPWORDS or len(token) <= 1:
            continue
        tokens.add(_light_stem(token))
    return tokens


def _light_stem(token: str) -> str:
    for suffix in ("researchers", "researcher", "evaluations", "evaluation", "proposals", "proposal", "submitted"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    for suffix in ("ing", "ed", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[: -len(suffix)]
    return token


def _prefer_edge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return left if _edge_quality_score(left) >= _edge_quality_score(right) else right


def _edge_quality_score(edge: dict[str, Any]) -> int:
    source_text = str(edge.get("source_text") or "")
    activation = str(edge.get("activation_condition") or "")
    execution = str(edge.get("execution_constraint") or "")
    fact = str(edge.get("fact") or "")
    return len(source_text) * 2 + (len(activation) + len(execution)) * 2 + len(fact)


def _drop_empty_unreferenced_placeholders(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    referenced = {edge.get("source") for edge in edges} | {edge.get("target") for edge in edges}
    retained = []
    for node in nodes:
        if node.get("id") in referenced:
            retained.append(node)
            continue
        if node.get("source_spans") or node.get("attributes") or node.get("summary"):
            retained.append(node)
            continue
        warnings.append(f"Dropped empty unreferenced placeholder node: {node.get('id')}")
    return retained


def _drop_unreferenced_nodes(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the authoritative KG edge-backed; move orphan node candidates to audit."""
    referenced = {edge.get("source") for edge in edges} | {edge.get("target") for edge in edges}
    retained: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for node in nodes:
        if node.get("id") in referenced:
            retained.append(node)
            continue
        record = {
            "node_id": node.get("id"),
            "name": node.get("name"),
            "type": node.get("type"),
            "iad_category": node.get("iad_category"),
            "node_family": node.get("node_family"),
            "is_actor": node.get("is_actor"),
            "persona_eligible": node.get("persona_eligible"),
            "source_spans": node.get("source_spans") or [],
            "reason_code": "unreferenced_by_supported_edges",
            "reason": (
                "Node was not connected to any validation-supported edge, so it is not part of "
                "the authoritative Stage 1 KG."
            ),
        }
        dropped.append({key: value for key, value in record.items() if value not in (None, "", [])})
        warnings.append(f"Dropped unreferenced node from authoritative KG: {node.get('id')}")
    return retained, dropped


def _promote_supported_edge_source_actors(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    """Recover actor status for panels/committees that are validated as action sources."""
    node_map = {node.get("id"): node for node in nodes}
    has_evaluator_actor = any(
        bool(node.get("is_actor"))
        and _normalize_role_family((node.get("attributes") or {}).get("role_family")) == "evaluator"
        for node in nodes
    )
    evaluator_edges = {"EVALUATES", "MUST_EVALUATE", "APPROVES", "SELECTS", "MONITORS", "USES_INFORMATION"}
    for edge in edges:
        edge_type = str(edge.get("type") or "").upper()
        if edge_type not in evaluator_edges:
            continue
        source = node_map.get(edge.get("source"))
        if not source or source.get("is_actor"):
            continue
        text = _node_text(source)
        if not any(marker in text for marker in ("panel", "committee", "review board", "evaluationpanel")):
            continue
        if has_evaluator_actor:
            continue
        source["is_actor"] = True
        source["persona_eligible"] = True
        source["iad_category"] = "Participants"
        source["node_family"] = "ActorParticipant"
        attributes = source.setdefault("attributes", {})
        attributes["agent_modality"] = "committee_member"
        attributes["role_family"] = "evaluator"
        has_evaluator_actor = True
        warnings.append(f"Promoted supported evaluator source to actor anchor: {source.get('id')}")


def _enforce_persona_anchor_scope(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    """Persona anchors must be supported actor/position nodes in the final KG."""
    referenced = {edge.get("source") for edge in edges} | {edge.get("target") for edge in edges}
    for node in nodes:
        original = node.get("persona_eligible")
        is_actor = bool(node.get("is_actor"))
        if node.get("id") not in referenced or not is_actor:
            node["persona_eligible"] = False
        elif _looks_like_non_actor_policy_object(node):
            node["persona_eligible"] = False
        if original and not node.get("persona_eligible"):
            warnings.append(f"Disabled unsupported persona anchor: {node.get('id')}")


def _renumber_edges(edges: list[dict[str, Any]]) -> None:
    for index, edge in enumerate(edges, 1):
        edge["id"] = f"edge_{index:04d}"


def _placeholder_node(endpoint: Any, side: str) -> dict[str, Any]:
    raw = str(endpoint or f"unknown_{side}")
    if ":" in raw:
        node_type, name = raw.split(":", 1)
    else:
        node_type, name = "UnclassifiedPolicyObject", raw
    name = canonical_node_name(node_type, name or raw)
    return {
        "id": make_node_id(node_type, name),
        "name": name,
        "type": node_type,
        "iad_category": "Information",
        "node_family": "InformationEvidence",
        "is_actor": False,
        "persona_eligible": False,
        "summary": f"Recovered unresolved {side} endpoint.",
        "attributes": {"recovered_from_validation": True},
        "rationale": "Preserved unresolved endpoint for inspection instead of dropping edge.",
        "source_spans": [],
    }


def _looks_like_non_actor_policy_object(node: dict[str, Any]) -> bool:
    if not (node.get("is_actor") or node.get("persona_eligible")):
        return False
    attributes = node.get("attributes") or {}
    if (
        str(attributes.get("agent_modality") or "").strip().lower() == "individual"
        or _normalize_role_family(attributes.get("role_family")) == "direct_beneficiary"
    ):
        return False
    name = str(node.get("name") or "").lower()
    node_type = str(node.get("type") or "").lower()
    summary = str(node.get("summary") or "").lower()
    text = f"{name} {summary}"
    if any(marker in name for marker in ("support", "choice", "mobility", "continuity", "visit", "network")):
        return True
    if "mentoring" in name and "officer" not in name and "panel" not in name:
        return True
    if any(marker in f"{name} {node_type}" for marker in ("agency", "ministry", "foundation", "officer", "panel")):
        return False
    if any(marker in text for marker in ("support service", "support for", "allows researchers", "online sharing")):
        return True
    return False
