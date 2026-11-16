"""Mechanism-covering persona panel construction.

The legacy Stage 1 path samples scenarios from every persona-eligible KG actor.
This additive mode first builds policy-grounded functional role prototypes,
generates candidates for those slots, and deterministically freezes one panel.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from models.schemas import RolePrototype, StakeholderGroup


DEFAULT_ROLE_DEFINITIONS = (
    {
        "prototype_id": "patient",
        "label": "Patient or covered beneficiary",
        "domain_role": "patient",
        "keywords": ("patient", "beneficiary", "enrollee", "consumer", "complainant"),
        "node_types": ("BeneficiaryOrEnrollee", "PolicyRole"),
        "anchor_keywords": ("patient", "beneficiary"),
        "mechanism_functions": ("affected", "decision"),
        "role_family": "direct_beneficiary",
        "agent_modality": "individual",
        "toc_phase_relevance": ("Activities", "Outputs", "Outcomes", "Impact"),
        "objectives": ("Receive effective policy protection and avoid preventable financial harm.",),
        "constraints": ("Must act with incomplete information and limited control over policy implementation.",),
        "causal_leverage_points": ("access to understandable information", "access to complaint or remedy channels"),
    },
    {
        "prototype_id": "provider",
        "label": "Service provider or facility representative",
        "domain_role": "provider",
        "keywords": ("provider", "facility", "hospital", "physician", "ambulance"),
        "node_types": ("NonparticipatingProvider", "ParticipatingFacility", "PolicyRole"),
        "anchor_keywords": ("nonparticipatingprovider", "nonparticipating provider"),
        "mechanism_functions": ("implementation", "affected"),
        "role_family": "affected_actor",
        "agent_modality": "organization_representative",
        "toc_phase_relevance": ("Activities", "Outputs", "Outcomes"),
        "objectives": ("Deliver services while maintaining operational and financial viability.",),
        "constraints": ("Must adapt billing, disclosure, compliance, and dispute practices to policy rules.",),
        "causal_leverage_points": ("front-line compliance", "billing and payment response"),
    },
    {
        "prototype_id": "payer",
        "label": "Health plan, insurer, or payer representative",
        "domain_role": "payer",
        "keywords": ("health plan", "issuer", "payer", "insurer", "carrier", "third-party administrator"),
        "node_types": ("GroupHealthPlanOrIssuer", "PolicyRole"),
        "anchor_keywords": ("group health plan or health insurance issuer", "health plan or issuer"),
        "mechanism_functions": ("decision", "implementation"),
        "role_family": "policy_administrator",
        "agent_modality": "organization_representative",
        "toc_phase_relevance": ("Inputs", "Activities", "Outputs", "Outcomes"),
        "objectives": ("Administer coverage and payment decisions consistently with policy requirements.",),
        "constraints": ("Must reconcile compliance, claims operations, provider payment, and administrative cost.",),
        "causal_leverage_points": ("claims adjudication", "provider payment and network administration"),
    },
    {
        "prototype_id": "regulator",
        "label": "Policy regulator or implementation official",
        "domain_role": "regulator",
        "keywords": (
            "regulatory", "department", "agency", " hhs", "cms", "treasury", "internal revenue",
            "office of personnel management", " opm", "department of labor",
        ),
        "node_types": ("RegulatoryAgency",),
        "anchor_keywords": ("federal regulatory departments", "hhs"),
        "mechanism_functions": ("decision", "implementation", "oversight"),
        "role_family": "policy_administrator",
        "agent_modality": "organization_representative",
        "toc_phase_relevance": ("Inputs", "Activities", "Outputs", "Impact"),
        "objectives": ("Implement, monitor, and enforce the policy as intended.",),
        "constraints": ("Must coordinate authority, guidance, oversight capacity, and accountability.",),
        "causal_leverage_points": ("implementation guidance", "monitoring and enforcement"),
    },
    {
        "prototype_id": "mediator",
        "label": "Independent dispute-resolution representative",
        "domain_role": "mediator",
        "keywords": ("independent dispute", "idr entity", "arbitrator", "arbitration", "dispute resolution"),
        "node_types": ("IndependentDisputeResolutionEntity",),
        "anchor_keywords": ("idr entity", "independent dispute resolution entity"),
        "mechanism_functions": ("mediation", "decision"),
        "role_family": "evaluator",
        "agent_modality": "organization_representative",
        "toc_phase_relevance": ("Activities", "Outputs", "Outcomes"),
        "objectives": ("Resolve disputes consistently, independently, and within procedural requirements.",),
        "constraints": ("Must make defensible determinations from limited submissions under timing and rule constraints.",),
        "causal_leverage_points": ("access to dispute resolution", "consistency and timeliness of determinations"),
    },
)


def mechanism_panel_enabled(config: dict) -> bool:
    return str((config.get("persona") or {}).get("selection_mode") or "legacy") == "mechanism_panel"


def _definitions(config: dict) -> list[dict[str, Any]]:
    configured = ((config.get("persona") or {}).get("mechanism_panel") or {}).get("roles")
    source = configured or DEFAULT_ROLE_DEFINITIONS
    definitions = []
    for raw in source:
        item = dict(raw)
        item["keywords"] = tuple(str(v).lower() for v in item.get("keywords") or ())
        item["node_types"] = tuple(str(v) for v in item.get("node_types") or ())
        item["anchor_keywords"] = tuple(str(v).lower() for v in item.get("anchor_keywords") or ())
        definitions.append(item)
    return definitions


def _node_text(node: dict, include_summary: bool = False) -> str:
    values = [node.get("id"), node.get("name"), node.get("type")]
    if include_summary:
        values.append(node.get("summary"))
    return " ".join(str(value or "") for value in values).lower()


def _classification_score(node: dict, definition: dict) -> int:
    allowed_types = definition.get("node_types") or ()
    if allowed_types and str(node.get("type") or "") not in allowed_types:
        return 0
    identity = _node_text(node)
    summary = str(node.get("summary") or "").lower()
    score = 0
    for keyword in definition["keywords"]:
        if keyword in identity:
            score += 5
        elif keyword in summary:
            score += 1
    return score


def _dedupe(values: list[str], limit: int | None = None) -> list[str]:
    result = []
    seen = set()
    for value in values:
        text = " ".join(str(value or "").split())
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def build_role_prototypes(
    policy_graph_path: str | Path,
    config: dict,
) -> tuple[list[RolePrototype], list[dict[str, Any]]]:
    """Build evidence-backed role prototypes without mutating the source KG."""
    graph = json.loads(Path(policy_graph_path).read_text(encoding="utf-8"))
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    definitions = _definitions(config)
    assignments: dict[str, list[tuple[int, dict]]] = {d["prototype_id"]: [] for d in definitions}
    overlay = []

    for node in nodes:
        scored = [(definition, _classification_score(node, definition)) for definition in definitions]
        definition, score = max(scored, key=lambda item: item[1])
        if score <= 0:
            continue
        assignments[definition["prototype_id"]].append((score, node))
        overlay.append({
            "entity_id": node.get("id"),
            "source_is_actor": bool(node.get("is_actor")),
            "source_persona_eligible": bool(node.get("persona_eligible")),
            "panel_role_candidate": True,
            "role_prototype_id": definition["prototype_id"],
            "classification_score": score,
            "classification_basis": [
                keyword for keyword in definition["keywords"]
                if keyword in _node_text(node, include_summary=True)
            ],
        })

    edge_by_entity: dict[str, list[dict]] = {}
    for edge in edges:
        for endpoint in (edge.get("source"), edge.get("target")):
            if endpoint:
                edge_by_entity.setdefault(str(endpoint), []).append(edge)

    prototypes = []
    strict = bool(((config.get("persona") or {}).get("mechanism_panel") or {}).get("strict_role_coverage", True))
    anchor_overrides = (
        ((config.get("persona") or {}).get("mechanism_panel") or {}).get("anchor_overrides") or {}
    )
    for definition in definitions:
        candidates = assignments[definition["prototype_id"]]
        if not candidates:
            if strict and definition.get("required", True):
                raise ValueError(f"No KG actor candidate found for mechanism role {definition['prototype_id']!r}")
            continue
        ranked = sorted(
            candidates,
            key=lambda item: (
                -sum(
                    1 for keyword in definition.get("anchor_keywords") or ()
                    if keyword in _node_text(item[1])
                ),
                -item[0],
                -len(edge_by_entity.get(str(item[1].get("id")), [])),
                str(item[1].get("id") or ""),
            ),
        )
        override_id = anchor_overrides.get(definition["prototype_id"])
        override_match = next(
            (node for _, node in candidates if str(node.get("id")) == str(override_id)),
            None,
        )
        if override_id and override_match is None:
            raise ValueError(
                f"Configured anchor override {override_id!r} is not a classified candidate "
                f"for role {definition['prototype_id']!r}"
            )
        anchor = override_match or ranked[0][1]
        source_nodes = [node for _, node in ranked]
        source_ids = [str(node.get("id")) for node in source_nodes if node.get("id")]
        related_edges = [edge for source_id in source_ids for edge in edge_by_entity.get(source_id, [])]
        outgoing_edges = [edge for edge in related_edges if str(edge.get("source")) in source_ids]
        anchor_outgoing = [
            edge for edge in outgoing_edges
            if str(edge.get("source")) == str(anchor.get("id"))
        ]
        action_edges = anchor_outgoing + [
            edge for edge in outgoing_edges if edge not in anchor_outgoing
        ] + [edge for edge in related_edges if edge not in outgoing_edges]
        actions = _dedupe([
            edge.get("fact") or edge.get("source_text")
            for edge in action_edges
            if edge.get("fact") or edge.get("source_text")
        ], limit=3)
        evidence = _dedupe(
            [str(anchor.get("summary") or "")[:600]]
            + [edge.get("source_text") or edge.get("fact") for edge in related_edges],
            limit=5,
        )
        prototypes.append(RolePrototype(
            prototype_id=definition["prototype_id"],
            label=definition.get("label") or definition["domain_role"],
            domain_role=definition["domain_role"],
            mechanism_functions=list(definition.get("mechanism_functions") or []),
            role_family=definition.get("role_family") or "affected_actor",
            agent_modality=definition.get("agent_modality") or "individual",
            required=bool(definition.get("required", True)),
            anchor_entity_id=str(anchor["id"]),
            source_entity_ids=source_ids,
            source_node_types=_dedupe([node.get("type") for node in source_nodes]),
            toc_phase_relevance=list(definition.get("toc_phase_relevance") or []),
            objectives=list(definition.get("objectives") or [])[:2],
            constraints=list(definition.get("constraints") or [])[:2],
            actions=actions,
            causal_leverage_points=list(definition.get("causal_leverage_points") or [])[:2],
            evidence=evidence,
            classification_basis=_dedupe([
                keyword
                for _, node in candidates
                for keyword in definition["keywords"]
                if keyword in _node_text(node, include_summary=True)
            ] + ([f"researcher_validated_anchor:{override_id}"] if override_id else [])),
        ))
    return prototypes, overlay


def compact_role_context(prototype: RolePrototype, persona: Any) -> dict[str, Any]:
    """Compile the bounded role payload that is safe to inject into Stage 2."""
    objectives = prototype.objectives or list(getattr(persona, "core_objectives", []) or [])
    constraints = prototype.constraints or list(getattr(persona, "core_constraints", []) or [])
    demographics = getattr(persona, "demographics", {}) or {}
    context = {
        "role_label": prototype.label,
        "domain_role": prototype.domain_role,
        "role_family": prototype.role_family,
        "agent_modality": prototype.agent_modality,
        "constraint_scope": demographics.get("constraint_scope"),
        "mechanism_functions": prototype.mechanism_functions[:3],
        "objectives": objectives[:2],
        "constraints": constraints[:2],
        "actions": prototype.actions[:3],
        "causal_leverage_points": prototype.causal_leverage_points[:2],
        "toc_phase_relevance": prototype.toc_phase_relevance,
    }
    return {key: value for key, value in context.items() if value not in (None, "", [], {})}


def attach_role_prototype(persona: Any, prototype: RolePrototype) -> None:
    demographics = dict(getattr(persona, "demographics", {}) or {})
    demographics.update({
        "role_prototype_id": prototype.prototype_id,
        "role_anchor_entity_id": prototype.anchor_entity_id,
        "entity_id": prototype.anchor_entity_id,
        "role_family": prototype.role_family,
        "agent_modality": prototype.agent_modality,
        "policy_role_context": compact_role_context(prototype, persona),
    })
    persona.demographics = demographics


def _candidate_score(persona: Any) -> tuple[float, dict[str, float]]:
    demographics = getattr(persona, "demographics", {}) or {}
    components = {
        "grounded_profile": 2.0 if demographics.get("professional_persona") else 0.0,
        "skills": 1.0 if demographics.get("skills_and_expertise") else 0.0,
        "goal": 1.0 if demographics.get("career_goals_and_ambitions") else 0.0,
        "match_score": min(float(demographics.get("match_score") or 0.0), 10.0) / 5.0,
        "direct_tier": 1.0 if demographics.get("tier_used") == "direct" else 0.0,
        "no_fallback": 1.0 if not demographics.get("used_fallback", False) else -2.0,
    }
    compatibility = demographics.get("compatibility_assessment") or {}
    status = compatibility.get("status") if isinstance(compatibility, dict) else None
    if status in {"direct_match", "accepted"}:
        components["compatibility"] = 1.0
    elif status in {"uncertain_but_retained", "fallback_after_error"}:
        components["compatibility"] = -1.0
    return sum(components.values()), components


def _diversity_bonus(persona: Any, selected: list[Any]) -> float:
    if not selected:
        return 0.0
    demographics = getattr(persona, "demographics", {}) or {}
    bonus = 0.0
    for field in ("occupation", "province", "sex", "education_level"):
        value = demographics.get(field)
        if value and all(value != (getattr(other, "demographics", {}) or {}).get(field) for other in selected):
            bonus += 0.25
    name = " ".join(str(demographics.get("name") or "").lower().split())
    if name and any(
        name == " ".join(
            str((getattr(other, "demographics", {}) or {}).get("name") or "").lower().split()
        )
        for other in selected
    ):
        # Duplicate display names are especially confusing in a user-study
        # discussion interface, even when the underlying personas differ.
        bonus -= 1.0
    return bonus


def _mechanism_fit_bonus(
    persona: Any,
    prototype_id: str,
    config: dict,
) -> tuple[float, dict[str, list[str]]]:
    """Score optional, policy-specific preferences for a panel role.

    Role coverage remains the primary selection criterion. These bounded terms
    only resolve cases where several complete candidates represent materially
    different parts of the same organizational role.
    """
    panel_cfg = ((config.get("persona") or {}).get("mechanism_panel") or {})
    preferences = (panel_cfg.get("selection_preferences") or {}).get(prototype_id) or {}
    demographics = getattr(persona, "demographics", {}) or {}
    candidate_text = " ".join(
        str(demographics.get(field) or "")
        for field in (
            "occupation",
            "professional_persona",
            "skills_and_expertise",
            "career_goals_and_ambitions",
        )
    ).lower()
    occupation_text = str(demographics.get("occupation") or "").lower()
    preferred_occupation = [
        str(term).lower() for term in preferences.get("preferred_occupation_terms") or []
        if str(term).strip()
    ]
    deprioritized_occupation = [
        str(term).lower() for term in preferences.get("deprioritized_occupation_terms") or []
        if str(term).strip()
    ]
    preferred = [
        str(term).lower() for term in preferences.get("preferred_terms") or []
        if str(term).strip()
    ]
    deprioritized = [
        str(term).lower() for term in preferences.get("deprioritized_terms") or []
        if str(term).strip()
    ]
    preferred_matches = [term for term in preferred if term in candidate_text]
    deprioritized_matches = [term for term in deprioritized if term in candidate_text]
    preferred_occupation_matches = [term for term in preferred_occupation if term in occupation_text]
    deprioritized_occupation_matches = [
        term for term in deprioritized_occupation if term in occupation_text
    ]
    positive_weight = float(preferences.get("preferred_term_weight", 0.5))
    negative_weight = float(preferences.get("deprioritized_term_weight", 0.5))
    max_bonus = float(preferences.get("max_bonus", 1.5))
    max_penalty = float(preferences.get("max_penalty", 1.0))
    preferred_occupation_weight = float(preferences.get("preferred_occupation_weight", 1.5))
    deprioritized_occupation_weight = float(
        preferences.get("deprioritized_occupation_weight", 1.0)
    )
    score = min(len(preferred_matches) * positive_weight, max_bonus)
    score -= min(len(deprioritized_matches) * negative_weight, max_penalty)
    score += len(preferred_occupation_matches) * preferred_occupation_weight
    score -= len(deprioritized_occupation_matches) * deprioritized_occupation_weight
    return score, {
        "preferred_matches": preferred_matches,
        "deprioritized_matches": deprioritized_matches,
        "preferred_occupation_matches": preferred_occupation_matches,
        "deprioritized_occupation_matches": deprioritized_occupation_matches,
    }


def select_mechanism_panel(
    persona_pool: list[StakeholderGroup],
    prototypes: list[RolePrototype],
    config: dict,
) -> tuple[list[StakeholderGroup], dict[str, Any]]:
    """Select one candidate per required role with an auditable score."""
    candidates_by_role: dict[str, list[tuple[str, Any]]] = {}
    for group in persona_pool:
        for persona in group.personas:
            role_id = (getattr(persona, "demographics", {}) or {}).get("role_prototype_id")
            if role_id:
                candidates_by_role.setdefault(str(role_id), []).append((group.stakeholder_type, persona))

    selected_groups = []
    selected_personas = []
    selections = []
    exclusions = []
    panel_cfg = ((config.get("persona") or {}).get("mechanism_panel") or {})
    candidate_overrides = panel_cfg.get("candidate_overrides") or {}
    name_overrides = panel_cfg.get("name_overrides") or {}
    for position, prototype in enumerate(prototypes, 1):
        candidates = candidates_by_role.get(prototype.prototype_id) or []
        if not candidates:
            if prototype.required:
                raise ValueError(f"No persona candidate generated for required role {prototype.prototype_id!r}")
            continue
        ranked = []
        for stakeholder_type, persona in candidates:
            base_score, components = _candidate_score(persona)
            diversity = _diversity_bonus(persona, selected_personas)
            mechanism_fit, mechanism_fit_basis = _mechanism_fit_bonus(
                persona, prototype.prototype_id, config
            )
            demographics = persona.demographics or {}
            stable_id = str(
                demographics.get("nemotron_uuid")
                or f"{prototype.prototype_id}:{demographics.get('candidate_index', persona.representative_type)}"
            )
            tie_break = hashlib.sha256(stable_id.encode("utf-8")).hexdigest()
            ranked.append((
                -(base_score + diversity + mechanism_fit),
                tie_break,
                stakeholder_type,
                persona,
                components,
                diversity,
                mechanism_fit,
                mechanism_fit_basis,
            ))
        ranked.sort(key=lambda item: (item[0], item[1]))
        override = candidate_overrides.get(prototype.prototype_id)
        selected_index = 0
        if override:
            selected_index = next(
                (
                    index
                    for index, item in enumerate(ranked)
                    if str(item[3].representative_type) == str(override)
                    or str((item[3].demographics or {}).get("persona_id") or "") == str(override)
                ),
                -1,
            )
            if selected_index < 0:
                raise ValueError(
                    f"Configured candidate override {override!r} was not found for "
                    f"mechanism role {prototype.prototype_id!r}"
                )
        selected_row = ranked[selected_index]
        _, _, stakeholder_type, chosen, components, diversity, mechanism_fit, mechanism_fit_basis = selected_row
        chosen.demographics = {
            **(chosen.demographics or {}),
            "panel_position": position,
            "panel_selection_score": -selected_row[0],
            **(
                {"name": name_overrides[prototype.prototype_id]}
                if name_overrides.get(prototype.prototype_id)
                else {}
            ),
        }
        selected_groups.append(StakeholderGroup(stakeholder_type=stakeholder_type, personas=[chosen]))
        selected_personas.append(chosen)
        selections.append({
            "slot": prototype.prototype_id,
            "domain_role": prototype.domain_role,
            "selected_representative_type": chosen.representative_type,
            "anchor_entity_id": prototype.anchor_entity_id,
            "score": -selected_row[0],
            "selection_method": "candidate_override" if override else "deterministic_score",
            "name_override": name_overrides.get(prototype.prototype_id),
            "score_components": {
                **components,
                "panel_diversity": diversity,
                "mechanism_fit": mechanism_fit,
            },
            "mechanism_fit_basis": mechanism_fit_basis,
            "candidate_count": len(candidates),
        })
        for index, rejected in enumerate(ranked):
            if index == selected_index:
                continue
            exclusions.append({
                "slot": prototype.prototype_id,
                "representative_type": rejected[3].representative_type,
                "reason": "lower deterministic compatibility/coverage score",
                "score": -rejected[0],
                "mechanism_fit_basis": rejected[7],
            })

    manifest = {
        "format_version": 1,
        "selection_mode": "mechanism_panel",
        "panel_id": panel_cfg.get("panel_id") or "primary_mechanism_panel",
        "selection_policy": "role-balanced_mechanism-covering",
        "panel_size": len(selected_personas),
        "role_order": [prototype.prototype_id for prototype in prototypes],
        "selections": selections,
        "excluded_candidates": exclusions,
        "ebc_blind_selection": True,
        "seed": (config.get("persona") or {}).get("simulation_sample_seed"),
    }
    return selected_groups, manifest
