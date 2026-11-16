import json
import os
from typing import Any

from llm.client import chat_completion
from llm.pdf_loader import load_pdfs
from models.prompts import (
    ECONOMY_CONSTRUCTOR_SYSTEM, ECONOMY_CONSTRUCTOR_USER,
    POLICY_READER_SYSTEM, POLICY_READER_USER,
    POLICY_CONTEXT_SYSTEM, POLICY_CONTEXT_USER,
    ENTITY_TO_PERSONA_SYSTEM, ENTITY_TO_PERSONA_USER,
    KG_NEMOTRON_PERSONA_SYNTHESIS_SYSTEM, KG_NEMOTRON_PERSONA_SYNTHESIS_USER,
    PERSONAS_FROM_TEXT_SYSTEM, PERSONAS_FROM_TEXT_USER,
)
from models.schemas import (
    EconomicContext,
    PolicyContext,
    PolicyPlan,
    PersonaProfile,
    RolePrototype,
    StakeholderGroup,
)
from persona.name_utils import infer_persona_name


def build_economic_context(config: dict) -> EconomicContext:
    pdf_text = load_pdfs(config["paths"]["background_dir"], config)
    messages = [
        {"role": "system", "content": ECONOMY_CONSTRUCTOR_SYSTEM},
        {"role": "user", "content": ECONOMY_CONSTRUCTOR_USER.format(pdf_text=pdf_text)},
    ]
    return EconomicContext(**json.loads(chat_completion(messages, config)))


def build_policy_plan(config: dict) -> PolicyPlan:
    pdf_text = load_pdfs(config["paths"]["input_policy_dir"], config)
    messages = [
        {"role": "system", "content": POLICY_READER_SYSTEM},
        {"role": "user", "content": POLICY_READER_USER.format(pdf_text=pdf_text)},
    ]
    return PolicyPlan(**json.loads(chat_completion(messages, config)))


def build_policy_context(
    policy: PolicyPlan | None,  # legacy stub kept for build_policy_context signature back-compat
    economic_context: EconomicContext | None,
    config: dict,
    policy_graph_path: str | None = None,
) -> PolicyContext:
    """Pass None for `policy` and/or `economic_context` to omit that section
    from the prompt entirely; the LLM then grounds policy_context purely on
    the remaining inputs (KG summary alone, in the KG-only ablation case)."""
    kg_summary = _policy_graph_summary(policy_graph_path)
    sections: list[str] = []
    if policy is not None:
        sections.append("--- Policy Plan ---\n" + json.dumps(policy.model_dump(), ensure_ascii=False))
    if economic_context is not None:
        sections.append("--- Economic Context ---\n" + json.dumps(economic_context.model_dump(), ensure_ascii=False))
    sections.append("--- KG Summary ---\n" + json.dumps(kg_summary, ensure_ascii=False))
    messages = [
        {"role": "system", "content": POLICY_CONTEXT_SYSTEM},
        {
            "role": "user",
            "content": POLICY_CONTEXT_USER.format(sections="\n\n".join(sections)),
        },
    ]
    return PolicyContext(**json.loads(chat_completion(messages, config)))


def _policy_graph_summary(policy_graph_path: str | None) -> dict[str, Any]:
    if not policy_graph_path:
        return {}
    try:
        with open(policy_graph_path, "r", encoding="utf-8") as f:
            graph = json.load(f)
    except OSError:
        return {}

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    ontology = graph.get("ontology") or {}
    actor_nodes = [
        {
            "id": node.get("id"),
            "name": node.get("name"),
            "type": node.get("type"),
            "iad_category": node.get("iad_category"),
            "node_family": node.get("node_family"),
            "persona_eligible": node.get("persona_eligible"),
            "summary": node.get("summary"),
        }
        for node in nodes
        if node.get("iad_category") in {"Participants", "Positions"} or node.get("persona_eligible")
    ]
    edge_summaries = [
        {
            "type": edge.get("type"),
            "edge_category": edge.get("edge_category"),
            "edge_family": edge.get("edge_family"),
            "source": edge.get("source"),
            "target": edge.get("target"),
            "deontic": edge.get("deontic"),
            "fact": edge.get("fact"),
            "activation_condition": edge.get("activation_condition") or edge.get("condition_text"),
            "execution_constraint": edge.get("execution_constraint"),
            "or_else_text": edge.get("or_else_text"),
            "source_text": edge.get("source_text"),
        }
        for edge in edges
    ]
    return {
        "ontology_analysis_summary": ontology.get("analysis_summary"),
        "entity_types": [
            {
                "name": item.get("name"),
                "iad_category": item.get("iad_category"),
                "node_family": item.get("node_family"),
                "is_actor": item.get("is_actor"),
                "persona_eligible": item.get("persona_eligible"),
                "description": item.get("description"),
            }
            for item in ontology.get("entity_types", [])
        ],
        "actor_nodes": actor_nodes,
        "edge_summaries": edge_summaries,
    }


def build_policy_entity_context(entity: Any) -> str:
    """Build a policy-focused context string from a graph entity for the persona prompt."""
    sections = []

    # Section 1: Entity Information
    entity_type = entity.get_entity_type() or "Unknown"
    iad_category = entity.attributes.get("iad_category") if entity.attributes else None
    node_family = entity.attributes.get("node_family") if entity.attributes else None
    rationale = entity.attributes.get("rationale") if entity.attributes else None
    lines = [
        "### Entity Information",
        f"- Name: {entity.name}",
        f"- Type: {entity_type}",
        f"- IAD Category: {iad_category or 'Unknown'}",
        f"- Node Family: {node_family or 'Unknown'}",
        f"- Summary: {entity.summary or '(no summary)'}",
    ]
    sections.append("\n".join(lines))

    # Section 2: IAD/IG role rationale
    iad_lines = ["### IAD/IG Rationale"]
    if rationale:
        iad_lines.append(f"- IAD rationale: {rationale}")
    policy_context = _extract_policy_context(entity)
    rule_context = policy_context["rules"]
    if rule_context["obligations"]:
        iad_lines.append(f"- Obligations: {json.dumps(rule_context['obligations'], ensure_ascii=False)}")
    if rule_context["eligible_actions"]:
        iad_lines.append(f"- Permissions: {json.dumps(rule_context['eligible_actions'], ensure_ascii=False)}")
    if rule_context["prohibited_actions"]:
        iad_lines.append(f"- Prohibitions: {json.dumps(rule_context['prohibited_actions'], ensure_ascii=False)}")
    for section_key, label in (
        ("resources", "Resources/benefits"),
        ("actions", "Policy actions"),
        ("information", "Information/documents"),
        ("evaluation", "Evaluation context"),
        ("structural_relations", "Other structural relations"),
    ):
        if policy_context.get(section_key):
            iad_lines.append(f"- {label}: {json.dumps(policy_context[section_key], ensure_ascii=False)}")
    if len(iad_lines) > 1:
        sections.append("\n".join(iad_lines))

    # Section 3: Policy-Related Facts (from related_edges)
    if entity.related_edges:
        fact_lines = ["### Policy-Related Facts"]
        for edge in entity.related_edges:
            fact = edge.get("fact", "").strip()
            edge_name = edge.get("edge_name", "")
            direction = edge.get("direction", "")
            deontic = edge.get("deontic")
            source_text = edge.get("source_text", "")
            if fact:
                suffix = f" | deontic={deontic}" if deontic else ""
                fact_lines.append(f"- [{direction}/{edge_name}{suffix}] {fact}")
            if source_text:
                fact_lines.append(f"  source: {source_text}")
        if len(fact_lines) > 1:
            sections.append("\n".join(fact_lines))

    # Section 4: Related Stakeholders and Policy Objects (from related_nodes)
    if entity.related_nodes:
        node_lines = ["### Related Stakeholders and Policy Objects"]
        for nb in entity.related_nodes:
            nb_name = nb.get("name", "")
            nb_category = nb.get("iad_category", "")
            nb_summary = nb.get("summary", "").strip()
            if nb_name:
                entry = f"- {nb_name}"
                if nb_category:
                    entry += f" ({nb_category})"
                if nb_summary:
                    entry += f": {nb_summary}"
                node_lines.append(entry)
        if len(node_lines) > 1:
            sections.append("\n".join(node_lines))

    # Section 5: Entity Attributes
    if entity.attributes:
        attr_lines = ["### Entity Attributes"]
        for k, v in entity.attributes.items():
            if v is not None:
                attr_lines.append(f"- {k}: {v}")
        if len(attr_lines) > 1:
            sections.append("\n".join(attr_lines))

    combined = "\n\n".join(sections)
    # Cap to stay within prompt budget while preserving richer IAD/IG context.
    max_chars = 5000
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n...(truncated)"
    return combined


def _extract_policy_context(entity: Any) -> dict[str, Any]:
    context = {
        "rules": {
            "obligations": [],
            "eligible_actions": [],
            "prohibited_actions": [],
        },
        "resources": [],
        "actions": [],
        "information": [],
        "evaluation": [],
        "structural_relations": [],
    }
    for edge in getattr(entity, "related_edges", []):
        item = _edge_policy_item(edge)
        attributes = edge.get("attributes") or {}
        if (
            attributes.get("evidence_alignment") not in {None, "supported"}
            or attributes.get("role_assignment_warning")
        ):
            continue
        deontic = edge.get("deontic")
        # IG regulative edges are actor-scoped. Assign permissions,
        # obligations, and prohibitions only to the edge source actor. Incoming
        # regulative edges remain visible in contextual sections below, but are
        # not treated as this persona's own rule.
        actor_scoped_rule = edge.get("direction") == "outgoing"
        if deontic == "must" and actor_scoped_rule:
            context["rules"]["obligations"].append(item)
        elif deontic == "can" and actor_scoped_rule:
            context["rules"]["eligible_actions"].append(item)
        elif deontic == "cannot" and actor_scoped_rule:
            context["rules"]["prohibited_actions"].append(item)
        elif _is_evaluation_edge(edge):
            context["evaluation"].append(item)
        elif _is_information_edge(edge):
            context["information"].append(item)
        elif _is_action_edge(edge):
            context["actions"].append(item)
        elif _is_resource_edge(edge):
            context["resources"].append(item)
        else:
            context["structural_relations"].append(item)
    return context


def _extract_institutional_rules(entity: Any) -> dict[str, list[dict[str, Any]]]:
    return _extract_policy_context(entity)["rules"]


def _edge_policy_item(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        "edge_type": edge.get("edge_name", ""),
        "edge_category": edge.get("edge_category", ""),
        "edge_family": edge.get("edge_family", ""),
        "direction": edge.get("direction", ""),
        "fact": edge.get("fact", ""),
        "deontic": edge.get("deontic"),
        "source": edge.get("source"),
        "source_name": edge.get("source_name"),
        "source_type": edge.get("source_type"),
        "source_iad_category": edge.get("source_iad_category"),
        "source_node_family": edge.get("source_node_family"),
        "source_iad_rule_type": edge.get("source_iad_rule_type"),
        "target": edge.get("target"),
        "target_name": edge.get("target_name"),
        "target_type": edge.get("target_type"),
        "target_iad_category": edge.get("target_iad_category"),
        "target_node_family": edge.get("target_node_family"),
        "target_iad_rule_type": edge.get("target_iad_rule_type"),
        "activation_condition": edge.get("activation_condition"),
        "execution_constraint": edge.get("execution_constraint"),
        "condition_text": edge.get("condition_text"),
        "or_else_text": edge.get("or_else_text"),
        "source_text": edge.get("source_text", ""),
        "attributes": edge.get("attributes", {}),
    }


def _is_resource_edge(edge: dict[str, Any]) -> bool:
    edge_type = str(edge.get("edge_name") or "").upper()
    categories = {edge.get("source_iad_category"), edge.get("target_iad_category")}
    return (
        "Costs_and_Benefits" in categories
        or edge_type in {"FUNDS", "RECEIVES", "PAYS", "SUPPORTS", "ALLOCATES"}
    )


def _is_evaluation_edge(edge: dict[str, Any]) -> bool:
    edge_type = str(edge.get("edge_name") or "").upper()
    text = f"{edge_type} {edge.get('fact', '')}"
    return "EVALUAT" in edge_type or "SELECT" in edge_type or "APPROV" in edge_type or "MONITOR" in edge_type


def _is_information_edge(edge: dict[str, Any]) -> bool:
    edge_type = str(edge.get("edge_name") or "").upper()
    categories = {edge.get("source_iad_category"), edge.get("target_iad_category")}
    return (
        "Information" in categories
        or edge_type in {"SUBMITS", "USES_INFORMATION", "REPORTS", "DISCLOSES"}
    )


def _is_action_edge(edge: dict[str, Any]) -> bool:
    edge_type = str(edge.get("edge_name") or "").upper()
    categories = {edge.get("source_iad_category"), edge.get("target_iad_category")}
    return (
        "Actions" in categories
        or edge_type in {"IMPLEMENTS", "PRECEDES", "APPLIES", "SELECTS", "PERFORMS"}
    )


def entity_to_persona(
    entity: Any,
    entity_type: str,
    policy_context: PolicyContext,
    context: EconomicContext,
    config: dict,
) -> PersonaProfile:
    """Convert a single graph entity into a PersonaProfile via LLM."""
    entity_context = build_policy_entity_context(entity)
    user_content = ENTITY_TO_PERSONA_USER.format(
        entity_name=entity.name,
        entity_type=entity_type,
        entity_summary=entity.summary or "",
        entity_context=entity_context,
        policy_context=json.dumps(policy_context.model_dump(), ensure_ascii=False),
    )
    if config["stage1"].get("use_economic_context_in_persona", True):
        user_content += f"\n\n--- Economic Context ---\n{json.dumps(context.model_dump(), ensure_ascii=False)}"
    messages = [
        {"role": "system", "content": ENTITY_TO_PERSONA_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    data = json.loads(chat_completion(messages, config))
    policy_context = _extract_policy_context(entity)
    rules = policy_context["rules"]
    role_family = _infer_role_family(entity, entity_type, None)
    return PersonaProfile(
        representative_type=entity.name,
        core_objectives=data["core_objectives"],
        core_constraints=_scope_constraints_to_role(
            data["core_constraints"],
            role_family,
            policy_context,
        ),
        role_entity_id=getattr(entity, "uuid", None),
        iad_category=entity.attributes.get("iad_category") if entity.attributes else None,
        obligations=rules["obligations"],
        eligible_actions=rules["eligible_actions"],
        prohibited_actions=rules["prohibited_actions"],
        kg_context=policy_context,
        source="policy_graph" if (entity.attributes or {}).get("source_spans") else "zep_graph",
    )


def entity_to_personas(
    entity: Any,
    entity_type: str,
    policy_context: PolicyContext,
    context: EconomicContext,
    config: dict,
    role_prototype: RolePrototype | None = None,
) -> list[PersonaProfile]:
    """Convert a graph entity into one or more personas."""
    base = entity_to_persona(entity, entity_type, policy_context, context, config)
    if role_prototype is not None:
        from stage1.mechanism_panel import attach_role_prototype

        # Candidate generation should be conditioned on the mechanism role, not
        # merely on the often broad KG entity type (for example,
        # QualifiedManufacturer).
        attach_role_prototype(base, role_prototype)
    sampler_name = str(config.get("persona", {}).get("sampler", "llm")).lower()
    if sampler_name == "nemotron":
        return _expand_with_nemotron(base, entity, entity_type, policy_context, config)
    if sampler_name == "scope_definer":
        return _expand_with_scope_definer(base, entity, entity_type, policy_context, config)
    if sampler_name == "kg_gen":
        return _expand_with_kg_gen(base, entity, entity_type, policy_context, config)
    return [base]


def _expand_with_nemotron(
    base: PersonaProfile,
    entity: Any,
    entity_type: str,
    policy_context: PolicyContext,
    config: dict,
) -> list[PersonaProfile]:
    actor_modality = _infer_actor_modality(entity, entity_type, base)
    role_family = _infer_role_family(entity, entity_type, base)
    if actor_modality == "organization":
        from stage1.mechanism_panel import mechanism_panel_enabled
        if mechanism_panel_enabled(config):
            return _expand_with_kg_gen(base, entity, entity_type, policy_context, config)
        return [
            base.model_copy(update={
                "demographics": {
                    **(base.demographics or {}),
                    "agent_modality": "organization",
                    "role_family": role_family,
                    "constraint_scope": _constraint_scope_for_role(role_family),
                    "synthesis_status": "institutional_actor_profile",
                    "compatibility_assessment": {
                        "status": "not_applicable_to_individual_seed",
                        "basis": [
                            "The KG actor is an organization or institution.",
                            "Nemotron-Personas-Korea contains individual rows, so no individual seed is attached.",
                        ],
                        "uncertainties": [],
                    },
                },
                "source": base.source or "policy_graph",
            })
        ]

    persona_config = config.get("persona", {})
    count = int(persona_config.get("personas_per_role", 1))
    try:
        from persona.nemotron_sampler import NemotronPersonaSampler

        sampler_config = dict(config)
        sampler_config["_current_policy_context"] = policy_context.model_dump()
        seeds = NemotronPersonaSampler(sampler_config).sample_for_entity(entity, entity_type, count)
    except Exception as exc:
        if persona_config.get("fallback_to_llm", True):
            print(f"  Nemotron sampler failed; falling back to LLM-only persona: {exc}")
            return [base]
        raise

    if not seeds:
        return [base]

    synthesis_config = persona_config.get("nemotron", {}).get("synthesis", {})
    merge_seed_into_role = not (
        synthesis_config.get("enabled", False)
        and synthesis_config.get("scope", "selected") == "selected"
    )
    return [
        _merge_nemotron_seed(
            base,
            seed,
            index,
            actor_modality=actor_modality,
            role_family=role_family,
            merge_seed_into_role=merge_seed_into_role,
        )
        for index, seed in enumerate(seeds, 1)
    ]


def _expand_with_scope_definer(
    base: PersonaProfile,
    entity: Any,
    entity_type: str,
    policy_context: PolicyContext,
    config: dict,
) -> list[PersonaProfile]:
    """Definer + tiered DuckDB pre-filter path. Shares `_merge_nemotron_seed` for the merge."""
    actor_modality = _infer_actor_modality(entity, entity_type, base)
    role_family = _infer_role_family(entity, entity_type, base)
    if actor_modality == "organization":
        from stage1.mechanism_panel import mechanism_panel_enabled
        if mechanism_panel_enabled(config):
            return _expand_with_kg_gen(base, entity, entity_type, policy_context, config)
        # Even though Nemotron sampling doesn't apply, save the org's KG context
        # to disk so the Stage 2 ToC scenario builder can pick it up as
        # shared scenario context (`shared_context.org_kg_contexts`).
        try:
            from persona.scope_definer import build_kg_context, save_kg_context
            kg_ctx = build_kg_context(entity, entity_type)
            save_kg_context(kg_ctx, config.get("_current_run_dir"), entity, entity_type)
        except Exception as exc:
            print(f"  [scope_definer] failed to save org kg_context for {entity_type}: {exc}")

        # The persona record itself also carries entity_id so the scenario
        # builder can correlate this org actor to its kg_context dump.
        org_entity_id = getattr(entity, "id", None) or getattr(entity, "uuid", None) or entity_type
        return [
            base.model_copy(update={
                "demographics": {
                    **(base.demographics or {}),
                    "agent_modality": "organization",
                    "role_family": role_family,
                    "constraint_scope": _constraint_scope_for_role(role_family),
                    "synthesis_status": "institutional_actor_profile",
                    "entity_id": org_entity_id,
                    "entity_type": entity_type,
                    "compatibility_assessment": {
                        "status": "not_applicable_to_individual_seed",
                        "basis": [
                            "The KG actor is an organization or institution.",
                            "Nemotron-Personas-Korea contains individual rows, so no individual seed is attached.",
                        ],
                        "uncertainties": [],
                    },
                },
                "source": base.source or "policy_graph",
            })
        ]

    persona_config = config.get("persona", {})
    try:
        from persona.scope_sampler import sample_for_entity as scope_sample_for_entity

        sampler_config = dict(config)
        sampler_config["_current_policy_context"] = policy_context.model_dump()
        seeds = scope_sample_for_entity(entity, entity_type, sampler_config)
    except Exception as exc:
        if persona_config.get("fallback_to_llm", True):
            print(f"  Scope definer failed; falling back to LLM-only persona: {exc}")
            return [base]
        raise

    if not seeds:
        return [base]

    return [
        _merge_nemotron_seed(
            base,
            seed,
            index,
            actor_modality=actor_modality,
            role_family=role_family,
            merge_seed_into_role=False,
        )
        for index, seed in enumerate(seeds, 1)
    ]


# ---------------------------------------------------------------------------
# kg_gen sampler: LLM-only persona generation in Nemotron-compatible schema
# ---------------------------------------------------------------------------

_KG_GEN_LANG = {
    "kor": "Korean", "singapore": "English", "usa": "English",
    "fr": "French", "ger": "German", "jp": "Japanese", "scot": "English",
}
_KG_GEN_COUNTRY_LABEL = {
    "kor": "Republic of Korea", "singapore": "Singapore",
    "usa": "United States", "fr": "France", "ger": "Germany", "jp": "Japan",
    "scot": "Scotland, United Kingdom",
}


def _kg_gen_build_prompt(
    base: "PersonaProfile",
    entity_type: str,
    policy_context: "PolicyContext",
    n: int,
    language: str,
    country_label: str,
) -> list[dict[str, str]]:
    kg = base.kg_context or {}
    rules = kg.get("rules") or {}
    obligations = rules.get("obligations") or []
    eligible_actions = rules.get("eligible_actions") or []

    actor_lines = [f"Role: {entity_type}"]
    role_context = (base.demographics or {}).get("policy_role_context") or {}
    if role_context:
        actor_lines.append(f"Functional role: {role_context.get('role_label')}")
        if role_context.get("mechanism_functions"):
            actor_lines.append(
                "Mechanism functions: " + ", ".join(role_context["mechanism_functions"])
            )
        if role_context.get("causal_leverage_points"):
            actor_lines.append(
                "Causal leverage points: " + "; ".join(role_context["causal_leverage_points"])
            )
    if base.core_objectives:
        actor_lines.append("Objectives: " + "; ".join(str(o) for o in base.core_objectives[:3]))
    if base.core_constraints:
        actor_lines.append("Constraints: " + "; ".join(str(c) for c in base.core_constraints[:3]))
    if obligations:
        actor_lines.append("Obligations: " + "; ".join(str(o.get("fact") or o) for o in obligations[:3]))
    if eligible_actions:
        actor_lines.append("Eligible actions: " + "; ".join(str(a.get("fact") or a) for a in eligible_actions[:3]))
    actor_block = "\n".join(actor_lines)

    ecosystem_lines = [
        f"Policy name: {policy_context.policy_name}",
        f"Domain: {policy_context.policy_domain}",
    ]
    if policy_context.target_population:
        ecosystem_lines.append("Target population: " + ", ".join(policy_context.target_population[:3]))
    if policy_context.expected_outcomes:
        ecosystem_lines.append("Expected outcomes: " + "; ".join(policy_context.expected_outcomes[:2]))
    ecosystem_block = "\n".join(ecosystem_lines)

    system = (
        f"You generate personas grounded in policy knowledge graph context. "
        f"Each persona is a distinct individual with self-consistent demographics, "
        f"professional narrative, skills, and goals. Narrative fields must be in {language}. "
        f"Demographic values must be plausible for {country_label} and consistent with the "
        f"actor's eligibility inferred from the policy context. "
        f"Output valid JSON only — no markdown, no code fences."
    )
    user = (
        f"Generate {n} personas who would plausibly participate in this policy as the focal actor ({entity_type}).\n\n"
        f"--- POLICY ECOSYSTEM ---\n{ecosystem_block}\n\n"
        f"--- FOCAL ACTOR ---\n{actor_block}\n\n"
        f"--- PERSONA SCHEMA ---\n"
        f"Each persona is a JSON object with these fields:\n"
        f"- name: a plausible full name for {country_label}\n"
        f"- professional_persona: 2-3 sentences in {language}\n"
        f"- skills_and_expertise: 1-2 sentences in {language}\n"
        f"- career_goals_and_ambitions: 1-2 sentences in {language}\n"
        f"- sex\n- age (integer)\n- marital_status\n- military_status\n"
        f"- education_level\n- occupation\n- district\n- province\n- bachelors_field\n\n"
        f"All demographic values must be plausible {country_label} values consistent with the actor role and eligibility.\n"
        f"Vary background, motivation, life circumstances, region, and field across the {n} personas.\n"
        f"Across the candidate set, cover the functional role's distinct causal leverage points. "
        f"At least one candidate must directly manage front-line operations or production when those are listed as a leverage point; "
        f"do not make every organizational representative a compliance, legal, tax, or government-relations specialist.\n\n"
        f'Return a JSON object: {{"personas": [...]}}. No prose, no markdown.'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _expand_with_kg_gen(
    base: "PersonaProfile",
    entity: Any,
    entity_type: str,
    policy_context: "PolicyContext",
    config: dict,
) -> list["PersonaProfile"]:
    import random
    import string

    actor_modality = _infer_actor_modality(entity, entity_type, base)
    role_family = _infer_role_family(entity, entity_type, base)

    if actor_modality == "organization":
        from stage1.mechanism_panel import mechanism_panel_enabled
        if mechanism_panel_enabled(config):
            actor_modality = "organization_representative"
        else:
            return [
                base.model_copy(update={
                    "demographics": {
                        **(base.demographics or {}),
                        "agent_modality": "organization",
                        "role_family": role_family,
                        "constraint_scope": _constraint_scope_for_role(role_family),
                        "synthesis_status": "institutional_actor_profile",
                    },
                    "source": base.source or "policy_graph",
                })
            ]

    kg_gen_cfg = config.get("persona", {}).get("kg_gen", {})
    from stage1.mechanism_panel import mechanism_panel_enabled
    if mechanism_panel_enabled(config):
        panel_cfg = (config.get("persona", {}).get("mechanism_panel") or {})
        default_count = int(panel_cfg.get("candidates_per_role", 5))
    else:
        default_count = int(config.get("persona", {}).get("personas_per_role", 5))
    n = int(kg_gen_cfg.get("personas_per_entity", default_count))
    country = kg_gen_cfg.get("country", "usa")
    language = _KG_GEN_LANG.get(country, "English")
    country_label = _KG_GEN_COUNTRY_LABEL.get(country, "United States")
    temperature = float(kg_gen_cfg.get("temperature", 0.9))

    prompt_actor = (
        f"{entity_type} human organization representative"
        if actor_modality == "organization_representative"
        else entity_type
    )
    messages = _kg_gen_build_prompt(base, prompt_actor, policy_context, n, language, country_label)
    call_config = {**config, "llm": {**config.get("llm", {}), "temperature": temperature}}

    try:
        raw = chat_completion(messages, call_config)
        personas_raw = json.loads(raw).get("personas") or []
    except Exception as exc:
        if config.get("persona", {}).get("fallback_to_llm", True):
            print(f"  [kg_gen] failed for {entity_type}; falling back to base: {exc}")
            return [base]
        raise

    if not personas_raw:
        return [base]

    _DEMO_FIELDS = (
        "name",
        "professional_persona", "skills_and_expertise", "career_goals_and_ambitions",
        "sex", "age", "marital_status", "military_status", "education_level",
        "occupation", "district", "province", "bachelors_field",
    )
    result = []
    for i, p in enumerate(personas_raw[:n], 1):
        if not isinstance(p, dict):
            continue
        uid = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        demographics: dict[str, Any] = {
            "agent_modality": actor_modality,
            "role_family": role_family,
            "constraint_scope": _constraint_scope_for_role(role_family),
            "kg_gen": True,
            "candidate_index": i,
        }
        for field in _DEMO_FIELDS:
            val = p.get(field)
            if val not in (None, ""):
                demographics[field] = val
        career = p.get("career_goals_and_ambitions")
        result.append(base.model_copy(update={
            "representative_type": f"{base.representative_type} | KG-Gen-{uid}",
            "core_objectives": _merge_limited(base.core_objectives, [career], limit=4),
            "demographics": demographics,
            "source": f"{base.source or 'policy_graph'}+kg_gen",
        }))
    return result or [base]


def _merge_nemotron_seed(
    base: PersonaProfile,
    seed: dict[str, Any],
    index: int,
    actor_modality: str = "individual",
    role_family: str = "affected_actor",
    merge_seed_into_role: bool = True,
) -> PersonaProfile:
    from persona.scope_sql_filter import PERSONA_OUTPUT_FIELDS as _COUNTRY_FIELDS
    name = _extract_seed_name(seed) or f"Nemotron-{index:03d}"
    seed_uuid = str(seed.get("uuid") or "")[:8]
    suffix = name if not seed_uuid else f"{name}-{seed_uuid}"
    demographics: dict[str, Any] = {
        "name": name,
        "nemotron_uuid": seed.get("uuid"),
        "agent_modality": actor_modality,
        "role_family": role_family,
        "constraint_scope": _constraint_scope_for_role(role_family),
        "candidate_index": index,
    }
    # Country-specific persona/demographic fields. Driven by PERSONA_OUTPUT_FIELDS
    # so SG/USA seeds don't try to pull KR-only columns like military_status.
    for field in _COUNTRY_FIELDS:
        demographics[field] = seed.get(field)
    # Cross-country common metadata (present in all Nemotron variants or written
    # by the sampler regardless of country).
    _META_FIELDS = (
        "hobbies_and_interests", "persona",
        "match_score", "match_strategy", "match_reasons", "match_warnings",
        "alignment_status", "adaptive_matching", "observable_eligibility",
        "eligibility_match", "rerank_score", "rerank_strategy",
        # Scope-definer provenance (populated only by the scope_definer sampler).
        "tier_used", "tier_pool_sizes", "used_fallback", "scope_definer_rationale",
        "entity_id", "entity_type",
    )
    for field in _META_FIELDS:
        demographics[field] = seed.get(field)
    demographics = {key: value for key, value in demographics.items() if value not in (None, "")}

    if not merge_seed_into_role:
        return base.model_copy(update={
            "representative_type": f"{base.representative_type} | {suffix}",
            "demographics": demographics,
            "source": f"{base.source or 'policy_graph'}+nemotron_personas_korea",
        })

    career_goal = seed.get("career_goals_and_ambitions")
    demographic_context = _demographic_sentence(seed)
    return base.model_copy(update={
        "representative_type": f"{base.representative_type} | {suffix}",
        "core_objectives": _merge_limited(base.core_objectives, [career_goal], limit=4),
        "core_constraints": _merge_limited(base.core_constraints, [demographic_context], limit=4),
        "demographics": demographics,
        "source": f"{base.source or 'policy_graph'}+nemotron_personas",
    })


def synthesize_simulation_participants(
    stakeholder_groups: list[StakeholderGroup],
    policy_context: PolicyContext,
    config: dict,
) -> list[StakeholderGroup]:
    """Use the KG+Nemotron synthesis prompt on the selected Stage 2 personas."""
    synthesis_config = (
        config.get("persona", {})
        .get("nemotron", {})
        .get("synthesis", {})
    )
    if not synthesis_config.get("enabled", False):
        return stakeholder_groups
    if synthesis_config.get("scope", "selected") != "selected":
        return stakeholder_groups

    max_calls = int(synthesis_config.get("max_calls", 0) or 0)
    calls = 0
    synthesized_groups = []
    for group in stakeholder_groups:
        synthesized_personas = []
        for persona in group.personas:
            if max_calls and calls >= max_calls:
                synthesized_personas.append(persona)
                continue
            if not (persona.demographics or {}).get("nemotron_uuid"):
                synthesized_personas.append(persona)
                continue
            synthesized_personas.append(_synthesize_nemotron_persona(persona, policy_context, config))
            calls += 1
        synthesized_groups.append(StakeholderGroup(
            stakeholder_type=group.stakeholder_type,
            personas=synthesized_personas,
        ))
    return synthesized_groups


def _synthesize_nemotron_persona(
    persona: PersonaProfile,
    policy_context: PolicyContext,
    config: dict,
) -> PersonaProfile:
    demographics = persona.demographics or {}
    role_profile = {
        "representative_type": persona.representative_type.split("|", 1)[0].strip(),
        "role_entity_id": persona.role_entity_id,
        "iad_category": persona.iad_category,
        "agent_modality": demographics.get("agent_modality") or _fallback_agent_modality(persona),
        "role_family": demographics.get("role_family") or _infer_role_family(None, None, persona),
        "constraint_scope": demographics.get("constraint_scope") or _constraint_scope_for_role(
            demographics.get("role_family") or _infer_role_family(None, None, persona)
        ),
        "core_objectives": persona.core_objectives,
        "core_constraints": persona.core_constraints,
    }
    messages = [
        {"role": "system", "content": KG_NEMOTRON_PERSONA_SYNTHESIS_SYSTEM},
        {
            "role": "user",
            "content": KG_NEMOTRON_PERSONA_SYNTHESIS_USER.format(
                role_profile=json.dumps(role_profile, ensure_ascii=False),
                kg_context=json.dumps(_authoritative_kg_context(persona.kg_context), ensure_ascii=False),
                policy_context=json.dumps(policy_context.model_dump(), ensure_ascii=False),
                nemotron_seed=json.dumps(demographics, ensure_ascii=False),
            ),
        },
    ]
    try:
        data = json.loads(chat_completion(messages, config))
    except Exception as exc:
        updated_demographics = _fallback_synthesis_demographics(
            persona,
            demographics,
            status="fallback_after_error",
            warning=str(exc),
        )
        return persona.model_copy(update={"demographics": updated_demographics})

    status = data.get("status")
    synthesized = data.get("persona") or {}
    if status != "accepted" or not synthesized:
        updated_demographics = _fallback_synthesis_demographics(
            persona,
            demographics,
            status="retained_without_prompt_synthesis",
            warning=data.get("rejection_reason") or "Synthesis prompt did not return a persona.",
        )
        return persona.model_copy(update={"demographics": updated_demographics})

    policy_position = synthesized.get("policy_position") or {}
    synthesized_demographics = synthesized.get("demographics") or {}
    translated_seed_profile = synthesized.get("translated_seed_profile") or {}
    translated_demographics = {
        f"{key}_en": value
        for key, value in translated_seed_profile.items()
        if value not in (None, "", [])
    }
    synthesized_demographics_en = {
        f"{key}_en": value
        for key, value in synthesized_demographics.items()
        if key != "age" and value not in (None, "", [])
    }
    updated_demographics = {
        **demographics,
        **synthesized_demographics_en,
        **translated_demographics,
        "agent_modality": synthesized.get("agent_modality") or _fallback_agent_modality(persona),
        "role_family": synthesized.get("role_family") or role_profile.get("role_family"),
        "constraint_scope": synthesized.get("constraint_scope") or role_profile.get("constraint_scope"),
        "persona_profile": _normalize_role_mapping_language(
            synthesized.get("persona_profile"),
            str(synthesized.get("policy_role") or role_profile["representative_type"]),
        ),
        "compatibility_assessment": synthesized.get("compatibility_assessment"),
        "synthesis_status": "accepted",
        "synthesis_source": "kg_nemotron_persona_synthesis_prompt",
    }
    scoped_constraints = _scope_constraints_to_role(
        policy_position.get("constraints") or persona.core_constraints,
        str(role_profile.get("role_family") or "affected_actor"),
        persona.kg_context,
    )
    return persona.model_copy(update={
        "core_objectives": policy_position.get("objectives") or persona.core_objectives,
        "core_constraints": scoped_constraints,
        "demographics": {
            key: value for key, value in updated_demographics.items()
            if value not in (None, "", [])
        },
    })


def _fallback_synthesis_demographics(
    persona: PersonaProfile,
    demographics: dict[str, Any],
    status: str,
    warning: str,
) -> dict[str, Any]:
    updated = {
        **demographics,
        "agent_modality": _fallback_agent_modality(persona),
        "role_family": demographics.get("role_family") or _infer_role_family(None, None, persona),
        "constraint_scope": demographics.get("constraint_scope") or _constraint_scope_for_role(
            demographics.get("role_family") or _infer_role_family(None, None, persona)
        ),
        "persona_profile": _fallback_persona_profile(persona, demographics),
        "compatibility_assessment": {
            "status": status,
            "basis": [
                "Selected by deterministic Nemotron sampling and ranking.",
                "KG actor profile retained as the authoritative policy role.",
            ],
            "uncertainties": [warning] if warning else [],
        },
        "synthesis_status": status,
        "synthesis_warning": warning,
        "synthesis_source": "deterministic_fallback_after_kg_nemotron_prompt",
    }
    return {key: value for key, value in updated.items() if value not in (None, "", [])}


def _authoritative_kg_context(context: dict[str, Any]) -> dict[str, Any]:
    """Remove non-authoritative legacy evidence data before LLM persona synthesis."""
    if not context:
        return {}
    return dict(context)


def _scope_constraints_to_role(
    constraints: list[str],
    role_family: str,
    policy_context: dict[str, Any],
) -> list[str]:
    """Prevent direct-beneficiary eligibility from becoming another role's own constraint."""
    if role_family == "direct_beneficiary":
        return constraints

    scoped: list[str] = []
    for constraint in constraints or []:
        text = str(constraint or "").strip()
        if not text:
            continue
        lowered = text.lower()
        mentions_beneficiary_eligibility = any(
            marker in lowered
            for marker in (
                "eligibility",
                "eligible",
                "ineligible",
                "degree",
                "age",
                "nationality",
                "employment status",
                "applicant",
                "beneficiary",
                "fellow",
            )
        )
        if mentions_beneficiary_eligibility:
            replacement = _role_scoped_eligibility_constraint(role_family)
            if replacement and replacement not in scoped:
                scoped.append(replacement)
            continue
        if "approval" in lowered and not _policy_context_has_edge(policy_context, "APPROV"):
            continue
        scoped.append(text)

    if scoped:
        return scoped[:4]
    fallback = _role_scoped_eligibility_constraint(role_family)
    return [fallback] if fallback else list(constraints or [])[:4]


def _role_scoped_eligibility_constraint(role_family: str) -> str | None:
    if role_family == "evaluator":
        return (
            "Must apply documented evaluation and eligibility criteria consistently when reviewing applicants; "
            "beneficiary eligibility is not the evaluator's own demographic constraint."
        )
    if role_family == "host_organization":
        return (
            "Can host or support only participants selected under the fellowship rules; individual beneficiary "
            "eligibility is not an institutional demographic constraint."
        )
    if role_family == "policy_administrator":
        return (
            "Must administer eligibility, selection, budget, and accountability rules without becoming subject to "
            "beneficiary eligibility."
        )
    if role_family == "mentor":
        return (
            "Must provide expertise and career guidance within the program rules without inheriting beneficiary "
            "eligibility constraints."
        )
    return None


def _policy_context_has_edge(policy_context: dict[str, Any], marker: str) -> bool:
    marker = marker.upper()
    for section_value in (policy_context or {}).values():
        if isinstance(section_value, dict):
            iterable = [item for values in section_value.values() for item in (values or [])]
        elif isinstance(section_value, list):
            iterable = section_value
        else:
            continue
        for item in iterable:
            edge_type = str((item or {}).get("edge_type") or "").upper()
            if marker in edge_type:
                return True
    return False


def _normalize_role_mapping_language(profile: Any, policy_role: str) -> str | None:
    if profile in (None, ""):
        return None
    text = str(profile)
    replacements = {
        f"currently a {policy_role}": f"modeled under the {policy_role} role",
        f"currently an {policy_role}": f"modeled under the {policy_role} role",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _fallback_agent_modality(persona: PersonaProfile) -> str:
    label = " ".join(
        str(value or "")
        for value in (persona.role_entity_id, persona.iad_category, persona.representative_type)
    ).lower()
    if any(term in label for term in ("committee", "panel", "review board", " rb")):
        return "committee_member"
    if persona.iad_category == "Positions":
        return "official_position_holder"
    if any(
        term in label
        for term in (
            "agency",
            "committee",
            "company",
            "foundation",
            "government",
            "host",
            "institution",
            "ministry",
            "office",
            "organization",
            "panel",
            "university",
        )
    ):
        return "organization_representative"
    if any(term in label for term in ("applicants", "beneficiaries", "fellows", "group", "population", "stakeholder")):
        return "stakeholder_group_member"
    return "individual"


def _infer_actor_modality(
    entity: Any | None,
    entity_type: str | None,
    persona: PersonaProfile | None = None,
) -> str:
    attributes = getattr(entity, "attributes", {}) or {}
    configured = attributes.get("agent_modality") or attributes.get("actor_modality")
    text = " ".join(
        str(value or "")
        for value in (
            entity_type,
            getattr(entity, "name", ""),
            getattr(entity, "summary", ""),
            getattr(entity, "uuid", ""),
            getattr(persona, "representative_type", ""),
            getattr(persona, "role_entity_id", ""),
        )
    ).lower()
    iad_category = attributes.get("iad_category") or getattr(persona, "iad_category", None)

    organization_markers = (
        "institution",
        "agency",
        "ministry",
        "foundation",
        "university",
        "organization",
        "institute",
        "researchinstitution",
        "fundingagency",
    )
    if any(marker in text for marker in organization_markers):
        inferred = "organization"
    elif any(marker in text for marker in ("panel", "committee", "review board", " rb")):
        inferred = "committee_member"
    elif iad_category == "Positions" or any(marker in text for marker in ("authority", "approvalauthority", "officer", "reviewer", "administrator")):
        inferred = "official_position_holder"
    elif any(marker in text for marker in ("group", "students", "applicants", "candidates", "population")):
        inferred = "stakeholder_group_member"
    else:
        inferred = "individual"
    if configured:
        configured_text = str(configured)
        if configured_text == "individual" and inferred in {
            "organization",
            "committee_member",
            "official_position_holder",
            "stakeholder_group_member",
        }:
            return inferred
        return configured_text
    return inferred


def _infer_role_family(
    entity: Any | None,
    entity_type: str | None,
    persona: PersonaProfile | None = None,
) -> str:
    attributes = getattr(entity, "attributes", {}) or {}
    configured = attributes.get("role_family") or attributes.get("policy_role_family")
    if configured:
        return _normalize_role_family(configured)

    text = " ".join(
        str(value or "")
        for value in (
            entity_type,
            getattr(entity, "name", ""),
            getattr(entity, "summary", ""),
            getattr(entity, "uuid", ""),
            getattr(persona, "representative_type", ""),
            getattr(persona, "role_entity_id", ""),
        )
    ).lower()
    if any(term in text for term in ("evaluation", "panel", "reviewer", "committee", "rb")):
        return "evaluator"
    if any(term in text for term in ("institution", "university", "institute", "host")):
        return "host_organization"
    if any(term in text for term in ("ministry", "foundation", "agency", "program officer", "administrator", "authority", "approval")):
        return "policy_administrator"
    if any(term in text for term in ("mentor", "mentoring")):
        return "mentor"
    if any(term in text for term in ("postdoctoral", "researcher", "fellow", "beneficiary", "applicant", "candidate")):
        return "direct_beneficiary"
    return "affected_actor"


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


def _constraint_scope_for_role(role_family: str | None) -> str:
    if role_family == "direct_beneficiary":
        return "Apply direct beneficiary eligibility and exclusion constraints when observable; keep unobservable eligibility as uncertainty."
    if role_family == "evaluator":
        return "Do not apply beneficiary eligibility constraints. Use evaluation criteria, expertise, fairness, and panel legitimacy."
    if role_family == "host_organization":
        return "Do not apply individual eligibility constraints. Use hosting capacity, employment/contracting, research administration, and compliance constraints."
    if role_family == "policy_administrator":
        return "Use budget, implementation, governance, selection, and accountability constraints."
    if role_family == "mentor":
        return "Use expertise, mentoring capacity, network support, and career guidance constraints."
    return "Use only constraints directly scoped to this KG role; do not inherit constraints from adjacent actors."


def _fallback_persona_profile(persona: PersonaProfile, demographics: dict[str, Any]) -> str:
    role = persona.representative_type.split("|", 1)[0].strip()
    role_family = demographics.get("role_family") or _infer_role_family(None, None, persona)
    occupation = demographics.get("occupation") or "an unspecified occupation"
    education = demographics.get("education_level") or "unspecified education"
    field = demographics.get("bachelors_field") or "an unspecified field"
    age = demographics.get("age")
    age_text = f"{age}-year-old " if age not in (None, "") else ""
    return (
        f"This participant is modeled as a {role} simulation actor ({role_family}) using a Nemotron seed with "
        f"{age_text}{occupation}, {education}, and {field} background. The policy-facing role, "
        "objectives, constraints, and expected stance remain grounded in the KG; the Nemotron row "
        "is retained only to add realistic individual variation and uncertainty. Role-specific constraints "
        "must not be transferred from other KG actor roles."
    )


def _extract_seed_name(seed: dict[str, Any]) -> str | None:
    return infer_persona_name(seed)


def _demographic_sentence(seed: dict[str, Any]) -> str:
    """Stitch a one-line demographic descriptor from whichever country fields are present.

    Country-aware: walks the active Nemotron schema's PERSONA_OUTPUT_FIELDS so SG/USA
    seeds don't waste lookups on KR-only columns (family_type / province / etc.).
    """
    from persona.scope_sql_filter import PERSONA_OUTPUT_FIELDS as _COUNTRY_FIELDS
    parts = []
    # Prioritize anchors first so the sentence reads naturally, then fill in
    # remaining country fields in their declared order.
    priority = ("age", "sex", "occupation")
    walked: set[str] = set()
    for key in priority:
        if key in _COUNTRY_FIELDS:
            value = seed.get(key)
            if value not in (None, ""):
                parts.append(str(value))
            walked.add(key)
    for key in _COUNTRY_FIELDS:
        if key in walked:
            continue
        # Skip the long free-text persona fields — we only want demographic anchors.
        if key in {"professional_persona", "skills_and_expertise",
                   "family_persona", "career_goals_and_ambitions"}:
            continue
        value = seed.get(key)
        if value not in (None, ""):
            parts.append(str(value))
    return "Nemotron demographic context: " + ", ".join(parts)


def _merge_limited(base_items: list[str], extra_items: list[Any], limit: int) -> list[str]:
    merged = list(base_items)
    for item in extra_items:
        text = str(item or "").strip()
        if text and text not in merged:
            merged.append(text)
    return merged[:limit]


def _append_context(base_text: str, context: Any) -> str:
    context_text = str(context or "").strip()
    if not context_text:
        return base_text
    return f"{base_text} Individual background: {context_text}"


def generate_personas_from_graph(
    graph_id: str,
    ontology: dict,
    policy_context: PolicyContext,
    context: EconomicContext,
    config: dict,
) -> list[StakeholderGroup]:
    """
    Read entities from the Zep graph, group by entity type (= stakeholder group),
    and generate one PersonaProfile per entity using entity-grounded context.

    Filters out non-actor types listed in config['stage1']['non_actor_types'].
    Caps each group at config['stage1']['personas_per_category'] entities.
    """
    from graph.entity_reader import EntityReader

    zep_api_key = os.environ.get(config["zep"]["api_key_env"])
    if not zep_api_key:
        raise ValueError(
            f"Zep API key not found. Set the '{config['zep']['api_key_env']}' environment variable."
        )

    reader = EntityReader(zep_api_key)

    allowed_types = [t["name"] for t in ontology["entity_types"]]
    persona_eligible_types = {
        t["name"]
        for t in ontology["entity_types"]
        if t.get("persona_eligible", t.get("is_actor", True))
    }
    non_actor = set(config["stage1"].get("non_actor_types", []))
    max_per_group = config["stage1"]["personas_per_category"]

    print(f"Reading entities from graph {graph_id}...")
    entities = reader.get_filtered_entities(graph_id, allowed_types)
    print(f"  {len(entities)} entities retrieved.")

    # Group by entity type, excluding non-actor types
    by_type: dict[str, list[Any]] = {}
    for entity in entities:
        etype = entity.get_entity_type()
        if etype and etype not in non_actor and etype in persona_eligible_types:
            by_type.setdefault(etype, []).append(entity)

    stakeholder_groups = []
    for entity_type, type_entities in by_type.items():
        # Rank by KG degree (edge connectivity) — entities mentioned more often in
        # the policy text tend to have more edges and make better anchors than
        # generic catch-all definitions extracted from the policy intro.
        ranked = sorted(type_entities, key=lambda e: -len(getattr(e, "related_edges", []) or []))
        capped = ranked[:max_per_group]
        print(f"  Generating {len(capped)} persona(s) for type '{entity_type}'...")
        for e in capped:
            print(f"    anchor: {e.name} (degree={len(getattr(e, 'related_edges', []) or [])})")
        personas = []
        for entity in capped:
            personas.extend(entity_to_personas(entity, entity_type, policy_context, context, config))
        stakeholder_groups.append(StakeholderGroup(
            stakeholder_type=entity_type,
            personas=personas,
        ))

    return stakeholder_groups


def generate_personas_from_policy_graph(
    policy_graph_path: str,
    policy_context: PolicyContext,
    context: EconomicContext,
    config: dict,
    ontology: dict | None = None,
    role_prototypes: list[RolePrototype] | None = None,
) -> list[StakeholderGroup]:
    """
    Generate personas from a local policy_graph.json artifact.

    The local graph keeps actor nodes central but also carries IAD/IG context from
    non-actor policy objects and regulative edges.
    """
    from graph.local_graph_reader import LocalGraphReader

    reader = LocalGraphReader(policy_graph_path)
    if role_prototypes:
        from stage1.mechanism_panel import attach_role_prototype

        all_entities = reader.get_filtered_entities(allowed_types=None, persona_only=False)
        entity_by_id = {
            str(getattr(entity, "id", None) or getattr(entity, "uuid", None)): entity
            for entity in all_entities
        }
        stakeholder_groups = []
        for prototype in role_prototypes:
            entity = entity_by_id.get(prototype.anchor_entity_id)
            if entity is None:
                raise ValueError(
                    f"Role prototype {prototype.prototype_id!r} references missing KG anchor "
                    f"{prototype.anchor_entity_id!r}"
                )
            entity_type = entity.get_entity_type() or prototype.domain_role
            print(
                f"  Generating mechanism-panel candidates for '{prototype.prototype_id}' "
                f"from anchor {prototype.anchor_entity_id}..."
            )
            if prototype.agent_modality == "organization_representative":
                base = entity_to_persona(entity, entity_type, policy_context, context, config)
                attach_role_prototype(base, prototype)
                personas = _expand_with_kg_gen(base, entity, entity_type, policy_context, config)
            else:
                personas = entity_to_personas(
                    entity,
                    entity_type,
                    policy_context,
                    context,
                    config,
                    role_prototype=prototype,
                )
            candidate_count = int(
                ((config.get("persona") or {}).get("mechanism_panel") or {})
                .get("candidates_per_role", 5)
            )
            personas = personas[:candidate_count]
            if not prototype.objectives and personas:
                prototype.objectives = list(personas[0].core_objectives)[:2]
            if not prototype.constraints and personas:
                prototype.constraints = list(personas[0].core_constraints)[:2]
            for persona in personas:
                attach_role_prototype(persona, prototype)
            stakeholder_groups.append(StakeholderGroup(
                stakeholder_type=prototype.domain_role,
                personas=personas,
            ))
        return stakeholder_groups

    # The validated local graph is authoritative for actor handoff. Some useful
    # actor anchors are recovered from source-supported edges even when the
    # generated ontology omitted that exact type, so do not re-prune by ontology.
    allowed_types = None
    max_per_group = config["stage1"]["personas_per_category"]

    anchor_policy = config.get("persona", {}).get("actor_anchor_policy", "persona_eligible")
    persona_only = anchor_policy == "persona_eligible"
    print(f"Reading simulation actor anchors from local policy graph: {policy_graph_path}")
    entities = reader.get_filtered_entities(allowed_types=allowed_types, persona_only=persona_only)
    entities = [entity for entity in entities if _is_simulation_actor_anchor(entity, config)]
    print(f"  {len(entities)} simulation actor anchor entities retrieved.")

    by_type: dict[str, list[Any]] = {}
    for entity in entities:
        etype = entity.get_entity_type()
        if etype:
            by_type.setdefault(etype, []).append(entity)

    stakeholder_groups = []
    for entity_type, type_entities in by_type.items():
        # Rank by KG degree (edge connectivity) — entities mentioned more often in
        # the policy text tend to have more edges and make better anchors than
        # generic catch-all definitions extracted from the policy intro.
        ranked = sorted(type_entities, key=lambda e: -len(getattr(e, "related_edges", []) or []))
        capped = ranked[:max_per_group]
        print(f"  Generating {len(capped)} persona(s) for type '{entity_type}'...")
        for e in capped:
            print(f"    anchor: {e.name} (degree={len(getattr(e, 'related_edges', []) or [])})")
        personas = []
        for entity in capped:
            personas.extend(entity_to_personas(entity, entity_type, policy_context, context, config))
        stakeholder_groups.append(StakeholderGroup(
            stakeholder_type=entity_type,
            personas=personas,
        ))

    return stakeholder_groups


def _is_simulation_actor_anchor(entity: Any, config: dict) -> bool:
    etype = entity.get_entity_type()
    if etype in set(config["stage1"].get("non_actor_types", [])):
        return False

    attributes = getattr(entity, "attributes", {}) or {}
    if attributes.get("persona_eligible") is False:
        return False
    if attributes.get("persona_eligible", False):
        return True

    anchor_policy = config.get("persona", {}).get("actor_anchor_policy", "persona_eligible")
    if anchor_policy not in {"iad_actor", "eligible_or_iad_actor"}:
        return False

    iad_category = attributes.get("iad_category")
    if iad_category not in {"Participants", "Positions"}:
        return False

    # IAD actor anchors may be individuals, groups, organizations, or official
    # positions. They need not be statutory beneficiaries, but they should have
    # some policy relation to avoid turning passive objects into personas.
    return bool(
        attributes.get("is_actor", False)
        or iad_category == "Positions"
        or getattr(entity, "related_edges", [])
    )


def generate_personas_no_graph(
    policy_context: PolicyContext,
    config: dict,
) -> list[StakeholderGroup]:
    """
    Generate stakeholder personas from past policy text only — no Zep graph, no entity grounding.
    Used for the no-entity ablation. Grounds personas in past_policy_dir documents.
    """
    past_policy_text = load_pdfs(config["paths"]["past_policy_dir"], config)
    n_personas = config["stage1"]["personas_per_category"]

    messages = [
        {"role": "system", "content": PERSONAS_FROM_TEXT_SYSTEM},
        {"role": "user", "content": PERSONAS_FROM_TEXT_USER.format(
            past_policy_text=past_policy_text,
            policy_context=json.dumps(policy_context.model_dump(), ensure_ascii=False),
            n_personas=n_personas,
        )},
    ]
    data = json.loads(chat_completion(messages, config))

    stakeholder_groups = []
    for group in data["stakeholder_groups"]:
        personas = [PersonaProfile(**p) for p in group["personas"]]
        stakeholder_groups.append(StakeholderGroup(
            stakeholder_type=group["stakeholder_type"],
            personas=personas,
        ))
    return stakeholder_groups
