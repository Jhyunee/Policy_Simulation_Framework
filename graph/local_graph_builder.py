"""Local IAD/IG policy KG builder that does not depend on Zep."""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

from llm.client import chat_completion
from models.prompts import (
    KG_EDGE_EXTRACTOR_SYSTEM,
    KG_EDGE_EXTRACTOR_USER,
    KG_NODE_EXTRACTOR_SYSTEM,
    KG_NODE_EXTRACTOR_USER,
    KG_EDGE_EXTRACTOR_SYSTEM_FREEFORM,
    KG_EDGE_EXTRACTOR_USER_FREEFORM,
    KG_NODE_EXTRACTOR_SYSTEM_FREEFORM,
    KG_NODE_EXTRACTOR_USER_FREEFORM,
)
from graph.normalizer import make_node_id, merge_nodes, normalize_name, normalize_node
from graph.validators import validate_policy_graph

_NODE_FAMILY_NAMES = {
    "ActorParticipant",
    "PositionRole",
    "EligibilityCondition",
    "RestrictionConstraint",
    "PolicyActionProcess",
    "InformationEvidence",
    "ResourceBenefitCost",
    "AuthorityControl",
    "TemporalMilestone",
    "EvaluationMetricOutcome",
}


def build_policy_graph(
    text: str,
    ontology: dict[str, Any],
    config: dict[str, Any],
    simulation_requirement: str = "",
) -> dict[str, Any]:
    """
    Build a local policy KG with a two-phase extraction process.

    Phase 1 extracts all nodes from chunks and deduplicates them into a global node pool.
    Phase 2 extracts edges with the full node pool available, allowing cross-chunk references.
    """
    kg_config = config.get("kg", {})
    chunk_size = kg_config.get("chunk_size", 3500)
    chunk_overlap = kg_config.get("chunk_overlap", 300)
    max_workers = max(1, int(kg_config.get("max_workers", 1)))
    chunks = _split_text(text, chunk_size, chunk_overlap)

    type_map = {t["name"]: t for t in ontology.get("entity_types", [])}
    extraction_warnings: list[str] = []

    # Phase 1: extract nodes per chunk (parallelizable, chunks are independent).
    node_results = _run_chunk_phase(
        chunks,
        max_workers,
        lambda chunk, chunk_id: _extract_nodes_from_chunk(chunk, chunk_id, ontology, config, simulation_requirement),
        phase_label="phase1-nodes",
    )
    mode = _kg_mode(config)
    all_nodes: list[dict[str, Any]] = []
    for chunk_id, extracted in node_results:
        extraction_warnings.extend(extracted.get("warnings", []))
        for raw_node in extracted.get("nodes", []):
            if not raw_node.get("name"):
                extraction_warnings.append(f"Skipped node with empty name in {chunk_id}: {raw_node}")
                continue
            if mode == "freeform":
                # Free-form ablation: the LLM did not emit IAD/IG enum slots.
                # Pre-populate placeholders so downstream IAD-typed lookups
                # (normalize_node defaults, persona/scope_definer.build_kg_context,
                # stage1 modules) all see well-formed dicts and silently degrade
                # via their existing .get() lookups instead of crashing.
                raw_node.setdefault("iad_category", "Information")
                raw_node.setdefault("node_family", "")
                raw_node.setdefault("iad_rule_type", None)
                type_meta = _fallback_node_type_meta(raw_node)
                # CRITICAL: normalize_node reads is_actor/persona_eligible from
                # type_meta only, NOT from raw_node. The fallback defaults both
                # to False, so without this passthrough the free-form graph would
                # have zero persona-eligible nodes (the persona generator would
                # find no actor anchors). Honor what the LLM emitted.
                if raw_node.get("is_actor") is not None:
                    type_meta["is_actor"] = bool(raw_node["is_actor"])
                if raw_node.get("persona_eligible") is not None:
                    type_meta["persona_eligible"] = bool(raw_node["persona_eligible"])
            else:
                node_type = _recover_node_type(raw_node, type_map)
                raw_node["type"] = node_type
                type_meta = type_map.get(node_type) or _fallback_node_type_meta(raw_node)
                if node_type not in type_map:
                    extraction_warnings.append(f"Kept node with non-ontology type in {chunk_id}: {raw_node}")
            all_nodes.append(normalize_node(raw_node, type_meta, chunk_id=chunk_id))

    # Barrier: phase 1 must finish (so phase 2 sees the full node pool) before phase 2 begins.
    nodes = merge_nodes(all_nodes)
    max_node_pool_context = kg_config.get("max_node_pool_context", 120)
    compact_pool = _compact_nodes_for_prompt(nodes)

    # Phase 2: extract edges per chunk against the global node pool (parallelizable).
    # When the graph fits within max_node_pool_context, every chunk gets the full pool (legacy
    # behavior). When it does not, each chunk gets the top-N most relevant nodes selected by
    # token overlap with the chunk text, so no node is silently dropped from the prompt.
    if len(compact_pool) <= max_node_pool_context:
        per_chunk_pool: dict[str, list[dict[str, Any]]] = {
            f"chunk_{idx:03d}": compact_pool for idx in range(1, len(chunks) + 1)
        }
    else:
        per_chunk_pool = {
            f"chunk_{idx:03d}": _select_relevant_nodes(compact_pool, chunk, max_node_pool_context)
            for idx, chunk in enumerate(chunks, 1)
        }

    edge_results = _run_chunk_phase(
        chunks,
        max_workers,
        lambda chunk, chunk_id: _extract_edges_from_chunk(
            chunk, chunk_id, ontology, per_chunk_pool[chunk_id], nodes, config, simulation_requirement,
        ),
        phase_label="phase2-edges",
    )
    all_edges: list[dict[str, Any]] = []
    for chunk_id, extracted in edge_results:
        extraction_warnings.extend(extracted.get("warnings", []))
        all_edges.extend(extracted.get("edges", []))

    nodes = _materialize_missing_endpoint_nodes(nodes, all_edges, ontology, extraction_warnings)
    builder_name = "local_freeform" if mode == "freeform" else "local_iad_ig"
    graph = {
        "graph_id": f"local_policy_graph_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "ontology": ontology,
        "nodes": nodes,
        "edges": _dedupe_edges(all_edges),
        "metadata": {
            "builder": builder_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "chunk_count": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "max_workers": max_workers,
            "simulation_requirement": simulation_requirement,
        },
        "validation_warnings": extraction_warnings,
    }
    # Skip IAD/IG-typed validation in free-form mode: validator's
    # _repair_iad_category and _NODE_FAMILY_NAMES checks would otherwise
    # re-cast our free-form attributes into IAD slots, defeating the ablation.
    if (config.get("kg") or {}).get("skip_iadig_validation"):
        return graph
    return validate_policy_graph(graph, ontology, config)


def _run_chunk_phase(
    chunks: list[str],
    max_workers: int,
    fn,
    phase_label: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Apply `fn(chunk_text, chunk_id)` over chunks, sequential or threaded.

    Always returns results in chunk order (chunk_001, chunk_002, ...) regardless of completion order
    so that downstream node/edge ordering and dedup keys remain deterministic across runs.
    """
    chunk_jobs = [(idx, f"chunk_{idx:03d}", chunk) for idx, chunk in enumerate(chunks, 1)]

    if max_workers <= 1 or len(chunk_jobs) <= 1:
        return [(chunk_id, fn(chunk, chunk_id)) for _, chunk_id, chunk in chunk_jobs]

    results: dict[int, tuple[str, dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=phase_label) as pool:
        future_to_idx = {
            pool.submit(fn, chunk, chunk_id): (idx, chunk_id)
            for idx, chunk_id, chunk in chunk_jobs
        }
        for future in as_completed(future_to_idx):
            idx, chunk_id = future_to_idx[future]
            results[idx] = (chunk_id, future.result())

    return [results[idx] for idx, _, _ in chunk_jobs]


def _kg_mode(config: dict[str, Any]) -> str:
    """Return 'iadig' (default) or 'freeform' for the 2×2 IAD/IG ablation."""
    return (config.get("kg") or {}).get("mode", "iadig")


def _extract_nodes_from_chunk(
    chunk_text: str,
    chunk_id: str,
    ontology: dict[str, Any],
    config: dict[str, Any],
    simulation_requirement: str,
) -> dict[str, Any]:
    if _kg_mode(config) == "freeform":
        messages = [
            {"role": "system", "content": KG_NODE_EXTRACTOR_SYSTEM_FREEFORM},
            {
                "role": "user",
                "content": KG_NODE_EXTRACTOR_USER_FREEFORM.format(
                    simulation_requirement=simulation_requirement,
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                ),
            },
        ]
    else:
        messages = [
            {"role": "system", "content": KG_NODE_EXTRACTOR_SYSTEM},
            {
                "role": "user",
                "content": KG_NODE_EXTRACTOR_USER.format(
                    simulation_requirement=simulation_requirement,
                    ontology=json.dumps(ontology, ensure_ascii=False),
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                ),
            },
        ]
    return json.loads(chat_completion(messages, config))


def _extract_edges_from_chunk(
    chunk_text: str,
    chunk_id: str,
    ontology: dict[str, Any],
    node_pool: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    config: dict[str, Any],
    simulation_requirement: str,
) -> dict[str, Any]:
    if _kg_mode(config) == "freeform":
        messages = [
            {"role": "system", "content": KG_EDGE_EXTRACTOR_SYSTEM_FREEFORM},
            {
                "role": "user",
                "content": KG_EDGE_EXTRACTOR_USER_FREEFORM.format(
                    node_pool=json.dumps(node_pool, ensure_ascii=False),
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    max_edges_per_chunk=int(config.get("kg", {}).get("max_edges_per_chunk", 20)),
                ),
            },
        ]
        return json.loads(chat_completion(messages, config))
    messages = [
        {"role": "system", "content": KG_EDGE_EXTRACTOR_SYSTEM},
        {
            "role": "user",
            "content": KG_EDGE_EXTRACTOR_USER.format(
                simulation_requirement=simulation_requirement,
                ontology=json.dumps(ontology, ensure_ascii=False),
                node_pool=json.dumps(node_pool, ensure_ascii=False),
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                max_edges_per_chunk=int(config.get("kg", {}).get("max_edges_per_chunk", 20)),
            ),
        },
    ]
    data = json.loads(chat_completion(messages, config))
    edges = []
    edge_types = {e["name"]: e for e in ontology.get("edge_types", [])}
    entity_types = {e["name"] for e in ontology.get("entity_types", [])}
    node_ids = {n["id"] for n in nodes}
    node_lookup = {(n["type"], normalize_name(n["name"])): n["id"] for n in nodes}
    node_name_lookup = {normalize_name(n["name"]): n["id"] for n in nodes}
    nodes_by_type: dict[str, list[str]] = {}
    for node in nodes:
        nodes_by_type.setdefault(node["type"], []).append(node["id"])

    for raw_edge in data.get("edges", []):
        edge_type = raw_edge.get("type") or "UNSPECIFIED_RELATION"
        if edge_type not in edge_types:
            data.setdefault("warnings", []).append(f"Kept edge with non-ontology type in {chunk_id}: {raw_edge}")
        source = _resolve_endpoint(raw_edge, "source", node_ids, node_lookup, node_name_lookup, nodes_by_type, entity_types)
        target = _resolve_endpoint(raw_edge, "target", node_ids, node_lookup, node_name_lookup, nodes_by_type, entity_types)
        if not source or not target:
            data.setdefault("warnings", []).append(f"Recovered edge endpoint placeholder in {chunk_id}: {raw_edge}")
        source = source or _placeholder_endpoint(raw_edge, "source")
        target = target or _placeholder_endpoint(raw_edge, "target")
        meta = edge_types.get(edge_type, {})
        edge_category = raw_edge.get("edge_category") or meta.get("edge_category", "structural")
        attributes = raw_edge.get("attributes") if isinstance(raw_edge.get("attributes"), dict) else {}
        activation_condition, execution_constraint = _condition_fields_from_edge(raw_edge, attributes)
        edges.append({
            "id": "",
            "type": edge_type,
            "edge_category": edge_category,
            "edge_family": raw_edge.get("edge_family") or meta.get("edge_family", ""),
            "source": source,
            "target": target,
            "fact": raw_edge.get("fact", ""),
            "deontic": raw_edge.get("deontic") or meta.get("ig_deontic"),
            "activation_condition": activation_condition,
            "execution_constraint": execution_constraint,
            "or_else_text": raw_edge.get("or_else_text"),
            "source_text": raw_edge.get("source_text", ""),
            "attributes": attributes,
        })

    data["edges"] = edges
    return data


def _fallback_node_type_meta(raw_node: dict[str, Any]) -> dict[str, Any]:
    category = raw_node.get("iad_category")
    node_type = raw_node.get("type") or "UnclassifiedPolicyObject"
    if category not in {"Participants", "Positions", "Actions", "Information", "Costs_and_Benefits"}:
        category = "Participants" if node_type in {"Participants", "Positions"} else "Information"
    is_actor = category in {"Participants", "Positions"}
    return {
        "name": node_type,
        "iad_category": category,
        "node_family": _fallback_node_family(category, is_actor),
        "is_actor": is_actor,
        "persona_eligible": False,
        "rationale": "Preserved non-ontology type for inspection instead of dropping it.",
    }


def _recover_node_type(raw_node: dict[str, Any], type_map: dict[str, dict[str, Any]]) -> str | None:
    """Recover common LLM mistakes where IAD category is emitted as type."""
    node_type = raw_node.get("type")
    if node_type in type_map:
        return node_type

    name = raw_node.get("name")
    if name in type_map:
        return name

    normalized_name = normalize_name(name or "")
    for type_name in type_map:
        if normalize_name(type_name) == normalized_name:
            return type_name

    category = raw_node.get("iad_category") or node_type
    if node_type in _NODE_FAMILY_NAMES:
        family_candidates = [
            type_name for type_name, meta in type_map.items()
            if meta.get("node_family") == node_type
        ]
        for type_name in family_candidates:
            if _shares_token(normalized_name, type_name):
                return type_name
        if family_candidates:
            return family_candidates[0]

    if category:
        candidates = [
            type_name for type_name, meta in type_map.items()
            if meta.get("iad_category") == category
        ]
        for type_name in candidates:
            if normalized_name.lower() in normalize_name(type_name).lower():
                return type_name
    return node_type


def _shares_token(left: str, right: str) -> bool:
    left_tokens = {token.lower() for token in normalize_name(left).replace("_", " ").split() if len(token) >= 3}
    right_tokens = {token.lower() for token in normalize_name(right).replace("_", " ").split() if len(token) >= 3}
    return bool(left_tokens & right_tokens)


def _fallback_node_family(category: str, is_actor: bool) -> str:
    if category == "Positions":
        return "PositionRole"
    if category == "Participants" or is_actor:
        return "ActorParticipant"
    if category == "Actions":
        return "PolicyActionProcess"
    if category == "Costs_and_Benefits":
        return "ResourceBenefitCost"
    return "InformationEvidence"


def _resolve_endpoint(
    raw_edge: dict[str, Any],
    side: str,
    node_ids: set[str],
    node_lookup: dict[tuple[str, str], str],
    node_name_lookup: dict[str, str],
    nodes_by_type: dict[str, list[str]],
    entity_types: set[str],
) -> str | None:
    value = raw_edge.get(side) or raw_edge.get(f"{side}_id")
    if value in node_ids:
        return value
    if isinstance(value, str) and ":" in value:
        return value
    if isinstance(value, str):
        normalized_value = normalize_name(value)
        if normalized_value in node_name_lookup:
            return node_name_lookup[normalized_value]
        if value in entity_types:
            existing = nodes_by_type.get(value)
            if existing:
                return existing[0]
            return make_node_id(value, value)

    node_type = raw_edge.get(f"{side}_type")
    node_name = raw_edge.get(f"{side}_name")
    if node_type and node_name:
        return node_lookup.get((node_type, normalize_name(node_name))) or make_node_id(node_type, node_name)
    return None


def _placeholder_endpoint(raw_edge: dict[str, Any], side: str) -> str:
    value = raw_edge.get(side) or raw_edge.get(f"{side}_id") or raw_edge.get(f"{side}_name") or f"unknown_{side}"
    if isinstance(value, str) and ":" in value:
        return value
    node_type = raw_edge.get(f"{side}_type") or "UnclassifiedPolicyObject"
    return make_node_id(node_type, str(value))


def _condition_fields_from_edge(
    raw_edge: dict[str, Any],
    attributes: dict[str, Any],
) -> tuple[Any, Any]:
    """Return IG 2.0 Cac/Cex fields, migrating deprecated condition_text when needed."""
    activation = _normalize_nullable(
        raw_edge.get("activation_condition")
        or raw_edge.get("activationCondition")
        or attributes.get("activation_condition")
    )
    execution = _normalize_nullable(
        raw_edge.get("execution_constraint")
        or raw_edge.get("executionConstraint")
        or attributes.get("execution_constraint")
    )
    legacy_condition = _normalize_nullable(_legacy_condition_text(raw_edge, attributes))
    if legacy_condition:
        attributes.setdefault("deprecated_condition_text", legacy_condition)
        if not activation and not execution:
            activation, execution = _split_legacy_condition_text(
                str(legacy_condition),
                edge_type=str(raw_edge.get("type") or ""),
            )
    return activation, execution


def _legacy_condition_text(raw_edge: dict[str, Any], attributes: dict[str, Any]) -> Any:
    for container in (raw_edge, attributes):
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


def _split_legacy_condition_text(condition_text: str, edge_type: str = "") -> tuple[str | None, str | None]:
    """Best-effort migration for old single-field condition_text values.

    Ambiguous clauses default to activation_condition because Cac is the safer
    backward-compatible interpretation of an old generic condition.
    """
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


def _materialize_missing_endpoint_nodes(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    ontology: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Create conservative placeholder nodes for valid ontology endpoints referenced by extracted edges."""
    node_ids = {node["id"] for node in nodes}
    type_map = {t["name"]: t for t in ontology.get("entity_types", [])}
    recovered = list(nodes)

    for edge in edges:
        for side in ("source", "target"):
            endpoint = edge.get(side)
            if not isinstance(endpoint, str) or endpoint in node_ids or ":" not in endpoint:
                continue
            node_type, name = endpoint.split(":", 1)
            if not name:
                continue
            meta = type_map.get(node_type) or {
                "iad_category": "Information",
                "is_actor": False,
                "persona_eligible": False,
                "rationale": "Recovered non-ontology endpoint for inspection.",
            }
            recovered.append({
                "id": endpoint,
                "name": normalize_name(name),
                "type": node_type,
                "iad_category": meta.get("iad_category", "Participants"),
                "node_family": meta.get("node_family", ""),
                "is_actor": bool(meta.get("is_actor", True)),
                "persona_eligible": bool(meta.get("persona_eligible", meta.get("is_actor", True))),
                "summary": f"Recovered from edge endpoint: {edge.get('type', '')}",
                "attributes": {"recovered_from_edge": True},
                "rationale": meta.get("rationale", ""),
                "source_spans": [],
            })
            node_ids.add(endpoint)
            warnings.append(f"Recovered missing endpoint node from edge: {endpoint}")

    return recovered


def _compact_nodes_for_prompt(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact projection of every node, used as the candidate pool for edge prompts."""
    return [
        {
            "id": node["id"],
            "name": node["name"],
            "type": node["type"],
            "iad_category": node["iad_category"],
            "node_family": node.get("node_family", ""),
            "summary": node.get("summary", "")[:240],
        }
        for node in nodes
    ]


_RELEVANCE_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힯]{2,}")


def _select_relevant_nodes(
    compact_pool: list[dict[str, Any]],
    chunk_text: str,
    max_nodes: int,
) -> list[dict[str, Any]]:
    """Pick the top-N nodes whose name/type/summary tokens overlap most with the chunk text.

    Falls back to head slicing only when the chunk has no scoreable tokens (extremely short or
    pure-symbol chunk). Otherwise every node is scored and the top-N kept; ties break on insertion
    order so that determinism is preserved across runs.
    """
    chunk_tokens = {tok.lower() for tok in _RELEVANCE_TOKEN_RE.findall(chunk_text or "")}
    if not chunk_tokens:
        return compact_pool[:max_nodes]

    scored: list[tuple[int, int, dict[str, Any]]] = []
    for index, node in enumerate(compact_pool):
        node_tokens = {
            tok.lower()
            for tok in _RELEVANCE_TOKEN_RE.findall(
                f"{node.get('name','')} {node.get('type','')} {node.get('summary','')}"
            )
        }
        score = len(chunk_tokens & node_tokens)
        # Negate score for stable descending sort by (score, -index) — higher score first,
        # earlier insertion wins ties.
        scored.append((-score, index, node))

    scored.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in scored[:max_nodes]]


def _dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for edge in edges:
        edge["deontic"] = _normalize_nullable(edge.get("deontic"))
        edge["activation_condition"] = _normalize_nullable(edge.get("activation_condition"))
        edge["execution_constraint"] = _normalize_nullable(edge.get("execution_constraint"))
        edge["or_else_text"] = _normalize_nullable(edge.get("or_else_text"))
        key = (
            edge.get("type"),
            edge.get("source"),
            edge.get("target"),
            edge.get("deontic"),
            _normalized_key_text(edge.get("fact")),
            _normalized_key_text(edge.get("activation_condition")),
            _normalized_key_text(edge.get("execution_constraint")),
        )
        if key in seen:
            continue
        seen.add(key)
        edge["id"] = f"edge_{len(deduped) + 1:04d}"
        deduped.append(edge)
    return deduped


def _normalize_nullable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "null", "none", "n/a"}:
        return None
    return value


def _normalized_key_text(value: Any) -> str:
    return str(value or "").strip().lower()


_SENTENCE_TERMINATORS = (
    "다.\n", "함.\n", "음.\n", "임.\n", "요.\n",
    ".\n", "!\n", "?\n",
    "다. ", "함. ", "음. ", "임. ", "요. ",
    "\n\n",
    "다.", "함.", "음.", "임.", "요.",
    "。", "！", "？",
    ". ", "! ", "? ",
)
_MIN_CHUNK_RATIO = 0.3


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks at sentence/paragraph boundaries when possible.

    The splitter tries each terminator in priority order (paragraph break > sentence-final endings >
    weaker punctuation), falling back to a hard cut only when no boundary is available within the
    last (1 - _MIN_CHUNK_RATIO) of the window. This avoids cutting Korean policy sentences mid-clause,
    which previously broke condition_text/source_text reconstruction at chunk seams.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    chunks: list[str] = []
    start = 0
    text_len = len(text)
    min_progress = max(1, int(chunk_size * _MIN_CHUNK_RATIO))

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            window = text[start:end]
            split_at = -1
            for terminator in _SENTENCE_TERMINATORS:
                position = window.rfind(terminator)
                if position >= min_progress:
                    split_at = position + len(terminator)
                    break
            if split_at > 0:
                end = start + split_at

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = start + max(1, chunk_size - chunk_overlap)
        start = next_start

    return chunks
