import json
import random
import re
from pathlib import Path

from models.schemas import Stage1Result, StakeholderGroup
from stage1.modules import (
    build_economic_context,
    build_policy_context,
    build_policy_plan,
    generate_personas_from_graph,
    generate_personas_from_policy_graph,
    generate_personas_no_graph,
    synthesize_simulation_participants,
)
from persona.role_actions import compute_for_entity as compute_role_actions_for_entity
from persona.name_utils import infer_persona_name
from persona.simulation_schema import sanitize_simulation_persona
from persona.scope_sql_filter import PERSONA_OUTPUT_FIELDS as _COUNTRY_PERSONA_FIELDS
from persona.refiner import refine_persona
from stage1.mechanism_panel import (
    build_role_prototypes,
    mechanism_panel_enabled,
    select_mechanism_panel,
)


def _build_kg_lookup(policy_graph_path: str) -> tuple[dict, dict]:
    """Return (entity_by_id, kg_context_by_entity_id) for policy_graph.json.

    The kg_context dict mirrors `persona.scope_definer.build_kg_context` so the
    role-actions synthesizer and refiner see the same structure as the Definer.
    """
    from graph.local_graph_reader import LocalGraphReader
    from persona.scope_definer import build_kg_context

    reader = LocalGraphReader(policy_graph_path)
    entities = reader.get_filtered_entities(persona_only=False)
    by_id: dict[str, object] = {}
    kg_by_id: dict[str, dict] = {}
    for entity in entities:
        eid = (
            getattr(entity, "id", None)
            or getattr(entity, "uuid", None)
            or entity.get_entity_type()
        )
        if not eid:
            continue
        etype = entity.get_entity_type() or ""
        by_id[eid] = entity
        kg_by_id[eid] = build_kg_context(entity, etype)
    return by_id, kg_by_id


def _attach_policy_role_actions(
    stakeholder_groups: list[StakeholderGroup],
    policy_graph_path: str,
    config: dict,
    run_dir: Path,
) -> None:
    """Compute role-actions ONCE per entity and attach to every persona of that entity.

    Writes per-entity action dumps under `<run_dir>/role_actions/<entity_id>.json` for inspection.
    Skipped when `persona.role_actions.enabled: false` — used for ablation studies.
    """
    role_actions_cfg = (config.get("persona", {}) or {}).get("role_actions", {}) or {}
    if not role_actions_cfg.get("enabled", True):
        print("  [role_actions] disabled via config; skipping.")
        return
    entity_by_id, kg_by_id = _build_kg_lookup(policy_graph_path)
    actions_dir = run_dir / "role_actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    for group in stakeholder_groups:
        for persona in group.personas:
            demographics = persona.demographics or {}
            if (demographics.get("agent_modality") or "").lower() == "organization":
                continue
            entity_id = demographics.get("entity_id")
            if not entity_id or entity_id not in entity_by_id:
                continue
            if entity_id not in cache:
                print(f"  [role_actions] synthesizing for entity {entity_id}")
                cache[entity_id] = compute_role_actions_for_entity(
                    entity_by_id[entity_id],
                    kg_by_id[entity_id],
                    config,
                )
                # Persist for reviewer inspection.
                safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(entity_id)).strip("_") or "entity"
                with (actions_dir / f"{safe}.json").open("w", encoding="utf-8") as f:
                    json.dump({"entity_id": entity_id, **cache[entity_id]}, f, ensure_ascii=False, indent=2)
            actions = cache[entity_id]
            new_demos = dict(demographics)
            new_demos["policy_role_actions_raw"] = actions["policy_role_actions_raw"]
            new_demos["policy_role_actions"] = actions["policy_role_actions"]
            persona.demographics = new_demos


def _project_persona_for_stage2(persona, stakeholder_type: str) -> dict:
    """Project a persona to the fixed Stage 2 simulation-persona contract."""
    demographics = persona.demographics or {}
    record = {
        "persona_id": demographics.get("persona_id") or _generate_persona_id(persona, stakeholder_type),
        "name": demographics.get("name") or infer_persona_name(demographics),
        "stakeholder_type": stakeholder_type,
    }
    for field in (
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
    ):
        if demographics.get(field) not in (None, "", [], {}):
            record[field] = demographics[field]
    return sanitize_simulation_persona(record)


def _generate_persona_id(persona, stakeholder_type: str) -> str:
    """Mirror flatten_persona_groups' persona_id derivation when one is missing."""
    role_label = (persona.role_entity_id or "").split(":", 1)[-1] or stakeholder_type
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", role_label).strip("_") or "persona"
    demographics = persona.demographics or {}
    suffix = str(demographics.get("nemotron_uuid") or "")[:8]
    if not suffix:
        match = re.search(r"\bKG-Gen-([0-9A-Za-z]+)", persona.representative_type or "")
        if match:
            suffix = match.group(1)
    if not suffix and demographics.get("candidate_index") not in (None, ""):
        suffix = f"{int(demographics['candidate_index']):03d}"
    return f"P_{safe}__{suffix or 'XXXXXXXX'}"


def _build_stage2_scenarios(
    persona_pool: list[StakeholderGroup],
    policy_graph_path: str,
    config: dict,
    run_dir: Path,
) -> dict:
    """Build the scenario-grouped simulation_personas.json bundle.

    Layout:
      {
        "scenario_count": N,
        "shared_context": {"org_kg_contexts": [...]},
        "scenarios": [{"scenario_id": "...", "participants": [...]}, ...],
        "sampling_metadata": {...}
      }

    Sampling rule: stratified — each scenario contains ≥1 individual-modality
    persona per stakeholder_type, then fills remaining slots uniformly without
    replacement within the scenario. Across scenarios, sampling is independent,
    unless `scope_definer.unique_participants_across_scenarios: true`, in which
    case a persona picked in any scenario is excluded from every later one.
    """
    scope_cfg = (config.get("persona", {}) or {}).get("scope_definer", {}) or {}
    num_scenarios = int(scope_cfg.get("num_scenarios", 5))
    participants_per_scenario = int(scope_cfg.get("participants_per_scenario", 5))
    base_seed = int(
        config.get("persona", {}).get("simulation_sample_seed")
        or scope_cfg.get("sample_seed")
        or 42
    )
    unique_across_scenarios = bool(scope_cfg.get("unique_participants_across_scenarios", False))

    # Bucket personas by (stakeholder_type, anchor_entity_id). Each anchor is one
    # of the up-to-K entities chosen per type in stage1; round-robin across
    # scenarios so every anchor surfaces in at least one scenario.
    individuals_by_pair: dict[tuple[str, str], list] = {}
    individuals_by_type: dict[str, list] = {}
    anchors_by_type: dict[str, list[str]] = {}
    for group in persona_pool:
        for persona in group.personas:
            modality = ((persona.demographics or {}).get("agent_modality") or "").lower()
            if modality == "organization":
                continue
            anchor_id = persona.role_entity_id or ""
            key = (group.stakeholder_type, anchor_id)
            individuals_by_pair.setdefault(key, []).append(persona)
            individuals_by_type.setdefault(group.stakeholder_type, []).append(persona)

    stakeholder_types = sorted(individuals_by_type)
    for stype in stakeholder_types:
        anchors_by_type[stype] = sorted({a for (t, a) in individuals_by_pair if t == stype})
    pool_size = sum(len(v) for v in individuals_by_type.values())
    if pool_size == 0:
        print("  [stage2_scenarios] empty individual-modality pool; emitting empty scenarios.")

    K = len(stakeholder_types)
    max_anchors = max((len(anchors_by_type[t]) for t in stakeholder_types), default=0)
    required = K * max_anchors
    available = num_scenarios * participants_per_scenario
    if required and available < required:
        print(
            f"  [stage2_scenarios] WARNING: N*P={available} < K*A_max={required}; "
            "some (type,anchor) pairs may not appear in any scenario."
        )
    if unique_across_scenarios and pool_size < available:
        print(
            f"  [stage2_scenarios] WARNING: unique_participants_across_scenarios=true but "
            f"pool_size={pool_size} < N*P={available}; later scenarios may end up short "
            "of participants_per_scenario once the pool is exhausted."
        )

    scenarios: list[dict] = []
    globally_used_ids: set[int] = set()
    for scenario_index in range(1, num_scenarios + 1):
        rng = random.Random(base_seed + scenario_index)
        picked: list = []
        used_ids: set[int] = set(globally_used_ids) if unique_across_scenarios else set()
        s_zero = scenario_index - 1  # 0-indexed offset for rotations

        # Step 1: rotate which types fill the guaranteed slots (handles K > P),
        # and rotate which anchor inside each chosen type (handles A > 1).
        if K > 0:
            rotated_types = stakeholder_types[s_zero % K :] + stakeholder_types[: s_zero % K]
            chosen_types = rotated_types[: min(participants_per_scenario, K)]
            for stype in chosen_types:
                anchors = anchors_by_type.get(stype) or [""]
                anchor = anchors[s_zero % len(anchors)]
                candidates = individuals_by_pair.get((stype, anchor)) or individuals_by_type.get(stype, [])
                candidates = [p for p in candidates if id(p) not in used_ids]
                if not candidates:
                    continue
                choice = rng.choice(candidates)
                picked.append((stype, choice))
                used_ids.add(id(choice))
                if len(picked) >= participants_per_scenario:
                    break

        # Step 2: fill remaining slots uniformly without replacement (within scenario)
        flat_pool = [
            (stype, p)
            for stype, items in individuals_by_type.items()
            for p in items
            if id(p) not in used_ids
        ]
        rng.shuffle(flat_pool)
        for stype, p in flat_pool:
            if len(picked) >= participants_per_scenario:
                break
            picked.append((stype, p))
            used_ids.add(id(p))

        participants = [_project_persona_for_stage2(p, stype) for stype, p in picked]
        scenarios.append({
            "scenario_id": f"scenario_{scenario_index}",
            "participants": participants,
        })
        if unique_across_scenarios:
            globally_used_ids = used_ids

    # Build org_kg_contexts from the saved scope_definer dumps (or directly).
    org_kg_contexts = _collect_org_kg_contexts(persona_pool, policy_graph_path, run_dir)

    return {
        "schema_version": "simulation_persona_v1",
        "scenario_count": num_scenarios,
        "shared_context": {"org_kg_contexts": org_kg_contexts},
        "scenarios": scenarios,
        "sampling_metadata": {
            "strategy": "stratified_round_robin_type_and_anchor",
            "stakeholder_types": stakeholder_types,
            "anchors_by_type": anchors_by_type,
            "pool_size": pool_size,
            "participants_per_scenario": participants_per_scenario,
            "seed": base_seed,
            "num_scenarios": num_scenarios,
            "unique_participants_across_scenarios": unique_across_scenarios,
        },
    }


def _build_fixed_mechanism_panel_bundle(
    selected_groups: list[StakeholderGroup],
    policy_graph_path: str,
    config: dict,
    run_dir: Path,
    panel_manifest: dict,
) -> dict:
    """Build one ordered scenario from the frozen mechanism-covering panel."""
    ordered = sorted(
        (
            (group.stakeholder_type, persona)
            for group in selected_groups
            for persona in group.personas
        ),
        key=lambda item: int((item[1].demographics or {}).get("panel_position") or 999),
    )
    participants = [_project_persona_for_stage2(persona, stype) for stype, persona in ordered]
    return {
        "schema_version": "simulation_persona_v1",
        "scenario_count": 1,
        "shared_context": {
            "org_kg_contexts": _collect_org_kg_contexts(selected_groups, policy_graph_path, run_dir),
        },
        "scenarios": [{
            "scenario_id": panel_manifest["panel_id"],
            "participants": participants,
        }],
        "sampling_metadata": {
            "strategy": "fixed_role_balanced_mechanism_panel",
            "panel_id": panel_manifest["panel_id"],
            "role_order": panel_manifest["role_order"],
            "participants_per_scenario": len(participants),
            "num_scenarios": 1,
            "ebc_blind_selection": True,
        },
    }


def _collect_org_kg_contexts(persona_pool, policy_graph_path: str, run_dir: Path) -> list[dict]:
    """Pull KG-context dumps for organization-modality entities from the run dir.

    scope_sampler now writes <run_dir>/scope_definer/<id>_kg_context.json for
    org-modality entities too (not just persona-eligible ones). Read those.
    """
    org_entities: dict[str, str] = {}  # entity_id -> name
    for group in persona_pool:
        for persona in group.personas:
            demo = persona.demographics or {}
            if (demo.get("agent_modality") or "").lower() != "organization":
                continue
            entity_id = demo.get("entity_id") or persona.role_entity_id
            if not entity_id:
                continue
            label = persona.representative_type or group.stakeholder_type
            org_entities.setdefault(str(entity_id), label)

    out: list[dict] = []
    seen = set()
    for entity_id, label in org_entities.items():
        safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(entity_id)).strip("_") or "entity"
        ctx_path = run_dir / "scope_definer" / f"{safe}_kg_context.json"
        if not ctx_path.exists():
            continue
        try:
            with ctx_path.open("r", encoding="utf-8") as f:
                kg_ctx = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        key = (kg_ctx.get("name") or label, kg_ctx.get("entity_id") or entity_id)
        if key in seen:
            continue
        seen.add(key)
        # Compress KG context to KG-derived essentials only. Drops type-level
        # rationale (ontology metadata, redundant with policy_context) plus the
        # bulky related_edges/related_nodes (already in policy_context KG summary)
        # and keeps the org-specific grounding signals: role_scope, the verbatim
        # policy spans mentioning this org, and the source_text of every edge
        # this org participates in. Cuts ~88% of org_kg_contexts payload size.
        spans_seen: set[str] = set()
        spans_text: list[str] = []
        for sp_ in kg_ctx.get("source_spans") or []:
            t = (sp_.get("text") or "").strip()
            if t and t not in spans_seen:
                spans_seen.add(t)
                spans_text.append(t)
        edges_seen: set[str] = set()
        edge_source: list[str] = []
        for e in kg_ctx.get("related_edges") or []:
            t = (e.get("source_text") or "").strip()
            if t and t not in edges_seen:
                edges_seen.add(t)
                edge_source.append(t)
        out.append({
            "name": kg_ctx.get("name") or label,
            "entity_type": kg_ctx.get("entity_type"),
            "role_scope": kg_ctx.get("role_scope"),
            "source_spans_text": "\n".join(spans_text),
            "edge_source": "\n".join(edge_source),
        })
    return out


def _refine_selected_personas(
    stakeholder_groups: list[StakeholderGroup],
    policy_graph_path: str,
    config: dict,
) -> None:
    """Run the persona refiner on each selected simulation_participant in place.

    Skipped when `persona.refiner.enabled: false` — used for ablation studies.
    """
    persona_cfg = config.get("persona", {}) or {}
    refiner_cfg = persona_cfg.get("refiner", {}) or {}
    # Role-prototype generation already grounds mechanism-panel personas. Keep
    # the legacy refiner default for older modes, but require an explicit opt-in
    # for mechanism panels so a second LLM pass cannot rewrite role identity.
    default_enabled = persona_cfg.get("selection_mode") != "mechanism_panel"
    if not refiner_cfg.get("enabled", default_enabled):
        print("  [refiner] disabled via config; skipping.")
        return
    _, kg_by_id = _build_kg_lookup(policy_graph_path)
    total = sum(len(g.personas) for g in stakeholder_groups)
    if not total:
        return
    print(f"  [refiner] refining {total} selected persona(s)...")
    for group in stakeholder_groups:
        for persona in group.personas:
            demographics = persona.demographics or {}
            if (demographics.get("agent_modality") or "").lower() == "organization":
                continue
            entity_id = demographics.get("entity_id")
            kg_context = kg_by_id.get(entity_id) if entity_id else None
            if not kg_context:
                continue
            refine_persona(persona, kg_context, config)


def save_json(data: dict | list, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_english_json(data: dict | list, path: Path) -> None:
    save_json(_sanitize_english_only(data), path)


def select_simulation_participants(
    stakeholder_groups: list[StakeholderGroup],
    config: dict,
) -> tuple[list[StakeholderGroup], dict]:
    """Randomly sample the Stage 2 participant cohort from the full persona pool."""
    max_count = int(config.get("persona", {}).get("max_simulation_participants") or 0)
    flat = [
        (group.stakeholder_type, persona)
        for group in stakeholder_groups
        for persona in group.personas
    ]
    if max_count <= 0 or len(flat) <= max_count:
        return stakeholder_groups, {
            "strategy": "all_personas",
            "max_simulation_participants": max_count,
            "pool_size": len(flat),
            "selection_pool_size": len(flat),
            "selected_count": len(flat),
            "seed": None,
        }

    persona_config = config.get("persona", {})
    selection_pool = flat
    selection_filter = {}
    min_match_score = persona_config.get("simulation_min_match_score")
    if min_match_score is not None:
        scored_pool = [
            item for item in flat
            if _persona_match_score(item[1]) >= float(min_match_score)
        ]
        if len(scored_pool) >= max_count:
            selection_pool = scored_pool
            selection_filter.update({
                "simulation_min_match_score": float(min_match_score),
                "eligible_pool_size": len(scored_pool),
            })

    role_gate = persona_config.get("simulation_role_gate", {})
    if role_gate.get("enabled", False):
        role_filtered_pool = [
            item for item in selection_pool
            if _passes_simulation_role_filter(item[0], item[1], config)
        ]
        if len(role_filtered_pool) >= max_count:
            selection_pool = role_filtered_pool
            selection_filter.update(_simulation_role_filter_metadata(config))

    seed = persona_config.get(
        "simulation_sample_seed",
        persona_config.get("nemotron", {}).get("seed"),
    )
    rng = random.Random(seed)
    sampling_strategy = persona_config.get("simulation_sampling_strategy", "role_stratified_weighted")
    if sampling_strategy == "role_stratified_weighted":
        selected = _role_stratified_weighted_sample(selection_pool, max_count, rng)
    else:
        selected = rng.sample(selection_pool, max_count)

    grouped: dict[str, list] = {}
    for stakeholder_type, persona in selected:
        grouped.setdefault(stakeholder_type, []).append(persona)

    selected_groups = [
        StakeholderGroup(stakeholder_type=stakeholder_type, personas=personas)
        for stakeholder_type, personas in grouped.items()
    ]
    return selected_groups, {
        "strategy": sampling_strategy,
        "max_simulation_participants": max_count,
        "pool_size": len(flat),
        "selection_pool_size": len(selection_pool),
        "selection_filter": selection_filter or None,
        "selected_count": len(selected),
        "seed": seed,
    }


def _role_stratified_weighted_sample(
    pool: list[tuple[str, object]],
    max_count: int,
    rng: random.Random,
) -> list[tuple[str, object]]:
    """Sample without hard filtering while preserving actor-role coverage."""
    if len(pool) <= max_count:
        return list(pool)

    remaining = list(pool)
    selected: list[tuple[str, object]] = []
    by_role: dict[str, list[tuple[str, object]]] = {}
    for item in remaining:
        role_key = getattr(item[1], "role_entity_id", None) or item[0]
        by_role.setdefault(str(role_key), []).append(item)

    # Cover as many IAD actor anchors as the Stage 2 cohort size allows.
    for role in sorted(by_role):
        if len(selected) >= max_count:
            break
        picked = _weighted_pick(by_role[role], rng)
        selected.append(picked)
        remaining = [item for item in remaining if item is not picked]

    while remaining and len(selected) < max_count:
        picked = _weighted_pick(remaining, rng)
        selected.append(picked)
        remaining = [item for item in remaining if item is not picked]

    return selected


def _weighted_pick(items: list[tuple[str, object]], rng: random.Random) -> tuple[str, object]:
    weights = []
    for _, persona in items:
        score = max(_persona_match_score(persona), 0.0)
        # Soft preference, not exclusion: even low-score rows remain sampleable.
        weights.append((score + 1.0) ** 2)
    total = sum(weights)
    threshold = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += weight
        if cumulative >= threshold:
            return item
    return items[-1]


def _persona_match_score(persona) -> float:
    demographics = getattr(persona, "demographics", {}) or {}
    try:
        return float(demographics.get("match_score", 0))
    except (TypeError, ValueError):
        return 0.0


def _passes_simulation_role_filter(stakeholder_type: str, persona, config: dict) -> bool:
    role_filter = (
        config.get("persona", {})
        .get("nemotron", {})
        .get("role_filters", {})
        .get(stakeholder_type, {})
    )
    demographics = getattr(persona, "demographics", {}) or {}

    if role_filter.get("simulation_require_age_range", False):
        age = _as_float(demographics.get("age"))
        min_age = role_filter.get("simulation_age_min", role_filter.get("age_min"))
        max_age = role_filter.get("simulation_age_max", role_filter.get("age_max"))
        if age is None:
            return False
        if min_age is not None and age < float(min_age):
            return False
        if max_age is not None and age > float(max_age):
            return False

    education_keywords = role_filter.get("simulation_education_keywords", [])
    if education_keywords:
        education = str(demographics.get("education_level") or "")
        if not any(str(keyword) in education for keyword in education_keywords):
            return False

    occupation_keywords = role_filter.get("simulation_occupation_keywords", [])
    if role_filter.get("simulation_require_occupation_match", False) and occupation_keywords:
        occupation = str(demographics.get("occupation") or "")
        if not any(str(keyword) in occupation for keyword in occupation_keywords):
            return False

    return True


def _simulation_role_filter_metadata(config: dict) -> dict:
    filters = (
        config.get("persona", {})
        .get("nemotron", {})
        .get("role_filters", {})
    )
    active = {}
    for role, role_filter in filters.items():
        metadata = {}
        if role_filter.get("simulation_require_age_range", False):
            metadata["simulation_age_min"] = role_filter.get("simulation_age_min", role_filter.get("age_min"))
            metadata["simulation_age_max"] = role_filter.get("simulation_age_max", role_filter.get("age_max"))
        if role_filter.get("simulation_education_keywords"):
            metadata["simulation_education_keywords"] = role_filter.get("simulation_education_keywords")
        if role_filter.get("simulation_require_occupation_match", False):
            metadata["simulation_occupation_keywords"] = role_filter.get("simulation_occupation_keywords", [])
        if metadata:
            active[role] = metadata
    return {"simulation_role_filters": active} if active else {}


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def flatten_persona_groups(
    stakeholder_groups: list[StakeholderGroup],
    id_lookup: dict[tuple[str, str, str], str] | None = None,
) -> tuple[list[dict], dict[tuple[str, str, str], str]]:
    """
    Export personas as a flat Stage 2 handoff list.

    The nested Stage1Result keeps the original group structure, while this flat
    form mirrors the legacy persona schema used by downstream simulation code.
    """
    records = []
    generated_lookup = dict(id_lookup or {})
    role_counts: dict[str, int] = {}

    for group in stakeholder_groups:
        for persona in group.personas:
            demographics = persona.demographics or {}
            key = _persona_key(group.stakeholder_type, persona)
            persona_id = generated_lookup.get(key)
            if not persona_id:
                role_label = _role_label(persona.role_entity_id, group.stakeholder_type)
                role_counts[role_label] = role_counts.get(role_label, 0) + 1
                persona_id = f"P_{_safe_id_part(role_label)}__{role_counts[role_label]:03d}"
                generated_lookup[key] = persona_id

            record = {
                "persona_id": persona_id,
                "name": demographics.get("name"),
                "stakeholder_type": group.stakeholder_type,
                "representative_type": persona.representative_type,
                "policy_role": _role_label(persona.role_entity_id, group.stakeholder_type),
                "policy_role_source": "policy_graph" if persona.role_entity_id else None,
                "role_entity_id": persona.role_entity_id,
                "iad_category": persona.iad_category,
                "source": _flat_source(persona.source, demographics),
                "augmentation_source": "policy_graph" if "policy_graph" in str(persona.source or "") else None,
                "agent_modality": demographics.get("agent_modality"),
                "role_family": demographics.get("role_family"),
                "constraint_scope": demographics.get("constraint_scope"),
                "nemotron_uuid": demographics.get("nemotron_uuid"),
                "nemotron_profile": _nemotron_profile(demographics),
                "sex": demographics.get("sex"),
                "sex_en": demographics.get("sex_en") or _english_value(demographics.get("sex")),
                "age": demographics.get("age"),
                "education_level": demographics.get("education_level"),
                "education_level_en": demographics.get("education_level_en") or _english_value(demographics.get("education_level")),
                "bachelors_field": demographics.get("bachelors_field"),
                "bachelors_field_en": demographics.get("bachelors_field_en") or _english_value(demographics.get("bachelors_field")),
                "marital_status": demographics.get("marital_status"),
                "marital_status_en": demographics.get("marital_status_en") or _english_value(demographics.get("marital_status")),
                "military_status": demographics.get("military_status"),
                "military_status_en": demographics.get("military_status_en") or _english_value(demographics.get("military_status")),
                "family_type": demographics.get("family_type"),
                "family_type_en": demographics.get("family_type_en") or _english_value(demographics.get("family_type")),
                "occupation": demographics.get("occupation"),
                "occupation_en": demographics.get("occupation_en"),
                "province": demographics.get("province"),
                "province_en": demographics.get("province_en"),
                "district": demographics.get("district"),
                "district_en": demographics.get("district_en"),
                "persona": demographics.get("persona"),
                "persona_en": demographics.get("persona_en"),
                "persona_profile": demographics.get("persona_profile"),
                "professional_persona": demographics.get("professional_persona"),
                "professional_persona_en": demographics.get("professional_persona_en"),
                "family_persona": demographics.get("family_persona"),
                "family_persona_en": demographics.get("family_persona_en"),
                "skills_and_expertise": demographics.get("skills_and_expertise"),
                "skills_and_expertise_en": demographics.get("skills_and_expertise_en"),
                "professional_persona_pre_refine": demographics.get("professional_persona_pre_refine"),
                "skills_and_expertise_pre_refine": demographics.get("skills_and_expertise_pre_refine"),
                "refiner_rationale": demographics.get("refiner_rationale"),
                "refiner_kg_evidence": demographics.get("refiner_kg_evidence"),
                "policy_role_actions": demographics.get("policy_role_actions"),
                "policy_role_actions_raw": demographics.get("policy_role_actions_raw"),
                "career_goals_and_ambitions": demographics.get("career_goals_and_ambitions"),
                "career_goals_and_ambitions_en": demographics.get("career_goals_and_ambitions_en"),
                "hobbies_and_interests": demographics.get("hobbies_and_interests"),
                "hobbies_and_interests_en": demographics.get("hobbies_and_interests_en"),
                "match_score": demographics.get("match_score"),
                "match_strategy": demographics.get("match_strategy"),
                "compatibility_assessment": demographics.get("compatibility_assessment"),
                "sampling_metadata": _sampling_metadata(demographics),
                "rerank_score": demographics.get("rerank_score"),
                "rerank_strategy": demographics.get("rerank_strategy"),
                "core_objectives": persona.core_objectives,
                "core_constraints": persona.core_constraints,
                "obligations": _compact_rules(persona.obligations),
                "eligible_actions": _compact_rules(persona.eligible_actions),
                "prohibited_actions": _compact_rules(persona.prohibited_actions),
                "kg_obligations_full": persona.obligations,
                "kg_eligible_actions_full": persona.eligible_actions,
                "kg_prohibited_actions_full": persona.prohibited_actions,
                "kg_context": _compact_context(persona.kg_context),
                "policy_resources": _compact_context_section(persona.kg_context, "resources"),
                "policy_actions": _compact_context_section(persona.kg_context, "actions"),
                "policy_information": _compact_context_section(persona.kg_context, "information"),
                "evaluation_context": _compact_context_section(persona.kg_context, "evaluation"),
                "policy_structural_relations": _compact_context_section(persona.kg_context, "structural_relations"),
            }
            records.append({key: value for key, value in record.items() if value not in (None, "", [])})

    return records, generated_lookup


def _persona_key(stakeholder_type: str, persona) -> tuple[str, str, str]:
    demographics = persona.demographics or {}
    return (
        stakeholder_type,
        str(persona.role_entity_id or ""),
        str(demographics.get("nemotron_uuid") or persona.representative_type),
    )


def _role_label(role_entity_id: str | None, fallback: str) -> str:
    if role_entity_id and ":" in role_entity_id:
        return role_entity_id.split(":", 1)[1]
    return role_entity_id or fallback


def _safe_id_part(value: str) -> str:
    text = re.sub(r"\s+", "_", str(value).strip())
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "persona"


def _flat_source(source: str | None, demographics: dict) -> str:
    if demographics.get("nemotron_uuid"):
        return "nemotron-personas"
    return source or "policy_graph"


def _nemotron_profile(demographics: dict) -> dict:
    """Build the persona's full Nemotron profile dict, dropping empty fields.

    Country-aware: persona/demographic columns are driven by PERSONA_OUTPUT_FIELDS
    (set by the active Nemotron values module). Cross-country common identifiers
    and runtime/eligibility metadata are appended verbatim. ``_en`` variants are
    surfaced only when present (Korean → English transliteration; for SG/USA
    where data is already English they're naturally absent).
    """
    profile: dict = {
        "uuid": demographics.get("uuid"),
        "nemotron_uuid": demographics.get("nemotron_uuid"),
        "persona": demographics.get("persona"),
        "persona_profile": demographics.get("persona_profile"),
        "hobbies_and_interests": demographics.get("hobbies_and_interests"),
    }
    # Country-specific persona/demographic fields plus stable runtime metadata.
    extra_keys = (
        *_COUNTRY_PERSONA_FIELDS,
        "agent_modality",
        "role_family",
        "constraint_scope",
        "compatibility_assessment",
        "adaptive_matching",
        "observable_eligibility",
        "eligibility_match",
        "synthesis_status",
        "synthesis_rejection_reason",
        "synthesis_warning",
        "synthesis_error",
        "synthesis_source",
    )
    for key in extra_keys:
        profile[key] = demographics.get(key)
    # Pick up English-translated variants when they exist (KR-only in practice).
    for key in ("persona", "hobbies_and_interests", *_COUNTRY_PERSONA_FIELDS):
        translated_key = f"{key}_en"
        if demographics.get(translated_key) not in (None, "", []):
            profile[translated_key] = demographics.get(translated_key)
    return {key: value for key, value in profile.items() if value not in (None, "", [])}


def _sampling_metadata(demographics: dict) -> dict:
    metadata = {
        "match_score": demographics.get("match_score"),
        "match_strategy": demographics.get("match_strategy"),
        "match_reasons": demographics.get("match_reasons"),
        "match_warnings": demographics.get("match_warnings"),
        "alignment_status": demographics.get("alignment_status"),
        "adaptive_matching": demographics.get("adaptive_matching"),
        "observable_eligibility": demographics.get("observable_eligibility"),
        "eligibility_match": demographics.get("eligibility_match"),
        "role_family": demographics.get("role_family"),
        "constraint_scope": demographics.get("constraint_scope"),
        "synthesis_status": demographics.get("synthesis_status"),
        "synthesis_rejection_reason": demographics.get("synthesis_rejection_reason"),
        "synthesis_warning": demographics.get("synthesis_warning"),
        "synthesis_error": demographics.get("synthesis_error"),
        "compatibility_assessment": demographics.get("compatibility_assessment"),
        "rerank_score": demographics.get("rerank_score"),
        "rerank_strategy": demographics.get("rerank_strategy"),
        # Scope-definer provenance
        "tier_used": demographics.get("tier_used"),
        "tier_pool_sizes": demographics.get("tier_pool_sizes"),
        "used_fallback": demographics.get("used_fallback"),
        "scope_definer_rationale": demographics.get("scope_definer_rationale"),
    }
    return {key: value for key, value in metadata.items() if value not in (None, "", [])}


def _english_value(value):
    if value in (None, "", []):
        return value
    text = str(value)
    translations = {
        "\ub0a8\uc790": "male",
        "\uc5ec\uc790": "female",
        "\ubbf8\ud63c": "unmarried",
        "\uae30\ud63c": "married",
        "\ube44\ud604\uc5ed": "not active-duty",
        "\ud604\uc5ed": "active-duty",
        "\ud574\ub2f9\uc5c6\uc74c": "not applicable",
        "\ub300\ud559\uc6d0": "graduate school",
        "4\ub144\uc81c \ub300\ud559\uad50": "four-year university",
        "2~3\ub144\uc81c \uc804\ubb38\ub300\ud559": "two- or three-year college",
        "\uc5b4\uba38\ub2c8\uc640 \ub3d9\uac70": "living with mother",
    }
    if text in translations:
        return translations[text]
    if re.search(r"[\uac00-\ud7a3]", text):
        return None
    return value


_NEMOTRON_PROFILE_KEYS = (
    "persona",
    "professional_persona",
    "family_persona",
    "skills_and_expertise",
    "career_goals_and_ambitions",
    "hobbies_and_interests",
)


_SOURCE_EVIDENCE_KEYS = {"source_text", "text"}
_NEMOTRON_SOURCE_KEYS = {
    "persona",
    "professional_persona",
    "professional_persona_pre_refine",
    "family_persona",
    "skills_and_expertise",
    "skills_and_expertise_pre_refine",
    "career_goals_and_ambitions",
    "hobbies_and_interests",
    "sex",
    "marital_status",
    "military_status",
    "family_type",
    "education_level",
    "bachelors_field",
    "occupation",
    "district",
    "province",
}
_RAW_SOURCE_KEYS = _SOURCE_EVIDENCE_KEYS | _NEMOTRON_SOURCE_KEYS


def _sanitize_english_only(value, key: str | None = None):
    if key in _RAW_SOURCE_KEYS:
        return value
    if isinstance(value, dict):
        data = dict(value)
        cleaned = {}
        for item_key, item in data.items():
            sanitized = _sanitize_english_only(item, item_key)
            if sanitized not in (None, "", [], {}):
                cleaned[item_key] = sanitized
        return cleaned
    if isinstance(value, list):
        cleaned = []
        for item in value:
            sanitized = _sanitize_english_only(item, key)
            if sanitized not in (None, "", [], {}):
                cleaned.append(sanitized)
        return cleaned
    if isinstance(value, str):
        return _english_value(value)
    return value


def _compact_rules(rules: list[dict]) -> list[dict]:
    compacted = []
    for rule in rules:
        edge_type = rule.get("edge_type") or "RULE"
        value = rule.get("fact") or rule.get("source_text") or edge_type
        compacted.append({edge_type: value})
    return compacted


def _compact_context(context: dict) -> dict:
    if not context:
        return {}
    compacted = {}
    rules = context.get("rules") or {}
    rule_sections = {
        key: _compact_rules(value)
        for key, value in rules.items()
        if value
    }
    if rule_sections:
        compacted["rules"] = rule_sections
    for section in (
        "resources",
        "actions",
        "information",
        "evaluation",
        "structural_relations",
    ):
        values = _compact_context_section(context, section)
        if values:
            compacted[section] = values
    return compacted


def _compact_context_section(context: dict, section: str) -> list[dict]:
    values = (context or {}).get(section) or []
    return _compact_rules(values)


def run_stage1(
    config: dict,
    run_dir: Path,
    simulation_requirement: str,
    graph_id: str | None = None,
    ontology: dict | None = None,
    policy_graph_path: str | Path | None = None,
    use_graph: bool = True,
) -> Stage1Result:
    """
    Run Stage 1: economic context → policy plan → graph-grounded persona pool
    → random Stage 2 participant handoff.

    Args:
        config: Pipeline config dict.
        run_dir: Directory to save intermediate and final outputs.
        simulation_requirement: Short description of the downstream simulation target or requirement.
        graph_id: Zep graph ID created by build-graph command.
        ontology: Ontology dict from generate_ontology(), saved alongside graph_id.
        policy_graph_path: Local policy_graph.json path created by build-kg.
    """
    # Economic context (optional)
    if config.get("stage1", {}).get("use_economic_context", False):
        context = build_economic_context(config)
        save_json(context.model_dump(), run_dir / "economiccontext.json")
        print("Economic context built.")
    else:
        context = None
        print("Economic context skipped (stage1.use_economic_context=false).")

    # Policy plan (optional — Level 3: skip entirely; policy_context grounds Stage 2)
    if config.get("stage1", {}).get("use_policy_plan", False):
        policy = build_policy_plan(config)
        save_json(policy.model_dump(), run_dir / "policyplan.json")
        print("Policy plan built.")
    else:
        policy = None
        print("Policy plan skipped (stage1.use_policy_plan=false) — Stage 2 grounds on policy_context only.")

    # Policy context — primary KG-grounded artifact for Stage 2
    policy_for_context = policy if config.get("stage1", {}).get("use_policy_plan_in_context", False) else None
    policy_context = build_policy_context(policy_for_context, context, config, str(policy_graph_path) if policy_graph_path else None)
    save_json(policy_context.model_dump(), run_dir / "policycontext.json")
    print("Policy context built.")
    runtime_config = {
        **config,
        "_current_policy_context": policy_context.model_dump(),
        "_current_run_dir": str(run_dir),
    }

    role_prototypes = []
    actor_overlay = []
    panel_manifest = None
    if mechanism_panel_enabled(runtime_config):
        if not use_graph or not policy_graph_path:
            raise ValueError("persona.selection_mode='mechanism_panel' requires a local policy_graph.json")
        role_prototypes, actor_overlay = build_role_prototypes(policy_graph_path, runtime_config)
        save_json([prototype.model_dump() for prototype in role_prototypes], run_dir / "role_prototypes.json")
        save_json(actor_overlay, run_dir / "actor_classification_overlay.json")
        print(
            "Mechanism panel blueprint built: "
            + ", ".join(prototype.prototype_id for prototype in role_prototypes)
        )

    if use_graph:
        if policy_graph_path:
            stakeholder_groups = generate_personas_from_policy_graph(
                str(policy_graph_path), policy_context, context, runtime_config,
                ontology=ontology,
                role_prototypes=role_prototypes or None,
            )
        else:
            if not graph_id or not ontology:
                raise ValueError("graph_id/ontology or policy_graph_path is required when use_graph=True")
            stakeholder_groups = generate_personas_from_graph(graph_id, ontology, policy_context, context, runtime_config)
    else:
        print("  [ablation] Generating personas from past policy text (no graph)...")
        stakeholder_groups = generate_personas_no_graph(policy_context, runtime_config)

    persona_pool = stakeholder_groups
    if use_graph and policy_graph_path:
        _attach_policy_role_actions(persona_pool, str(policy_graph_path), runtime_config, run_dir)
    if role_prototypes:
        simulation_participants, panel_manifest = select_mechanism_panel(
            persona_pool, role_prototypes, runtime_config,
        )
        sampling_metadata = {
            "strategy": "fixed_role_balanced_mechanism_panel",
            "panel_id": panel_manifest["panel_id"],
            "selected_count": panel_manifest["panel_size"],
        }
        save_json(panel_manifest, run_dir / "panel_manifest.json")
    else:
        simulation_participants, sampling_metadata = select_simulation_participants(persona_pool, runtime_config)
    simulation_participants = synthesize_simulation_participants(
        simulation_participants,
        policy_context,
        runtime_config,
    )
    # The automatic refiner pass is intentionally disabled. Role grounding is
    # completed during role-prototype generation and persona synthesis; applying
    # a second LLM rewrite here can alter identity and confound EBC comparisons.
    save_english_json([g.model_dump() for g in persona_pool], run_dir / "personas_pool.json")
    save_english_json([g.model_dump() for g in simulation_participants], run_dir / "simulation_participants.json")
    save_english_json([g.model_dump() for g in simulation_participants], run_dir / "personas_initial.json")
    persona_pool_flat, persona_id_lookup = flatten_persona_groups(persona_pool)
    simulation_personas, _ = flatten_persona_groups(simulation_participants, persona_id_lookup)
    save_english_json(persona_pool_flat, run_dir / "personas_pool_flat.json")

    # Scenario-grouped output for Stage 2 ToC. Replaces the legacy flat
    # simulation_personas.json shape (now `simulation_personas_flat.json`
    # for back-compat / human inspection).
    save_english_json(simulation_personas, run_dir / "simulation_personas_flat.json")
    if use_graph and policy_graph_path:
        if panel_manifest:
            scenario_bundle = _build_fixed_mechanism_panel_bundle(
                selected_groups=simulation_participants,
                policy_graph_path=str(policy_graph_path),
                config=runtime_config,
                run_dir=run_dir,
                panel_manifest=panel_manifest,
            )
        else:
            scenario_bundle = _build_stage2_scenarios(
                persona_pool=persona_pool,
                policy_graph_path=str(policy_graph_path),
                config=runtime_config,
                run_dir=run_dir,
            )
        # Use plain save_json (NOT save_english_json) — the org_kg_contexts and
        # persona narrative fields are intentionally Korean; the English-only
        # sanitizer would strip them and collapse the bundle to empty objects.
        save_json(scenario_bundle, run_dir / "simulation_personas.json")
    else:
        # Fallback: keep the flat shape under the canonical name when running
        # the no-graph ablation path (no scope_definer / org context to bundle).
        save_english_json(simulation_personas, run_dir / "simulation_personas.json")
    simulation_context = {
        "simulation_requirement": simulation_requirement,
        "policy_context": policy_context.model_dump(),
        "policy_plan": policy.model_dump() if policy is not None else None,
        "economic_context": context.model_dump() if context is not None else None,
        "policy_graph_path": str(policy_graph_path) if policy_graph_path else None,
        "simulation_personas_path": "simulation_personas.json",
        "stage2_usage": (
            "Use this file as the shared scenario context for every Stage 2 prompt. "
            "Use simulation_personas.json for participant-specific profiles."
        ),
    }
    save_json(simulation_context, run_dir / "simulation_context.json")
    pool_count = sum(len(g.personas) for g in persona_pool)
    selected_count = sum(len(g.personas) for g in simulation_participants)
    print(f"Persona pool generated: {pool_count} total.")
    if panel_manifest:
        print(f"Mechanism-covering panel frozen: {selected_count} participant(s).")
    else:
        print(f"Simulation participants randomly selected: {selected_count} total.")

    result = Stage1Result(
        economic_context=context,
        policy_plan=policy,  # may be None when stage1.use_policy_plan=false
        policy_context=policy_context,
        stakeholder_groups=simulation_participants,
        persona_pool=persona_pool,
        simulation_participants=simulation_participants,
        predictions=[],
        critic_results=[],
        handoff_metadata={
            "stage1_scope": "kg_and_persona_handoff",
            "simulation_requirement": simulation_requirement,
            "critic_refine_removed": True,
            "sampling": sampling_metadata,
            "artifacts": {
                "economic_context": "economiccontext.json",
                "policy_plan": "policyplan.json",
                "policy_context": "policycontext.json",
                "simulation_context": "simulation_context.json",
                "persona_pool": "personas_pool.json",
                "persona_pool_flat": "personas_pool_flat.json",
                "simulation_participants": "simulation_participants.json",
                "simulation_personas": "simulation_personas.json",
                "legacy_personas_initial": "personas_initial.json",
                "stage1_result": "stage1_result.json",
                **({
                    "actor_classification_overlay": "actor_classification_overlay.json",
                    "role_prototypes": "role_prototypes.json",
                    "panel_manifest": "panel_manifest.json",
                } if panel_manifest else {}),
            },
        },
    )
    save_english_json(result.model_dump(), run_dir / "stage1_result.json")

    return result
