"""Repeated inner-schema generation with an immutable Outcomes phase.

This module is additive: the original single-call ``build_inner_toc_schema``
path remains unchanged for reproducibility.
"""

from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from typing import Callable, Literal

from llm.client import chat_completion
from models.prompts import (
    INNER_SCHEMA_COMPACT_SYSTEM,
    INNER_SCHEMA_COMPACT_USER,
    INNER_SCHEMA_DEDUP_SYSTEM,
    INNER_SCHEMA_DEDUP_USER,
    INNER_SCHEMA_ENSEMBLE_SYSTEM,
    INNER_SCHEMA_ENSEMBLE_USER,
)
from models.schemas import (
    TOC_PHASES,
    PhaseInnerSpec,
    PhaseVariableSpec,
    PolicyContext,
    PredictionResult,
    ToCInnerSchema,
)


GENERATED_PHASES = ("Inputs", "Activities", "Outputs", "Impact")
GROUNDED_INPUT_GENERATED_PHASES = ("Activities", "Outputs", "Impact")
CompletionFn = Callable[[list[dict], dict], str]


INNER_SCHEMA_GROUNDED_INPUT_ENSEMBLE_USER = """Design one independent candidate schema for the three generated phases of a policy Theory of Change.

The Inputs and Outcomes phases are fixed by the researcher and shown only as immutable causal anchors. Do not return, revise, rename, add to, or remove any Input or Outcome field.

For each phase in this exact order — Activities, Outputs, Impact — return:
- prediction_fields: 2–4 concrete, measurable variables. Each field has name (snake_case), unit, type (int | float | str), and description.
- discussion_points: 2–4 questions about mechanisms, constraints, implementation variation, or causal links relevant to that phase.

Requirements:
- Activities are actions performed using the fixed Inputs; Outputs are direct products, delivery, coverage, or reach; Impact is longer-term systemic change beyond the fixed Outcomes.
- Every Impact field must specify an explicit measurement horizon at least 3 years after policy implementation.
- Do not restate a fixed Input or Outcome by changing its subgroup, unit, aggregation, or wording.
- Ground fields in the policy context and preserve policy-stated quantities exactly when referenced.
- It is valid to infer a simulation variable when the document does not state it, but do not present an inferred quantity as policy fact.
- Seek a useful alternative operationalization in this independent run and avoid trivial synonyms.

Return exactly:
{{
  "phases": [
    {{"phase": "Activities", "prediction_fields": [...], "discussion_points": [...]}},
    {{"phase": "Outputs", "prediction_fields": [...], "discussion_points": [...]}},
    {{"phase": "Impact", "prediction_fields": [...], "discussion_points": [...]}}
  ]
}}

--- Policy Context ---
{policy_context}

--- Fixed Inputs (document-grounded; immutable; not part of your response) ---
{fixed_inputs}

--- Fixed Outcomes (immutable; not part of your response) ---
{fixed_outcomes}

--- Stage 1 Predictions (context for scale only; do not copy unsupported values as policy facts) ---
{stage1_predictions}

Independent candidate run: {run_index} of {run_count}
"""


INNER_SCHEMA_GROUNDED_INPUT_COMPACT_USER = """Select a compact schema from existing consolidated candidates for a policy simulation.

Inputs and Outcomes are fixed and excluded from selection. Select exactly 3 prediction fields and exactly 3 discussion points for each generated phase: Activities, Outputs, and Impact. Evaluate all three phases jointly so the same construct does not appear in multiple phases.

Selection principles:
- Activities: actions performed using the fixed Inputs.
- Outputs: direct products, delivery, coverage, or reach produced by Activities.
- Impact: longer-term systemic consequences downstream of the fixed Outcomes. Select only candidates marked impact_horizon_eligible=true.
- Do not select candidates that restate fixed Input or Outcome constructs.
- Prefer a coherent causal chain and fields supported across independent runs.
- Use only supplied candidate IDs; do not rename or invent fields.

Return exactly:
{{
  "phases": [
    {{"phase": "Activities", "selected_field_ids": ["...", "...", "..."], "selected_discussion_ids": ["...", "...", "..."], "rationale": "..."}},
    {{"phase": "Outputs", "selected_field_ids": ["...", "...", "..."], "selected_discussion_ids": ["...", "...", "..."], "rationale": "..."}},
    {{"phase": "Impact", "selected_field_ids": ["...", "...", "..."], "selected_discussion_ids": ["...", "...", "..."], "rationale": "..."}}
  ],
  "cross_phase_rationale": "Explain how phase leakage and overlap with fixed phases were avoided."
}}

--- Policy Context ---
{policy_context}

--- Fixed Inputs ---
{fixed_inputs}

--- Fixed Outcomes ---
{fixed_outcomes}

--- Consolidated Candidates ---
{candidates}
"""


INNER_SCHEMA_RAW_JOINT_SELECTION_USER = """Jointly consolidate five independent Theory of Change candidate runs into one compact final schema.

Inputs and Outcomes are fixed and immutable. For each generated phase — Activities, Outputs, and Impact — select exactly 3 prediction-field IDs and exactly 3 discussion-point IDs from the supplied raw candidates.

Selection rules:
- Treat semantically equivalent candidates across runs as support for the same construct, then select one representative ID. Do not select two synonyms or trivial reformulations.
- Activities must be implementation actions using the fixed Inputs.
- Outputs must be direct products, delivery, coverage, or reach resulting from Activities.
- Impact must be a distinct longer-term systemic consequence downstream of the fixed Outcomes and have an explicit horizon of at least 3 years.
- Do not restate or rescale any fixed Input or Outcome.
- Evaluate all three phases jointly to prevent causal-level leakage and cross-phase duplication.
- Prefer constructs recurring across independent runs when they are causally valid, but retain a less frequent construct when needed for coverage.
- Use only supplied IDs. Do not invent, rename, or rewrite fields.

Return exactly:
{{
  "phases": [
    {{"phase": "Activities", "selected_field_ids": ["...", "...", "..."], "selected_discussion_ids": ["...", "...", "..."], "rationale": "..."}},
    {{"phase": "Outputs", "selected_field_ids": ["...", "...", "..."], "selected_discussion_ids": ["...", "...", "..."], "rationale": "..."}},
    {{"phase": "Impact", "selected_field_ids": ["...", "...", "..."], "selected_discussion_ids": ["...", "...", "..."], "rationale": "..."}}
  ],
  "cross_phase_rationale": "Explain the resulting causal chain, semantic consolidation, and coverage choices."
}}

--- Policy Context ---
{policy_context}

--- Fixed Inputs ---
{fixed_inputs}

--- Fixed Outcomes ---
{fixed_outcomes}

--- Raw Candidates from Five Independent Runs ---
{candidates}
"""


def _normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _normalized_unit(value: str) -> str:
    """Normalize notation aliases without collapsing substantively different units."""
    normalized = _normalized_text(value)
    if normalized.startswith("number of "):
        normalized = normalized[len("number of ") :]
    aliases = {
        "usd": "usd",
        "dollar": "usd",
        "dollars": "usd",
        "us dollar": "usd",
        "us dollars": "usd",
        "fte": "fte",
        "ftes": "fte",
        "full time equivalent": "fte",
        "full time equivalents": "fte",
        "percent": "%",
        "percentage": "%",
        "percentage point": "percentage_points",
        "percentage points": "percentage_points",
        "pp": "percentage_points",
    }
    if value.strip() == "%":
        return "%"
    return aliases.get(normalized, normalized)


def coalesce_canonical_groups(
    groups: list[dict], kind: Literal["field", "discussion"]
) -> list[dict]:
    """Remove exact duplicates introduced by independently canonicalized LLM groups."""
    coalesced: dict[tuple, dict] = {}
    for group in groups:
        if kind == "field":
            field = group["canonical_field"]
            key = (
                _normalized_text(field["name"]),
                _normalized_unit(field["unit"]),
                field["type"],
            )
        else:
            key = (_normalized_text(group["canonical_text"]),)
        if key not in coalesced:
            coalesced[key] = copy.deepcopy(group)
            continue
        target = coalesced[key]
        target["source_ids"] = list(dict.fromkeys(target["source_ids"] + group["source_ids"]))
        rationales = [target.get("rationale", ""), group.get("rationale", "")]
        target["rationale"] = " | ".join(dict.fromkeys(item for item in rationales if item))
        if kind == "field":
            target["canonical_coalesced"] = True
    return list(coalesced.values())


def _has_explicit_long_term_horizon(field: dict) -> bool:
    text = _normalized_text(
        " ".join(
            str(field.get(key, "")) for key in ("name", "description")
        )
    )
    markers = (
        "long term",
        "lifetime",
        "intergenerational",
        "multi year",
        "3 year",
        "three year",
        "over 3 years",
        "over three years",
        "5 year",
        "five year",
        "10 year",
        "ten year",
        "over 5 years",
        "over 10 years",
        "cumulative",
        "over time",
        "market transformation",
        "structural shift",
        "vehicle stock",
        "fleet share",
        "supply chain growth",
    )
    return any(marker in text for marker in markers)


def _phase_map(schema: ToCInnerSchema) -> dict[str, PhaseInnerSpec]:
    phases = {phase.phase: phase for phase in schema.phases}
    if tuple(phase.phase for phase in schema.phases) != TOC_PHASES:
        raise ValueError(f"Fixed schema phases must be ordered as {TOC_PHASES}")
    return phases


def fixed_outcomes(schema: ToCInnerSchema) -> PhaseInnerSpec:
    return copy.deepcopy(_phase_map(schema)["Outcomes"])


def _parse_generated_phases(
    payload: dict,
    generated_phases: tuple[str, ...] = GENERATED_PHASES,
) -> list[PhaseInnerSpec]:
    phases = [PhaseInnerSpec(**item) for item in payload.get("phases", [])]
    actual = tuple(phase.phase for phase in phases)
    if actual != generated_phases:
        raise ValueError(
            f"Generated phase order must be {generated_phases}; got {actual}."
        )
    for phase in phases:
        if not 2 <= len(phase.prediction_fields) <= 4:
            raise ValueError(f"{phase.phase} must contain 2-4 prediction fields")
        if not 2 <= len(phase.discussion_points) <= 4:
            raise ValueError(f"{phase.phase} must contain 2-4 discussion points")
    return phases


def _assemble_schema(
    generated: list[PhaseInnerSpec],
    outcome: PhaseInnerSpec,
    fixed_input: PhaseInnerSpec | None = None,
) -> ToCInnerSchema:
    by_phase = {phase.phase: phase for phase in generated}
    phases = [
        copy.deepcopy(outcome)
        if phase == "Outcomes"
        else copy.deepcopy(fixed_input)
        if phase == "Inputs" and fixed_input is not None
        else copy.deepcopy(by_phase[phase])
        for phase in TOC_PHASES
    ]
    return ToCInnerSchema(phases=phases)


def generate_schema_samples(
    policy_context: PolicyContext,
    prediction_results: list[PredictionResult],
    fixed_schema: ToCInnerSchema,
    config: dict,
    runs: int = 5,
    fix_inputs: bool = False,
    completion_fn: CompletionFn = chat_completion,
) -> list[ToCInnerSchema]:
    """Generate ``runs`` independent schemas while preserving Outcomes verbatim."""
    if runs < 1:
        raise ValueError("runs must be at least 1")
    outcome = fixed_outcomes(fixed_schema)
    fixed_input = copy.deepcopy(_phase_map(fixed_schema)["Inputs"]) if fix_inputs else None
    generated_phases = GROUNDED_INPUT_GENERATED_PHASES if fix_inputs else GENERATED_PHASES
    samples: list[ToCInnerSchema] = []
    for run_index in range(1, runs + 1):
        messages = [
            {"role": "system", "content": INNER_SCHEMA_ENSEMBLE_SYSTEM},
            {
                "role": "user",
                "content": (
                    INNER_SCHEMA_GROUNDED_INPUT_ENSEMBLE_USER
                    if fix_inputs else INNER_SCHEMA_ENSEMBLE_USER
                ).format(
                    policy_context=json.dumps(policy_context.model_dump(), ensure_ascii=False),
                    fixed_inputs=json.dumps(
                        fixed_input.model_dump() if fixed_input else {}, ensure_ascii=False
                    ),
                    fixed_outcomes=json.dumps(outcome.model_dump(), ensure_ascii=False),
                    stage1_predictions=json.dumps(
                        [result.model_dump() for result in prediction_results],
                        ensure_ascii=False,
                    ),
                    run_index=run_index,
                    run_count=runs,
                ),
            },
        ]
        payload = json.loads(completion_fn(messages, config), strict=False)
        generated = _parse_generated_phases(payload, generated_phases)
        sample = _assemble_schema(generated, outcome, fixed_input)
        if sample.phases[3].model_dump() != outcome.model_dump():
            raise AssertionError("The fixed Outcomes phase changed during generation")
        samples.append(sample)
    return samples


def collect_candidates(
    samples: list[ToCInnerSchema],
    generated_phases: tuple[str, ...] = GENERATED_PHASES,
) -> dict[str, dict]:
    """Create stable candidate IDs and retain run provenance for consolidation."""
    candidates: dict[str, dict] = {}
    for phase_name in generated_phases:
        field_rows = []
        discussion_rows = []
        for run_index, schema in enumerate(samples, start=1):
            phase = _phase_map(schema)[phase_name]
            phase_slug = phase_name.lower()
            for index, field in enumerate(phase.prediction_fields, start=1):
                field_rows.append(
                    {
                        "id": f"r{run_index:02d}_{phase_slug}_f{index:02d}",
                        "run": run_index,
                        **field.model_dump(),
                    }
                )
            for index, point in enumerate(phase.discussion_points, start=1):
                discussion_rows.append(
                    {
                        "id": f"r{run_index:02d}_{phase_slug}_d{index:02d}",
                        "run": run_index,
                        "text": point,
                    }
                )
        candidates[phase_name] = {
            "fields": field_rows,
            "discussion_points": discussion_rows,
        }
    return candidates


def _exact_groups(rows: list[dict], kind: Literal["field", "discussion"]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if kind == "field":
            key = (_normalized_text(row["name"]), row["unit"].lower(), row["type"])
        else:
            key = (_normalized_text(row["text"]),)
        grouped[key].append(row)

    groups = []
    for members in grouped.values():
        first = members[0]
        if kind == "field":
            groups.append(
                {
                    "source_ids": [member["id"] for member in members],
                    "canonical_field": {
                        key: first[key] for key in ("name", "unit", "type", "description")
                    },
                    "rationale": "Deterministic normalized-name, unit, and type match.",
                }
            )
        else:
            groups.append(
                {
                    "source_ids": [member["id"] for member in members],
                    "canonical_text": first["text"],
                    "rationale": "Deterministic normalized-text match.",
                }
            )
    return groups


def _repair_partition(
    proposed: list[dict], rows: list[dict], kind: Literal["field", "discussion"]
) -> list[dict]:
    """Make LLM groups a complete partition and enforce field compatibility."""
    by_id = {row["id"]: row for row in rows}
    used: set[str] = set()
    repaired: list[dict] = []

    for group in proposed:
        ids = [item for item in group.get("source_ids", []) if item in by_id and item not in used]
        if not ids:
            continue
        if kind == "field":
            compatible: dict[tuple[str, str], list[str]] = defaultdict(list)
            for item_id in ids:
                row = by_id[item_id]
                compatible[(_normalized_unit(row["unit"]), row["type"])].append(item_id)
            for split_index, split_ids in enumerate(compatible.values()):
                first = by_id[split_ids[0]]
                canonical = group.get("canonical_field", {}) if split_index == 0 else {}
                canonical = {
                    "name": canonical.get("name") or first["name"],
                    "unit": first["unit"],
                    "type": first["type"],
                    "description": canonical.get("description") or first["description"],
                }
                PhaseVariableSpec(**canonical)
                repaired.append(
                    {
                        "source_ids": split_ids,
                        "canonical_field": canonical,
                        "rationale": group.get("rationale", "Semantic consolidation."),
                        "compatibility_split": len(compatible) > 1,
                    }
                )
                used.update(split_ids)
        else:
            canonical_text = group.get("canonical_text") or by_id[ids[0]]["text"]
            repaired.append(
                {
                    "source_ids": ids,
                    "canonical_text": canonical_text,
                    "rationale": group.get("rationale", "Semantic consolidation."),
                }
            )
            used.update(ids)

    for item_id, row in by_id.items():
        if item_id in used:
            continue
        if kind == "field":
            repaired.append(
                {
                    "source_ids": [item_id],
                    "canonical_field": {
                        key: row[key] for key in ("name", "unit", "type", "description")
                    },
                    "rationale": "Preserved as a singleton because it was not validly grouped.",
                    "compatibility_split": False,
                }
            )
        else:
            repaired.append(
                {
                    "source_ids": [item_id],
                    "canonical_text": row["text"],
                    "rationale": "Preserved as a singleton because it was not validly grouped.",
                }
            )
    return repaired


def consolidate_candidates(
    candidates: dict[str, dict],
    config: dict,
    mode: Literal["llm", "exact"] = "llm",
    generated_phases: tuple[str, ...] = GENERATED_PHASES,
    completion_fn: CompletionFn = chat_completion,
) -> dict[str, dict]:
    """Consolidate candidates phase by phase, preserving a complete provenance map."""
    audit: dict[str, dict] = {}
    for phase_name in generated_phases:
        phase_candidates = candidates[phase_name]
        if mode == "exact":
            field_groups = _exact_groups(phase_candidates["fields"], "field")
            discussion_groups = _exact_groups(
                phase_candidates["discussion_points"], "discussion"
            )
        elif mode == "llm":
            messages = [
                {"role": "system", "content": INNER_SCHEMA_DEDUP_SYSTEM},
                {
                    "role": "user",
                    "content": INNER_SCHEMA_DEDUP_USER.format(
                        phase=phase_name,
                        field_candidates=json.dumps(
                            phase_candidates["fields"], ensure_ascii=False
                        ),
                        discussion_candidates=json.dumps(
                            phase_candidates["discussion_points"], ensure_ascii=False
                        ),
                    ),
                },
            ]
            payload = json.loads(completion_fn(messages, config), strict=False)
            field_groups = _repair_partition(
                payload.get("field_groups", []), phase_candidates["fields"], "field"
            )
            discussion_groups = _repair_partition(
                payload.get("discussion_groups", []),
                phase_candidates["discussion_points"],
                "discussion",
            )
        else:
            raise ValueError(f"Unsupported consolidation mode: {mode}")

        field_groups = coalesce_canonical_groups(field_groups, "field")
        discussion_groups = coalesce_canonical_groups(discussion_groups, "discussion")

        field_runs = {row["id"]: row["run"] for row in phase_candidates["fields"]}
        point_runs = {
            row["id"]: row["run"] for row in phase_candidates["discussion_points"]
        }
        for group in field_groups:
            group["support_runs"] = sorted({field_runs[item] for item in group["source_ids"]})
            group["support_count"] = len(group["support_runs"])
        for group in discussion_groups:
            group["support_runs"] = sorted({point_runs[item] for item in group["source_ids"]})
            group["support_count"] = len(group["support_runs"])
        audit[phase_name] = {
            "field_groups": field_groups,
            "discussion_groups": discussion_groups,
        }
    return audit


def build_ensemble_schema(
    fixed_schema: ToCInnerSchema,
    consolidation: dict[str, dict],
    fix_inputs: bool = False,
) -> ToCInnerSchema:
    """Build the final schema; fixed Outcomes are copied byte-for-byte by value."""
    outcome = fixed_outcomes(fixed_schema)
    fixed_input = copy.deepcopy(_phase_map(fixed_schema)["Inputs"]) if fix_inputs else None
    generated_phases = GROUNDED_INPUT_GENERATED_PHASES if fix_inputs else GENERATED_PHASES
    generated = []
    for phase_name in generated_phases:
        phase = consolidation[phase_name]
        field_groups = sorted(
            phase["field_groups"], key=lambda item: (-item["support_count"], item["source_ids"][0])
        )
        point_groups = sorted(
            phase["discussion_groups"],
            key=lambda item: (-item["support_count"], item["source_ids"][0]),
        )
        generated.append(
            PhaseInnerSpec(
                phase=phase_name,
                prediction_fields=[
                    PhaseVariableSpec(**group["canonical_field"]) for group in field_groups
                ],
                discussion_points=[group["canonical_text"] for group in point_groups],
            )
        )
    schema = _assemble_schema(generated, outcome, fixed_input)
    if schema.phases[3].model_dump() != outcome.model_dump():
        raise AssertionError("The fixed Outcomes phase changed during consolidation")
    return schema


def select_joint_schema_from_raw_candidates(
    policy_context: PolicyContext,
    fixed_schema: ToCInnerSchema,
    candidates: dict[str, dict],
    config: dict,
    completion_fn: CompletionFn = chat_completion,
) -> tuple[ToCInnerSchema, dict]:
    """One-call semantic consolidation and selection with fixed Inputs/Outcomes."""
    generated_phases = GROUNDED_INPUT_GENERATED_PHASES
    fixed_input = copy.deepcopy(_phase_map(fixed_schema)["Inputs"])
    outcome = fixed_outcomes(fixed_schema)
    catalog = copy.deepcopy(candidates)
    catalog["Impact"]["fields"] = [
        {**row, "impact_horizon_eligible": _has_explicit_long_term_horizon(row)}
        for row in catalog["Impact"]["fields"]
        if _has_explicit_long_term_horizon(row)
    ]
    if len(catalog["Impact"]["fields"]) < 3:
        raise ValueError("Fewer than 3 raw Impact candidates have an explicit long-term horizon")

    messages = [
        {"role": "system", "content": INNER_SCHEMA_COMPACT_SYSTEM},
        {
            "role": "user",
            "content": INNER_SCHEMA_RAW_JOINT_SELECTION_USER.format(
                policy_context=json.dumps(policy_context.model_dump(), ensure_ascii=False),
                fixed_inputs=json.dumps(fixed_input.model_dump(), ensure_ascii=False),
                fixed_outcomes=json.dumps(outcome.model_dump(), ensure_ascii=False),
                candidates=json.dumps(catalog, ensure_ascii=False),
            ),
        },
    ]
    selection = json.loads(completion_fn(messages, config), strict=False)
    phase_selections = selection.get("phases", [])
    actual_order = tuple(item.get("phase") for item in phase_selections)
    if actual_order != generated_phases:
        raise ValueError(f"Joint selection phase order must be {generated_phases}")

    generated: list[PhaseInnerSpec] = []
    selected_names: set[str] = set()
    for item in phase_selections:
        phase_name = item["phase"]
        field_ids = item.get("selected_field_ids", [])
        discussion_ids = item.get("selected_discussion_ids", [])
        if len(field_ids) != 3 or len(set(field_ids)) != 3:
            raise ValueError(f"{phase_name} must select exactly 3 unique fields")
        if len(discussion_ids) != 3 or len(set(discussion_ids)) != 3:
            raise ValueError(f"{phase_name} must select exactly 3 unique discussion points")
        fields_by_id = {row["id"]: row for row in catalog[phase_name]["fields"]}
        points_by_id = {
            row["id"]: row for row in catalog[phase_name]["discussion_points"]
        }
        if any(item_id not in fields_by_id for item_id in field_ids):
            raise ValueError(f"{phase_name} selected an unknown field ID")
        if any(item_id not in points_by_id for item_id in discussion_ids):
            raise ValueError(f"{phase_name} selected an unknown discussion ID")
        fields = []
        for item_id in field_ids:
            row = fields_by_id[item_id]
            normalized_name = _normalized_text(row["name"])
            if normalized_name in selected_names:
                raise ValueError(f"Duplicate field name selected across phases: {row['name']}")
            selected_names.add(normalized_name)
            fields.append(PhaseVariableSpec(**{
                key: row[key] for key in ("name", "unit", "type", "description")
            }))
        generated.append(PhaseInnerSpec(
            phase=phase_name,
            prediction_fields=fields,
            discussion_points=[points_by_id[item_id]["text"] for item_id in discussion_ids],
        ))

    schema = _assemble_schema(generated, outcome, fixed_input)
    if schema.phases[0].model_dump() != fixed_input.model_dump():
        raise AssertionError("The fixed Inputs phase changed during joint selection")
    if schema.phases[3].model_dump() != outcome.model_dump():
        raise AssertionError("The fixed Outcomes phase changed during joint selection")
    return schema, {"selection": selection, "candidate_catalog": catalog}


def build_compact_candidates(
    consolidation: dict[str, dict],
    fallback_impact: PhaseInnerSpec | None = None,
    generated_phases: tuple[str, ...] = GENERATED_PHASES,
) -> dict[str, dict]:
    """Assign stable IDs to consolidated groups for joint cross-phase selection."""
    payload: dict[str, dict] = {}
    for phase_name in generated_phases:
        phase_slug = phase_name.lower()
        field_groups = sorted(
            consolidation[phase_name]["field_groups"],
            key=lambda item: (-item["support_count"], item["source_ids"][0]),
        )
        discussion_groups = sorted(
            consolidation[phase_name]["discussion_groups"],
            key=lambda item: (-item["support_count"], item["source_ids"][0]),
        )
        payload[phase_name] = {
            "fields": [
                {
                    "candidate_id": f"{phase_slug}_f{index:02d}",
                    "support_count": group["support_count"],
                    "support_runs": group["support_runs"],
                    **group["canonical_field"],
                    **(
                        {
                            "impact_horizon_eligible": _has_explicit_long_term_horizon(
                                group["canonical_field"]
                            )
                        }
                        if phase_name == "Impact"
                        else {}
                    ),
                }
                for index, group in enumerate(field_groups, start=1)
            ],
            "discussion_points": [
                {
                    "candidate_id": f"{phase_slug}_d{index:02d}",
                    "support_count": group["support_count"],
                    "support_runs": group["support_runs"],
                    "text": group["canonical_text"],
                }
                for index, group in enumerate(discussion_groups, start=1)
            ],
        }

        unique_fields = []
        seen_field_names: set[str] = set()
        for row in payload[phase_name]["fields"]:
            normalized_name = _normalized_text(row["name"])
            if normalized_name in seen_field_names:
                continue
            seen_field_names.add(normalized_name)
            unique_fields.append(row)
        payload[phase_name]["fields"] = unique_fields

    eligible_impact = [
        row for row in payload["Impact"]["fields"]
        if row.get("impact_horizon_eligible", False)
    ]
    if len(eligible_impact) < 3 and fallback_impact is not None:
        existing_names = {_normalized_text(row["name"]) for row in eligible_impact}
        for index, field in enumerate(fallback_impact.prediction_fields, start=1):
            row = {
                "candidate_id": f"impact_seed_f{index:02d}",
                "support_count": 0,
                "support_runs": [],
                **field.model_dump(),
                "impact_horizon_eligible": _has_explicit_long_term_horizon(
                    field.model_dump()
                ),
                "candidate_source": "fixed_schema_fallback",
            }
            if (
                row["impact_horizon_eligible"]
                and _normalized_text(row["name"]) not in existing_names
            ):
                eligible_impact.append(row)
                existing_names.add(_normalized_text(row["name"]))
    payload["Impact"]["fields"] = eligible_impact
    return payload


def _select_compact_schema_once(
    policy_context: PolicyContext,
    fixed_schema: ToCInnerSchema,
    consolidation: dict[str, dict],
    config: dict,
    fix_inputs: bool = False,
    completion_fn: CompletionFn = chat_completion,
) -> tuple[ToCInnerSchema, dict]:
    """Jointly select exactly three fields and discussion points per generated phase."""
    generated_phases = GROUNDED_INPUT_GENERATED_PHASES if fix_inputs else GENERATED_PHASES
    fixed_input = copy.deepcopy(_phase_map(fixed_schema)["Inputs"]) if fix_inputs else None
    candidates = build_compact_candidates(
        consolidation,
        fallback_impact=_phase_map(fixed_schema)["Impact"],
        generated_phases=generated_phases,
    )
    outcome = fixed_outcomes(fixed_schema)
    messages = [
        {"role": "system", "content": INNER_SCHEMA_COMPACT_SYSTEM},
        {
            "role": "user",
                "content": (
                    INNER_SCHEMA_GROUNDED_INPUT_COMPACT_USER
                    if fix_inputs else INNER_SCHEMA_COMPACT_USER
                ).format(
                    policy_context=json.dumps(policy_context.model_dump(), ensure_ascii=False),
                    fixed_inputs=json.dumps(
                        fixed_input.model_dump() if fixed_input else {}, ensure_ascii=False
                    ),
                fixed_outcomes=json.dumps(outcome.model_dump(), ensure_ascii=False),
                candidates=json.dumps(candidates, ensure_ascii=False),
            ),
        },
    ]
    selection = json.loads(completion_fn(messages, config), strict=False)
    phase_selections = selection.get("phases", [])
    actual_order = tuple(item.get("phase") for item in phase_selections)
    if actual_order != generated_phases:
        raise ValueError(f"Compact selection phase order must be {generated_phases}")

    generated: list[PhaseInnerSpec] = []
    selected_names: set[str] = set()
    for item in phase_selections:
        phase_name = item["phase"]
        field_ids = item.get("selected_field_ids", [])
        discussion_ids = item.get("selected_discussion_ids", [])
        if len(field_ids) != 3 or len(set(field_ids)) != 3:
            raise ValueError(f"{phase_name} must select exactly 3 unique fields")
        if len(discussion_ids) != 3 or len(set(discussion_ids)) != 3:
            raise ValueError(f"{phase_name} must select exactly 3 unique discussion points")

        fields_by_id = {
            row["candidate_id"]: row for row in candidates[phase_name]["fields"]
        }
        points_by_id = {
            row["candidate_id"]: row
            for row in candidates[phase_name]["discussion_points"]
        }
        if any(candidate_id not in fields_by_id for candidate_id in field_ids):
            raise ValueError(f"{phase_name} selected an unknown or cross-phase field ID")
        if any(candidate_id not in points_by_id for candidate_id in discussion_ids):
            raise ValueError(f"{phase_name} selected an unknown or cross-phase discussion ID")

        fields = []
        for candidate_id in field_ids:
            row = fields_by_id[candidate_id]
            if phase_name == "Impact" and not row.get("impact_horizon_eligible", False):
                raise ValueError(
                    f"Impact field lacks an explicit long-term horizon: {row['name']}"
                )
            normalized_name = _normalized_text(row["name"])
            if normalized_name in selected_names:
                raise ValueError(f"Duplicate field construct name selected: {row['name']}")
            selected_names.add(normalized_name)
            fields.append(
                PhaseVariableSpec(
                    name=row["name"],
                    unit=row["unit"],
                    type=row["type"],
                    description=row["description"],
                )
            )
        generated.append(
            PhaseInnerSpec(
                phase=phase_name,
                prediction_fields=fields,
                discussion_points=[points_by_id[candidate_id]["text"] for candidate_id in discussion_ids],
            )
        )

    compact_schema = _assemble_schema(generated, outcome, fixed_input)
    if compact_schema.phases[3].model_dump() != outcome.model_dump():
        raise AssertionError("The fixed Outcomes phase changed during compact selection")
    audit = {"selection": selection, "candidate_catalog": candidates}
    return compact_schema, audit


def select_compact_schema(
    policy_context: PolicyContext,
    fixed_schema: ToCInnerSchema,
    consolidation: dict[str, dict],
    config: dict,
    fix_inputs: bool = False,
    completion_fn: CompletionFn = chat_completion,
) -> tuple[ToCInnerSchema, dict]:
    """Select a compact schema, retrying malformed or duplicate selections."""
    last_error: ValueError | None = None
    for _ in range(3):
        try:
            return _select_compact_schema_once(
                policy_context=policy_context,
                fixed_schema=fixed_schema,
                consolidation=consolidation,
                config=config,
                fix_inputs=fix_inputs,
                completion_fn=completion_fn,
            )
        except ValueError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error
