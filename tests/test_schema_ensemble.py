import copy
import json

from models.schemas import PolicyContext, ToCInnerSchema
from stage2_toc.schema_ensemble import (
    _normalized_unit,
    build_compact_candidates,
    build_ensemble_schema,
    coalesce_canonical_groups,
    collect_candidates,
    consolidate_candidates,
    generate_schema_samples,
    select_compact_schema,
)


def _policy_context() -> PolicyContext:
    return PolicyContext(
        policy_name="Test policy",
        policy_domain="social policy",
        policy_type="benefit",
        delivery_mechanism="monthly payments",
        eligibility_logic="eligible households",
        resource_logic="administrative capacity",
        theory_of_change_summary="Resources enable delivery and household outcomes.",
        simulation_relevance="Tests implementation pathways.",
    )


def _fixed_schema() -> ToCInnerSchema:
    phases = []
    for phase in ("Inputs", "Activities", "Outputs", "Outcomes", "Impact"):
        phases.append(
            {
                "phase": phase,
                "prediction_fields": [
                    {
                        "name": f"{phase.lower()}_fixed",
                        "unit": "%" if phase == "Outcomes" else "count",
                        "type": "float" if phase == "Outcomes" else "int",
                        "description": f"Fixed {phase} field",
                    }
                ],
                "discussion_points": [f"Fixed {phase} point"],
            }
        )
    return ToCInnerSchema(phases=phases)


def _generation_response(messages, config):
    del messages, config
    phases = []
    for phase in ("Inputs", "Activities", "Outputs", "Impact"):
        phases.append(
            {
                "phase": phase,
                "prediction_fields": [
                    {
                        "name": f"{phase.lower()}_a",
                        "unit": "count",
                        "type": "int",
                        "description": "First construct",
                    },
                    {
                        "name": f"{phase.lower()}_b",
                        "unit": "%",
                        "type": "float",
                        "description": "Second construct",
                    },
                ],
                "discussion_points": ["How much?", "Under what constraint?"],
            }
        )
    return json.dumps({"phases": phases})


def test_generation_keeps_outcomes_immutable():
    fixed = _fixed_schema()
    samples = generate_schema_samples(
        _policy_context(), [], fixed, {}, runs=2, completion_fn=_generation_response
    )
    assert len(samples) == 2
    for sample in samples:
        assert sample.phases[3].model_dump() == fixed.phases[3].model_dump()


def test_exact_consolidation_retains_provenance_and_fixed_outcome():
    fixed = _fixed_schema()
    samples = generate_schema_samples(
        _policy_context(), [], fixed, {}, runs=2, completion_fn=_generation_response
    )
    candidates = collect_candidates(samples)
    audit = consolidate_candidates(candidates, {}, mode="exact")
    final = build_ensemble_schema(fixed, audit)

    assert len(audit["Inputs"]["field_groups"]) == 2
    assert audit["Inputs"]["field_groups"][0]["support_count"] == 2
    assert final.phases[3].model_dump() == fixed.phases[3].model_dump()


def test_llm_consolidation_splits_incompatible_units():
    fixed = _fixed_schema()
    samples = generate_schema_samples(
        _policy_context(), [], fixed, {}, runs=1, completion_fn=_generation_response
    )
    candidates = collect_candidates(samples)

    def bad_merge(messages, config):
        del config
        content = messages[1]["content"]
        phase = next(name for name in ("Inputs", "Activities", "Outputs", "Impact") if f"Phase: {name}" in content)
        rows = candidates[phase]
        return json.dumps(
            {
                "field_groups": [
                    {
                        "source_ids": [row["id"] for row in rows["fields"]],
                        "canonical_field": {
                            "name": "incorrect_merge",
                            "unit": "count",
                            "type": "int",
                            "description": "Should be split",
                        },
                        "rationale": "Incorrectly treated as duplicates",
                    }
                ],
                "discussion_groups": [
                    {
                        "source_ids": [row["id"] for row in rows["discussion_points"]],
                        "canonical_text": "Canonical point",
                        "rationale": "Merged",
                    }
                ],
            }
        )

    audit = consolidate_candidates(candidates, {}, mode="llm", completion_fn=bad_merge)
    assert len(audit["Inputs"]["field_groups"]) == 2
    assert all(group["compatibility_split"] for group in audit["Inputs"]["field_groups"])


def test_unit_normalization_only_collapses_notation_aliases():
    assert _normalized_unit("USD") == _normalized_unit("dollars")
    assert _normalized_unit("FTEs") == _normalized_unit("full-time equivalents")
    assert _normalized_unit("percentage points") != _normalized_unit("%")
    assert _normalized_unit("families") != _normalized_unit("households")


def test_canonical_groups_with_notation_aliases_are_coalesced():
    groups = [
        {
            "source_ids": ["a"],
            "canonical_field": {
                "name": "payments_count",
                "unit": "payments",
                "type": "int",
                "description": "Payments",
            },
            "rationale": "First",
        },
        {
            "source_ids": ["b"],
            "canonical_field": {
                "name": "payments_count",
                "unit": "number of payments",
                "type": "int",
                "description": "Payments",
            },
            "rationale": "Second",
        },
    ]
    result = coalesce_canonical_groups(groups, "field")
    assert len(result) == 1
    assert result[0]["source_ids"] == ["a", "b"]


def test_compact_selection_keeps_three_fields_and_fixed_outcomes():
    fixed = _fixed_schema()
    samples = generate_schema_samples(
        _policy_context(), [], fixed, {}, runs=2, completion_fn=_generation_response
    )
    consolidation = consolidate_candidates(collect_candidates(samples), {}, mode="exact")

    for phase in ("Inputs", "Activities", "Outputs", "Impact"):
        slug = phase.lower()
        if phase == "Impact":
            for group in consolidation[phase]["field_groups"]:
                group["canonical_field"]["description"] += " over a 10-year horizon"
        existing_field = consolidation[phase]["field_groups"][0]
        consolidation[phase]["field_groups"].append(
            {
                **existing_field,
                "source_ids": [f"extra_{slug}"],
                "canonical_field": {
                    **existing_field["canonical_field"],
                        "name": f"{slug}_c",
                        "description": (
                            "Long-term outcome over 10 years"
                            if phase == "Impact"
                            else existing_field["canonical_field"]["description"]
                        ),
                    },
                "support_runs": [1],
                "support_count": 1,
            }
        )
        existing_point = consolidation[phase]["discussion_groups"][0]
        consolidation[phase]["discussion_groups"].append(
            {
                **existing_point,
                "source_ids": [f"extra_d_{slug}"],
                "canonical_text": f"Third {phase} point?",
                "support_runs": [1],
                "support_count": 1,
            }
        )

    def select_first_three(messages, config):
        del messages, config
        phases = []
        for phase in ("Inputs", "Activities", "Outputs", "Impact"):
            slug = phase.lower()
            phases.append(
                {
                    "phase": phase,
                    "selected_field_ids": [f"{slug}_f01", f"{slug}_f02", f"{slug}_f03"],
                    "selected_discussion_ids": [f"{slug}_d01", f"{slug}_d02", f"{slug}_d03"],
                    "rationale": "Test",
                }
            )
        return json.dumps({"phases": phases, "cross_phase_rationale": "Distinct phases"})

    compact, _ = select_compact_schema(
        _policy_context(), fixed, consolidation, {}, completion_fn=select_first_three
    )
    assert [len(phase.prediction_fields) for phase in compact.phases] == [3, 3, 3, 1, 3]
    assert compact.phases[3].model_dump() == fixed.phases[3].model_dump()


def test_compact_candidates_fall_back_to_fixed_long_term_impact_fields():
    fixed = _fixed_schema()
    fixed.phases[4] = fixed.phases[4].model_copy(
        update={
            "prediction_fields": [
                fixed.phases[4].prediction_fields[0].model_copy(
                    update={
                        "name": f"impact_seed_{index}",
                        "unit": "%",
                        "type": "float",
                        "description": "Long-term change three years after policy launch.",
                    }
                )
                for index in range(1, 4)
            ]
        }
    )
    samples = generate_schema_samples(
        _policy_context(), [], fixed, {}, runs=1, completion_fn=_generation_response
    )
    consolidation = consolidate_candidates(collect_candidates(samples), {}, mode="exact")

    candidates = build_compact_candidates(
        consolidation,
        fallback_impact=fixed.phases[4],
    )

    assert len(candidates["Impact"]["fields"]) == 3
    assert all(
        row["candidate_source"] == "fixed_schema_fallback"
        for row in candidates["Impact"]["fields"]
    )


def test_compact_candidates_remove_duplicate_canonical_field_names():
    fixed = _fixed_schema()
    samples = generate_schema_samples(
        _policy_context(), [], fixed, {}, runs=1, completion_fn=_generation_response
    )
    consolidation = consolidate_candidates(collect_candidates(samples), {}, mode="exact")
    duplicate = copy.deepcopy(consolidation["Activities"]["field_groups"][0])
    duplicate["source_ids"] = ["duplicate_activity"]
    duplicate["canonical_field"]["unit"] = "different unit"
    consolidation["Activities"]["field_groups"].append(duplicate)

    candidates = build_compact_candidates(consolidation)

    names = [row["name"] for row in candidates["Activities"]["fields"]]
    assert len(names) == len(set(names))
    assert all(
        row["impact_horizon_eligible"]
        for row in candidates["Impact"]["fields"]
    )
