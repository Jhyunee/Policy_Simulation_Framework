"""Build disjoint follow-up panels from a mechanism-panel persona pool."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona.name_utils import infer_persona_name
from persona.simulation_schema import sanitize_simulation_persona


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _participant(candidate: dict[str, Any], position: int) -> dict[str, Any]:
    demographics = candidate.get("demographics") or {}
    source = {**candidate, **demographics}
    representative_type = str(candidate.get("representative_type") or "")
    suffix_match = re.search(r"-(?P<suffix>[0-9A-Za-z]+)$", representative_type)
    suffix = suffix_match.group("suffix") if suffix_match else f"candidate{position}"
    role_id = str(demographics.get("role_prototype_id") or candidate["stakeholder_type"])
    source.update(
        {
            "persona_id": f"P_{role_id}__{suffix}",
            "name": infer_persona_name(source),
            "stakeholder_type": candidate["stakeholder_type"],
            "panel_position": position,
            "policy_role_context": demographics.get("policy_role_context"),
        }
    )
    participant = sanitize_simulation_persona(source)
    if not participant.get("name"):
        raise ValueError(f"Could not infer name for {candidate.get('representative_type')}")
    return participant


def build_bundle(pool: list[dict[str, Any]], manifest: dict[str, Any]) -> tuple[dict, dict]:
    role_order = list(manifest["role_order"])
    selected_by_role = {
        item["slot"]: item["selected_representative_type"]
        for item in manifest["selections"]
    }
    groups = {group["stakeholder_type"]: group["personas"] for group in pool}

    remaining: dict[str, list[dict[str, Any]]] = {}
    for role in role_order:
        chosen = selected_by_role[role]
        candidates = [
            {**candidate, "stakeholder_type": role}
            for candidate in groups[role]
            if candidate.get("representative_type") != chosen
        ]
        if len(candidates) != 4:
            raise ValueError(f"Expected four remaining {role} candidates; got {len(candidates)}")
        remaining[role] = candidates

    scenarios = []
    manifest_panels = []
    for offset in range(4):
        panel_number = offset + 2
        participants = [
            _participant(remaining[role][offset], position)
            for position, role in enumerate(role_order, 1)
        ]
        scenario_id = f"nsa_disjoint_mechanism_panel_v{panel_number}"
        scenarios.append({"scenario_id": scenario_id, "participants": participants})
        manifest_panels.append(
            {
                "panel_number": panel_number,
                "scenario_id": scenario_id,
                "participants": [
                    {
                        "persona_id": person["persona_id"],
                        "name": person["name"],
                        "stakeholder_type": person["stakeholder_type"],
                    }
                    for person in participants
                ],
            }
        )

    bundle = {
        "schema_version": "simulation_persona_v1",
        "scenario_count": len(scenarios),
        "shared_context": {"org_kg_contexts": []},
        "scenarios": scenarios,
        "sampling_metadata": {
            "strategy": "disjoint_role_balanced_mechanism_panels",
            "source_panel_id": manifest["panel_id"],
            "role_order": role_order,
            "participants_per_scenario": 5,
            "num_scenarios": len(scenarios),
            "persona_reuse_across_panels": False,
            "ebc_blind_selection": True,
        },
    }
    audit = {
        "source_panel_id": manifest["panel_id"],
        "construction_policy": "exclude primary selection, then zip remaining candidates in source order",
        "role_order": role_order,
        "panels": manifest_panels,
    }
    return bundle, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    bundle, audit = build_bundle(_load(args.pool), _load(args.primary_manifest))
    args.output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    args.manifest_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}: {bundle['scenario_count']} panels, 20 unique personas")


if __name__ == "__main__":
    main()
