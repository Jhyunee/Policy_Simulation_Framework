from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.schemas import RolePrototype, Stage1Result, StakeholderGroup
from stage1.mechanism_panel import select_mechanism_panel
from stage1.pipeline import (
    _build_fixed_mechanism_panel_bundle,
    flatten_persona_groups,
    save_english_json,
    save_json,
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-select a completed mechanism panel from its saved candidate pool."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy-graph", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pool = [
        StakeholderGroup.model_validate(item)
        for item in _load(run_dir / "personas_pool.json")
    ]
    prototypes = [
        RolePrototype.model_validate(item)
        for item in _load(run_dir / "role_prototypes.json")
    ]
    selected, manifest = select_mechanism_panel(pool, prototypes, config)

    save_json(manifest, run_dir / "panel_manifest.json")
    selected_dump = [group.model_dump() for group in selected]
    save_english_json(selected_dump, run_dir / "simulation_participants.json")
    save_english_json(selected_dump, run_dir / "personas_initial.json")

    _, persona_id_lookup = flatten_persona_groups(pool)
    selected_flat, _ = flatten_persona_groups(selected, persona_id_lookup)
    save_english_json(selected_flat, run_dir / "simulation_personas_flat.json")

    bundle = _build_fixed_mechanism_panel_bundle(
        selected_groups=selected,
        policy_graph_path=str(args.policy_graph.resolve()),
        config=config,
        run_dir=run_dir,
        panel_manifest=manifest,
    )
    save_json(bundle, run_dir / "simulation_personas.json")

    stage1 = Stage1Result.model_validate(_load(run_dir / "stage1_result.json"))
    stage1.stakeholder_groups = selected
    stage1.simulation_participants = selected
    stage1.handoff_metadata.setdefault("sampling", {}).update({
        "strategy": "fixed_role_balanced_mechanism_panel",
        "panel_id": manifest["panel_id"],
        "selected_count": manifest["panel_size"],
    })
    save_english_json(stage1.model_dump(), run_dir / "stage1_result.json")

    run_log_path = run_dir / "run_log.json"
    if run_log_path.exists():
        run_log = _load(run_log_path)
        run_log["panel_reselection"] = {
            "reselected_at": datetime.now().isoformat(timespec="seconds"),
            "candidate_overrides": (
                ((config.get("persona") or {}).get("mechanism_panel") or {})
                .get("candidate_overrides", {})
            ),
        }
        save_json(run_log, run_log_path)

    print(f"Reselected {manifest['panel_size']} participants in {run_dir}")
    for selection in manifest["selections"]:
        print(
            f"  {selection['slot']}: {selection['selected_representative_type']} "
            f"({selection['selection_method']})"
        )


if __name__ == "__main__":
    main()
