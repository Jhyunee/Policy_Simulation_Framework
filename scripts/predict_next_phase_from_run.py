"""Predict one ToC phase from a completed run's inherited phase history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_phase_branch_tree import ROOT, load_runtime
from models.schemas import PhaseRoundResult, ToCInnerSchema
from stage2_toc.branching import generate_phase_children


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--inner-schema", type=Path, required=True)
    parser.add_argument("--parent-phase", required=True)
    parser.add_argument("--transition", default="baseline")
    parser.add_argument("--interaction-mode", default="independent")
    parser.add_argument("--simulation-target", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_dir = args.source_dir if args.source_dir.is_absolute() else ROOT / args.source_dir
    schema_path = args.inner_schema if args.inner_schema.is_absolute() else ROOT / args.inner_schema
    output = args.output if args.output.is_absolute() else ROOT / args.output

    runtime = load_runtime(source_dir, args.scenario)
    inner_schema = ToCInnerSchema.model_validate(
        json.loads(schema_path.read_text(encoding="utf-8"))
    )
    phase_names = [spec.phase for spec in inner_schema.phases]
    try:
        parent_index = phase_names.index(args.parent_phase)
    except ValueError as exc:
        raise ValueError(
            f"parent phase {args.parent_phase!r} not found in {phase_names}"
        ) from exc
    if parent_index >= len(phase_names) - 1:
        raise ValueError("parent phase must have a following phase")

    source_forward = runtime["scenario"].get("forward_pass") or []
    if len(source_forward) <= parent_index:
        raise ValueError(
            f"source scenario has {len(source_forward)} forward phases; "
            f"cannot inherit through index {parent_index}"
        )
    prior_phases = [
        PhaseRoundResult.model_validate(item)
        for item in source_forward[: parent_index + 1]
    ]

    config = runtime["config"]
    stage2 = config.setdefault("stage2_toc", {})
    stage2["phase_interaction_mode"] = args.interaction_mode
    stage2["stance"] = "baseline"
    stage2.setdefault("branch_mechanism_planner", {})["enabled"] = False

    simulation_target = args.simulation_target or runtime["simulation_target"]
    children = generate_phase_children(
        parent_id=f"source/scenario-{args.scenario}/{args.parent_phase}",
        prior_phases=prior_phases,
        personas=runtime["personas"],
        inner_schema=inner_schema,
        policy_context=runtime["policy_context"],
        simulation_target=simulation_target,
        config=config,
        transitions=(args.transition,),
        parallel_transitions=False,
    )

    payload = {
        "source_run_dir": str(source_dir),
        "source_scenario": args.scenario,
        "inherited_phases": [result.phase for result in prior_phases],
        "predicted_phase": phase_names[parent_index + 1],
        "transition": args.transition,
        "interaction_mode": args.interaction_mode,
        "simulation_target": simulation_target,
        "inner_schema": str(schema_path),
        "results": [child.model_dump(mode="json") for child in children],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
