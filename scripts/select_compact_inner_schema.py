#!/usr/bin/env python3
"""Select a 3-field-per-phase schema from a consolidated schema ensemble."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.schemas import GroundedInputState, Stage1Result, ToCInnerSchema
from stage2_toc.grounded_inputs import apply_grounded_inputs_to_schema
from stage2_toc.schema_ensemble import select_compact_schema


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage1-result", type=Path, required=True)
    parser.add_argument("--fixed-schema", type=Path, required=True)
    parser.add_argument("--dedupe-audit", type=Path, required=True)
    parser.add_argument("--grounded-input-state", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stage1 = Stage1Result(**_load(args.stage1_result))
    if stage1.policy_context is None:
        raise ValueError("stage1_result.policy_context is required")
    fixed_schema = ToCInnerSchema(**_load(args.fixed_schema))
    fix_inputs = args.grounded_input_state is not None
    if fix_inputs:
        grounded_state = GroundedInputState(**_load(args.grounded_input_state))
        fixed_schema = apply_grounded_inputs_to_schema(fixed_schema, grounded_state)
    compact, audit = select_compact_schema(
        policy_context=stage1.policy_context,
        fixed_schema=fixed_schema,
        consolidation=_load(args.dedupe_audit),
        config=config,
        fix_inputs=fix_inputs,
    )
    _save(args.output_dir / "inner_schema_compact.json", compact.model_dump())
    _save(args.output_dir / "selection_audit.json", audit)
    _save(
        args.output_dir / "run_manifest.json",
        {
            "stage1_result": str(args.stage1_result.resolve()),
            "fixed_schema": str(args.fixed_schema.resolve()),
            "dedupe_audit": str(args.dedupe_audit.resolve()),
            "selection": "joint_llm_selection",
            "fields_per_generated_phase": 3,
            "discussion_points_per_generated_phase": 3,
            "outcomes_fixed": True,
            "inputs_fixed": fix_inputs,
            "grounded_input_state": (
                str(args.grounded_input_state.resolve()) if args.grounded_input_state else None
            ),
        },
    )
    print(f"Saved compact schema to {args.output_dir}")


if __name__ == "__main__":
    main()
