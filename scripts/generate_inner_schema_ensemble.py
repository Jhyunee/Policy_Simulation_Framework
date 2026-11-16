#!/usr/bin/env python3
"""Generate repeated non-Outcome ToC schemas and semantically consolidate them."""

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

from models.schemas import GroundedInputState, Stage1Result, ToCInnerSchema
from stage2_toc.grounded_inputs import apply_grounded_inputs_to_schema
from stage2_toc.schema_ensemble import (
    GROUNDED_INPUT_GENERATED_PHASES,
    build_ensemble_schema,
    collect_candidates,
    consolidate_candidates,
    generate_schema_samples,
    select_joint_schema_from_raw_candidates,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat Inputs/Activities/Outputs/Impact schema generation while fixing Outcomes."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage1-result", type=Path, required=True)
    parser.add_argument("--fixed-schema", type=Path, required=True)
    parser.add_argument(
        "--grounded-input-state",
        type=Path,
        default=None,
        help="Fix Inputs from a validated grounded_input_state.json instead of generating them.",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--dedupe-mode", choices=("llm", "exact"), default="llm")
    parser.add_argument(
        "--joint-finalize",
        action="store_true",
        help="Use one final LLM call to consolidate and select directly from all raw candidates.",
    )
    parser.add_argument(
        "--reuse-candidates",
        type=Path,
        default=None,
        help="Skip generation and re-consolidate an existing candidate_fields.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stage1 = Stage1Result(**_load_json(args.stage1_result))
    if stage1.policy_context is None:
        raise ValueError("stage1_result.policy_context is required")
    fixed_schema = ToCInnerSchema(**_load_json(args.fixed_schema))
    fix_inputs = args.grounded_input_state is not None
    if fix_inputs:
        grounded_state = GroundedInputState(**_load_json(args.grounded_input_state))
        fixed_schema = apply_grounded_inputs_to_schema(fixed_schema, grounded_state)

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model = config.get("llm", {}).get("model", "llm")
        output_dir = REPO_ROOT / "outputs" / "usa" / "chi_ctc" / "inner_schema_ensemble" / f"{stamp}_{model}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if fix_inputs:
        _save_json(output_dir / "fixed_schema_effective.json", fixed_schema.model_dump())

    if args.reuse_candidates is None:
        samples = generate_schema_samples(
            policy_context=stage1.policy_context,
            prediction_results=stage1.predictions,
            fixed_schema=fixed_schema,
            config=config,
            runs=args.runs,
            fix_inputs=fix_inputs,
        )
        for index, sample in enumerate(samples, start=1):
            _save_json(output_dir / f"run_{index:02d}.json", sample.model_dump())
        candidates = collect_candidates(
            samples,
            generated_phases=(GROUNDED_INPUT_GENERATED_PHASES if fix_inputs else None),
        ) if fix_inputs else collect_candidates(samples)
    else:
        candidates = _load_json(args.reuse_candidates)
    _save_json(output_dir / "candidate_fields.json", candidates)
    if args.joint_finalize:
        if not fix_inputs:
            raise ValueError("--joint-finalize currently requires --grounded-input-state")
        final_schema, selection_audit = select_joint_schema_from_raw_candidates(
            policy_context=stage1.policy_context,
            fixed_schema=fixed_schema,
            candidates=candidates,
            config=config,
        )
        _save_json(output_dir / "selection_audit.json", selection_audit)
        _save_json(output_dir / "inner_schema_compact.json", final_schema.model_dump())
        _save_json(output_dir / "inner_schema_ensemble.json", final_schema.model_dump())
        _save_json(
            output_dir / "run_manifest.json",
            {
                "config": str(args.config.resolve()),
                "stage1_result": str(args.stage1_result.resolve()),
                "fixed_schema": str(args.fixed_schema.resolve()),
                "grounded_input_state": str(args.grounded_input_state.resolve()),
                "runs": args.runs,
                "generation_calls": args.runs,
                "joint_finalization_calls": 1,
                "expected_total_llm_calls": args.runs + 1,
                "inputs_fixed": True,
                "outcomes_fixed": True,
                "selection": "joint_raw_candidate_selection",
                "artifacts": {
                    "samples": [f"run_{index:02d}.json" for index in range(1, args.runs + 1)],
                    "candidates": "candidate_fields.json",
                    "selection_audit": "selection_audit.json",
                    "final_schema": "inner_schema_compact.json",
                },
            },
        )
        print(f"Saved 5+1 joint ensemble schema artifacts to {output_dir}")
        return
    consolidation = consolidate_candidates(
        candidates,
        config,
        mode=args.dedupe_mode,
        generated_phases=(GROUNDED_INPUT_GENERATED_PHASES if fix_inputs else tuple(candidates)),
    )
    _save_json(output_dir / "dedupe_audit.json", consolidation)

    final_schema = build_ensemble_schema(fixed_schema, consolidation, fix_inputs=fix_inputs)
    _save_json(output_dir / "inner_schema_ensemble.json", final_schema.model_dump())
    _save_json(
        output_dir / "run_manifest.json",
        {
            "config": str(args.config.resolve()),
            "stage1_result": str(args.stage1_result.resolve()),
            "fixed_schema": str(args.fixed_schema.resolve()),
            "runs": args.runs,
            "dedupe_mode": args.dedupe_mode,
            "reused_candidates": (
                str(args.reuse_candidates.resolve()) if args.reuse_candidates else None
            ),
            "outcomes_fixed": True,
            "inputs_fixed": fix_inputs,
            "grounded_input_state": (
                str(args.grounded_input_state.resolve()) if args.grounded_input_state else None
            ),
            "artifacts": {
                "samples": (
                    [f"run_{index:02d}.json" for index in range(1, args.runs + 1)]
                    if args.reuse_candidates is None
                    else []
                ),
                "candidates": "candidate_fields.json",
                "dedupe_audit": "dedupe_audit.json",
                "final_schema": "inner_schema_ensemble.json",
            },
        },
    )
    print(f"Saved ensemble schema artifacts to {output_dir}")


if __name__ == "__main__":
    main()
