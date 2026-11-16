"""Replace a generated ToC Inputs phase with a validated grounded state."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.schemas import GroundedInputState, ToCInnerSchema
from stage2_toc.grounded_inputs import apply_grounded_inputs_to_schema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--grounded-input-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    schema = ToCInnerSchema.model_validate(
        json.loads(args.schema.read_text(encoding="utf-8"))
    )
    state = GroundedInputState.model_validate(
        json.loads(args.grounded_input_state.read_text(encoding="utf-8"))
    )
    finalized = apply_grounded_inputs_to_schema(schema, state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(finalized.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved grounded schema to {args.output}")


if __name__ == "__main__":
    main()
