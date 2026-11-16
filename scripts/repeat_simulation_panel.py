"""Repeat one fixed simulation-persona panel as independent scenarios."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--label", default="repeat")
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    bundle = json.loads(args.input.read_text(encoding="utf-8"))
    scenarios = bundle.get("scenarios") or []
    if len(scenarios) != 1:
        raise ValueError(f"Expected one source scenario; got {len(scenarios)}")

    source = scenarios[0]
    repeated = []
    for index in range(1, args.repeats + 1):
        scenario = copy.deepcopy(source)
        scenario["scenario_id"] = f"{source['scenario_id']}__{args.label}_{index}"
        repeated.append(scenario)

    output = {
        **bundle,
        "scenario_count": args.repeats,
        "scenarios": repeated,
        "sampling_metadata": {
            **(bundle.get("sampling_metadata") or {}),
            "strategy": "fixed_panel_repeated_runs",
            "source_scenario_id": source.get("scenario_id"),
            "num_scenarios": args.repeats,
            "repeat_count": args.repeats,
            "persona_reuse_across_scenarios": True,
        },
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}: {args.repeats} repeats of {source.get('scenario_id')}")


if __name__ == "__main__":
    main()
