"""Normalize a scenario bundle to the fixed Stage 2 simulation-persona schema."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona.simulation_schema import sanitize_simulation_persona


def _atomic_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def _repair_legacy_persona_id(record: dict[str, Any]) -> None:
    persona_id = str(record.get("persona_id") or "")
    if not persona_id.endswith("__XXXXXXXX"):
        return
    match = re.search(r"\bKG-Gen-([0-9A-Za-z]+)", str(record.get("representative_type") or ""))
    if match:
        record["persona_id"] = persona_id.removesuffix("XXXXXXXX") + match.group(1)


def normalize_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(bundle)
    scenarios = []
    for scenario in bundle.get("scenarios") or []:
        compact_scenario = dict(scenario)
        participants = []
        for source in scenario.get("participants") or []:
            record = dict(source)
            _repair_legacy_persona_id(record)
            participants.append(sanitize_simulation_persona(record))
        compact_scenario["participants"] = participants
        scenarios.append(compact_scenario)
    normalized["scenarios"] = scenarios
    normalized["schema_version"] = "simulation_persona_v1"
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = json.loads(args.input.read_text(encoding="utf-8"))
    normalized = normalize_bundle(bundle)
    _atomic_write(args.output, normalized)
    count = sum(len(item.get("participants") or []) for item in normalized.get("scenarios") or [])
    print(f"Wrote {args.output} ({count} participants)")


if __name__ == "__main__":
    main()
