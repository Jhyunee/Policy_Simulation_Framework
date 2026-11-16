"""Build and inspect a mechanism-panel role blueprint without LLM calls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage1.mechanism_panel import build_role_prototypes  # noqa: E402


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-graph", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prototypes, overlay = build_role_prototypes(args.policy_graph, config)
    write_json(args.output_dir / "role_prototypes.json", [p.model_dump() for p in prototypes])
    write_json(args.output_dir / "actor_classification_overlay.json", overlay)
    print(f"roles={len(prototypes)} classified_entities={len(overlay)}")
    for prototype in prototypes:
        print(f"  {prototype.prototype_id}: {prototype.anchor_entity_id}")


if __name__ == "__main__":
    main()
