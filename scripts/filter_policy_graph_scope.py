"""Create a scoped policy graph while preserving the source KG unchanged."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _text(value: object) -> str:
    if isinstance(value, dict):
        return " ".join(_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_text(item) for item in value)
    return str(value or "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-node-pattern", action="append", default=[])
    parser.add_argument("--exclude-edge-pattern", action="append", default=[])
    parser.add_argument("--scope-label", required=True)
    args = parser.parse_args()

    graph = json.loads(args.input.read_text(encoding="utf-8"))
    node_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in args.exclude_node_pattern]
    edge_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in args.exclude_edge_pattern]

    def excluded(value: object, patterns: list[re.Pattern[str]]) -> bool:
        content = _text(value)
        return any(pattern.search(content) for pattern in patterns)

    def node_identity(node: dict) -> dict:
        # Scope membership is determined by entity identity. Evidence spans can
        # mention an excluded policy extension while still grounding a core actor.
        return {key: node.get(key) for key in ("id", "name", "type")}

    removed_nodes = [
        node for node in graph.get("nodes", [])
        if excluded(node_identity(node), node_patterns)
    ]
    removed_ids = {str(node.get("id")) for node in removed_nodes}
    kept_nodes = [node for node in graph.get("nodes", []) if str(node.get("id")) not in removed_ids]

    removed_edges = []
    kept_edges = []
    for edge in graph.get("edges", []):
        endpoint_removed = str(edge.get("source")) in removed_ids or str(edge.get("target")) in removed_ids
        if endpoint_removed or excluded(edge, edge_patterns):
            removed_edges.append(edge)
        else:
            kept_edges.append(edge)

    graph["nodes"] = kept_nodes
    graph["edges"] = kept_edges
    metadata = graph.setdefault("metadata", {})
    metadata["scope_filter"] = {
        "label": args.scope_label,
        "source_graph": str(args.input),
        "exclude_node_patterns": args.exclude_node_pattern,
        "exclude_edge_patterns": args.exclude_edge_pattern,
        "removed_node_count": len(removed_nodes),
        "removed_edge_count": len(removed_edges),
        "retained_node_count": len(kept_nodes),
        "retained_edge_count": len(kept_edges),
        "removed_node_ids": sorted(removed_ids),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"nodes={len(kept_nodes)} (-{len(removed_nodes)}) "
        f"edges={len(kept_edges)} (-{len(removed_edges)})"
    )


if __name__ == "__main__":
    main()
