#!/usr/bin/env python3
"""Create a policy-scoped KG while preserving the original graph and an audit trail."""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _matches(value: Any, patterns: list[re.Pattern[str]]) -> bool:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return any(pattern.search(text) for pattern in patterns)


def _exclusive_to_excluded_scope(
    value: Any,
    excluded_patterns: list[re.Pattern[str]],
    target_patterns: list[re.Pattern[str]],
) -> bool:
    return _matches(value, excluded_patterns) and not _matches(value, target_patterns)


def _filter_summary(summary: str, excluded_patterns: list[re.Pattern[str]]) -> str:
    if not summary:
        return summary
    sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
    kept = [sentence for sentence in sentences if not _matches(sentence, excluded_patterns)]
    return " ".join(kept).strip()


def scope_graph(graph: dict[str, Any], scope: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    scoped = deepcopy(graph)
    excluded_node_ids = set(scope.get("excluded_node_ids", []))
    excluded_patterns = [re.compile(value, re.IGNORECASE) for value in scope["excluded_text_patterns"]]
    target_patterns = [re.compile(value, re.IGNORECASE) for value in scope["target_text_patterns"]]

    original_nodes = scoped.get("nodes", [])
    original_edges = scoped.get("edges", [])
    removed_nodes = [node for node in original_nodes if node.get("id") in excluded_node_ids]
    kept_nodes: list[dict[str, Any]] = []
    cleaned_source_spans: list[dict[str, Any]] = []
    cleaned_summaries: list[dict[str, str]] = []

    for node in original_nodes:
        if node.get("id") in excluded_node_ids:
            continue

        node = deepcopy(node)
        original_spans = node.get("source_spans", [])
        kept_spans = [
            span
            for span in original_spans
            if not _exclusive_to_excluded_scope(span, excluded_patterns, target_patterns)
        ]
        if len(kept_spans) != len(original_spans):
            cleaned_source_spans.append(
                {
                    "node_id": node.get("id"),
                    "removed_count": len(original_spans) - len(kept_spans),
                }
            )
            node["source_spans"] = kept_spans

        old_summary = node.get("summary", "")
        new_summary = _filter_summary(old_summary, excluded_patterns)
        if new_summary != old_summary:
            node["summary"] = new_summary
            cleaned_summaries.append({"node_id": node.get("id"), "removed_scope": scope["excluded_scope"]})
        kept_nodes.append(node)

    removed_edges: list[dict[str, Any]] = []
    kept_edges: list[dict[str, Any]] = []
    for edge in original_edges:
        incident_to_removed_node = edge.get("source") in excluded_node_ids or edge.get("target") in excluded_node_ids
        excluded_evidence = _exclusive_to_excluded_scope(
            {"source_text": edge.get("source_text"), "fact": edge.get("fact")},
            excluded_patterns,
            target_patterns,
        )
        if incident_to_removed_node or excluded_evidence:
            removed_edges.append(edge)
        else:
            kept_edges.append(edge)

    original_warnings = scoped.get("validation_warnings", [])
    kept_warnings = [warning for warning in original_warnings if not _matches(warning, excluded_patterns)]

    scoped["graph_id"] = f'{graph.get("graph_id", "policy_graph")}__{scope["scope_id"]}'
    scoped["nodes"] = kept_nodes
    scoped["edges"] = kept_edges
    scoped["validation_warnings"] = kept_warnings
    scoped.setdefault("metadata", {})["policy_scope"] = {
        "scope_id": scope["scope_id"],
        "target_policy": scope["target_policy"],
        "excluded_scope": scope["excluded_scope"],
        "method": "deterministic node and evidence filtering",
        "source_graph_id": graph.get("graph_id"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    kept_node_ids = {node.get("id") for node in kept_nodes}
    dangling_edges = [
        edge.get("id")
        for edge in kept_edges
        if edge.get("source") not in kept_node_ids or edge.get("target") not in kept_node_ids
    ]
    if dangling_edges:
        raise ValueError(f"Scoped graph contains dangling edges: {dangling_edges[:5]}")

    audit = {
        "scope_id": scope["scope_id"],
        "target_policy": scope["target_policy"],
        "excluded_scope": scope["excluded_scope"],
        "source_graph_id": graph.get("graph_id"),
        "counts": {
            "nodes_before": len(original_nodes),
            "nodes_after": len(kept_nodes),
            "edges_before": len(original_edges),
            "edges_after": len(kept_edges),
            "warnings_before": len(original_warnings),
            "warnings_after": len(kept_warnings),
        },
        "removed_nodes": [
            {"id": node.get("id"), "name": node.get("name"), "type": node.get("type")}
            for node in removed_nodes
        ],
        "removed_edge_ids": [edge.get("id") for edge in removed_edges],
        "cleaned_source_spans": cleaned_source_spans,
        "cleaned_summaries": cleaned_summaries,
    }
    return scoped, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    graph = _load_json(args.graph)
    scope = _load_json(args.scope)
    scoped, audit = scope_graph(graph, scope)
    _dump_json(scoped, args.out)
    _dump_json(audit, args.out.with_name("policy_scope_audit.json"))

    counts = audit["counts"]
    print(
        f"Scoped graph written to {args.out} "
        f"(nodes {counts['nodes_before']} -> {counts['nodes_after']}, "
        f"edges {counts['edges_before']} -> {counts['edges_after']})"
    )


if __name__ == "__main__":
    main()
