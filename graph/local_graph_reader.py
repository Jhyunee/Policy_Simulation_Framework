"""Reader for local policy_graph.json artifacts."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LocalGraphEntity:
    uuid: str
    name: str
    labels: list[str]
    summary: str
    attributes: dict[str, Any]
    related_edges: list[dict[str, Any]] = field(default_factory=list)
    related_nodes: list[dict[str, Any]] = field(default_factory=list)

    def get_entity_type(self) -> str | None:
        for label in self.labels:
            if label not in ("Entity", "Node"):
                return label
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }


class LocalGraphReader:
    """Reads local policy KG nodes and enriches them with edge/neighbor context."""

    def __init__(self, policy_graph: str | Path | dict[str, Any]):
        if isinstance(policy_graph, (str, Path)):
            with open(policy_graph, "r", encoding="utf-8") as f:
                self.graph = json.load(f)
        else:
            self.graph = policy_graph

    def get_filtered_entities(
        self,
        allowed_types: list[str] | None = None,
        persona_only: bool = False,
    ) -> list[LocalGraphEntity]:
        allowed_set = set(allowed_types or [])
        raw_nodes = self.graph.get("nodes", [])
        raw_edges = self.graph.get("edges", [])
        node_map = {node["id"]: node for node in raw_nodes}
        entities = []

        for node in raw_nodes:
            if allowed_set and node.get("type") not in allowed_set:
                continue
            if persona_only and not node.get("persona_eligible", False):
                continue

            related_edges = []
            related_node_ids: set[str] = set()
            for edge in raw_edges:
                if edge.get("source") == node["id"]:
                    related_edges.append(_edge_context(edge, "outgoing", node_map))
                    related_node_ids.add(edge.get("target"))
                elif edge.get("target") == node["id"]:
                    related_edges.append(_edge_context(edge, "incoming", node_map))
                    related_node_ids.add(edge.get("source"))

            related_nodes = []
            for related_id in related_node_ids:
                related = node_map.get(related_id)
                if not related:
                    continue
                related_nodes.append({
                    "name": related.get("name", ""),
                    "labels": ["Entity", related.get("type", "")],
                    "summary": related.get("summary", ""),
                    "iad_category": related.get("iad_category", ""),
                    "node_family": related.get("node_family", ""),
                    "iad_rule_type": related.get("iad_rule_type"),
                })

            attributes = dict(node.get("attributes", {}))
            attributes.update({
                "iad_category": node.get("iad_category"),
                "node_family": node.get("node_family"),
                "iad_rule_type": node.get("iad_rule_type"),
                "is_actor": node.get("is_actor"),
                "persona_eligible": node.get("persona_eligible"),
                "rationale": node.get("rationale", ""),
                "source_spans": node.get("source_spans", []),
            })

            entities.append(LocalGraphEntity(
                uuid=node["id"],
                name=node.get("name", ""),
                labels=["Entity", node.get("type", "")],
                summary=node.get("summary", ""),
                attributes=attributes,
                related_edges=related_edges,
                related_nodes=related_nodes,
            ))

        return entities


def _edge_context(
    edge: dict[str, Any],
    direction: str,
    node_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source = node_map.get(edge.get("source"), {})
    target = node_map.get(edge.get("target"), {})
    return {
        "direction": direction,
        "edge_name": edge.get("type", ""),
        "fact": edge.get("fact", ""),
        "edge_category": edge.get("edge_category", ""),
        "edge_family": edge.get("edge_family", ""),
        "deontic": edge.get("deontic"),
        "source": edge.get("source", ""),
        "source_name": source.get("name", ""),
        "source_type": source.get("type", ""),
        "source_iad_category": source.get("iad_category", ""),
        "source_node_family": source.get("node_family", ""),
        "source_iad_rule_type": source.get("iad_rule_type"),
        "target": edge.get("target", ""),
        "target_name": target.get("name", ""),
        "target_type": target.get("type", ""),
        "target_iad_category": target.get("iad_category", ""),
        "target_node_family": target.get("node_family", ""),
        "target_iad_rule_type": target.get("iad_rule_type"),
        "activation_condition": edge.get("activation_condition") or edge.get("condition_text"),
        "execution_constraint": edge.get("execution_constraint"),
        "condition_text": edge.get("condition_text"),
        "or_else_text": edge.get("or_else_text"),
        "source_text": edge.get("source_text", ""),
        "attributes": edge.get("attributes", {}),
    }
