"""
Zep entity reader for IITP_Mirofish_v1.
Reads typed EntityNodes from a Zep graph and enriches them with edge/neighbor context.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from pathlib import Path
from dotenv import load_dotenv
from zep_cloud.client import Zep

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

_DEFAULT_PAGE_SIZE = 100
_MAX_NODES = 2000
_DEFAULT_MAX_RETRIES = 5
_DEFAULT_RETRY_DELAY = 15.0


@dataclass
class EntityNode:
    """A typed entity node extracted from the Zep knowledge graph."""
    uuid: str
    name: str
    labels: list[str]
    summary: str
    attributes: dict[str, Any]
    related_edges: list[dict[str, Any]] = field(default_factory=list)
    # Each entry: {fact, edge_name, direction}
    related_nodes: list[dict[str, Any]] = field(default_factory=list)
    # Each entry: {name, labels, summary}

    def get_entity_type(self) -> str | None:
        """Return the first label that is not 'Entity' or 'Node'."""
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


class EntityReader:
    """
    Reads and filters entity nodes from a Zep standalone graph.

    Usage:
        reader = EntityReader(api_key)
        entities = reader.get_filtered_entities(graph_id, allowed_types=["University", "GraduateStudent"])
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Zep API key must not be empty.")
        self.client = Zep(api_key=api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_filtered_entities(
        self,
        graph_id: str,
        allowed_types: list[str],
    ) -> list[EntityNode]:
        """
        Fetch all nodes from the graph, filter to those whose labels include
        at least one type in allowed_types, and enrich with edge/neighbor context.

        Args:
            graph_id: Zep graph ID.
            allowed_types: List of entity type names to keep (e.g. from ontology).

        Returns:
            List of EntityNode objects.
        """
        print(f"  Fetching all nodes from graph {graph_id}...")
        raw_nodes = self._fetch_all_nodes(graph_id)
        print(f"  {len(raw_nodes)} total nodes fetched.")

        # Build a lookup from UUID → raw node dict for neighbor resolution
        node_map: dict[str, dict] = {n["uuid"]: n for n in raw_nodes}

        # Fetch all edges once (more efficient than per-node API calls)
        print("  Fetching all edges...")
        raw_edges = self._fetch_all_edges(graph_id)
        print(f"  {len(raw_edges)} total edges fetched.")

        allowed_set = set(allowed_types)
        entities: list[EntityNode] = []

        for node in raw_nodes:
            labels = node.get("labels", [])
            custom_labels = [lb for lb in labels if lb not in ("Entity", "Node")]

            if not custom_labels:
                continue

            matching = [lb for lb in custom_labels if lb in allowed_set]
            if not matching:
                continue

            # Collect edges touching this node
            related_edges = []
            related_node_uuids: set[str] = set()

            for edge in raw_edges:
                if edge["source_node_uuid"] == node["uuid"]:
                    related_edges.append({
                        "direction": "outgoing",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                    })
                    related_node_uuids.add(edge["target_node_uuid"])
                elif edge["target_node_uuid"] == node["uuid"]:
                    related_edges.append({
                        "direction": "incoming",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                    })
                    related_node_uuids.add(edge["source_node_uuid"])

            # Resolve neighbor node summaries from the collected node map
            related_nodes = []
            for nb_uuid in related_node_uuids:
                nb = node_map.get(nb_uuid)
                if nb:
                    related_nodes.append({
                        "name": nb.get("name", ""),
                        "labels": nb.get("labels", []),
                        "summary": nb.get("summary", ""),
                    })

            entities.append(EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=labels,
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {}),
                related_edges=related_edges,
                related_nodes=related_nodes,
            ))

        print(f"  {len(entities)} entities matched allowed types: {allowed_types}")
        return entities

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_all_nodes(self, graph_id: str) -> list[dict]:
        """Paginate through all nodes in the graph."""
        all_nodes: list[dict] = []
        cursor: str | None = None
        page = 0

        while True:
            page += 1
            kwargs: dict[str, Any] = {"limit": _DEFAULT_PAGE_SIZE}
            if cursor is not None:
                kwargs["uuid_cursor"] = cursor

            batch = self._retry(
                lambda kw=kwargs: self.client.graph.node.get_by_graph_id(graph_id, **kw),
                description=f"fetch nodes page {page}",
            )
            if not batch:
                break

            for node in batch:
                all_nodes.append({
                    "uuid": getattr(node, "uuid_", None) or getattr(node, "uuid", ""),
                    "name": node.name or "",
                    "labels": node.labels or [],
                    "summary": node.summary or "",
                    "attributes": node.attributes or {},
                })

            if len(all_nodes) >= _MAX_NODES:
                all_nodes = all_nodes[:_MAX_NODES]
                print(f"    Warning: node limit ({_MAX_NODES}) reached, stopping pagination.")
                break

            if len(batch) < _DEFAULT_PAGE_SIZE:
                break

            last = batch[-1]
            cursor = getattr(last, "uuid_", None) or getattr(last, "uuid", None)
            if cursor is None:
                break

        return all_nodes

    def _fetch_all_edges(self, graph_id: str) -> list[dict]:
        """Paginate through all edges in the graph."""
        all_edges: list[dict] = []
        cursor: str | None = None
        page = 0

        while True:
            page += 1
            kwargs: dict[str, Any] = {"limit": _DEFAULT_PAGE_SIZE}
            if cursor is not None:
                kwargs["uuid_cursor"] = cursor

            batch = self._retry(
                lambda kw=kwargs: self.client.graph.edge.get_by_graph_id(graph_id, **kw),
                description=f"fetch edges page {page}",
            )
            if not batch:
                break

            for edge in batch:
                all_edges.append({
                    "uuid": getattr(edge, "uuid_", None) or getattr(edge, "uuid", ""),
                    "name": edge.name or "",
                    "fact": edge.fact or "",
                    "source_node_uuid": edge.source_node_uuid or "",
                    "target_node_uuid": edge.target_node_uuid or "",
                    "attributes": edge.attributes or {},
                })

            if len(batch) < _DEFAULT_PAGE_SIZE:
                break

            last = batch[-1]
            cursor = getattr(last, "uuid_", None) or getattr(last, "uuid", None)
            if cursor is None:
                break

        return all_edges

    def _retry(self, func, description: str = "Zep API call") -> Any:
        """Call func with exponential-backoff retry on transient errors."""
        delay = _DEFAULT_RETRY_DELAY
        last_exc: Exception | None = None

        for attempt in range(_DEFAULT_MAX_RETRIES):
            try:
                return func()
            except Exception as e:
                last_exc = e
                if attempt < _DEFAULT_MAX_RETRIES - 1:
                    print(
                        f"    Warning: {description} attempt {attempt + 1} failed: "
                        f"{str(e)[:80]}, retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    print(f"    Error: {description} failed after {_DEFAULT_MAX_RETRIES} attempts.")

        raise last_exc
