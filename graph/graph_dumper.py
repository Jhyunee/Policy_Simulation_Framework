"""
Dumps all nodes and edges from a Zep graph to a JSON file for offline inspection.
"""

import os
from typing import Any

from graph.entity_reader import EntityReader


def dump_graph(graph_id: str, config: dict) -> dict:
    """
    Fetch all nodes and edges from a Zep graph.

    Returns a dict with keys:
        - graph_id: str
        - nodes: list of {uuid, name, labels, summary, attributes}
        - edges: list of {uuid, name, fact, source_node_uuid, target_node_uuid}
    """
    zep_api_key = os.environ.get(config["zep"]["api_key_env"])
    if not zep_api_key:
        raise ValueError(
            f"Zep API key not found. Set the '{config['zep']['api_key_env']}' environment variable."
        )

    reader = EntityReader(zep_api_key)

    print(f"Fetching all nodes from graph {graph_id}...")
    nodes = reader._fetch_all_nodes(graph_id)
    print(f"  {len(nodes)} nodes fetched.")

    print("Fetching all edges...")
    edges = reader._fetch_all_edges(graph_id)
    print(f"  {len(edges)} edges fetched.")

    return {
        "graph_id": graph_id,
        "nodes": nodes,
        "edges": edges,
    }
