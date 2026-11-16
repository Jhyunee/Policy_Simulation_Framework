"""
Synchronous Zep graph builder for IITP_Mirofish_v1.
Creates a knowledge graph from policy document text using a pre-generated ontology.
"""

import os
import time
import uuid
import warnings
from typing import Optional

from pathlib import Path
from dotenv import load_dotenv
from zep_cloud.client import Zep
from zep_cloud import EpisodeData, EntityEdgeSourceTarget
from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel
from pydantic import Field

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Zep reserved attribute names that cannot be used as field names
RESERVED_NAMES = {"uuid", "name", "group_id", "name_embedding", "summary", "created_at"}


def build_graph(
    text: str,
    ontology: dict,
    config: dict,
    graph_name: str = "IITP_Mirofish Graph",
) -> str:
    """
    Build a Zep knowledge graph from policy text.

    Pipeline:
      1. Create graph in Zep
      2. Set ontology (entity + edge types)
      3. Split text into overlapping chunks
      4. Add chunks in batches
      5. Wait for Zep processing to complete
      6. Return graph_id

    Args:
        text: Full policy document text.
        ontology: Ontology dict from generate_ontology().
        config: Pipeline config dict.
        graph_name: Human-readable name for the graph.

    Returns:
        graph_id string.
    """
    api_key = os.environ.get(config["zep"]["api_key_env"])
    if not api_key:
        raise ValueError(
            f"Zep API key not found. Set the '{config['zep']['api_key_env']}' environment variable."
        )

    client = Zep(api_key=api_key)

    chunk_size = config["zep"]["chunk_size"]
    chunk_overlap = config["zep"]["chunk_overlap"]
    batch_size = config["zep"]["batch_size"]
    timeout = config["zep"]["wait_timeout"]

    print("Creating Zep graph...")
    graph_id = _create_graph(client, graph_name)
    print(f"  Graph created: {graph_id}")

    print("Setting ontology...")
    _set_ontology(client, graph_id, ontology)
    print("  Ontology set.")

    print("Splitting text into chunks...")
    chunks = _split_text(text, chunk_size, chunk_overlap)
    print(f"  {len(chunks)} chunks created.")

    print("Adding text batches to graph...")
    episode_uuids = _add_text_batches(client, graph_id, chunks, batch_size)
    print(f"  {len(episode_uuids)} episodes submitted.")

    print("Waiting for Zep to process episodes...")
    _wait_for_processing(client, episode_uuids, timeout)
    print("  Processing complete.")

    return graph_id


def _create_graph(client: Zep, name: str) -> str:
    """Create a new Zep graph and return its graph_id."""
    graph_id = f"iitp_mirofish_{uuid.uuid4().hex[:16]}"
    client.graph.create(
        graph_id=graph_id,
        name=name,
        description="IITP Mirofish policy simulation knowledge graph",
    )
    return graph_id


def _safe_attr_name(attr_name: str) -> str:
    """Rename reserved attribute names to avoid Zep API conflicts."""
    if attr_name.lower() in RESERVED_NAMES:
        return f"entity_{attr_name}"
    return attr_name


def _set_ontology(client: Zep, graph_id: str, ontology: dict) -> None:
    """Build EntityModel/EdgeModel classes dynamically and register with Zep."""
    # Suppress pydantic Field(default=None) warnings from dynamic class creation
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    # Build entity type classes
    entity_types: dict = {}
    for entity_def in ontology.get("entity_types", []):
        name = entity_def["name"]
        description = entity_def.get("description", f"A {name} entity.")

        attrs: dict = {"__doc__": description}
        annotations: dict = {}

        for attr_def in entity_def.get("attributes", []):
            attr_name = _safe_attr_name(attr_def["name"])
            attr_desc = attr_def.get("description", attr_name)
            attrs[attr_name] = Field(description=attr_desc, default=None)
            annotations[attr_name] = Optional[EntityText]

        attrs["__annotations__"] = annotations
        entity_class = type(name, (EntityModel,), attrs)
        entity_class.__doc__ = description
        entity_types[name] = entity_class

    # Build edge type classes
    edge_definitions: dict = {}
    for edge_def in ontology.get("edge_types", []):
        name = edge_def["name"]
        description = edge_def.get("description", f"A {name} relationship.")

        attrs = {"__doc__": description}
        annotations = {}

        for attr_def in edge_def.get("attributes", []):
            attr_name = _safe_attr_name(attr_def["name"])
            attr_desc = attr_def.get("description", attr_name)
            attrs[attr_name] = Field(description=attr_desc, default=None)
            annotations[attr_name] = Optional[str]

        attrs["__annotations__"] = annotations
        class_name = "".join(word.capitalize() for word in name.split("_"))
        edge_class = type(class_name, (EdgeModel,), attrs)
        edge_class.__doc__ = description

        source_targets = []
        for st in edge_def.get("source_targets", []):
            source_targets.append(
                EntityEdgeSourceTarget(
                    source=st.get("source", "Entity"),
                    target=st.get("target", "Entity"),
                )
            )

        if source_targets:
            edge_definitions[name] = (edge_class, source_targets)

    if entity_types or edge_definitions:
        client.graph.set_ontology(
            graph_ids=[graph_id],
            entities=entity_types if entity_types else None,
            edges=edge_definitions if edge_definitions else None,
        )


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks using a sliding window."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    chunks = []
    step = chunk_size - chunk_overlap
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def _add_text_batches(
    client: Zep,
    graph_id: str,
    chunks: list[str],
    batch_size: int,
) -> list[str]:
    """Send chunks to Zep in batches. Returns list of episode UUIDs."""
    episode_uuids = []
    total_chunks = len(chunks)
    total_batches = (total_chunks + batch_size - 1) // batch_size

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Sending batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...")

        episodes = [EpisodeData(data=chunk, type="text") for chunk in batch_chunks]

        try:
            batch_result = client.graph.add_batch(graph_id=graph_id, episodes=episodes)
            if batch_result and isinstance(batch_result, list):
                for ep in batch_result:
                    ep_uuid = getattr(ep, "uuid_", None) or getattr(ep, "uuid", None)
                    if ep_uuid:
                        episode_uuids.append(ep_uuid)
        except Exception as e:
            raise RuntimeError(f"Failed to add batch {batch_num}: {e}") from e

        # Rate-limit buffer between batches
        time.sleep(1)

    return episode_uuids


def _wait_for_processing(
    client: Zep,
    episode_uuids: list[str],
    timeout: int,
) -> None:
    """Poll until all submitted episodes are marked as processed, or timeout."""
    if not episode_uuids:
        print("  No episodes to wait for.")
        return

    start_time = time.time()
    pending = set(episode_uuids)
    total = len(episode_uuids)

    while pending:
        elapsed = int(time.time() - start_time)
        if elapsed > timeout:
            print(
                f"  Warning: timeout reached after {elapsed}s. "
                f"{total - len(pending)}/{total} episodes processed."
            )
            break

        for ep_uuid in list(pending):
            try:
                episode = client.graph.episode.get(uuid_=ep_uuid)
                if getattr(episode, "processed", False):
                    pending.discard(ep_uuid)
            except Exception:
                pass  # transient error — will retry next iteration

        if pending:
            completed = total - len(pending)
            print(
                f"  Waiting... {completed}/{total} processed, "
                f"{len(pending)} pending ({elapsed}s elapsed)"
            )
            time.sleep(5)

    completed = total - len(pending)
    print(f"  Done: {completed}/{total} episodes confirmed processed.")
