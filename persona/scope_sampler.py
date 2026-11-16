"""Per-entity orchestrator: KG context -> Definer -> SQL filter -> tiered widening -> diversity-aware pick.

Returns a list of seed dicts shaped like the existing Nemotron sampler output so
that `stage1.modules._merge_nemotron_seed` can consume them unchanged.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from persona.scope_definer import (
    PersonaScopeDefiner,
    build_kg_context,
    save_kg_context,
)
from persona.scope_sql_filter import (
    PERSONA_FIELDS_SQL,
    PERSONA_OUTPUT_FIELDS,
    PARQUET_TRACEABILITY_FIELDS,
    filter_personas,
)


# Tier label -> numeric score, surfaced as `match_score` so the existing
# role_stratified_weighted_sample at simulation_participants selection time
# (pipeline.py:_role_stratified_weighted_sample) keeps working without
# special-casing.
TIER_SCORE: dict[str, int] = {
    "direct": 5,
    "plausible": 3,
    "weak": 2,
    "fallback": 1,
}


def sample_for_entity(
    entity: Any,
    entity_type: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run the full Definer + tiered SQL pipeline for one persona-eligible entity.

    Returns a list of seed dicts (Nemotron-shaped) ready for `_merge_nemotron_seed`.
    Returns `[]` for organization-modality entities so the caller falls back to
    the existing `institutional_actor_profile` path.
    """
    attributes = getattr(entity, "attributes", {}) or {}
    if attributes.get("agent_modality") == "organization":
        # Definer + SQL filtering doesn't apply (Nemotron is individuals only),
        # but we still build + save kg_context so Stage 2 can inject the org's
        # policy-implementer backdrop into round prompts.
        kg_context = build_kg_context(entity, entity_type)
        save_kg_context(kg_context, config.get("_current_run_dir"), entity, entity_type)
        return []

    scope_config = config.get("persona", {}).get("scope_definer", {}) or {}
    parquet_glob = scope_config.get(
        "parquet_glob",
        "dataset/processed/nemotron_kor/Nemotron-Personas-Korea/data/*.parquet",
    )
    min_pool = int(scope_config.get("min_pool_size", 5))
    target_pool = int(scope_config.get("target_pool_size", 20))
    fallback_enabled = bool((scope_config.get("fallback") or {}).get("enabled", True))
    seed = int(
        config.get("persona", {}).get("simulation_sample_seed")
        or scope_config.get("sample_seed")
        or 42
    )

    run_dir = config.get("_current_run_dir")

    # 1. Build & save KG context for inspection.
    kg_context = build_kg_context(entity, entity_type)
    save_kg_context(kg_context, run_dir, entity, entity_type)

    # 2. Definer LLM call (cached on disk per entity).
    definer = PersonaScopeDefiner(config)
    definer_output = definer.define(entity, entity_type, kg_context)
    if definer_output is None:
        return []  # caller routes to institutional path

    # 3. Tiered DuckDB SQL filter.
    df = filter_personas(definer_output, parquet_glob)

    # 4. Tiered widening.
    pool, tier_pool_sizes, used_fallback = _walk_tiers(df, min_pool, parquet_glob, fallback_enabled, seed)

    # 5. Diversity-aware random pick down to target_pool.
    diversity_penalty = config.get("persona", {}).get("diversity_penalty", {}) or {}
    picked = _diversity_aware_pick(pool, target_pool, diversity_penalty, seed)

    # 6. Build seed dicts in the shape `_merge_nemotron_seed` expects.
    return _project_seeds(
        picked,
        entity=entity,
        entity_type=entity_type,
        definer_output=definer_output,
        tier_pool_sizes=tier_pool_sizes,
        used_fallback=used_fallback,
    )


# --------------------------------------------------------------------- helpers

def _walk_tiers(
    df,
    min_pool: int,
    parquet_glob: str,
    fallback_enabled: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int], bool]:
    """Walk direct -> plausible -> weak; backfill from a random fallback if needed."""
    counts = df["tier"].value_counts().to_dict() if len(df) else {}
    tier_pool_sizes = {
        "direct": int(counts.get("direct", 0)),
        "plausible": int(counts.get("plausible", 0)),
        "weak": int(counts.get("weak", 0)),
    }

    direct_rows = df[df["tier"] == "direct"].to_dict(orient="records")
    if len(direct_rows) >= min_pool:
        return direct_rows, tier_pool_sizes, False

    plausible_rows = df[df["tier"].isin(["direct", "plausible"])].to_dict(orient="records")
    if len(plausible_rows) >= min_pool:
        return plausible_rows, tier_pool_sizes, False

    all_rows = df.to_dict(orient="records")
    if len(all_rows) >= min_pool or not fallback_enabled:
        return all_rows, tier_pool_sizes, False

    # Fallback: random sample from full Nemotron, tagged as 'fallback'.
    needed = min_pool - len(all_rows)
    fallback_rows = _random_fallback_rows(parquet_glob, needed, seed)
    return all_rows + fallback_rows, tier_pool_sizes, True


def _random_fallback_rows(parquet_glob: str, n: int, seed: int) -> list[dict[str, Any]]:
    """Pull `n` random rows from the parquet shards, tagged with tier='fallback'."""
    if n <= 0:
        return []
    import duckdb

    cols = ", ".join(PARQUET_TRACEABILITY_FIELDS + PERSONA_FIELDS_SQL)
    sql = f"""
        SELECT {cols}, 'fallback' AS tier
        FROM read_parquet('{parquet_glob}')
        USING SAMPLE {n} ROWS (reservoir, {seed})
    """
    return duckdb.sql(sql).fetchdf().to_dict(orient="records")


def _diversity_aware_pick(
    pool: list[dict[str, Any]],
    target: int,
    diversity_penalty: dict[str, float],
    seed: int,
) -> list[dict[str, Any]]:
    """Greedy pick that prefers diversity along configured columns.

    Mirrors `nemotron_sampler._diversified_ranked_candidates` (line 769-795) but
    without an LLM-derived role_fit_score: each row starts with a uniform random
    base and is penalized for repeating values already picked along the
    diversity axes (occupation, province, sex by default).
    """
    if len(pool) <= target:
        return list(pool)

    rng = random.Random(seed)
    remaining = list(pool)
    # Precompute a stable per-row base score so order doesn't shift across loops.
    bases = {id(row): rng.random() for row in remaining}

    selected: list[dict[str, Any]] = []
    usage: dict[str, dict[Any, int]] = {key: {} for key in diversity_penalty}

    while remaining and len(selected) < target:
        best_index = 0
        best_score = float("-inf")
        for index, row in enumerate(remaining):
            score = bases[id(row)]
            for axis, weight in diversity_penalty.items():
                value = row.get(axis)
                if value in (None, ""):
                    continue
                score -= float(weight) * usage.get(axis, {}).get(value, 0)
            if score > best_score:
                best_score = score
                best_index = index
        picked = remaining.pop(best_index)
        selected.append(picked)
        for axis in usage:
            value = picked.get(axis)
            if value not in (None, ""):
                usage[axis][value] = usage[axis].get(value, 0) + 1
    return selected


def _project_seeds(
    picked: list[dict[str, Any]],
    entity: Any,
    entity_type: str,
    definer_output: dict[str, Any],
    tier_pool_sizes: dict[str, int],
    used_fallback: bool,
) -> list[dict[str, Any]]:
    """Shape each picked row into a seed dict compatible with `_merge_nemotron_seed`."""
    seeds: list[dict[str, Any]] = []
    rationale = (definer_output or {}).get("rationale", "")
    entity_id = getattr(entity, "id", None) or entity_type
    for row in picked:
        tier_used = row.get("tier") or "fallback"
        seed = {field: row.get(field) for field in PERSONA_OUTPUT_FIELDS}
        seed["uuid"] = row.get("uuid")
        # Map tier to the existing match_score scale (5/3/2/1) so downstream
        # role_stratified_weighted_sample weighting still applies.
        seed["match_score"] = TIER_SCORE.get(tier_used, 0)
        seed["match_strategy"] = "scope_definer_tier"
        seed["tier_used"] = tier_used
        seed["tier_pool_sizes"] = tier_pool_sizes
        seed["used_fallback"] = used_fallback
        seed["scope_definer_rationale"] = rationale
        seed["entity_id"] = entity_id
        seed["entity_type"] = entity_type
        seeds.append(seed)
    return seeds
