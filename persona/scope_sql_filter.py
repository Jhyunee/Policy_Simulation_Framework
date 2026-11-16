"""DuckDB-backed tiered SQL filter for PersonaScopeDefiner output.

Given a Definer JSON object with `direct / plausible / weak` tiers (each carrying
`occupation`, `education_level`, `age_range`), this module:

1. Computes cumulative value sets per tier in Python (Option A semantics: each
   tier's effective filter = union of all tiers up to and including it; an empty
   list at a tier means "inherit previous cumulative").
2. Issues ONE DuckDB CASE WHEN query against the local Nemotron parquet shards.
3. Returns a pandas DataFrame containing only the projected persona fields,
   tagged with the most-specific tier each row matches.

The reranker is intentionally not used: tier label IS the rank.
"""

from __future__ import annotations

import importlib
import os
from typing import Any


def _load_country_fields() -> tuple[list[str], list[str]]:
    """Load per-country ``PERSONA_FIELDS`` (DuckDB SELECT expressions) from the
    Nemotron values module pointed to by ``NEMOTRON_VALUES_MODULE``.

    Returns ``(sql_exprs, output_names)`` where ``sql_exprs`` go straight into
    the SELECT projection (may contain ``... AS target`` aliases) and
    ``output_names`` are the resulting column names downstream code reads.
    Falls back to Korea defaults if the module is missing.
    """
    module_path = os.environ.get("NEMOTRON_VALUES_MODULE", "dataset.processed.nemotron_kor")
    fields_mod = importlib.import_module(f"{module_path}.fields")
    sql_exprs: list[str] = list(fields_mod.PERSONA_FIELDS)
    output_names = [expr.split(" AS ")[-1].strip() for expr in sql_exprs]
    return sql_exprs, output_names


PERSONA_FIELDS_SQL, PERSONA_OUTPUT_FIELDS = _load_country_fields()

# Always-projected traceability columns (carried from parquet alongside the persona fields).
PARQUET_TRACEABILITY_FIELDS: list[str] = ["uuid"]


def _coalesce_axis(tier: dict[str, Any], axis: str, fallback: Any) -> Any:
    """Return the tier's value for `axis`, or the cumulative fallback if empty/None."""
    value = tier.get(axis)
    if value in (None, "", []):
        return fallback
    return value


def _cumulative_value_sets(definer_output: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build the cumulative IN-list / age range per tier.

    Returns:
        {
            "direct":    {"occupation": [...], "education_level": [...], "age_range": [lo, hi]},
            "plausible": {...},
            "weak":      {...},
        }

    Empty axis at a tier inherits the previous cumulative value (Option A).
    """
    result: dict[str, dict[str, Any]] = {}
    cum_occ: list[str] = []
    cum_edu: list[str] = []
    cum_age: list[int] | None = None

    for tier in ("direct", "plausible", "weak"):
        block = definer_output.get(tier) or {}
        new_occ = block.get("occupation") or []
        new_edu = block.get("education_level") or []
        new_age = block.get("age_range")

        # Cumulative occupation: previous union new (de-duplicated, order-stable).
        cum_occ = _dedupe(cum_occ + list(new_occ))
        cum_edu = _dedupe(cum_edu + list(new_edu))

        if isinstance(new_age, (list, tuple)) and len(new_age) == 2:
            lo, hi = int(new_age[0]), int(new_age[1])
            if cum_age is None:
                cum_age = [lo, hi]
            else:
                # Tier widens monotonically: take the union of the ranges.
                cum_age = [min(cum_age[0], lo), max(cum_age[1], hi)]
        elif cum_age is None:
            # No range emitted at any tier yet; default to a permissive range.
            cum_age = [0, 200]

        result[tier] = {
            "occupation": list(cum_occ),
            "education_level": list(cum_edu),
            "age_range": list(cum_age),
        }
    return result


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        key = str(v)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _format_in_list(values: list[str]) -> str:
    """SQL-quote each value for an IN (...) clause. Single quotes are doubled."""
    if not values:
        # Empty IN-list would yield 0 rows; use a sentinel that matches nothing.
        return "''"
    quoted = ["'" + str(v).replace("'", "''") + "'" for v in values]
    return ", ".join(quoted)


def build_tiered_query(
    cumulative: dict[str, dict[str, Any]],
    parquet_glob: str,
) -> str:
    """Construct the DuckDB CASE WHEN query that tags each parquet row with its tier."""
    direct = cumulative["direct"]
    plausible = cumulative["plausible"]
    weak = cumulative["weak"]

    select_fields = ", ".join(PARQUET_TRACEABILITY_FIELDS + PERSONA_FIELDS_SQL)

    sql = f"""
    SELECT
        {select_fields},
        CASE
            WHEN occupation IN ({_format_in_list(direct["occupation"])})
             AND education_level IN ({_format_in_list(direct["education_level"])})
             AND age BETWEEN {direct["age_range"][0]} AND {direct["age_range"][1]}
                THEN 'direct'
            WHEN occupation IN ({_format_in_list(plausible["occupation"])})
             AND education_level IN ({_format_in_list(plausible["education_level"])})
             AND age BETWEEN {plausible["age_range"][0]} AND {plausible["age_range"][1]}
                THEN 'plausible'
            WHEN occupation IN ({_format_in_list(weak["occupation"])})
             AND education_level IN ({_format_in_list(weak["education_level"])})
             AND age BETWEEN {weak["age_range"][0]} AND {weak["age_range"][1]}
                THEN 'weak'
        END AS tier
    FROM read_parquet('{parquet_glob}')
    """
    return sql


def filter_personas(definer_output: dict[str, Any], parquet_glob: str):
    """Run the tiered SQL filter and return a pandas DataFrame of tier-tagged rows.

    Rows that match no tier are filtered out (they fall outside even the weak
    cumulative scope).
    """
    import duckdb  # imported lazily so the rest of the module can be inspected without duckdb installed

    cumulative = _cumulative_value_sets(definer_output)
    sql = build_tiered_query(cumulative, parquet_glob)
    df = duckdb.sql(sql).fetchdf()
    df = df[df["tier"].notna()].reset_index(drop=True)
    return df
