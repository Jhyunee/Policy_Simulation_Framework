"""MD-driven Nemotron sampler for Condition (1) Baseline.

No scope_definer 2-stage call (that requires KG context). Per stakeholder_type:
1. LLM infers (age_range, education_keywords, occupation_keywords) hints.
2. DuckDB SELECT against the country's Nemotron parquet with PERSONA_FIELDS
   projection only — output rows already contain ONLY country-specific
   PERSONA_FIELDS (plus tier metadata column we drop).
"""
from __future__ import annotations

import json
import random
from typing import Any

from llm.client import chat_completion_with_usage
from ablation.baseline_md.prompts import (
    STAKEHOLDER_TO_DEMOGRAPHIC_HINTS_SYSTEM,
    STAKEHOLDER_TO_DEMOGRAPHIC_HINTS_USER,
)


def infer_demographic_hints(
    stakeholder_type: str,
    representative_type: str,
    core_objectives: list[str],
    core_constraints: list[str],
    policy_summary: dict,
    config: dict,
) -> tuple[dict, dict]:
    """LLM call → (hints dict, usage). Hints: age_range, education_level_keywords, occupation_keywords."""
    messages = [
        {"role": "system", "content": STAKEHOLDER_TO_DEMOGRAPHIC_HINTS_SYSTEM},
        {
            "role": "user",
            "content": STAKEHOLDER_TO_DEMOGRAPHIC_HINTS_USER.format(
                policy_summary=json.dumps(policy_summary, ensure_ascii=False, indent=2),
                stakeholder_type=stakeholder_type,
                representative_type=representative_type,
                core_objectives=json.dumps(core_objectives, ensure_ascii=False),
                core_constraints=json.dumps(core_constraints, ensure_ascii=False),
            ),
        },
    ]
    content, usage = chat_completion_with_usage(messages, config)
    hints = json.loads(content)
    # Normalize fields to defensive defaults if LLM omits any.
    hints.setdefault("age_range", [20, 65])
    hints.setdefault("education_level_keywords", [])
    hints.setdefault("occupation_keywords", [])
    return hints, usage


def _quote_sql_value(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def _ilike_or(column: str, keywords: list[str]) -> str:
    """SQL OR-of-ILIKE clause for a column. Returns 'TRUE' if no keywords (matches all)."""
    if not keywords:
        return "TRUE"
    parts = [f"{column} ILIKE '%' || {_quote_sql_value(k)} || '%'" for k in keywords]
    return "(" + " OR ".join(parts) + ")"


def sample_personas_for_stakeholder(
    hints: dict,
    parquet_glob: str,
    sql_select_exprs: list[str],
    n_target: int,
    seed: int,
) -> list[dict]:
    """Run DuckDB query with PERSONA_FIELDS projection + hint-driven WHERE.

    Returns N persona dicts (each containing ONLY PERSONA_FIELDS — no extras).
    Falls back to wider scope (drop education filter) if pool < n_target, then to
    age-only if still empty.
    """
    import duckdb

    age_lo, age_hi = int(hints["age_range"][0]), int(hints["age_range"][1])
    edu_clause = _ilike_or("education_level", hints.get("education_level_keywords") or [])
    occ_clause = _ilike_or("occupation", hints.get("occupation_keywords") or [])
    select_clause = ", ".join(sql_select_exprs)

    def _query(where: str, limit: int) -> list[dict]:
        sql = f"""
        SELECT {select_clause}
        FROM read_parquet('{parquet_glob}')
        WHERE {where}
        USING SAMPLE {limit}
        """
        try:
            df = duckdb.sql(sql).fetchdf()
        except Exception:
            return []
        return df.to_dict(orient="records")

    age_where = f"age BETWEEN {age_lo} AND {age_hi}"
    pool = _query(f"{age_where} AND {edu_clause} AND {occ_clause}", n_target * 3)
    if len(pool) < n_target:
        # Widen: drop education filter
        pool = _query(f"{age_where} AND {occ_clause}", n_target * 3)
    if len(pool) < n_target:
        # Widen further: drop occupation filter (age + edu only)
        pool = _query(f"{age_where} AND {edu_clause}", n_target * 3)
    if len(pool) < n_target:
        # Last resort: age only
        pool = _query(f"{age_where}", n_target * 3)

    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n_target]


def sample_pool(
    stakeholder_groups: list[dict],
    policy_summary: dict,
    parquet_glob: str,
    sql_select_exprs: list[str],
    n_per_stakeholder: int,
    config: dict,
    base_seed: int = 42,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Top-level: for each stakeholder_type, infer hints + sample n personas.

    Returns:
        pool_by_type: {stakeholder_type: [persona_dict, ...]} — each persona only has PERSONA_FIELDS.
        usage_log: list of {stakeholder_type, hints, usage} for the hints calls.
    """
    pool_by_type: dict[str, list[dict]] = {}
    usage_log: list[dict] = []

    for s_idx, group in enumerate(stakeholder_groups):
        stype = group["stakeholder_type"]
        # Use the FIRST persona's narrative for hint inference (PERSONAS_FROM_TEXT may emit several).
        first = (group.get("personas") or [{}])[0]
        hints, usage = infer_demographic_hints(
            stakeholder_type=stype,
            representative_type=first.get("representative_type") or stype,
            core_objectives=first.get("core_objectives") or [],
            core_constraints=first.get("core_constraints") or [],
            policy_summary=policy_summary,
            config=config,
        )
        rows = sample_personas_for_stakeholder(
            hints=hints,
            parquet_glob=parquet_glob,
            sql_select_exprs=sql_select_exprs,
            n_target=n_per_stakeholder,
            seed=base_seed + s_idx,
        )
        pool_by_type[stype] = rows
        usage_log.append({
            "stakeholder_type": stype,
            "hints": hints,
            "n_sampled": len(rows),
            "usage": usage,
        })

    return pool_by_type, usage_log
