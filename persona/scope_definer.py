"""PersonaScopeDefiner — two-stage demographic-anchored selection of tier scopes.

Stage 1 (`SCOPE_DEFINER_DEMOGRAPHIC_*` prompt): the LLM picks per-tier
`age_range` and `education_level` only. No occupation reasoning.

Bridge: a deterministic DuckDB query asks "given this tier's cumulative
(age_range, education_level), which occupations co-occur in Nemotron-Personas-
Korea, and how many rows does each have?" The result is a per-tier candidate
pool of (occupation, count) pairs — top-K by count.

Stage 2 (`SCOPE_DEFINER_OCCUPATION_*` prompt): the LLM picks <=3 NEW
occupations per tier from the corresponding candidate pool. Cumulative semantic
("don't repeat across tiers") is enforced in the prompt.

Final output: the same `{direct, plausible, weak}` shape the existing SQL
filter consumes. An `_audit` block carries the intermediate stages so reviewers
can inspect each step.

Reuses:
- `models.prompts.SCOPE_DEFINER_DEMOGRAPHIC_SYSTEM/USER`
- `models.prompts.SCOPE_DEFINER_OCCUPATION_SYSTEM/USER`
- `llm.client.chat_completion` (drives USAGE_TRACKER)
- `dataset.processed.nemotron.education_level_values.EDUCATION_LEVELS`
- `dataset.processed.nemotron.age_range.AGE_MIN/MAX`
"""

from __future__ import annotations

import importlib
import json
import os
import re
from pathlib import Path
from typing import Any

from llm.client import chat_completion
from models.prompts import (
    SCOPE_DEFINER_DEMOGRAPHIC_SYSTEM,
    SCOPE_DEFINER_DEMOGRAPHIC_USER,
    SCOPE_DEFINER_OCCUPATION_SYSTEM,
    SCOPE_DEFINER_OCCUPATION_USER,
)


def _load_nemotron_values(module_path: str) -> tuple[list[str], int, int]:
    """Import EDUCATION_LEVELS / AGE_MIN / AGE_MAX from a country-specific package.

    `module_path` is a dotted Python package (e.g. ``dataset.processed.nemotron_sg``)
    that contains ``education_level_values.py`` and ``age_range.py``.
    """
    edu = importlib.import_module(f"{module_path}.education_level_values")
    age = importlib.import_module(f"{module_path}.age_range")
    return edu.EDUCATION_LEVELS, age.AGE_MIN, age.AGE_MAX


# Bind at import time from env var (set by main.py:_apply_policy_override) or fall
# back to Korea defaults. This lets a process opt into the right Nemotron variant
# before scope_definer is exercised.
NEMOTRON_VALUES_MODULE = os.environ.get(
    "NEMOTRON_VALUES_MODULE", "dataset.processed.nemotron_kor"
)
EDUCATION_LEVELS, AGE_MIN, AGE_MAX = _load_nemotron_values(NEMOTRON_VALUES_MODULE)


SCOPE_DEFINER_SUBDIR = "scope_definer"

# Default per-tier candidate pool size handed to Stage 2.
DEFAULT_CANDIDATE_TOP_K = 100


class PersonaScopeDefiner:
    """Two-stage LLM-driven selector of tiered Nemotron filter scopes per KG actor."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.scope_config = (
            config.get("persona", {}).get("scope_definer", {}) if isinstance(config, dict) else {}
        )
        self.max_per_tier = int(self.scope_config.get("max_per_tier_occupation", 3))
        self.candidate_top_k = int(self.scope_config.get("candidate_pool_top_k", DEFAULT_CANDIDATE_TOP_K))
        self.parquet_glob = self.scope_config.get(
            "parquet_glob",
            "dataset/processed/nemotron_kor/Nemotron-Personas-Korea/data/*.parquet",
        )
        self.run_dir: Path | None = (
            Path(config["_current_run_dir"]) if config.get("_current_run_dir") else None
        )

    # ------------------------------------------------------------------ public

    def define(self, entity: Any, entity_type: str, kg_context: dict[str, Any]) -> dict[str, Any] | None:
        """Run the two-stage Definer and return the merged tier scope dict.

        Returns None for organization-modality entities so the caller can skip
        the SQL pipeline and route to `institutional_actor_profile`.
        """
        attributes = getattr(entity, "attributes", {}) or {}
        if attributes.get("agent_modality") == "organization":
            return None

        cached = self._read_cache(entity, entity_type)
        if cached is not None:
            return cached

        # --- Stage 1: demographic ---
        demographic_payload = self._call_demographic(kg_context)
        demographic_clean = self._sanitize_demographic(demographic_payload)

        # --- Bridge: query candidate occupation pools per tier (cumulative) ---
        candidate_pools = self._query_candidate_pools(demographic_clean)

        # --- Stage 2: occupation ---
        occupation_payload = self._call_occupation(kg_context, demographic_clean, candidate_pools)
        occupation_clean = self._sanitize_occupation(occupation_payload, candidate_pools)

        # --- Merge into the final shape the SQL filter expects ---
        final = self._merge(demographic_clean, occupation_clean)
        final["_audit"] = {
            "stage1_demographic": demographic_clean,
            "candidate_pools": candidate_pools,
            "stage2_occupation": occupation_clean,
        }
        self._write_cache(entity, entity_type, final)
        return final

    # ------------------------------------------------------------- Stage 1 LLM

    def _call_demographic(self, kg_context: dict[str, Any]) -> dict[str, Any]:
        education_list = "\n".join(EDUCATION_LEVELS)
        user = SCOPE_DEFINER_DEMOGRAPHIC_USER.format(
            actor_context=json.dumps(kg_context, ensure_ascii=False),
            policy_context=json.dumps(self.config.get("_current_policy_context", {}), ensure_ascii=False),
            education_list=education_list,
            education_count=len(EDUCATION_LEVELS),
            age_min=AGE_MIN,
            age_max=AGE_MAX,
        )
        messages = [
            {"role": "system", "content": SCOPE_DEFINER_DEMOGRAPHIC_SYSTEM},
            {"role": "user", "content": user},
        ]
        raw = chat_completion(messages, self.config)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Stage1 (demographic) returned invalid JSON: {exc}\nRaw prefix: {raw[:240]!r}"
            )

    def _sanitize_demographic(self, payload: dict[str, Any]) -> dict[str, Any]:
        education_set = set(EDUCATION_LEVELS)
        cleaned: dict[str, Any] = {}
        for tier in ("direct", "plausible", "weak"):
            block = payload.get(tier) if isinstance(payload, dict) else None
            block = block if isinstance(block, dict) else {}

            edu = [v for v in (block.get("education_level") or []) if v in education_set]
            age = block.get("age_range")
            if isinstance(age, (list, tuple)) and len(age) == 2:
                try:
                    lo = max(AGE_MIN, int(age[0]))
                    hi = min(AGE_MAX, int(age[1]))
                    if lo > hi:
                        lo, hi = hi, lo
                    age_range = [lo, hi]
                except (TypeError, ValueError):
                    age_range = [AGE_MIN, AGE_MAX]
            else:
                age_range = [AGE_MIN, AGE_MAX]
            cleaned[tier] = {"education_level": edu, "age_range": age_range}
        cleaned["rationale"] = str(payload.get("rationale") or "")
        return cleaned

    # ----------------------------------------------------- Bridge: candidate query

    def _query_candidate_pools(self, demographic: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        """Per tier, query DuckDB for distinct occupations co-occurring with the
        tier's CUMULATIVE (age, education_level) filter. Returns top-K by count.

        Cumulative semantics matches the eventual SQL filter: each tier's pool is
        the set of occupations potentially tagged with that tier or below.
        """
        import duckdb

        cum_age: list[int] | None = None
        cum_edu: list[str] = []
        pools: dict[str, list[dict[str, Any]]] = {}

        for tier in ("direct", "plausible", "weak"):
            block = demographic.get(tier) or {}
            new_age = block.get("age_range")
            new_edu = block.get("education_level") or []

            cum_edu = _dedupe(cum_edu + list(new_edu))
            if isinstance(new_age, (list, tuple)) and len(new_age) == 2:
                lo, hi = int(new_age[0]), int(new_age[1])
                cum_age = (
                    [lo, hi] if cum_age is None
                    else [min(cum_age[0], lo), max(cum_age[1], hi)]
                )
            elif cum_age is None:
                cum_age = [AGE_MIN, AGE_MAX]

            if not cum_edu:
                # No education constraint yet — query without the edu predicate.
                edu_predicate = "TRUE"
            else:
                edu_predicate = f"education_level IN ({_format_in_list(cum_edu)})"

            sql = f"""
                SELECT occupation, COUNT(*) AS n
                FROM read_parquet('{self.parquet_glob}')
                WHERE age BETWEEN {cum_age[0]} AND {cum_age[1]}
                  AND {edu_predicate}
                  AND occupation IS NOT NULL
                  AND occupation != ''
                GROUP BY occupation
                ORDER BY n DESC
                LIMIT {self.candidate_top_k}
            """
            rows = duckdb.sql(sql).fetchall()
            pools[tier] = [{"occupation": occ, "count": int(n)} for occ, n in rows]
        return pools

    # ------------------------------------------------------------- Stage 2 LLM

    def _call_occupation(
        self,
        kg_context: dict[str, Any],
        demographic: dict[str, Any],
        candidate_pools: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        user = SCOPE_DEFINER_OCCUPATION_USER.format(
            actor_context=json.dumps(kg_context, ensure_ascii=False),
            demographic_scope=json.dumps(demographic, ensure_ascii=False),
            direct_pool=_format_pool_for_prompt(candidate_pools["direct"]),
            direct_count=len(candidate_pools["direct"]),
            plausible_pool=_format_pool_for_prompt(candidate_pools["plausible"]),
            plausible_count=len(candidate_pools["plausible"]),
            weak_pool=_format_pool_for_prompt(candidate_pools["weak"]),
            weak_count=len(candidate_pools["weak"]),
            max_per_tier=self.max_per_tier,
        )
        messages = [
            {"role": "system", "content": SCOPE_DEFINER_OCCUPATION_SYSTEM},
            {"role": "user", "content": user},
        ]
        raw = chat_completion(messages, self.config)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Stage2 (occupation) returned invalid JSON: {exc}\nRaw prefix: {raw[:240]!r}"
            )

    def _sanitize_occupation(
        self,
        payload: dict[str, Any],
        candidate_pools: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Validate that each tier's picks come from that tier's pool, deduplicate
        across tiers, and clip to max_per_tier."""
        cleaned: dict[str, Any] = {}
        already_picked: set[str] = set()
        for tier in ("direct", "plausible", "weak"):
            allowed = {p["occupation"] for p in candidate_pools.get(tier, [])}
            picks_raw = payload.get(tier) if isinstance(payload, dict) else []
            picks_raw = picks_raw if isinstance(picks_raw, list) else []
            tier_picks: list[str] = []
            for v in picks_raw:
                if not isinstance(v, str):
                    continue
                if v in already_picked:
                    continue
                if v not in allowed:
                    continue
                tier_picks.append(v)
                if len(tier_picks) >= self.max_per_tier:
                    break
            cleaned[tier] = tier_picks
            already_picked.update(tier_picks)
        cleaned["rationale"] = str(payload.get("rationale") or "")
        return cleaned

    # ------------------------------------------------------------- merge result

    def _merge(self, demographic: dict[str, Any], occupation: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        rationale_parts = []
        for tier in ("direct", "plausible", "weak"):
            d = demographic.get(tier) or {}
            merged[tier] = {
                "occupation": occupation.get(tier) or [],
                "education_level": d.get("education_level") or [],
                "age_range": d.get("age_range") or [AGE_MIN, AGE_MAX],
            }
        if demographic.get("rationale"):
            rationale_parts.append(f"[demographic] {demographic['rationale']}")
        if occupation.get("rationale"):
            rationale_parts.append(f"[occupation] {occupation['rationale']}")
        merged["rationale"] = " | ".join(rationale_parts)
        return merged

    # -------------------------------------------------------------- caching IO

    def _cache_path(self, entity: Any, entity_type: str) -> Path | None:
        if not self.run_dir:
            return None
        key = _entity_cache_key(entity, entity_type)
        return self.run_dir / SCOPE_DEFINER_SUBDIR / f"{key}.json"

    def _read_cache(self, entity: Any, entity_type: str) -> dict[str, Any] | None:
        path = self._cache_path(entity, entity_type)
        if not path or not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_cache(self, entity: Any, entity_type: str, payload: dict[str, Any]) -> None:
        path = self._cache_path(entity, entity_type)
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------- entity util

def _entity_cache_key(entity: Any, entity_type: str) -> str:
    raw = getattr(entity, "id", None) or getattr(entity, "uuid", None) or entity_type or "entity"
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(raw)).strip("_")
    return safe or "entity"


def build_kg_context(entity: Any, entity_type: str) -> dict[str, Any]:
    """Assemble the JSON object the Definer (and the deferred Refiner) consumes."""
    attributes = getattr(entity, "attributes", {}) or {}
    return {
        "entity_id": getattr(entity, "id", None),
        "entity_type": entity_type,
        "name": getattr(entity, "name", ""),
        "summary": getattr(entity, "summary", ""),
        "iad_category": getattr(entity, "iad_category", None) or attributes.get("iad_category"),
        "node_family": getattr(entity, "node_family", None) or attributes.get("node_family"),
        "agent_modality": attributes.get("agent_modality"),
        "role_family": attributes.get("role_family"),
        "role_scope": attributes.get("role_scope"),
        "eligibility_criteria": attributes.get("eligibility_criteria"),
        "exclusion_criteria": attributes.get("exclusion_criteria"),
        "is_actor": getattr(entity, "is_actor", None) or attributes.get("is_actor"),
        "persona_eligible": getattr(entity, "persona_eligible", None) or attributes.get("persona_eligible"),
        "attributes": attributes,
        "related_edges": getattr(entity, "related_edges", []) or [],
        "related_nodes": getattr(entity, "related_nodes", []) or [],
        "source_spans": attributes.get("source_spans") or getattr(entity, "source_spans", []),
    }


def save_kg_context(kg_context: dict[str, Any], run_dir: Path | str | None, entity: Any, entity_type: str) -> Path | None:
    """Persist the per-entity KG context next to the definer cache for inspection."""
    if not run_dir:
        return None
    base = Path(run_dir) / SCOPE_DEFINER_SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{_entity_cache_key(entity, entity_type)}_kg_context.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(kg_context, f, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------- prompt utils

def _format_pool_for_prompt(pool: list[dict[str, Any]]) -> str:
    """Render a candidate pool as `   count  occupation` lines, sorted DESC by count."""
    if not pool:
        return "(no candidates)"
    return "\n".join(f"  {entry['count']:>6,}  {entry['occupation']}" for entry in pool)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _format_in_list(values: list[str]) -> str:
    if not values:
        return "''"
    quoted = ["'" + str(v).replace("'", "''") + "'" for v in values]
    return ", ".join(quoted)
