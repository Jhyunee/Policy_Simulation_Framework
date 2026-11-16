"""Sampler for NVIDIA Nemotron-Personas-Korea rows."""

from __future__ import annotations

import os
import re
import heapq
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from llm.client import chat_completion
from models.prompts import (
    NEMOTRON_RERANKER_SYSTEM,
    NEMOTRON_RERANKER_USER,
    NEMOTRON_VALUE_PROFILE_SYSTEM,
    NEMOTRON_VALUE_PROFILE_USER,
)


DEFAULT_COLUMNS = [
    "uuid",
    "persona",
    "professional_persona",
    "family_persona",
    "skills_and_expertise",
    "career_goals_and_ambitions",
    "sex",
    "age",
    "marital_status",
    "military_status",
    "family_type",
    "education_level",
    "bachelors_field",
    "occupation",
    "district",
    "province",
    "hobbies_and_interests",
]

LIGHT_PREFILTER_COLUMNS = [
    "uuid",
    "sex",
    "age",
    "education_level",
    "bachelors_field",
    "occupation",
    "district",
    "province",
]


class NemotronPersonaSampler:
    """Streams and filters Nemotron-Personas-Korea for a policy actor role."""

    def __init__(self, config: dict[str, Any]):
        self.full_config = config
        persona_config = config.get("persona", {})
        self.config = persona_config.get("nemotron", {})
        self.dataset_name = self.config.get("dataset_name", "nvidia/Nemotron-Personas-Korea")
        self.split = self.config.get("split", "train")
        self.streaming = bool(self.config.get("streaming", True))
        self.cache_dir = self.config.get("cache_dir")
        self.local_data_dir = self.config.get("local_data_dir")
        self.seed = int(self.config.get("seed", 42))
        self.shuffle_buffer_size = int(self.config.get("shuffle_buffer_size", 10000))
        self.scan_limit = int(self.config.get("scan_limit", 50000))
        self.candidate_pool_size = int(self.config.get("candidate_pool_size", 300))
        self.prefilter_strategy = str(self.config.get("prefilter_strategy", "first_match")).lower()
        self.min_match_score = int(self.config.get("min_match_score", 2))
        self.reranker = str(self.config.get("reranker", "heuristic")).lower()
        self.llm_reranker_config = self.config.get("llm_reranker", {})
        self.diversity_penalty = self.config.get("diversity_penalty", {})
        self.columns = list(self.config.get("columns") or DEFAULT_COLUMNS)
        self.top_m_light_scan = bool(self.config.get("top_m_light_scan", True))
        self.top_m_inline_text_columns = bool(self.config.get("top_m_inline_text_columns", False))
        self.adaptive_config = self.config.get("adaptive_matching", {})
        self.adaptive_enabled = bool(self.adaptive_config.get("enabled", False))
        self._role_filter_cache: dict[str, dict[str, Any]] = {}

    def sample_for_entity(
        self,
        entity: Any,
        entity_type: str,
        count: int,
    ) -> list[dict[str, Any]]:
        """Return up to count rows matched to a KG actor entity."""
        if count <= 0:
            return []

        if self.adaptive_enabled:
            self._get_role_filter(entity, entity_type)

        pool_size = max(count, self.candidate_pool_size)
        candidates = self._collect_pool(entity, entity_type, pool_size, self.min_match_score)
        if len(candidates) < count:
            candidates.extend(
                self._collect_pool(
                    entity,
                    entity_type,
                    pool_size - len(candidates),
                    min_score=1,
                    seen=candidates,
                )
            )
        if len(candidates) < count:
            candidates.extend(self._collect_any(count - len(candidates), seen=candidates))

        ranked = self._rank_candidates(candidates, count, entity, entity_type)
        return ranked[:count]

    def _collect_pool(
        self,
        entity: Any,
        entity_type: str,
        pool_size: int,
        min_score: int,
        seen: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if pool_size <= 0:
            return []
        if self.prefilter_strategy in {"top_m", "top_score", "top-score"}:
            return self._collect_top_pool(entity, entity_type, pool_size, min_score, seen=seen)
        selected = []
        seen_ids = {row.get("uuid") for row in (seen or []) if row.get("uuid")}
        for index, row in enumerate(self._iter_rows()):
            if index >= self.scan_limit:
                break
            slim = self._slim_row(row)
            row_id = slim.get("uuid")
            if row_id and row_id in seen_ids:
                continue
            score = self._score_row(slim, entity, entity_type)
            if score < min_score:
                continue
            slim["match_score"] = score
            slim["match_strategy"] = f"score>={min_score}"
            selected.append(slim)
            if row_id:
                seen_ids.add(row_id)
            if len(selected) >= pool_size:
                break
        return selected

    def _collect_top_pool(
        self,
        entity: Any,
        entity_type: str,
        pool_size: int,
        min_score: int,
        seen: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Scan up to scan_limit and keep the strongest rule-filtered Top-M candidates."""
        local_files = _local_parquet_files(self.local_data_dir)
        if local_files and self.top_m_light_scan:
            fast_local = self._collect_top_pool_local_fast(
                local_files,
                entity,
                entity_type,
                pool_size,
                min_score,
                seen=seen,
            )
            if fast_local is not None:
                return fast_local
            vectorized = self._collect_top_pool_local_vectorized(
                local_files,
                entity,
                entity_type,
                pool_size,
                min_score,
                seen=seen,
            )
            if vectorized is not None:
                return vectorized

        heap: list[tuple[tuple[float, float, str], int, dict[str, Any]]] = []
        seen_ids = {row.get("uuid") for row in (seen or []) if row.get("uuid")}
        rows = self._iter_prefilter_rows(local_files)
        for index, row in enumerate(rows):
            if index >= self.scan_limit:
                break
            slim = self._slim_row(row)
            row_id = str(slim.get("uuid") or "")
            if row_id and row_id in seen_ids:
                continue
            score = self._score_row(slim, entity, entity_type)
            if score < min_score:
                continue
            slim["match_score"] = score
            slim["match_strategy"] = f"top_m_score>={min_score}"
            raw_score = float(slim.get("rule_filter_raw_score") or score)
            rank_key = (raw_score, float(score), _stable_tie_breaker(row_id or str(index)))
            item = (rank_key, index, slim)
            if len(heap) < pool_size:
                heapq.heappush(heap, item)
            elif rank_key > heap[0][0]:
                heapq.heapreplace(heap, item)
            if row_id:
                seen_ids.add(row_id)

        ranked = sorted(heap, key=lambda item: item[0], reverse=True)
        selected = [item[2] for item in ranked]
        if local_files and self.top_m_light_scan:
            return self._hydrate_rows_by_uuid(selected, local_files)
        return selected

    def _collect_top_pool_local_fast(
        self,
        local_files: list[str],
        entity: Any,
        entity_type: str,
        pool_size: int,
        min_score: int,
        seen: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]] | None:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return None

        role_filter = self._get_role_filter(entity, entity_type)
        role_family = _infer_role_family_for_sampling(entity, entity_type)
        keywords = _lower_terms(_role_keywords(entity, entity_type, role_filter))
        occupation_keywords = _lower_terms(role_filter.get("occupation_keywords", []))
        occupation_exclude_keywords = _lower_terms(role_filter.get("occupation_exclude_keywords", []))
        education_keywords = _lower_terms(role_filter.get("education_keywords", []))
        field_keywords = _lower_terms(role_filter.get("bachelors_field_keywords", []))
        signal_keywords = _lower_terms([
            "research",
            "researcher",
            "scientist",
            "science",
            "doctoral",
            "doctor",
            "phd",
            "engineer",
            "연구",
            "연구원",
            "연구자",
            "과학",
            "공학",
            "기술",
            "기술자",
            "박사",
            "대학원",
            "정보통신",
            "소프트웨어",
            "프로그래머",
            "데이터",
            "컴퓨터",
        ])
        occupation_weights = _lower_weight_map(role_filter.get("adaptive_occupation_weights", {}))
        education_weights = _lower_weight_map(role_filter.get("adaptive_education_weights", {}))
        field_weights = _lower_weight_map(role_filter.get("adaptive_bachelors_field_weights", {}))
        age_constraints = role_filter.get("age_soft_constraints") or {}
        observable_profile = role_filter.get("observable_eligibility") or {}
        observable_age_constraints = observable_profile.get("age_constraints") or {}
        has_observable_indirect = bool(observable_profile.get("indirect_signals"))
        require_column_signal = bool(self.config.get("require_column_signal", True))

        if self.top_m_inline_text_columns:
            columns = list(self.columns)
        else:
            columns = [column for column in LIGHT_PREFILTER_COLUMNS if column in self.columns]
        if "uuid" not in columns:
            columns.append("uuid")

        heap: list[tuple[tuple[float, float, str], int, dict[str, Any]]] = []
        heap_limit = self._prefilter_heap_limit(pool_size)
        field_heap_limit = int(self.config.get("prefilter_field_heap_size", pool_size) or pool_size)
        field_heaps: dict[str, list[tuple[tuple[float, float, str], int, dict[str, Any]]]] = {}
        seen_ids = {str(row.get("uuid")) for row in (seen or []) if row.get("uuid")}
        batch_size = int(self.config.get("local_batch_size", 8192))
        scanned = 0
        sequence = 0

        for file_path in local_files:
            if scanned >= self.scan_limit:
                break
            parquet_file = pq.ParquetFile(file_path)
            available_columns = set(parquet_file.schema_arrow.names)
            selected_columns = [column for column in columns if column in available_columns]
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=selected_columns):
                if scanned >= self.scan_limit:
                    break
                data = batch.to_pydict()
                batch_len = batch.num_rows
                remaining = self.scan_limit - scanned
                if batch_len > remaining:
                    batch_len = remaining
                scanned += batch_len

                for offset in range(batch_len):
                    row_id = str(_column_value(data, "uuid", offset) or "")
                    if row_id and row_id in seen_ids:
                        continue
                    occupation = str(_column_value(data, "occupation", offset) or "").lower()
                    education = str(_column_value(data, "education_level", offset) or "").lower()
                    field = str(_column_value(data, "bachelors_field", offset) or "").lower()
                    age = _as_int(_column_value(data, "age", offset))

                    raw_score = 0.0
                    column_signal = False

                    if _contains_term(occupation, occupation_exclude_keywords):
                        raw_score -= 1.0

                    occupation_weight = _exact_weight(occupation, occupation_weights)
                    occupation_match = _contains_term(occupation, occupation_keywords)
                    keyword_match = _contains_term(occupation, keywords)
                    education_weight = _exact_weight(education, education_weights)
                    education_match = _contains_term(education, education_keywords)
                    field_weight = _exact_weight(field, field_weights)
                    field_match = _contains_term(field, field_keywords)

                    if occupation_weight > 0:
                        raw_score += min(occupation_weight, 3.0)
                        column_signal = True
                    elif occupation_match:
                        raw_score += 2.25
                        column_signal = True
                    if keyword_match:
                        raw_score += 1.0
                        column_signal = True

                    if education_weight > 0:
                        raw_score += min(education_weight, 2.0)
                        column_signal = True
                    elif education_match:
                        raw_score += 1.0
                        column_signal = True

                    if field_weight > 0:
                        raw_score += min(field_weight, 1.5)
                        column_signal = True
                    elif field_match:
                        raw_score += 0.75
                        column_signal = True

                    if _age_passes(age, age_constraints):
                        raw_score += 0.5
                        column_signal = True

                    if role_family == "direct_beneficiary":
                        if _age_passes(age, observable_age_constraints):
                            raw_score += 0.75
                            column_signal = True
                        if has_observable_indirect and _contains_term(
                            " ".join([occupation, education, field]),
                            signal_keywords,
                        ):
                            raw_score += 0.25
                            column_signal = True

                    representative_signal = occupation_weight > 0 or occupation_match or keyword_match
                    if role_family in {"evaluator", "policy_administrator"} and not representative_signal:
                        raw_score -= 0.5

                    if require_column_signal and not column_signal:
                        continue

                    score = _scale_role_fit_1_to_5(raw_score, [])
                    if score < min_score:
                        continue

                    slim = {
                        key: _column_value(data, key, offset)
                        for key in self.columns
                        if key in data
                    }
                    slim["rule_filter_raw_score"] = round(raw_score, 3)
                    slim["match_score"] = score
                    slim["match_strategy"] = f"top_m_fast_score>={min_score}"
                    rank_key = (raw_score, float(score), _stable_tie_breaker(row_id or str(sequence)))
                    item = (rank_key, sequence, slim)
                    sequence += 1
                    if len(heap) < heap_limit:
                        heapq.heappush(heap, item)
                    elif rank_key > heap[0][0]:
                        heapq.heapreplace(heap, item)
                    field_key = str(slim.get("bachelors_field") or "unknown")
                    field_heap = field_heaps.setdefault(field_key, [])
                    if len(field_heap) < field_heap_limit:
                        heapq.heappush(field_heap, item)
                    elif rank_key > field_heap[0][0]:
                        heapq.heapreplace(field_heap, item)
                    if row_id:
                        seen_ids.add(row_id)

        combined_items = list(heap)
        for field_heap in field_heaps.values():
            combined_items.extend(field_heap)
        by_row_id: dict[str, tuple[tuple[float, float, str], int, dict[str, Any]]] = {}
        for item in combined_items:
            row_id = str(item[2].get("uuid") or item[1])
            if row_id not in by_row_id or item[0] > by_row_id[row_id][0]:
                by_row_id[row_id] = item
        ranked = sorted(by_row_id.values(), key=lambda item: item[0], reverse=True)
        selected = [item[2] for item in ranked]
        hydrated = selected if self.top_m_inline_text_columns else self._hydrate_rows_by_uuid(selected, local_files)
        rescored: list[dict[str, Any]] = []
        for row in hydrated:
            exact_score = self._score_row(row, entity, entity_type)
            if exact_score < min_score:
                continue
            row["match_score"] = exact_score
            row["match_strategy"] = f"top_m_fast_score>={min_score}"
            rescored.append(row)
        rescored = sorted(
            rescored,
            key=lambda row: (
                float(row.get("rule_filter_raw_score") or 0),
                float(row.get("match_score") or 0),
                str(row.get("uuid") or ""),
            ),
            reverse=True,
        )
        return self._diversified_ranked_candidates(rescored, pool_size)

    def _prefilter_heap_limit(self, pool_size: int) -> int:
        multiplier = int(self.config.get("prefilter_heap_multiplier", 1) or 1)
        return max(pool_size, pool_size * max(multiplier, 1))

    def _collect_top_pool_local_vectorized(
        self,
        local_files: list[str],
        entity: Any,
        entity_type: str,
        pool_size: int,
        min_score: int,
        seen: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]] | None:
        try:
            import pandas as pd
            import pyarrow.parquet as pq
        except ImportError:
            return None

        role_filter = self._get_role_filter(entity, entity_type)
        role_family = _infer_role_family_for_sampling(entity, entity_type)
        keywords = _role_keywords(entity, entity_type, role_filter)
        columns = [column for column in LIGHT_PREFILTER_COLUMNS if column in self.columns]
        if "uuid" not in columns:
            columns.append("uuid")

        heap: list[tuple[tuple[float, float, str], int, dict[str, Any]]] = []
        seen_ids = {str(row.get("uuid")) for row in (seen or []) if row.get("uuid")}
        batch_size = int(self.config.get("local_batch_size", 8192))
        scanned = 0
        sequence = 0

        for file_path in local_files:
            if scanned >= self.scan_limit:
                break
            parquet_file = pq.ParquetFile(file_path)
            available_columns = set(parquet_file.schema_arrow.names)
            selected_columns = [column for column in columns if column in available_columns]
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=selected_columns):
                if scanned >= self.scan_limit:
                    break
                df = batch.to_pandas()
                remaining = self.scan_limit - scanned
                if len(df) > remaining:
                    df = df.iloc[:remaining].copy()
                scanned += len(df)
                if df.empty:
                    continue
                if "uuid" in df.columns and seen_ids:
                    id_series = df["uuid"].astype(str)
                    df = df.loc[~id_series.isin(seen_ids)].copy()
                    if df.empty:
                        continue

                raw_scores = self._vector_rule_raw_scores(df, role_filter, role_family, keywords, pd)
                scaled_scores = raw_scores.apply(lambda value: _scale_role_fit_1_to_5(float(value), []))
                matched = df.loc[scaled_scores >= min_score]
                if matched.empty:
                    continue

                for row_index, row in matched.iterrows():
                    row_id = str(row.get("uuid") or "")
                    if row_id and row_id in seen_ids:
                        continue
                    raw_score = float(raw_scores.loc[row_index])
                    score = int(scaled_scores.loc[row_index])
                    slim = {
                        key: row.get(key)
                        for key in self.columns
                        if key in row.index
                    }
                    slim["rule_filter_raw_score"] = round(raw_score, 3)
                    slim["match_score"] = score
                    slim["match_strategy"] = f"top_m_vector_score>={min_score}"
                    rank_key = (raw_score, float(score), _stable_tie_breaker(row_id or str(sequence)))
                    item = (rank_key, sequence, slim)
                    sequence += 1
                    if len(heap) < pool_size:
                        heapq.heappush(heap, item)
                    elif rank_key > heap[0][0]:
                        heapq.heapreplace(heap, item)
                    if row_id:
                        seen_ids.add(row_id)

        ranked = sorted(heap, key=lambda item: item[0], reverse=True)
        selected = [item[2] for item in ranked]
        hydrated = self._hydrate_rows_by_uuid(selected, local_files)
        rescored: list[dict[str, Any]] = []
        for row in hydrated:
            exact_score = self._score_row(row, entity, entity_type)
            if exact_score < min_score:
                continue
            row["match_score"] = exact_score
            row["match_strategy"] = f"top_m_vector_score>={min_score}"
            rescored.append(row)
        return sorted(
            rescored,
            key=lambda row: (
                float(row.get("rule_filter_raw_score") or 0),
                float(row.get("match_score") or 0),
                str(row.get("uuid") or ""),
            ),
            reverse=True,
        )

    def _vector_rule_raw_scores(
        self,
        df: Any,
        role_filter: dict[str, Any],
        role_family: str,
        keywords: list[str],
        pd: Any,
    ) -> Any:
        index = df.index
        raw = pd.Series(0.0, index=index)
        column_signal = pd.Series(False, index=index)

        occupation = _string_series(df, "occupation", pd)
        education = _string_series(df, "education_level", pd)
        field = _string_series(df, "bachelors_field", pd)
        combined_light_text = occupation + " " + education + " " + field

        exclude_match = _series_contains_any(occupation, role_filter.get("occupation_exclude_keywords", []), pd)
        raw.loc[exclude_match] -= 1.0

        occupation_weights = _series_exact_weights(
            occupation,
            role_filter.get("adaptive_occupation_weights", {}),
            pd,
        )
        occupation_match = _series_contains_any(occupation, role_filter.get("occupation_keywords", []), pd)
        keyword_match = _series_contains_any(occupation, keywords, pd)

        education_weights = _series_exact_weights(
            education,
            role_filter.get("adaptive_education_weights", {}),
            pd,
        )
        education_match = _series_contains_any(education, role_filter.get("education_keywords", []), pd)

        field_weights = _series_exact_weights(
            field,
            role_filter.get("adaptive_bachelors_field_weights", {}),
            pd,
        )
        field_match = _series_contains_any(field, role_filter.get("bachelors_field_keywords", []), pd)

        age_signal = _series_age_signal(df, role_filter, pd)
        eligibility_signal = _series_observable_eligibility_signal(
            df,
            role_filter,
            role_family,
            combined_light_text,
            pd,
        )

        column_signal |= occupation_weights > 0
        column_signal |= occupation_match
        column_signal |= education_weights > 0
        column_signal |= education_match
        column_signal |= field_weights > 0
        column_signal |= field_match
        column_signal |= age_signal > 0
        column_signal |= eligibility_signal > 0

        raw += eligibility_signal
        raw += occupation_weights.clip(upper=3.0)
        raw += occupation_match.where(occupation_weights <= 0, False).astype(float) * 2.25
        raw += keyword_match.astype(float)
        raw += education_weights.clip(upper=2.0)
        raw += education_match.where(education_weights <= 0, False).astype(float)
        raw += field_weights.clip(upper=1.5)
        raw += field_match.where(field_weights <= 0, False).astype(float) * 0.75
        raw += age_signal

        representative_signal = (occupation_weights > 0) | occupation_match | keyword_match
        if role_family in {"evaluator", "policy_administrator"}:
            raw.loc[~representative_signal] -= 0.5

        if bool(self.config.get("require_column_signal", True)):
            raw.loc[~column_signal] = 0.0
        return raw

    def _collect_any(
        self,
        count: int,
        seen: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        selected = []
        seen_ids = {row.get("uuid") for row in (seen or []) if row.get("uuid")}
        for row in self._iter_rows():
            slim = self._slim_row(row)
            row_id = slim.get("uuid")
            if row_id and row_id in seen_ids:
                continue
            slim["match_score"] = 0
            slim["match_strategy"] = "fallback_any"
            selected.append(slim)
            if row_id:
                seen_ids.add(row_id)
            if len(selected) >= count:
                break
        return selected

    def _rank_candidates(
        self,
        candidates: list[dict[str, Any]],
        count: int,
        entity: Any,
        entity_type: str,
    ) -> list[dict[str, Any]]:
        if self.reranker == "llm":
            try:
                return self._llm_rerank_candidates(candidates, count, entity, entity_type)
            except Exception as exc:
                for row in candidates:
                    row.setdefault("rerank_warnings", []).append(f"LLM reranker failed; heuristic fallback used: {exc}")
                return self._heuristic_rank_candidates(candidates, count)
        if self.reranker != "heuristic":
            raise ValueError(f"Unsupported Nemotron reranker: {self.reranker}")
        return self._heuristic_rank_candidates(candidates, count)

    def _heuristic_rank_candidates(self, candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        remaining = sorted(
            candidates,
            key=lambda row: (
                row.get("match_score", 0),
                str(row.get("occupation") or ""),
                str(row.get("uuid") or ""),
            ),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        usage = self._diversity_usage_template()

        while remaining and len(selected) < count:
            best_index = 0
            best_score = float("-inf")
            for index, row in enumerate(remaining):
                score = float(row.get("match_score", 0))
                score -= self._diversity_penalty(row, usage)
                if score > best_score:
                    best_score = score
                    best_index = index
            picked = remaining.pop(best_index)
            picked["rerank_score"] = round(best_score, 3)
            picked["rerank_strategy"] = "heuristic_pool_top_n"
            selected.append(picked)
            for key in usage:
                value = picked.get(key)
                if value not in (None, ""):
                    usage[key][value] = usage[key].get(value, 0) + 1
        return selected

    def _llm_rerank_candidates(
        self,
        candidates: list[dict[str, Any]],
        count: int,
        entity: Any,
        entity_type: str,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        batch_size = int(self.llm_reranker_config.get("batch_size", 30))
        batch_size = max(5, min(batch_size, 60))
        top_n = int(self.llm_reranker_config.get("top_n", count))
        keep_count = min(max(count, top_n), len(candidates))
        role_filter = self._get_role_filter(entity, entity_type)
        actor_context = _actor_context_for_reranker(entity, entity_type)
        scored_by_uuid: dict[str, dict[str, Any]] = {}

        total_batches = (len(candidates) + batch_size - 1) // batch_size
        for batch_index, start in enumerate(range(0, len(candidates), batch_size), 1):
            batch = candidates[start:start + batch_size]
            print(
                f"  LLM reranker batch {batch_index}/{total_batches} "
                f"for {entity_type}:{getattr(entity, 'name', '')}",
                flush=True,
            )
            messages = [
                {"role": "system", "content": NEMOTRON_RERANKER_SYSTEM},
                {
                    "role": "user",
                    "content": NEMOTRON_RERANKER_USER.format(
                        actor_context=json.dumps(actor_context, ensure_ascii=False),
                        policy_context=json.dumps(self.full_config.get("_current_policy_context", {}), ensure_ascii=False),
                        matching_profile=json.dumps(_matching_profile_for_prompt(role_filter), ensure_ascii=False),
                        candidates=json.dumps(
                            [_compact_candidate_for_rerank(row) for row in batch],
                            ensure_ascii=False,
                        ),
                    ),
                },
            ]
            data = json.loads(chat_completion(messages, self.full_config))
            for item in data.get("ranked_candidates", []):
                row_id = str(item.get("uuid") or "").strip()
                if not row_id:
                    continue
                scored_by_uuid[row_id] = {
                    "role_fit_score": _clamp_role_fit_score(item.get("role_fit_score")),
                    "llm_rerank_rationale": item.get("rationale"),
                    "llm_rerank_uncertainties": item.get("uncertainties") or [],
                }

        for row in candidates:
            row_id = str(row.get("uuid") or "")
            llm_score = scored_by_uuid.get(row_id)
            if llm_score:
                row.update(llm_score)
                row["rule_filter_score"] = row.get("match_score")
                capped_score, cap_notes = _apply_observable_constraint_score_cap(
                    row,
                    role_filter,
                    _infer_role_family_for_sampling(entity, entity_type),
                )
                row["role_fit_score"] = capped_score
                row["match_score"] = capped_score
                if cap_notes:
                    row.setdefault("llm_rerank_uncertainties", []).extend(cap_notes)
                    row.setdefault("score_adjustments", []).extend(cap_notes)
                row["match_strategy"] = "llm_role_fit_1_to_5"
            else:
                row["role_fit_score"] = _clamp_role_fit_score(row.get("match_score"))
                row["match_score"] = row["role_fit_score"]
                row["match_strategy"] = "llm_reranker_missing_score_fallback"

        ranked = sorted(
            candidates,
            key=lambda row: (
                float(row.get("role_fit_score") or row.get("match_score") or 0),
                float(row.get("rule_filter_raw_score") or 0),
                float(row.get("rule_filter_score") or 0),
                str(row.get("uuid") or ""),
            ),
            reverse=True,
        )
        for index, row in enumerate(ranked, 1):
            row["rerank_score"] = row.get("role_fit_score") or row.get("match_score")
            row["rerank_rank"] = index
            row["rerank_strategy"] = "llm_role_fit_top_n"
        diversified = self._diversified_ranked_candidates(ranked, keep_count)
        for index, row in enumerate(diversified, 1):
            row["rerank_rank"] = index
        return diversified

    def _diversified_ranked_candidates(
        self,
        ranked: list[dict[str, Any]],
        keep_count: int,
    ) -> list[dict[str, Any]]:
        if len(ranked) <= keep_count:
            return ranked[:keep_count]
        remaining = list(ranked)
        selected: list[dict[str, Any]] = []
        usage = self._diversity_usage_template()
        while remaining and len(selected) < keep_count:
            best_index = 0
            best_score = float("-inf")
            for index, row in enumerate(remaining):
                score = float(row.get("role_fit_score") or row.get("match_score") or 0) * 10.0
                score += float(row.get("rule_filter_raw_score") or 0)
                score -= self._diversity_penalty(row, usage)
                if score > best_score:
                    best_score = score
                    best_index = index
            picked = remaining.pop(best_index)
            selected.append(picked)
            for key in usage:
                value = picked.get(key)
                if value not in (None, ""):
                    usage[key][value] = usage[key].get(value, 0) + 1
        return selected

    def _diversity_usage_template(self) -> dict[str, dict[Any, int]]:
        keys = list(self.diversity_penalty.keys()) or ["occupation", "province", "sex"]
        return {key: {} for key in keys}

    def _diversity_penalty(self, row: dict[str, Any], usage: dict[str, dict[Any, int]]) -> float:
        penalty = 0.0
        for key, counts in usage.items():
            value = row.get(key)
            if value in (None, ""):
                continue
            weight = float(self.diversity_penalty.get(key, 0.0))
            penalty += weight * counts.get(value, 0)
        return penalty

    def _iter_rows(self) -> Iterable[dict[str, Any]]:
        local_files = _local_parquet_files(self.local_data_dir)
        if local_files:
            return self._iter_local_parquet_rows(local_files)

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Nemotron sampler requires the 'datasets' package. Install requirements.txt first."
            ) from exc

        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        kwargs: dict[str, Any] = {
            "split": self.split,
            "streaming": self.streaming,
        }
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = self.cache_dir

        dataset = load_dataset(self.dataset_name, **kwargs)
        if self.streaming and hasattr(dataset, "shuffle"):
            dataset = dataset.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)
        elif not self.streaming and hasattr(dataset, "shuffle"):
            dataset = dataset.shuffle(seed=self.seed)
        return iter(dataset)

    def _iter_prefilter_rows(self, local_files: list[str]) -> Iterable[dict[str, Any]]:
        if local_files and self.top_m_light_scan:
            columns = [column for column in LIGHT_PREFILTER_COLUMNS if column in self.columns]
            return self._iter_local_parquet_rows(local_files, columns=columns)
        return self._iter_rows()

    def _iter_local_parquet_rows(
        self,
        local_files: list[str],
        columns: list[str] | None = None,
    ) -> Iterable[dict[str, Any]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Local Nemotron parquet loading requires the 'pyarrow' package. Install requirements.txt first."
            ) from exc

        batch_size = int(self.config.get("local_batch_size", 8192))
        for file_path in local_files:
            parquet_file = pq.ParquetFile(file_path)
            available_columns = set(parquet_file.schema_arrow.names)
            requested_columns = columns or self.columns
            selected_columns = [column for column in requested_columns if column in available_columns]
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=selected_columns):
                yield from batch.to_pylist()

    def _hydrate_rows_by_uuid(
        self,
        rows: list[dict[str, Any]],
        local_files: list[str],
    ) -> list[dict[str, Any]]:
        if not rows:
            return rows
        ids = [str(row.get("uuid") or "") for row in rows if row.get("uuid")]
        if not ids:
            return rows
        by_id = {str(row.get("uuid")): dict(row) for row in rows if row.get("uuid")}
        try:
            import pyarrow.dataset as ds
        except ImportError:
            return rows

        dataset = ds.dataset(local_files, format="parquet")
        columns = [column for column in self.columns if column in dataset.schema.names]
        try:
            table = dataset.to_table(
                columns=columns,
                filter=ds.field("uuid").isin(ids),
            )
        except Exception:
            return rows

        for hydrated in table.to_pylist():
            row_id = str(hydrated.get("uuid") or "")
            if not row_id or row_id not in by_id:
                continue
            merged = by_id[row_id]
            for key, value in hydrated.items():
                if value not in (None, "") or key not in merged:
                    merged[key] = value
        return [by_id.get(str(row.get("uuid")), row) for row in rows]

    def _score_row(self, row: dict[str, Any], entity: Any, entity_type: str) -> int:
        role_filter = self._get_role_filter(entity, entity_type)
        role_family = _infer_role_family_for_sampling(entity, entity_type)
        keywords = _role_keywords(entity, entity_type, role_filter)
        professional_text = " ".join(
            str(row.get(key) or "")
            for key in ("professional_persona", "skills_and_expertise", "occupation")
        )
        broad_text = " ".join(
            str(row.get(key) or "")
            for key in ("career_goals_and_ambitions", "persona")
        )

        raw_score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []
        occupation_keywords = role_filter.get("occupation_keywords", [])
        occupation_text = str(row.get("occupation") or "")
        exclude_match = _contains_any(occupation_text, role_filter.get("occupation_exclude_keywords", []))
        if exclude_match:
            raw_score -= 1.0
            warnings.append("occupation matched a possible negative signal, but was retained for LLM reranking")
        occupation_value_weight = _value_weight(
            occupation_text,
            role_filter.get("adaptive_occupation_weights", {}),
        )
        occupation_match = _contains_any(occupation_text, occupation_keywords)
        professional_match = _contains_any(professional_text, occupation_keywords)
        broad_match = _contains_any(broad_text, [entity_type, getattr(entity, "name", "")])

        education_keywords = role_filter.get("education_keywords", [])
        education_value_weight = _value_weight(
            row.get("education_level", ""),
            role_filter.get("adaptive_education_weights", {}),
        )
        education_match = _contains_any(row.get("education_level", ""), education_keywords)
        field_keywords = role_filter.get("bachelors_field_keywords", [])
        field_value_weight = _value_weight(
            row.get("bachelors_field", ""),
            role_filter.get("adaptive_bachelors_field_weights", {}),
        )
        field_match = _contains_any(row.get("bachelors_field", ""), field_keywords)
        age_signal, age_warning = _age_signal(row, role_filter)
        if age_warning:
            warnings.append(age_warning)
        eligibility_signal, eligibility_reasons, eligibility_warnings, eligibility_match = _observable_eligibility_signal(
            row,
            role_filter,
            role_family,
            professional_text,
            broad_text,
        )
        if eligibility_signal > 0:
            raw_score += eligibility_signal
        reasons.extend(eligibility_reasons)
        warnings.extend(eligibility_warnings)

        column_signal = (
            occupation_value_weight > 0
            or occupation_match
            or education_value_weight > 0
            or education_match
            or field_value_weight > 0
            or field_match
            or age_signal > 0
            or eligibility_signal > 0
        )
        if bool(self.config.get("require_column_signal", True)) and not column_signal:
            row["rule_filter_raw_score"] = round(raw_score, 3)
            return 0
        representative_signal = (
            occupation_value_weight > 0
            or occupation_match
            or professional_match
            or _contains_any(professional_text, keywords)
        )
        if role_family in {"evaluator", "policy_administrator"} and not representative_signal:
            raw_score -= 0.5
            warnings.append("representative role signal is weak; retained for LLM judgment")

        if occupation_value_weight > 0:
            raw_score += min(occupation_value_weight, 3.0)
            reasons.append("adaptive occupation value match")
        elif occupation_match:
            raw_score += 2.25
            reasons.append("occupation keyword match")
        elif professional_match:
            raw_score += 1.25
            reasons.append("professional persona keyword match")
        if _contains_any(professional_text, keywords):
            raw_score += 1.0
            reasons.append("role keyword appears in professional text")
        if broad_match:
            raw_score += 0.5
            reasons.append("role keyword appears in broader persona text")
        if education_value_weight > 0:
            raw_score += min(education_value_weight, 2.0)
            reasons.append("adaptive education value match")
        elif education_match:
            raw_score += 1.0
            reasons.append("education keyword match")
        if field_value_weight > 0:
            raw_score += min(field_value_weight, 1.5)
            reasons.append("adaptive bachelors field value match")
        elif field_match:
            raw_score += 0.75
            reasons.append("bachelors field keyword match")
        elif (
            role_family == "direct_beneficiary"
            and (field_keywords or role_filter.get("adaptive_bachelors_field_weights"))
        ):
            warnings.append(
                "bachelors_field has no direct adaptive match for the direct-beneficiary role; "
                "row retained because other policy-role signals are stronger"
            )

        if age_signal > 0:
            raw_score += age_signal
            reasons.append("age is compatible with observable policy context")

        score = _scale_role_fit_1_to_5(raw_score, warnings)
        row["rule_filter_raw_score"] = round(raw_score, 3)
        row["match_reasons"] = reasons
        if warnings:
            row["match_warnings"] = warnings
        if role_filter.get("adaptive_profile"):
            profile = role_filter["adaptive_profile"]
            row["adaptive_matching"] = {
                "actor_role_summary": profile.get("actor_role_summary"),
                "sampling_rationale": profile.get("sampling_rationale"),
                "sampling_guidance": profile.get("sampling_guidance"),
            }
        if role_filter.get("observable_eligibility"):
            row["observable_eligibility"] = role_filter["observable_eligibility"]
            row["eligibility_match"] = eligibility_match
        row["alignment_status"] = _alignment_status(score, reasons, warnings)
        return score

    def _slim_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return {key: row.get(key) for key in self.columns if key in row}

    def _get_role_filter(self, entity: Any, entity_type: str) -> dict[str, Any]:
        cache_key = _entity_cache_key(entity, entity_type)
        if cache_key in self._role_filter_cache:
            return self._role_filter_cache[cache_key]

        base_filter: dict[str, Any] = {}
        if not self.adaptive_enabled:
            self._role_filter_cache[cache_key] = base_filter
            return base_filter

        try:
            profile = self._load_or_infer_value_profile(entity, entity_type)
        except Exception as exc:
            base_filter.setdefault("adaptive_profile_error", str(exc))
            self._role_filter_cache[cache_key] = base_filter
            return base_filter

        merged = dict(base_filter)
        merged.setdefault("require_occupation_match", False)
        merged.setdefault("require_education_match", False)
        merged.setdefault("require_bachelors_field_match", False)
        merged.setdefault("require_age_range", False)
        strict_occupation = bool(merged.get("require_occupation_match", False))
        strict_education = bool(merged.get("require_education_match", False))
        strict_field = bool(merged.get("require_bachelors_field_match", False))
        role_family = _infer_role_family_for_sampling(entity, entity_type)
        observable_eligibility = _observable_eligibility_profile(
            entity,
            role_family,
            self.full_config.get("_current_policy_context"),
        )
        domain_signals = _domain_sampling_signals(
            self.full_config.get("_current_policy_context"),
            role_family,
        )
        age_soft_constraints = _merge_age_constraints(
            _age_constraints_from_profile(profile.get("hard_constraints", [])),
            _age_constraints_from_actor_context(entity),
            observable_eligibility.get("age_constraints", {}),
        )
        if observable_eligibility.get("age_alternative_unobservable"):
            age_soft_constraints["alternative_unobservable"] = True
        merged.update({
            "adaptive_profile": profile,
            "observable_eligibility": observable_eligibility,
            "age_soft_constraints": age_soft_constraints,
            "adaptive_education_weights": _weights_from_profile(profile.get("education_values", [])),
            "adaptive_bachelors_field_weights": _merge_weight_maps(
                _weights_from_profile(profile.get("bachelors_field_values", [])),
                domain_signals.get("bachelors_field_weights", {}),
            ),
            "adaptive_occupation_weights": _weights_from_profile(profile.get("occupation_values", [])),
            "occupation_keywords": _dedupe(
                list(base_filter.get("occupation_keywords", []))
                + ([] if strict_occupation else list(profile.get("occupation_keywords", [])))
                + domain_signals.get("occupation_keywords", [])
            ),
            "occupation_exclude_keywords": _dedupe(
                list(base_filter.get("occupation_exclude_keywords", []))
                + list(profile.get("negative_keywords", []))
            ),
            "education_keywords": _dedupe(
                list(base_filter.get("education_keywords", []))
                + ([] if strict_education else [item.get("value", "") for item in profile.get("education_values", [])])
                + domain_signals.get("education_keywords", [])
            ),
            "bachelors_field_keywords": _dedupe(
                list(base_filter.get("bachelors_field_keywords", []))
                + ([] if strict_field else [item.get("value", "") for item in profile.get("bachelors_field_values", [])])
                + domain_signals.get("bachelors_field_keywords", [])
            ),
        })
        self._role_filter_cache[cache_key] = merged
        return merged

    def _load_or_infer_value_profile(self, entity: Any, entity_type: str) -> dict[str, Any]:
        cache_path = self._adaptive_cache_path(entity, entity_type)
        if cache_path and cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                return json.load(f)

        value_profile = self._collect_value_profile()
        role_family = _infer_role_family_for_sampling(entity, entity_type)
        actor_context = {
            "entity_type": entity_type,
            "name": getattr(entity, "name", ""),
            "summary": getattr(entity, "summary", ""),
            "role_family": role_family,
            "constraint_scope": _constraint_scope_for_sampling(role_family),
            "attributes": getattr(entity, "attributes", {}),
            "related_edges": getattr(entity, "related_edges", []),
            "related_nodes": getattr(entity, "related_nodes", []),
        }
        messages = [
            {"role": "system", "content": NEMOTRON_VALUE_PROFILE_SYSTEM},
            {
                "role": "user",
                "content": NEMOTRON_VALUE_PROFILE_USER.format(
                    actor_context=json.dumps(actor_context, ensure_ascii=False),
                    policy_context=json.dumps(self.full_config.get("_current_policy_context", {}), ensure_ascii=False),
                    value_profile=json.dumps(value_profile, ensure_ascii=False),
                ),
            },
        ]
        inferred = json.loads(chat_completion(messages, self.full_config))
        inferred["observed_value_profile"] = value_profile
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(inferred, f, ensure_ascii=False, indent=2)
        return inferred

    def _collect_value_profile(self) -> dict[str, Any]:
        scan_limit = int(self.adaptive_config.get("value_scan_limit", 5000))
        max_education = int(self.adaptive_config.get("max_education_values", 100))
        max_fields = int(self.adaptive_config.get("max_bachelors_field_values", 100))
        max_occupations = int(self.adaptive_config.get("max_occupation_values", 300))
        counters = {
            "education_level": Counter(),
            "bachelors_field": Counter(),
            "occupation": Counter(),
        }
        rows_seen = 0
        for index, row in enumerate(self._iter_rows()):
            if index >= scan_limit:
                break
            rows_seen += 1
            for key, counter in counters.items():
                value = str(row.get(key) or "").strip()
                if value:
                    counter[value] += 1
        return {
            "rows_scanned": rows_seen,
            "education_level": _counter_values(counters["education_level"], max_education),
            "bachelors_field": _counter_values(counters["bachelors_field"], max_fields),
            "occupation": _counter_values(counters["occupation"], max_occupations),
        }

    def _adaptive_cache_path(self, entity: Any, entity_type: str) -> Path | None:
        cache_dir = self.adaptive_config.get("cache_dir") or self.cache_dir
        if not cache_dir:
            return None
        key = _entity_cache_key(entity, entity_type)
        version = str(self.adaptive_config.get("profile_version", "iad_ig_role_scope_v2"))
        safe_version = re.sub(r"[^0-9A-Za-z_.-]+", "_", version).strip("_") or "default"
        return Path(cache_dir) / "role_value_profiles" / f"{safe_version}_{key}.json"


def _actor_context_for_reranker(entity: Any, entity_type: str) -> dict[str, Any]:
    role_family = _infer_role_family_for_sampling(entity, entity_type)
    return {
        "entity_type": entity_type,
        "name": getattr(entity, "name", ""),
        "summary": getattr(entity, "summary", ""),
        "role_family": role_family,
        "constraint_scope": _constraint_scope_for_sampling(role_family),
        "attributes": getattr(entity, "attributes", {}),
        "related_edges": getattr(entity, "related_edges", []),
        "related_nodes": getattr(entity, "related_nodes", []),
    }


def _local_parquet_files(local_data_dir: Any) -> list[str]:
    if not local_data_dir:
        return []
    path = Path(str(local_data_dir))
    if path.is_file() and path.suffix.lower() == ".parquet":
        return [str(path)]
    if not path.exists() or not path.is_dir():
        return []
    return [str(item) for item in sorted(path.glob("*.parquet"))]


def _compact_candidate_for_rerank(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "uuid": row.get("uuid"),
        "occupation": row.get("occupation"),
        "education_level": row.get("education_level"),
        "bachelors_field": row.get("bachelors_field"),
        "age": row.get("age"),
        "sex": row.get("sex"),
        "province": row.get("province"),
        "district": row.get("district"),
        "rule_filter_score": row.get("match_score"),
        "rule_filter_raw_score": row.get("rule_filter_raw_score"),
        "rule_filter_reasons": row.get("match_reasons"),
        "rule_filter_warnings": row.get("match_warnings"),
        "eligibility_match": row.get("eligibility_match"),
        "professional_persona": _clip_text(row.get("professional_persona"), 80),
        "skills_and_expertise": _clip_text(row.get("skills_and_expertise"), 70),
        "career_goals_and_ambitions": _clip_text(row.get("career_goals_and_ambitions"), 70),
        "family_persona": _clip_text(row.get("family_persona"), 50),
    }


def _clip_text(value: Any, limit: int) -> str | None:
    if value in (None, ""):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _clamp_role_fit_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 1
    return max(1, min(score, 5))


def _scale_role_fit_1_to_5(raw_score: float, warnings: list[str]) -> int:
    if raw_score >= 5.0:
        score = 5
    elif raw_score >= 3.5:
        score = 4
    elif raw_score >= 2.0:
        score = 3
    elif raw_score >= 0.75:
        score = 2
    else:
        score = 1
    if warnings and score > 4:
        score = 4
    return score


def _apply_observable_constraint_score_cap(
    row: dict[str, Any],
    role_filter: dict[str, Any],
    role_family: str,
) -> tuple[int, list[str]]:
    """Cap role-fit scores when a seed visibly conflicts with a role-scoped policy constraint."""
    score = _clamp_role_fit_score(row.get("role_fit_score") or row.get("match_score"))
    notes: list[str] = []
    if role_family != "direct_beneficiary":
        return score, notes

    age = _as_int(row.get("age"))
    age_constraints = role_filter.get("age_soft_constraints") or {}
    max_age = age_constraints.get("max")
    min_age = age_constraints.get("min")
    if age is not None and max_age is not None and age > int(max_age):
        if age_constraints.get("alternative_unobservable"):
            notes.append(
                "seed age exceeds the observable age route, but an alternative eligibility route such as PhD timing "
                "is not observable in Nemotron columns; score was not hard-capped"
            )
            return score, notes
        capped = min(score, 2)
        if capped < score:
            notes.append(
                "role_fit_score capped because the seed's observable age exceeds a direct-beneficiary age constraint; "
                "unobserved alternative eligibility such as degree timing remains uncertain"
            )
        score = capped
    if age is not None and min_age is not None and age < int(min_age):
        capped = min(score, 2)
        if capped < score:
            notes.append(
                "role_fit_score capped because the seed's observable age is below a direct-beneficiary age constraint"
            )
        score = capped
    return score, notes


def _matching_profile_for_prompt(role_filter: dict[str, Any]) -> dict[str, Any]:
    profile = dict(role_filter.get("adaptive_profile") or {})
    observable = role_filter.get("observable_eligibility")
    if observable:
        profile["observable_eligibility"] = observable
    return profile


def _domain_sampling_signals(policy_context: dict[str, Any] | None, role_family: str) -> dict[str, list[str]]:
    if role_family not in {"direct_beneficiary", "evaluator", "mentor", "pipeline_actor"}:
        return {}
    context_text = json.dumps(policy_context or {}, ensure_ascii=False).lower()
    if not any(
        marker in context_text
        for marker in (
            "science",
            "technology",
            "research",
            "scientist",
            "engineering",
            "과학",
            "기술",
            "연구",
            "공학",
        )
    ):
        return {}
    return {
        "occupation_keywords": [
            "연구",
            "연구원",
            "연구자",
            "과학",
            "공학",
            "기술",
            "기술자",
            "개발",
            "데이터",
            "소프트웨어",
            "프로그래머",
            "컴퓨터",
            "정보통신",
            "인공지능",
            "ai",
            "반도체",
            "researcher",
            "scientist",
            "engineer",
            "developer",
        ],
        "education_keywords": [
            "대학원",
            "박사",
            "석사",
        ],
        "bachelors_field_keywords": [
            "정보통신기술",
            "공학·제조·건설",
            "자연과학·수학",
        ],
        "bachelors_field_weights": {
            "정보통신기술": 2.75,
            "공학·제조·건설": 2.25,
            "자연과학·수학": 2.25,
        },
    }


def _observable_eligibility_profile(
    entity: Any,
    role_family: str,
    policy_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Map policy eligibility text to Nemotron-observable and unobservable signals."""
    if role_family != "direct_beneficiary":
        return {}

    source_texts = _eligibility_source_texts(entity, policy_context)
    if not source_texts:
        return {}

    joined = " ".join(source_texts)
    age_constraints = _age_constraints_from_text(joined)
    unobservable_constraints = _unobservable_eligibility_constraints_from_text(joined)
    indirect_signals = _indirect_eligibility_signals_from_text(joined)
    logic = "or" if _has_alternative_logic(joined) else "and_or_unspecified"
    age_alternative_unobservable = bool(
        age_constraints
        and unobservable_constraints
        and logic == "or"
    )

    column_mappings: list[dict[str, Any]] = []
    if age_constraints:
        column_mappings.append({
            "policy_field": "age eligibility route",
            "nemotron_columns": ["age"],
            "observable": True,
            "constraint": _age_constraint_label(age_constraints),
        })
    for item in unobservable_constraints:
        column_mappings.append({
            "policy_field": item["field"],
            "nemotron_columns": [],
            "observable": False,
            "constraint": item["constraint"],
        })
    for item in indirect_signals:
        column_mappings.append({
            "policy_field": item["field"],
            "nemotron_columns": item["columns"],
            "observable": "indirect",
            "constraint": item["constraint"],
        })

    return {
        "source": "policy_context_and_kg",
        "logic": logic,
        "source_texts": source_texts[:6],
        "age_constraints": age_constraints,
        "age_alternative_unobservable": age_alternative_unobservable,
        "unobservable_constraints": unobservable_constraints,
        "indirect_signals": indirect_signals,
        "column_mappings": column_mappings,
        "filtering_policy": (
            "soft_score_only: use observable columns for bonuses/warnings; do not hard reject when an OR-route "
            "depends on unobserved fields such as PhD timing"
        ),
    }


def _eligibility_source_texts(entity: Any, policy_context: dict[str, Any] | None) -> list[str]:
    texts: list[str] = []
    if isinstance(policy_context, dict):
        for key in ("eligibility_logic", "target_population", "primary_beneficiaries"):
            value = policy_context.get(key)
            if isinstance(value, list):
                texts.extend(str(item) for item in value if item)
            elif value:
                texts.append(str(value))

    attributes = getattr(entity, "attributes", {}) or {}
    for key in ("eligibility_criteria", "exclusion_criteria", "role_scope"):
        if attributes.get(key):
            texts.append(str(attributes[key]))

    for edge in getattr(entity, "related_edges", []) or []:
        if edge.get("direction") not in {None, "outgoing"}:
            continue
        edge_text = " ".join(
            str(edge.get(key) or "")
            for key in ("fact", "activation_condition", "execution_constraint", "condition_text", "source_text")
        )
        if _looks_like_eligibility_text(edge_text):
            texts.append(edge_text)

    return _dedupe([re.sub(r"\s+", " ", text).strip() for text in texts if str(text).strip()])


def _looks_like_eligibility_text(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
        marker in lowered
        for marker in (
            "eligible",
            "eligibility",
            "ineligible",
            "qualification",
            "qualifies",
            "applicant",
            "applicants",
            "selection target",
            "target population",
            "phd",
            "doctor",
            "postdoctoral",
            "years of age",
            "박사",
            "자격",
            "지원 대상",
            "신청 자격",
            "선정 대상",
            "이내",
            "이하",
            "미만",
        )
    )


def _has_alternative_logic(text: str) -> bool:
    lowered = str(text or "").lower()
    return bool(re.search(r"\b(or|either|one of)\b", lowered)) or any(
        marker in lowered for marker in ("또는", "혹은", "거나", "이나")
    )


def _unobservable_eligibility_constraints_from_text(text: str) -> list[dict[str, str]]:
    lowered = str(text or "").lower()
    constraints: list[dict[str, str]] = []
    if any(marker in lowered for marker in ("phd", "doctor", "doctoral", "박사")):
        numbers = re.findall(r"\d+", lowered)
        if numbers and any(marker in lowered for marker in ("within", "last", "past", "이내", "취득")):
            constraints.append({
                "field": "phd_completion_window",
                "constraint": f"PhD completion timing within {numbers[0]} years",
                "reason": "Nemotron has education level but no PhD award date column.",
            })
        else:
            constraints.append({
                "field": "phd_status_or_degree_timing",
                "constraint": "PhD/postdoctoral status or timing is required by policy context",
                "reason": "Nemotron does not directly expose PhD award date or verified postdoctoral status.",
            })
    return constraints


def _indirect_eligibility_signals_from_text(text: str) -> list[dict[str, Any]]:
    lowered = str(text or "").lower()
    signals: list[dict[str, Any]] = []
    if any(
        marker in lowered
        for marker in (
            "postdoctoral",
            "researcher",
            "scientist",
            "science",
            "engineering",
            "stem",
            "박사후",
            "연구자",
            "연구원",
            "과학자",
            "과학",
            "공학",
            "이공",
        )
    ):
        signals.append({
            "field": "researcher_or_postdoctoral_signal",
            "columns": ["occupation", "professional_persona", "skills_and_expertise", "education_level"],
            "constraint": "researcher, scientist, doctoral, or postdoctoral background appears relevant",
        })
    return signals


def _age_constraint_label(age_constraints: dict[str, Any]) -> str:
    parts = []
    if age_constraints.get("min") is not None:
        parts.append(f"age >= {age_constraints['min']}")
    if age_constraints.get("max") is not None:
        parts.append(f"age <= {age_constraints['max']}")
    return " and ".join(parts) or "age constraint"


def _observable_eligibility_signal(
    row: dict[str, Any],
    role_filter: dict[str, Any],
    role_family: str,
    professional_text: str,
    broad_text: str,
) -> tuple[float, list[str], list[str], dict[str, Any]]:
    if role_family != "direct_beneficiary":
        return 0.0, [], [], {}
    profile = role_filter.get("observable_eligibility") or {}
    if not profile:
        return 0.0, [], [], {}

    score = 0.0
    reasons: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []

    age = _as_int(row.get("age"))
    age_constraints = profile.get("age_constraints") or {}
    max_age = age_constraints.get("max")
    min_age = age_constraints.get("min")
    age_pass = None
    if age is not None and (max_age is not None or min_age is not None):
        age_pass = True
        if max_age is not None and age > int(max_age):
            age_pass = False
        if min_age is not None and age < int(min_age):
            age_pass = False
        checks.append({
            "field": "age",
            "value": age,
            "constraint": _age_constraint_label(age_constraints),
            "passed": age_pass,
            "observable": True,
        })
        if age_pass:
            score += 0.75
            reasons.append("matches observable age eligibility route")
        elif profile.get("age_alternative_unobservable"):
            warnings.append(
                "age misses the observable age route, but the policy has an OR eligibility route that Nemotron cannot verify"
            )
        else:
            warnings.append("age misses an observable direct-beneficiary eligibility route")
    elif age_constraints:
        checks.append({
            "field": "age",
            "value": None,
            "constraint": _age_constraint_label(age_constraints),
            "passed": None,
            "observable": True,
        })
        warnings.append("age eligibility route exists but seed age is missing")

    if profile.get("indirect_signals"):
        signal_keywords = [
            "research",
            "researcher",
            "scientist",
            "science",
            "doctoral",
            "doctor",
            "phd",
            "engineer",
            "연구",
            "연구원",
            "연구자",
            "과학",
            "공학",
            "기술",
            "기술자",
            "박사",
            "대학원",
            "정보통신",
            "소프트웨어",
            "프로그래머",
            "데이터",
            "컴퓨터",
        ]
        if _contains_any(" ".join([professional_text, broad_text]), signal_keywords):
            score += 0.25
            reasons.append("matches indirect researcher/postdoctoral role signal")
            checks.append({
                "field": "researcher_or_postdoctoral_signal",
                "passed": True,
                "observable": "indirect",
            })
        else:
            checks.append({
                "field": "researcher_or_postdoctoral_signal",
                "passed": False,
                "observable": "indirect",
            })

    status = "uncertain"
    has_true_check = any(item.get("passed") is True for item in checks)
    has_observable_condition = any(item.get("observable") is True for item in checks)
    if has_true_check and age_pass is None and has_observable_condition:
        status = "partial_observable_match"
    elif has_true_check:
        status = "observable_match"
    if age_pass is False and profile.get("age_alternative_unobservable"):
        status = "observable_age_miss_with_unobserved_or_route"
    elif any(item.get("passed") is False and item.get("observable") is True for item in checks):
        status = "observable_mismatch"

    return score, reasons, warnings, {
        "status": status,
        "checks": checks,
        "unobservable_constraints": profile.get("unobservable_constraints") or [],
        "logic": profile.get("logic"),
    }


def _age_constraints_from_profile(items: list[dict[str, Any]]) -> dict[str, int]:
    constraints: dict[str, int] = {}
    for item in items or []:
        field = str(item.get("field") or "").lower()
        text = " ".join(
            str(value or "")
            for value in (item.get("constraint"), item.get("source"), item.get("reason"))
        ).lower()
        if "age" not in field and "age" not in text and "years old" not in text and "세" not in text:
            continue
        numbers = [int(value) for value in re.findall(r"\d+", text)]
        if not numbers:
            continue
        number = numbers[0]
        if any(marker in text for marker in ("under", "below", "younger", "less than", "maximum", "max", "이하", "미만", "≤", "<=")):
            constraints["max"] = min(constraints.get("max", number), number)
        elif any(marker in text for marker in ("over", "older", "at least", "minimum", "min", "이상", "초과", "≥", ">=")):
            constraints["min"] = max(constraints.get("min", number), number)
    return constraints


def _age_constraints_from_actor_context(entity: Any) -> dict[str, int]:
    text_parts = [
        getattr(entity, "name", ""),
        getattr(entity, "summary", ""),
    ]
    attributes = getattr(entity, "attributes", {}) or {}
    text_parts.extend(str(value or "") for value in attributes.values())
    for edge in getattr(entity, "related_edges", []) or []:
        text_parts.extend(
            str(edge.get(key) or "")
            for key in ("fact", "activation_condition", "execution_constraint", "or_else_text", "source_text")
        )
    return _age_constraints_from_text(" ".join(text_parts))


def _age_constraints_from_text(text: str) -> dict[str, int]:
    lowered = str(text or "").lower()
    constraints: dict[str, int] = {}
    for match in re.finditer(
        r"(?:under|below|younger than|less than|max(?:imum)?|not older than)\s+(?:the\s+)?(?:age\s+of\s+)?(\d+)",
        lowered,
    ):
        number = int(match.group(1))
        constraints["max"] = min(constraints.get("max", number), number)
    for match in re.finditer(
        r"(\d+)\s*(?:years?\s*old|세)?\s*(under|below|younger|or younger|less than|maximum|max|이하|미만|까지)",
        lowered,
    ):
        number = int(match.group(1))
        constraints["max"] = min(constraints.get("max", number), number)
    for match in re.finditer(
        r"(?:age|aged)\s*(\d+)\s*(?:or\s+)?(?:under|below|younger|younger than|less)",
        lowered,
    ):
        number = int(match.group(1))
        constraints["max"] = min(constraints.get("max", number), number)
    for match in re.finditer(r"(under|below|younger than|less than|maximum|max|만|이하|미만)\s*(\d+)", lowered):
        number = int(match.group(2))
        constraints["max"] = min(constraints.get("max", number), number)
    for match in re.finditer(
        r"(?:over|above|older than|at least|min(?:imum)?|not younger than)\s+(?:the\s+)?(?:age\s+of\s+)?(\d+)",
        lowered,
    ):
        number = int(match.group(1))
        constraints["min"] = max(constraints.get("min", number), number)
    for match in re.finditer(
        r"(\d+)\s*(?:years?\s*old|세)?\s*(over|older|or older|at least|minimum|min|이상|초과|부터)",
        lowered,
    ):
        number = int(match.group(1))
        constraints["min"] = max(constraints.get("min", number), number)
    for match in re.finditer(r"(over|older than|at least|minimum|min|이상|초과)\s*(\d+)", lowered):
        number = int(match.group(2))
        constraints["min"] = max(constraints.get("min", number), number)
    return constraints


def _merge_age_constraints(*items: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for item in items:
        if not item:
            continue
        if item.get("max") is not None:
            merged["max"] = min(merged.get("max", int(item["max"])), int(item["max"]))
        if item.get("min") is not None:
            merged["min"] = max(merged.get("min", int(item["min"])), int(item["min"]))
    return merged


def _age_signal(row: dict[str, Any], role_filter: dict[str, Any]) -> tuple[float, str | None]:
    age = _as_int(row.get("age"))
    constraints = role_filter.get("age_soft_constraints") or {}
    if age is None or not constraints:
        return 0.0, None
    min_age = constraints.get("min")
    max_age = constraints.get("max")
    if min_age is not None and age < int(min_age):
        return 0.0, "age is below an observable policy-context range; retained for LLM judgment"
    if max_age is not None and age > int(max_age):
        if constraints.get("alternative_unobservable"):
            return (
                0.0,
                "age does not satisfy the observable age eligibility route; retained because another policy route "
                "is not observable in Nemotron columns",
            )
        return 0.0, "age is above an observable policy-context range; retained for LLM judgment"
    return 0.5, None


def _role_keywords(entity: Any, entity_type: str, role_filter: dict[str, Any]) -> list[str]:
    configured = []
    for key in ("occupation_keywords",):
        configured.extend(role_filter.get(key, []))

    raw_parts = [
        entity_type,
        getattr(entity, "name", ""),
    ]
    raw_text = " ".join(raw_parts)

    tokens = [
        token
        for token in re.split(r"[^0-9A-Za-z가-힣]+", raw_text)
        if len(token) >= 2
    ]
    return _dedupe(configured + tokens)


def _infer_role_family_for_sampling(entity: Any, entity_type: str) -> str:
    attributes = getattr(entity, "attributes", {}) or {}
    configured = attributes.get("role_family") or attributes.get("policy_role_family")
    if configured:
        return _normalize_role_family(configured)
    text = " ".join(
        str(value or "")
        for value in (
            entity_type,
            getattr(entity, "name", ""),
            getattr(entity, "summary", ""),
            getattr(entity, "uuid", ""),
        )
    ).lower()
    if any(term in text for term in ("evaluation", "panel", "reviewer", "committee", "rb")):
        return "evaluator"
    if any(term in text for term in ("institution", "university", "institute", "host")):
        return "host_organization"
    if any(term in text for term in ("ministry", "foundation", "agency", "program officer", "administrator", "authority", "approval")):
        return "policy_administrator"
    if any(term in text for term in ("mentor", "mentoring")):
        return "mentor"
    if any(term in text for term in ("student", "candidate", "pipeline", "trainee")):
        return "pipeline_actor"
    if any(term in text for term in ("postdoctoral", "researcher", "fellow", "beneficiary", "applicant")):
        return "direct_beneficiary"
    return "affected_actor"


def _normalize_role_family(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"direct_beneficiary", "beneficiary", "applicant", "fellow"}:
        return "direct_beneficiary"
    if text in {"evaluator", "evaluation", "reviewer", "committee_member", "panel_member"}:
        return "evaluator"
    if text in {"host_organization", "host", "institution", "research_institution", "research_organization"}:
        return "host_organization"
    if text in {"policy_administrator", "administrator", "program_officer", "funding_organization", "funder", "funding_agency"}:
        return "policy_administrator"
    if text in {"mentor", "mentoring"}:
        return "mentor"
    if text in {"pipeline_actor", "pipeline", "student", "trainee"}:
        return "pipeline_actor"
    if text in {"affected_actor", "affected", "stakeholder"}:
        return "affected_actor"
    return text or "affected_actor"


def _constraint_scope_for_sampling(role_family: str) -> str:
    scopes = {
        "direct_beneficiary": "Use observable beneficiary eligibility only when directly supported by policy and seed fields.",
        "evaluator": "Use expertise and evaluation legitimacy; do not apply beneficiary eligibility gates.",
        "host_organization": "Use hosting and research administration capacity; do not apply individual eligibility gates.",
        "policy_administrator": "Use governance, implementation, budget, and accountability constraints.",
        "mentor": "Use expertise, mentoring capacity, and network support constraints.",
        "pipeline_actor": "Use adjacent participation and future eligibility relevance without claiming direct eligibility.",
    }
    return scopes.get(role_family, "Use only constraints scoped to this KG actor role.")


def _lower_terms(values: list[Any]) -> tuple[str, ...]:
    return tuple(
        str(value).strip().lower()
        for value in values or []
        if str(value).strip()
    )


def _lower_weight_map(weights: dict[str, float]) -> dict[str, float]:
    return {
        str(key).strip().lower(): float(value)
        for key, value in (weights or {}).items()
        if str(key).strip()
    }


def _contains_term(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _exact_weight(text: str, weights: dict[str, float]) -> float:
    return weights.get(str(text or "").strip().lower(), 0.0)


def _column_value(data: dict[str, list[Any]], key: str, index: int) -> Any:
    values = data.get(key)
    if values is None or index >= len(values):
        return None
    return values[index]


def _stable_tie_breaker(value: str) -> float:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(0xFFFFFFFFFFFF)


def _age_passes(age: int | None, constraints: dict[str, Any]) -> bool:
    if age is None or not constraints:
        return False
    if constraints.get("min") is not None and age < int(constraints["min"]):
        return False
    if constraints.get("max") is not None and age > int(constraints["max"]):
        return False
    return True


def _string_series(df: Any, key: str, pd: Any) -> Any:
    if key not in df.columns:
        return pd.Series("", index=df.index)
    return df[key].fillna("").astype(str).str.lower()


def _series_contains_any(series: Any, keywords: list[str], pd: Any) -> Any:
    terms = [str(keyword).strip().lower() for keyword in keywords or [] if str(keyword).strip()]
    if not terms:
        return pd.Series(False, index=series.index)
    pattern = "|".join(re.escape(term) for term in terms)
    return series.str.contains(pattern, regex=True, na=False)


def _series_exact_weights(series: Any, weights: dict[str, float], pd: Any) -> Any:
    if not weights:
        return pd.Series(0.0, index=series.index)
    normalized = {
        str(key).strip().lower(): float(value)
        for key, value in weights.items()
        if str(key).strip()
    }
    if not normalized:
        return pd.Series(0.0, index=series.index)
    return series.str.strip().map(normalized).fillna(0.0).astype(float)


def _numeric_series(df: Any, key: str, pd: Any) -> Any:
    if key not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    return pd.to_numeric(df[key], errors="coerce")


def _series_age_signal(df: Any, role_filter: dict[str, Any], pd: Any) -> Any:
    result = pd.Series(0.0, index=df.index)
    constraints = role_filter.get("age_soft_constraints") or {}
    if not constraints:
        return result
    age = _numeric_series(df, "age", pd)
    passed = age.notna()
    if constraints.get("min") is not None:
        passed &= age >= float(constraints["min"])
    if constraints.get("max") is not None:
        passed &= age <= float(constraints["max"])
    result.loc[passed] = 0.5
    return result


def _series_observable_eligibility_signal(
    df: Any,
    role_filter: dict[str, Any],
    role_family: str,
    combined_light_text: Any,
    pd: Any,
) -> Any:
    result = pd.Series(0.0, index=df.index)
    if role_family != "direct_beneficiary":
        return result
    profile = role_filter.get("observable_eligibility") or {}
    if not profile:
        return result

    age_constraints = profile.get("age_constraints") or {}
    if age_constraints:
        age = _numeric_series(df, "age", pd)
        passed = age.notna()
        if age_constraints.get("min") is not None:
            passed &= age >= float(age_constraints["min"])
        if age_constraints.get("max") is not None:
            passed &= age <= float(age_constraints["max"])
        result.loc[passed] += 0.75

    if profile.get("indirect_signals"):
        signal_keywords = [
            "research",
            "researcher",
            "scientist",
            "science",
            "doctoral",
            "doctor",
            "phd",
            "engineer",
            "연구",
            "연구원",
            "연구자",
            "과학",
            "공학",
            "기술",
            "기술자",
            "박사",
            "대학원",
            "정보통신",
            "소프트웨어",
            "프로그래머",
            "데이터",
            "컴퓨터",
        ]
        result.loc[_series_contains_any(combined_light_text, signal_keywords, pd)] += 0.25
    return result


def _contains_any(value: Any, keywords: list[str]) -> bool:
    text = str(value or "").lower()
    return any(str(keyword).lower() in text for keyword in keywords if keyword)


def _value_weight(value: Any, weights: dict[str, float]) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    if text in weights:
        return float(weights[text])
    lowered = text.lower()
    for key, weight in weights.items():
        if str(key).lower() in lowered:
            return float(weight)
    return 0.0


def _weights_from_profile(items: list[dict[str, Any]]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in items or []:
        value = str(item.get("value") or "").strip()
        if not value:
            continue
        weights[value] = max(weights.get(value, 0.0), _relevance_weight(item.get("relevance")))
    return weights


def _merge_weight_maps(*maps: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for weights in maps:
        for key, value in (weights or {}).items():
            text = str(key).strip()
            if not text:
                continue
            merged[text] = max(merged.get(text, 0.0), float(value))
    return merged


def _relevance_weight(relevance: Any) -> float:
    normalized = str(relevance or "").strip().lower()
    if normalized == "direct":
        return 3.0
    if normalized == "adjacent":
        return 2.0
    if normalized == "broad":
        return 1.0
    if normalized == "uncertain":
        return 0.5
    return 0.5


def _counter_values(counter: Counter, limit: int) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in counter.most_common(limit)
    ]


def _entity_cache_key(entity: Any, entity_type: str) -> str:
    raw = "|".join([
        str(entity_type or ""),
        str(getattr(entity, "uuid", "")),
        str(getattr(entity, "name", "")),
        str(getattr(entity, "summary", ""))[:240],
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _alignment_status(score: int, reasons: list[str], warnings: list[str]) -> str:
    if warnings:
        return "broad_match_with_warning"
    if score >= 4:
        return "strong_match"
    if score >= 3:
        return "broad_match"
    return "minimal_match"


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
