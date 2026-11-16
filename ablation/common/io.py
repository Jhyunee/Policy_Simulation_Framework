"""Shared I/O + aggregation utilities for ablation studies.

- Country/Nemotron resolution from policy_key (kor/ssf, singapore/nais, ...)
- Run directory layout under outputs/ablation/<condition>/<policy>/<ts>_<tag>/
- Scenario-level mean + MAPE aggregation (lifted from baseline/predict.py).
"""
from __future__ import annotations

import importlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ABLATION_OUTPUTS_ROOT = REPO_ROOT / "outputs" / "ablation"
POLICY_TARGETS_PATH = REPO_ROOT / "config" / "policy_targets.json"

# Country prefix → (values_module, hf_dataset_dir). Mirrors main.py:_COUNTRY_TO_NEMOTRON.
_COUNTRY_TO_NEMOTRON = {
    "kor":       ("nemotron_kor", "Nemotron-Personas-Korea"),
    "singapore": ("nemotron_sg",  "Nemotron-Personas-Singapore"),
    "usa":       ("nemotron_usa", "Nemotron-Personas-USA"),
    "fr":        ("nemotron_fr",  "Nemotron-Personas-France"),
    "jp":        ("nemotron_jp",  "Nemotron-Personas-Japan"),
}


def policy_to_country(policy_key: str) -> str:
    """`kor/ssf` -> `kor`; raises if prefix is unknown."""
    country = policy_key.strip("/").split("/", 1)[0]
    if country not in _COUNTRY_TO_NEMOTRON:
        raise ValueError(
            f"Unknown country prefix '{country}' from policy_key='{policy_key}'. "
            f"Known: {list(_COUNTRY_TO_NEMOTRON.keys())}"
        )
    return country


def configure_nemotron_env(policy_key: str) -> tuple[str, str]:
    """Set NEMOTRON_VALUES_MODULE env var and return (parquet_glob, values_module).

    Must be called BEFORE importing persona.* modules that read PERSONA_FIELDS
    from the country module at import time.
    """
    country = policy_to_country(policy_key)
    pkg, hf_dir = _COUNTRY_TO_NEMOTRON[country]
    parquet_glob = str(REPO_ROOT / "dataset" / "processed" / pkg / hf_dir / "data" / "*.parquet")
    values_module = f"dataset.processed.{pkg}"
    os.environ["NEMOTRON_VALUES_MODULE"] = values_module
    return parquet_glob, values_module


def load_persona_fields(policy_key: str) -> tuple[list[str], list[str]]:
    """Load country-specific PERSONA_FIELDS (SELECT exprs + output column names).

    Returns:
        (sql_exprs, output_names) — sql_exprs go into a DuckDB SELECT,
        output_names are the resulting column names downstream code reads.
    """
    country = policy_to_country(policy_key)
    pkg, _ = _COUNTRY_TO_NEMOTRON[country]
    fields_mod = importlib.import_module(f"dataset.processed.{pkg}.fields")
    sql_exprs: list[str] = list(fields_mod.PERSONA_FIELDS)
    output_names = [expr.split(" AS ")[-1].strip() for expr in sql_exprs]
    return sql_exprs, output_names


def make_run_dir(condition: str, policy_key: str, tag: str | None) -> Path:
    """outputs/ablation/<condition>/<policy_underscore>/<ts>_<tag>/"""
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    suffix = f"_{tag}" if tag else ""
    safe_policy = policy_key.replace("/", "_")
    run_dir = ABLATION_OUTPUTS_ROOT / condition / safe_policy / f"{ts}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_policy_targets() -> dict:
    with open(POLICY_TARGETS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_gt(gt: Any) -> dict[str, float]:
    if isinstance(gt, dict):
        return {k: float(v) for k, v in gt.items()}
    return {"value": float(gt)}


def aggregate_scenario(
    persona_predictions: list[dict[str, float | None]],
    gt: dict[str, float],
    min_valid_personas: int = 3,
) -> dict:
    """Compute scenario mean + MAPE per target key. Mirrors baseline/predict.py.

    persona_predictions: list of dicts mapping target_key -> float or None.
    """
    target_keys = list(gt.keys())
    per_key_valid: dict[str, list[float]] = {k: [] for k in target_keys}
    for pred in persona_predictions:
        for k in target_keys:
            v = pred.get(k)
            if v is not None:
                per_key_valid[k].append(v)

    n_valid_per_key = {k: len(per_key_valid[k]) for k in target_keys}
    n_valid_overall = min(n_valid_per_key.values()) if n_valid_per_key else 0
    skipped = n_valid_overall < min_valid_personas

    scenario_mean: dict[str, float | None] = {}
    scenario_mape: dict[str, float | None] = {}
    for k in target_keys:
        vs = per_key_valid[k]
        if not vs or skipped:
            scenario_mean[k] = None
            scenario_mape[k] = None
            continue
        mean = sum(vs) / len(vs)
        scenario_mean[k] = mean
        scenario_mape[k] = abs(mean - gt[k]) / abs(gt[k]) * 100.0 if gt[k] != 0 else None

    return {
        "scenario_mean_prediction": scenario_mean,
        "scenario_mape": scenario_mape,
        "n_valid_personas_per_key": n_valid_per_key,
        "skipped": skipped,
    }


def aggregate_final_mape(scenario_records: list[dict], gt: dict[str, float]) -> tuple[dict[str, float | None], int]:
    """Average per-scenario MAPE across non-skipped scenarios per target key."""
    target_keys = list(gt.keys())
    final_mape: dict[str, float | None] = {}
    n_skipped = sum(1 for r in scenario_records if r.get("skipped"))
    for k in target_keys:
        vals = [r["scenario_mape"][k] for r in scenario_records if r.get("scenario_mape", {}).get(k) is not None]
        final_mape[k] = sum(vals) / len(vals) if vals else None
    return final_mape, n_skipped


def save_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def merge_usage(*usages: dict[str, int]) -> dict[str, int]:
    out = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in usages:
        if not u:
            continue
        for k in out:
            out[k] += int(u.get(k, 0) or 0)
    return out
