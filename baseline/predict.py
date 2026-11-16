"""Baseline experiment: policy summary -> 5 personas predict target -> scenario mean -> MAPE.

Reuses the existing POLICY_READER prompt for the per-policy summarization (one LLM call per policy).
For each of the first 200 scenarios in the stage1 simulation_personas.json, dispatches 5 persona
prediction calls. Concurrent dispatch is batched at the scenario level (default 25 scenarios per
wave = 125 in-flight calls). Per-call token usage is recorded.

Run:
    python -m baseline.predict --config-path config/base.yaml --run-tag baseline_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from baseline.policy_mapping import (
    POLICY_MAPPING,
    REPO_ROOT,
    all_policy_keys,
    resolve_md_path,
    resolve_personas_path,
)
from baseline.prompts import (
    BASELINE_PERSONA_PREDICT_SYSTEM,
    BASELINE_PERSONA_PREDICT_USER,
)
from llm.client import chat_completion_with_usage
from models.prompts import POLICY_READER_SYSTEM, POLICY_READER_USER

load_dotenv(REPO_ROOT / ".env")

POLICY_TARGETS_PATH = REPO_ROOT / "config" / "policy_targets.json"
BASELINE_OUTPUTS_ROOT = REPO_ROOT / "outputs" / "baseline"

# Fields stripped from a persona before it is shown to the LLM — these are
# KG-derived and would leak graph structure into the baseline.
KG_DERIVED_FIELDS = {"representative_type", "stakeholder_type", "persona_id"}


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_policy_targets() -> dict:
    with open(POLICY_TARGETS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_gt(gt: Any) -> dict[str, float]:
    """Convert a raw gt entry (number or dict) to {key: float}."""
    if isinstance(gt, dict):
        return {k: float(v) for k, v in gt.items()}
    return {"value": float(gt)}


def make_baseline_run_dir(model: str, tag: str | None) -> Path:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    suffix = f"_{tag}" if tag else ""
    safe_model = model.replace("/", "_")
    run_dir = BASELINE_OUTPUTS_ROOT / f"{ts}_{safe_model}_result{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def scrub_persona(persona: dict) -> dict:
    """Return a copy of the persona dict with KG-derived fields removed."""
    return {k: v for k, v in persona.items() if k not in KG_DERIVED_FIELDS}


def summarize_policy(md_text: str, config: dict) -> tuple[dict, dict]:
    """Run POLICY_READER once on the .md text. Returns (summary_dict, usage_dict)."""
    messages = [
        {"role": "system", "content": POLICY_READER_SYSTEM},
        {"role": "user", "content": POLICY_READER_USER.format(pdf_text=md_text)},
    ]
    content, usage = chat_completion_with_usage(messages, config)
    return json.loads(content), usage


def predict_one(
    policy_summary: dict,
    persona: dict,
    simulation_target: str,
    target_keys: list[str],
    config: dict,
) -> dict:
    """Single persona prediction. Returns {prediction, evidence, usage} or {error, usage}."""
    persona_for_llm = scrub_persona(persona)
    prediction_keys_hint = ", ".join(f'"{k}": <number>' for k in target_keys)
    messages = [
        {"role": "system", "content": BASELINE_PERSONA_PREDICT_SYSTEM},
        {
            "role": "user",
            "content": BASELINE_PERSONA_PREDICT_USER.format(
                policy_summary=json.dumps(policy_summary, ensure_ascii=False, indent=2),
                persona=json.dumps(persona_for_llm, ensure_ascii=False, indent=2),
                simulation_target=simulation_target,
                target_keys=target_keys,
                prediction_keys_hint=prediction_keys_hint,
            ),
        },
    ]
    try:
        content, usage = chat_completion_with_usage(messages, config)
        parsed = json.loads(content)
        prediction = parsed.get("prediction") or {}
        evidence = parsed.get("evidence") or ""
        # Coerce numeric strings to floats; missing keys -> None
        cleaned_pred: dict[str, float | None] = {}
        for k in target_keys:
            v = prediction.get(k)
            if v is None:
                cleaned_pred[k] = None
                continue
            try:
                cleaned_pred[k] = float(v)
            except (TypeError, ValueError):
                cleaned_pred[k] = None
        return {"prediction": cleaned_pred, "evidence": evidence, "usage": usage}
    except Exception as exc:  # noqa: BLE001 — driver-level, log + continue
        return {
            "prediction": {k: None for k in target_keys},
            "evidence": "",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "error": f"{type(exc).__name__}: {exc}",
        }


def aggregate_scenario(
    persona_results: list[dict],
    gt: dict[str, float],
    min_valid_personas: int = 3,
) -> dict:
    """Compute scenario mean + MAPE per target key. Returns scenario summary dict."""
    target_keys = list(gt.keys())
    per_key_valid: dict[str, list[float]] = {k: [] for k in target_keys}
    for r in persona_results:
        for k in target_keys:
            v = r["prediction"].get(k)
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


def run_policy(
    policy_key: str,
    config: dict,
    config_path: str,
    max_scenarios: int,
    batch_size: int,
    policy_run_dir: Path,
) -> dict:
    """Execute baseline for one policy. Writes results.json + policy_summary.json under policy_run_dir."""
    print(f"\n=== {policy_key} ===")
    md_path = resolve_md_path(policy_key)
    personas_path = resolve_personas_path(policy_key)
    print(f"  md:       {md_path}")
    print(f"  personas: {personas_path}")

    md_text = md_path.read_text(encoding="utf-8")

    targets_all = load_policy_targets()
    target_info = targets_all[policy_key]
    simulation_target = target_info["simulation_target"]
    gt = normalize_gt(target_info["gt"])
    target_keys = list(gt.keys())

    print(f"  simulation_target: {simulation_target}")
    print(f"  gt: {gt}")

    # --- Policy summary (one call) ---
    t0 = time.time()
    summary, summary_usage = summarize_policy(md_text, config)
    print(
        f"  summary done in {time.time()-t0:.1f}s — tokens: prompt={summary_usage['prompt_tokens']} "
        f"completion={summary_usage['completion_tokens']} total={summary_usage['total_tokens']}"
    )
    (policy_run_dir / "policy_summary.json").write_text(
        json.dumps(
            {
                "policy_key": policy_key,
                "md_path": str(md_path),
                "summary": summary,
                "usage": summary_usage,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --- Load scenarios ---
    sim = json.loads(personas_path.read_text(encoding="utf-8"))
    scenarios = sorted(sim["scenarios"], key=lambda s: s["scenario_id"])[:max_scenarios]
    print(f"  loaded {len(scenarios)} scenarios (5 personas each)")

    # --- Per-scenario aggregation containers ---
    scenario_records: list[dict] = []

    pred_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # --- Wave loop: batch_size scenarios per wave -> 5*batch_size concurrent calls ---
    concurrency = batch_size * 5
    wave_count = (len(scenarios) + batch_size - 1) // batch_size
    for wave_idx in range(wave_count):
        wave_start = wave_idx * batch_size
        wave_scens = scenarios[wave_start : wave_start + batch_size]
        wave_t0 = time.time()
        # task key: (scen_array_index_within_wave, persona_idx)
        tasks: dict[tuple[int, int], dict] = {}
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            future_to_key = {}
            for s_local, scen in enumerate(wave_scens):
                for p_idx, persona in enumerate(scen["participants"]):
                    fut = ex.submit(
                        predict_one,
                        summary,
                        persona,
                        simulation_target,
                        target_keys,
                        config,
                    )
                    future_to_key[fut] = (s_local, p_idx)
            for fut in as_completed(future_to_key):
                key = future_to_key[fut]
                tasks[key] = fut.result()

        # Build scenario records for this wave
        for s_local, scen in enumerate(wave_scens):
            persona_results = []
            for p_idx, persona in enumerate(scen["participants"]):
                r = tasks[(s_local, p_idx)]
                persona_results.append(
                    {
                        "persona_idx": p_idx,
                        "persona_id": persona.get("persona_id"),
                        "prediction": r["prediction"],
                        "evidence": r["evidence"],
                        "usage": r["usage"],
                        **({"error": r["error"]} if "error" in r else {}),
                    }
                )
                u = r["usage"]
                pred_tokens["prompt_tokens"] += u.get("prompt_tokens", 0)
                pred_tokens["completion_tokens"] += u.get("completion_tokens", 0)
                pred_tokens["total_tokens"] += u.get("total_tokens", 0)

            agg = aggregate_scenario(persona_results, gt)
            scenario_records.append(
                {
                    "scenario_id": scen["scenario_id"],
                    "persona_predictions": persona_results,
                    **agg,
                }
            )

        wave_done = wave_start + len(wave_scens)
        print(
            f"  wave {wave_idx+1}/{wave_count}: scenarios {wave_start+1}-{wave_done} "
            f"done in {time.time()-wave_t0:.1f}s — running tokens total={pred_tokens['total_tokens']}"
        )

        # --- Checkpoint partial results.json after each wave ---
        _write_results_json(
            policy_run_dir,
            policy_key,
            config,
            config_path,
            md_path,
            personas_path,
            simulation_target,
            gt,
            len(scenarios),
            batch_size,
            concurrency,
            summary_usage,
            scenario_records,
            pred_tokens,
            partial=True,
        )

    # --- Final aggregation: average MAPE per target key across non-skipped scenarios ---
    final_mape: dict[str, float | None] = {}
    n_skipped = sum(1 for r in scenario_records if r.get("skipped"))
    for k in target_keys:
        vals = [r["scenario_mape"][k] for r in scenario_records if r["scenario_mape"].get(k) is not None]
        final_mape[k] = sum(vals) / len(vals) if vals else None

    _write_results_json(
        policy_run_dir,
        policy_key,
        config,
        config_path,
        md_path,
        personas_path,
        simulation_target,
        gt,
        len(scenarios),
        batch_size,
        concurrency,
        summary_usage,
        scenario_records,
        pred_tokens,
        partial=False,
        final_mape=final_mape,
        n_skipped=n_skipped,
    )

    print(f"  final MAPE: {final_mape}   skipped={n_skipped}")

    return {
        "policy_key": policy_key,
        "gt": gt,
        "final_mape": final_mape,
        "n_scenarios_used": len(scenarios),
        "n_skipped_scenarios": n_skipped,
        "summary_usage": summary_usage,
        "pred_tokens": pred_tokens,
    }


def _write_results_json(
    policy_run_dir: Path,
    policy_key: str,
    config: dict,
    config_path: str,
    md_path: Path,
    personas_path: Path,
    simulation_target: str,
    gt: dict[str, float],
    n_scenarios_used: int,
    batch_size: int,
    concurrency: int,
    summary_usage: dict,
    scenario_records: list[dict],
    pred_tokens: dict,
    partial: bool,
    final_mape: dict | None = None,
    n_skipped: int | None = None,
) -> None:
    grand = {
        "prompt_tokens": summary_usage.get("prompt_tokens", 0) + pred_tokens["prompt_tokens"],
        "completion_tokens": summary_usage.get("completion_tokens", 0)
        + pred_tokens["completion_tokens"],
        "total_tokens": summary_usage.get("total_tokens", 0) + pred_tokens["total_tokens"],
    }
    out = {
        "policy_key": policy_key,
        "config_model": config.get("llm", {}).get("model"),
        "config_path": config_path,
        "used_paths": {
            "md": str(md_path),
            "personas": str(personas_path),
        },
        "simulation_target": simulation_target,
        "gt": gt,
        "n_scenarios_used": n_scenarios_used,
        "batch_size": batch_size,
        "concurrency": concurrency,
        "policy_summary_usage": summary_usage,
        "scenarios": scenario_records,
        "final_mape": final_mape,
        "n_skipped_scenarios": n_skipped,
        "total_usage": {
            "policy_summary": summary_usage,
            "predictions": pred_tokens,
            "grand_total": grand,
        },
        "partial": partial,
    }
    fname = "results.json"
    (policy_run_dir / fname).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_summary_csv(run_dir: Path, policy_results: list[dict]) -> None:
    """Per-target row aggregation across all policies."""
    csv_path = run_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "policy_key",
                "target_name",
                "gt",
                "final_mape",
                "n_scenarios_used",
                "n_skipped_scenarios",
                "summary_prompt_tokens",
                "summary_completion_tokens",
                "summary_total_tokens",
                "pred_prompt_tokens",
                "pred_completion_tokens",
                "pred_total_tokens",
            ]
        )
        for pr in policy_results:
            for k, gt_v in pr["gt"].items():
                w.writerow(
                    [
                        pr["policy_key"],
                        k,
                        gt_v,
                        pr["final_mape"].get(k),
                        pr["n_scenarios_used"],
                        pr["n_skipped_scenarios"],
                        pr["summary_usage"].get("prompt_tokens", 0),
                        pr["summary_usage"].get("completion_tokens", 0),
                        pr["summary_usage"].get("total_tokens", 0),
                        pr["pred_tokens"].get("prompt_tokens", 0),
                        pr["pred_tokens"].get("completion_tokens", 0),
                        pr["pred_tokens"].get("total_tokens", 0),
                    ]
                )
    print(f"\nSummary CSV: {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline: policy summary -> 5 personas -> MAPE")
    parser.add_argument(
        "--config-path",
        default="config/base.yaml",
        help="LLM config yaml (default: config/base.yaml)",
    )
    parser.add_argument(
        "--policies",
        default=",".join(all_policy_keys()),
        help="Comma-separated policy keys (default: all 5)",
    )
    parser.add_argument("--max-scenarios", type=int, default=200)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Scenarios per concurrent wave (default 25 = 125 in-flight LLM calls)",
    )
    parser.add_argument("--run-tag", default=None)
    parser.add_argument(
        "--policy-sleep",
        type=int,
        default=0,
        help="Seconds to sleep between policies (default 0). Use 10+ for rate-limited bridges.",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    model = config.get("llm", {}).get("model", "unknown")
    run_dir = make_baseline_run_dir(model, args.run_tag)
    print(f"Run dir: {run_dir}")

    policy_keys = [p.strip() for p in args.policies.split(",") if p.strip()]
    unknown = [p for p in policy_keys if p not in POLICY_MAPPING]
    if unknown:
        print(f"ERROR: unknown policy keys: {unknown}", file=sys.stderr)
        print(f"Available: {list(POLICY_MAPPING.keys())}", file=sys.stderr)
        return 2

    policy_results = []
    failed_policies = []
    for i, policy_key in enumerate(policy_keys):
        if i > 0 and args.policy_sleep > 0:
            print(f"\n  sleeping {args.policy_sleep}s between policies (rate-limit headroom)...")
            time.sleep(args.policy_sleep)
        policy_run_dir = run_dir / policy_key.replace("/", "_")
        policy_run_dir.mkdir(parents=True, exist_ok=True)
        try:
            pr = run_policy(
                policy_key=policy_key,
                config=config,
                config_path=args.config_path,
                max_scenarios=args.max_scenarios,
                batch_size=args.batch_size,
                policy_run_dir=policy_run_dir,
            )
            policy_results.append(pr)
        except Exception as exc:
            print(f"  !! policy {policy_key} FAILED: {type(exc).__name__}: {exc}")
            failed_policies.append(policy_key)
            continue

    if failed_policies:
        print(f"\nFailed policies (skipped): {failed_policies}")
    write_summary_csv(run_dir, policy_results)
    print(f"\nDone. Run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
