"""Condition (1) Baseline pipeline.

MD-driven Nemotron persona extraction + sequential 5-persona prediction + mean
aggregation + MAPE. Also produces the simulation_personas.json reused by (2).

Run:
    python -m ablation.baseline_md --policy kor/ssf --scenarios 100 \\
        --config-path config/base.yaml --run-tag smoke
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from ablation.common.io import (
    REPO_ROOT,
    aggregate_final_mape,
    aggregate_scenario,
    configure_nemotron_env,
    load_persona_fields,
    load_policy_targets,
    make_run_dir,
    merge_usage,
    normalize_gt,
    save_json,
)
from ablation.common.prompts import SEQ_PREDICT_MD_SYSTEM, SEQ_PREDICT_MD_USER
from ablation.common.sequential_runner import (
    extract_persona_predictions,
    run_all_scenarios,
)
from baseline.policy_mapping import POLICY_MAPPING, resolve_md_path

from llm.client import chat_completion_with_usage
from models.prompts import (
    PERSONAS_FROM_TEXT_SYSTEM,
    PERSONAS_FROM_TEXT_USER,
    POLICY_READER_SYSTEM,
    POLICY_READER_USER,
)

load_dotenv(REPO_ROOT / ".env")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def summarize_policy(md_text: str, config: dict) -> tuple[dict, dict]:
    """POLICY_READER → PolicyPlan-shaped dict + usage."""
    messages = [
        {"role": "system", "content": POLICY_READER_SYSTEM},
        {"role": "user", "content": POLICY_READER_USER.format(pdf_text=md_text)},
    ]
    content, usage = chat_completion_with_usage(messages, config)
    return json.loads(content), usage


def extract_stakeholder_groups(
    md_text: str,
    policy_summary: dict,
    n_personas_per_type: int,
    config: dict,
) -> tuple[list[dict], dict]:
    """PERSONAS_FROM_TEXT → list of {stakeholder_type, personas: [{representative_type, core_objectives, core_constraints}, ...]}."""
    messages = [
        {"role": "system", "content": PERSONAS_FROM_TEXT_SYSTEM},
        {
            "role": "user",
            "content": PERSONAS_FROM_TEXT_USER.format(
                past_policy_text=md_text,
                policy_context=json.dumps(policy_summary, ensure_ascii=False),
                n_personas=n_personas_per_type,
            ),
        },
    ]
    content, usage = chat_completion_with_usage(messages, config)
    data = json.loads(content)
    return data.get("stakeholder_groups") or [], usage


def build_scenarios(
    pool_by_type: dict[str, list[dict]],
    num_scenarios: int,
    participants_per_scenario: int,
    base_seed: int = 42,
    scenario_offset: int = 0,
) -> list[dict]:
    """Build N scenarios × K participants/scenario via stratified round-robin over stakeholder_types.

    Strategy:
    - Each scenario rotates the stakeholder_type order to give each type a chance at the early slots.
    - Within each type, sample WITHOUT replacement during a scenario.
    - Across scenarios, sampling is independent (with replacement at the pool level).

    Note: participant dicts contain ONLY PERSONA_FIELDS values from the parquet — no
    stakeholder_type, no representative_type, no persona_id (deliberate per ablation
    spec: KG-derived/routing metadata removed).
    """
    stakeholder_types = sorted(pool_by_type.keys())
    K = len(stakeholder_types)
    if K == 0:
        return []

    scenarios = []
    for i in range(1, num_scenarios + 1):
        s_idx = i + scenario_offset  # absolute scenario index (1..50 or 51..100)
        rng = random.Random(base_seed + s_idx)
        rotated = stakeholder_types[(s_idx - 1) % K :] + stakeholder_types[: (s_idx - 1) % K]

        # First, ensure ≥1 persona from each type up to participants_per_scenario.
        picked: list[dict] = []
        for stype in rotated:
            if len(picked) >= participants_per_scenario:
                break
            pool = pool_by_type.get(stype) or []
            if not pool:
                continue
            picked.append(dict(rng.choice(pool)))

        # Fill remaining slots from a flat pool, uniform without replacement (within scenario).
        if len(picked) < participants_per_scenario:
            flat = [dict(p) for stype in stakeholder_types for p in (pool_by_type.get(stype) or [])]
            rng.shuffle(flat)
            for p in flat:
                if len(picked) >= participants_per_scenario:
                    break
                picked.append(p)

        scenarios.append({
            "scenario_id": f"scenario_{s_idx}",
            "participants": picked[:participants_per_scenario],
        })
    return scenarios


def main() -> int:
    ap = argparse.ArgumentParser(description="Condition (1) Baseline — MD-driven Nemotron + sequential mean")
    ap.add_argument("--policy", required=True, help=f"Policy key. Available: {list(POLICY_MAPPING.keys())}")
    ap.add_argument("--scenarios", type=int, default=100)
    ap.add_argument("--scenario-offset", type=int, default=0,
                    help="Offset added to scenario indices (e.g., 50 → scenario_51..scenario_100). "
                         "Uses different seeds so the scenarios are genuinely different.")
    ap.add_argument("--participants-per-scenario", type=int, default=5)
    ap.add_argument("--n-personas-per-stakeholder", type=int, default=20,
                    help="Nemotron rows sampled per stakeholder_type (the pool from which scenarios draw)")
    ap.add_argument("--config-path", default="config/base.yaml")
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--max-concurrent-scenarios", type=int, default=5)
    args = ap.parse_args()

    if args.policy not in POLICY_MAPPING:
        print(f"ERROR: unknown policy '{args.policy}'. Available: {list(POLICY_MAPPING.keys())}", file=sys.stderr)
        return 2

    # MUST configure Nemotron env BEFORE we ever read PERSONA_FIELDS.
    parquet_glob, values_module = configure_nemotron_env(args.policy)
    sql_select_exprs, output_field_names = load_persona_fields(args.policy)

    config = load_config(args.config_path)
    model = config.get("llm", {}).get("model", "unknown")
    run_dir = make_run_dir("baseline_md", args.policy, args.run_tag)
    print(f"Run dir: {run_dir}")
    print(f"Nemotron values_module: {values_module}")
    print(f"Nemotron parquet:       {parquet_glob}")
    print(f"PERSONA_FIELDS:         {output_field_names}")

    # --- Policy summary ---
    md_path = resolve_md_path(args.policy)
    print(f"\n=== Policy summary === ({md_path.name})")
    md_text = md_path.read_text(encoding="utf-8")
    t0 = time.time()
    policy_summary, summary_usage = summarize_policy(md_text, config)
    print(f"  POLICY_READER done in {time.time()-t0:.1f}s — tokens total={summary_usage['total_tokens']}")
    save_json(run_dir / "policy_summary.json", {
        "policy_key": args.policy,
        "md_path": str(md_path),
        "summary": policy_summary,
        "usage": summary_usage,
    })
    # Stash PolicyPlan-shaped JSON for (2) reuse — POLICY_READER output IS PolicyPlan-shaped.
    save_json(run_dir / "policy_plan.json", policy_summary)

    # --- Stakeholder extraction ---
    print("\n=== Stakeholder extraction (PERSONAS_FROM_TEXT) ===")
    t0 = time.time()
    stakeholder_groups, sh_usage = extract_stakeholder_groups(
        md_text=md_text,
        policy_summary=policy_summary,
        n_personas_per_type=2,
        config=config,
    )
    print(f"  done in {time.time()-t0:.1f}s — {len(stakeholder_groups)} stakeholder types — tokens total={sh_usage['total_tokens']}")
    for g in stakeholder_groups:
        print(f"    - {g['stakeholder_type']}: {len(g.get('personas') or [])} representative_type(s)")
    save_json(run_dir / "stakeholder_extraction.json", {
        "stakeholder_groups": stakeholder_groups,
        "usage": sh_usage,
    })

    # --- Nemotron sampling per stakeholder_type ---
    print("\n=== Nemotron sampling (MD-driven, no scope_definer) ===")
    # Lazy import to defer DuckDB import until needed
    from ablation.baseline_md.nemotron_sampler import sample_pool

    t0 = time.time()
    pool_by_type, hint_usages = sample_pool(
        stakeholder_groups=stakeholder_groups,
        policy_summary=policy_summary,
        parquet_glob=parquet_glob,
        sql_select_exprs=sql_select_exprs,
        n_per_stakeholder=args.n_personas_per_stakeholder,
        config=config,
        base_seed=42,
    )
    print(f"  done in {time.time()-t0:.1f}s — pool sizes: {[(k, len(v)) for k, v in pool_by_type.items()]}")
    hints_total_usage = merge_usage(*[u["usage"] for u in hint_usages])
    print(f"  hints calls tokens total={hints_total_usage['total_tokens']}")
    save_json(run_dir / "demographic_hints.json", hint_usages)

    # --- Scenario assembly ---
    scenarios = build_scenarios(
        pool_by_type=pool_by_type,
        num_scenarios=args.scenarios,
        participants_per_scenario=args.participants_per_scenario,
        base_seed=42,
        scenario_offset=args.scenario_offset,
    )
    print(f"\n=== Scenarios: {len(scenarios)} × {args.participants_per_scenario} personas ===")
    sim_personas_payload = {
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "metadata": {
            "policy_key": args.policy,
            "country_values_module": values_module,
            "persona_fields": output_field_names,
            "n_personas_per_stakeholder": args.n_personas_per_stakeholder,
            "stakeholder_types": list(pool_by_type.keys()),
        },
    }
    save_json(run_dir / "simulation_personas.json", sim_personas_payload)

    # --- Sequential prediction ---
    targets_all = load_policy_targets()
    target_info = targets_all[args.policy]
    simulation_target = target_info["simulation_target"]
    gt = normalize_gt(target_info["gt"])
    target_keys = list(gt.keys())
    print(f"\n=== Sequential prediction ===")
    print(f"  simulation_target: {simulation_target}")
    print(f"  target_keys: {target_keys}")
    print(f"  gt: {gt}")

    # The LLM is prompted to emit a value per target_key. For single-key policies (e.g. ssf,
    # 3ia, aip) this is one number; for multi-key policies (bk21=2 keys, nais=2 keys) we ask
    # for each key explicitly. See sequential_runner._build_prediction_keys_hint.
    policy_context_text = json.dumps(policy_summary, ensure_ascii=False, indent=2)

    def _checkpoint(partial_results: list[dict]) -> None:
        # Persist partial scenario records each time one completes.
        save_json(run_dir / "scenario_records_partial.json", partial_results)

    t0 = time.time()
    scenario_runs = run_all_scenarios(
        scenarios=scenarios,
        policy_context_text=policy_context_text,
        org_kg_contexts_text=None,
        simulation_target=simulation_target,
        target_keys=target_keys,
        system_prompt=SEQ_PREDICT_MD_SYSTEM,
        user_template=SEQ_PREDICT_MD_USER,
        config=config,
        max_concurrent_scenarios=args.max_concurrent_scenarios,
        checkpoint_cb=_checkpoint,
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")

    # --- Aggregation ---
    scenario_records: list[dict] = []
    pred_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for run in scenario_runs:
        persona_preds = extract_persona_predictions(run, target_keys)
        agg = aggregate_scenario(persona_preds, gt)
        # Sum per-persona usage.
        usages = [r.get("usage") or {} for r in run.get("persona_records") or []]
        scenario_usage = merge_usage(*usages)
        for k in pred_tokens:
            pred_tokens[k] += scenario_usage[k]
        scenario_records.append({
            "scenario_id": run["scenario_id"],
            "persona_records": run.get("persona_records") or [],
            **agg,
        })
    final_mape, n_skipped = aggregate_final_mape(scenario_records, gt)
    print(f"  final_mape: {final_mape}   skipped={n_skipped}")

    grand = merge_usage(summary_usage, sh_usage, hints_total_usage, pred_tokens)
    save_json(run_dir / "results.json", {
        "policy_key": args.policy,
        "config_model": model,
        "config_path": args.config_path,
        "simulation_target": simulation_target,
        "gt": gt,
        "n_scenarios_used": len(scenarios),
        "participants_per_scenario": args.participants_per_scenario,
        "scenarios": scenario_records,
        "final_mape": final_mape,
        "n_skipped_scenarios": n_skipped,
        "total_usage": {
            "policy_summary": summary_usage,
            "stakeholder_extraction": sh_usage,
            "demographic_hints": hints_total_usage,
            "predictions": pred_tokens,
            "grand_total": grand,
        },
    })

    # Cleanup partial file
    partial = run_dir / "scenario_records_partial.json"
    if partial.exists():
        partial.unlink()

    save_json(run_dir / "run_log.json", {
        "policy_key": args.policy,
        "model": model,
        "config_path": args.config_path,
        "n_scenarios": len(scenarios),
        "participants_per_scenario": args.participants_per_scenario,
        "final_mape": final_mape,
        "elapsed_seconds": elapsed,
        "total_usage": grand,
    })
    print(f"\nDone. Run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
