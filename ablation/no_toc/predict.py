"""Condition (3) w/o ToC — full KG persona pipeline + sequential 5-persona average.

Reuses the existing v6 Stage 1 outputs (no Stage 1 re-run):
- `simulation_personas.json` — scenarios with 15-field participants (KG-routed)
- `policycontext.json` — full KG-grounded PolicyContext
- `personas_pool_flat.json` — per-persona records with KG fields (kg_context,
  policy_role_actions, policy_resources, policy_actions, policy_information, ...)
  JOIN'd into scenario participants by `persona_id`.
- `shared_context.org_kg_contexts` from simulation_personas.json — injected as
  the prompt's `{org_kg_contexts}` slot.

Runs the same sequential 5-persona discussion as Cond.1 (random shuffle +
prior-speaker visibility), but with the KG-enriched persona + KG policy_context
+ org_kg_contexts. Mean across 5 personas → MAPE.

Run:
    python -m ablation.no_toc --policy kor/ssf --scenarios 100 \\
        --config-path config/base.yaml --run-tag smoke
"""
from __future__ import annotations

import argparse
import json
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
    load_policy_targets,
    make_run_dir,
    merge_usage,
    normalize_gt,
    save_json,
)
from ablation.common.prompts import SEQ_PREDICT_KG_SYSTEM, SEQ_PREDICT_KG_USER
from ablation.common.sequential_runner import (
    extract_persona_predictions,
    run_all_scenarios,
)
from baseline.policy_mapping import POLICY_MAPPING, resolve_personas_path

load_dotenv(REPO_ROOT / ".env")


# Fields from personas_pool_flat.json we attach to each participant. Excludes
# administrative/sampling metadata; includes KG-derived narrative + structural
# fields that make the "full KG" condition meaningfully richer than (1)/(2).
KG_FIELDS_TO_ATTACH = [
    "core_objectives",
    "core_constraints",
    "kg_context",
    "policy_role",
    "policy_role_actions",
    "policy_resources",
    "policy_actions",
    "policy_information",
    "evaluation_context",
    "policy_structural_relations",
    "iad_category",
    "role_family",
    "obligations",
    "eligible_actions",
    "prohibited_actions",
]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def enrich_participants_with_kg(
    sim_data: dict,
    pool_flat: list[dict],
) -> dict:
    """JOIN by representative_type (primary) → professional_persona (fallback); attach
    KG_FIELDS_TO_ATTACH to each scenario participant.

    `personas_pool_flat.json` and `simulation_personas.json` use DIFFERENT
    persona_id schemes (the pool uses generic `P_PolicyRole__NNN` while the
    scenarios use `P_<role>__<uuid>`), so we can't JOIN by persona_id. The
    `representative_type` field IS preserved verbatim across both files and is
    unique per persona — that's the right key.
    """
    pool_by_rep: dict[str, dict] = {}
    pool_by_pp: dict[str, dict] = {}
    for rec in pool_flat:
        rep = rec.get("representative_type")
        if rep and rep not in pool_by_rep:
            pool_by_rep[rep] = rec
        pp = (rec.get("professional_persona") or "")[:60]
        if pp and pp not in pool_by_pp:
            pool_by_pp[pp] = rec

    out = json.loads(json.dumps(sim_data, ensure_ascii=False))
    n_enriched = 0
    n_missed = 0
    for scen in out.get("scenarios") or []:
        for p in scen.get("participants") or []:
            rep = p.get("representative_type")
            full = pool_by_rep.get(rep) if rep else None
            if not full:
                pp = (p.get("professional_persona") or "")[:60]
                full = pool_by_pp.get(pp) if pp else None
            if not full:
                n_missed += 1
                continue
            for k in KG_FIELDS_TO_ATTACH:
                v = full.get(k)
                if v not in (None, "", [], {}):
                    p[k] = v
            n_enriched += 1
    print(f"  [enrich] joined {n_enriched} participants from pool; {n_missed} missed (kept as-is)")
    return out


def org_kg_contexts_to_text(org_kg_contexts: list[dict]) -> str:
    """Pretty-format the org_kg_contexts list for the prompt `{org_kg_contexts}` slot.

    Each entry has: name, entity_type, role_scope, source_spans_text, edge_source.
    """
    if not org_kg_contexts:
        return "(none)"
    blocks = []
    for ctx in org_kg_contexts:
        lines = [f"- {ctx.get('name', '')} ({ctx.get('entity_type', '')})"]
        if ctx.get("role_scope"):
            lines.append(f"  role_scope: {ctx['role_scope']}")
        if ctx.get("source_spans_text"):
            lines.append(f"  source_spans: {ctx['source_spans_text'][:600]}")
        if ctx.get("edge_source"):
            lines.append(f"  edges: {ctx['edge_source'][:600]}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def main() -> int:
    ap = argparse.ArgumentParser(description="Condition (3) w/o ToC — reuses v6 stage1 + sequential mean")
    ap.add_argument("--policy", required=True, help=f"Policy key. Available: {list(POLICY_MAPPING.keys())}")
    ap.add_argument("--scenarios", type=int, default=100)
    ap.add_argument("--scenario-offset", type=int, default=0,
                    help="Offset into v6 simulation_personas (0 → scenarios[0:N], 50 → scenarios[50:50+N]).")
    ap.add_argument("--config-path", default="config/base.yaml")
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--max-concurrent-scenarios", type=int, default=5)
    args = ap.parse_args()

    if args.policy not in POLICY_MAPPING:
        print(f"ERROR: unknown policy '{args.policy}'", file=sys.stderr)
        return 2

    configure_nemotron_env(args.policy)
    config = load_config(args.config_path)
    model = config.get("llm", {}).get("model", "unknown")
    run_dir = make_run_dir("no_toc", args.policy, args.run_tag)
    print(f"Run dir: {run_dir}")

    # --- Locate the v6 stage1 outputs for this policy ---
    personas_path = resolve_personas_path(args.policy)
    stage1_dir = personas_path.parent
    policycontext_path = stage1_dir / "policycontext.json"
    pool_flat_path = stage1_dir / "personas_pool_flat.json"
    print(f"Stage1 source: {stage1_dir}")
    for p in (personas_path, policycontext_path, pool_flat_path):
        if not p.exists():
            print(f"ERROR: missing required file: {p}", file=sys.stderr)
            return 2

    with open(personas_path, "r", encoding="utf-8") as f:
        sim_data = json.load(f)
    with open(policycontext_path, "r", encoding="utf-8") as f:
        policy_context = json.load(f)
    with open(pool_flat_path, "r", encoding="utf-8") as f:
        pool_flat = json.load(f)
    print(f"  loaded: {len(sim_data.get('scenarios') or [])} scenarios, "
          f"{len(pool_flat)} persona pool records")

    # --- Slice scenarios by offset + enrich participants with KG fields ---
    all_scens = sim_data.get("scenarios") or []
    lo = args.scenario_offset
    hi = lo + args.scenarios
    sim_data["scenarios"] = all_scens[lo:hi]
    sim_data["scenario_count"] = len(sim_data["scenarios"])
    print(f"  scenario slice: [{lo}:{hi}] of {len(all_scens)} → {len(sim_data['scenarios'])} scenarios")
    sim_data = enrich_participants_with_kg(sim_data, pool_flat)

    # Snapshot the enriched personas into the run-dir for traceability.
    save_json(run_dir / "simulation_personas_enriched.json", sim_data)

    # --- Pre-compute org_kg_contexts text and policy_context text ---
    org_kg_contexts = (sim_data.get("shared_context") or {}).get("org_kg_contexts") or []
    org_kg_text = org_kg_contexts_to_text(org_kg_contexts)
    policy_context_text = json.dumps(policy_context, ensure_ascii=False, indent=2)

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
    print(f"  org_kg_contexts: {len(org_kg_contexts)}")

    def _checkpoint(partial_results: list[dict]) -> None:
        save_json(run_dir / "scenario_records_partial.json", partial_results)

    t0 = time.time()
    scenario_runs = run_all_scenarios(
        scenarios=sim_data["scenarios"],
        policy_context_text=policy_context_text,
        org_kg_contexts_text=org_kg_text,
        simulation_target=simulation_target,
        target_keys=target_keys,
        system_prompt=SEQ_PREDICT_KG_SYSTEM,
        user_template=SEQ_PREDICT_KG_USER,
        config=config,
        max_concurrent_scenarios=args.max_concurrent_scenarios,
        checkpoint_cb=_checkpoint,
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")

    # --- Aggregate ---
    scenario_records: list[dict] = []
    pred_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for run in scenario_runs:
        persona_preds = extract_persona_predictions(run, target_keys)
        agg = aggregate_scenario(persona_preds, gt)
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

    save_json(run_dir / "results.json", {
        "policy_key": args.policy,
        "condition": "no_toc",
        "config_model": model,
        "config_path": args.config_path,
        "stage1_source": str(stage1_dir),
        "simulation_target": simulation_target,
        "gt": gt,
        "n_scenarios_used": len(sim_data["scenarios"]),
        "scenarios": scenario_records,
        "final_mape": final_mape,
        "n_skipped_scenarios": n_skipped,
        "total_usage": {
            "predictions": pred_tokens,
            "grand_total": pred_tokens,  # only prediction calls in this condition
        },
    })
    partial = run_dir / "scenario_records_partial.json"
    if partial.exists():
        partial.unlink()
    save_json(run_dir / "run_log.json", {
        "policy_key": args.policy,
        "condition": "no_toc",
        "model": model,
        "stage1_source": str(stage1_dir),
        "n_scenarios": len(sim_data["scenarios"]),
        "final_mape": final_mape,
        "elapsed_seconds": elapsed,
        "total_usage": pred_tokens,
    })
    print(f"\nDone. Run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
