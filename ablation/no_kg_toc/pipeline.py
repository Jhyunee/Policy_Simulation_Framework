"""Condition (2) w/o Policy KG — runs ToC simulation over Condition (1)'s personas.

Inputs: (1)'s run-dir containing
- `policy_plan.json` (POLICY_READER output, PolicyPlan-shaped)
- `simulation_personas.json` (PERSONA_FIELDS-only participants in scenarios)

Steps:
1. Build KG-free PolicyContext via `stage1.modules.build_policy_context(policy=plan, ..., policy_graph_path=None)`.
2. Augment (1)'s simulation_personas with a synthetic `persona_id` per participant
   so ToC's dedup/routing plumbing works. (Opaque identifier — carries no KG semantics.)
3. Write a minimal Stage1Result + the augmented simulation_personas under (2)'s run-dir.
4. Call `stage2_toc.pipeline.run_stage2_toc(...)`.
5. Extract simulation_target value across scenarios from cross_checks → mean → MAPE.

Run:
    python -m ablation.no_kg_toc --policy kor/ssf \\
        --baseline-run-dir outputs/ablation/baseline_md/kor_ssf/<ts>_smoke \\
        --config-path config/base.yaml --run-tag smoke
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from ablation.common.io import (
    REPO_ROOT,
    configure_nemotron_env,
    load_policy_targets,
    make_run_dir,
    merge_usage,
    normalize_gt,
    save_json,
)
from baseline.policy_mapping import POLICY_MAPPING

load_dotenv(REPO_ROOT / ".env")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def slugify(text: str) -> str:
    """'number of SCI(E) publications in 2021-2022' → 'number_of_sci_e_publications_in_2021_2022'"""
    s = text.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def augment_personas_with_ids(sim_data: dict) -> dict:
    """Inject a synthetic persona_id (and minimal stakeholder_type='Unknown' fallback)
    into each participant for ToC plumbing. Opaque identifier — no KG semantics.

    Returns a new dict (does not mutate input).
    """
    out = json.loads(json.dumps(sim_data, ensure_ascii=False))
    counter = 0
    for scen in out.get("scenarios") or []:
        for p in scen.get("participants") or []:
            counter += 1
            if "persona_id" not in p:
                p["persona_id"] = f"ablation_P_{counter:05d}"
    # Ensure shared_context has empty org_kg_contexts (KG-free condition)
    out.setdefault("shared_context", {}).setdefault("org_kg_contexts", [])
    return out


def build_thin_stage1_result(policy_context_dict: dict) -> dict:
    """Construct a minimal Stage1Result JSON containing only the policy_context.

    stage2_toc/pipeline.py loads stage1_result.json and accesses .policy_context,
    .predictions, .stakeholder_groups. We need to pass schema-valid empty values.
    """
    return {
        "economic_context": None,
        "policy_plan": None,
        "policy_context": policy_context_dict,
        "stakeholder_groups": [],
        "persona_pool": [],
        "simulation_participants": [],
        "predictions": [],
        "critic_results": [],
        "handoff_metadata": {"source": "ablation/no_kg_toc"},
    }


_STOPWORDS = {"in", "of", "the", "a", "an", "and", "or", "for", "to", "by", "at", "on"}


def _tokens(slug: str) -> set[str]:
    """Slug → set of content tokens (drop stopwords, drop pure-numeric tokens)."""
    return {t for t in slug.split("_") if t and t not in _STOPWORDS and not t.isdigit()}


def _match_variable_for_key(candidate_names: list[str], hint_slug: str) -> str | None:
    """Pick the candidate variable whose slug best matches hint_slug.

    Ranks ALL candidates that contain ALL hint content tokens (subset rule), then
    picks the one with highest token-Jaccard with the hint — i.e. the candidate
    that adds the fewest extra words. This prefers `graduate_employment_rate`
    over `graduate_employment_rate_of_bk21_participants_in_2022` for the hint
    `employment_rate`, since the former adds fewer extraneous tokens.
    """
    # 1. exact
    for v in candidate_names:
        if slugify(v) == hint_slug:
            return v

    # 2. token-overlap: candidate must contain ALL hint tokens; highest Jaccard wins.
    hint_tokens = _tokens(hint_slug)
    if hint_tokens:
        best: tuple[float, int, str] | None = None
        for v in candidate_names:
            cand_tokens = _tokens(slugify(v))
            if not hint_tokens.issubset(cand_tokens):
                continue
            union = hint_tokens | cand_tokens
            jaccard = len(hint_tokens & cand_tokens) / len(union) if union else 0.0
            # Tie-break by SHORTER candidate (fewer added words → closer semantic match).
            key = (jaccard, -len(cand_tokens), v)
            if best is None or key > best:
                best = key
        if best is not None:
            return best[2]

    # 3. fallback substring (low-priority — only fires when token-subset fails)
    for v in candidate_names:
        vs = slugify(v)
        if vs and (vs in hint_slug or hint_slug in vs):
            return v
    return None


def extract_target_values_from_scenarios(
    scenarios: list[dict],
    simulation_target: str,
    target_keys: list[str],
) -> tuple[dict[str, list[float]], dict[str, str | None]]:
    """For each gt key, find the matching variable in cross_checks and pull per-scenario aggregated_value.

    Returns ({key: [per_scenario_values, ...]}, {key: matched_variable_name}).
    """
    candidate_names: list[str] = []
    for scen in scenarios:
        for cc in scen.get("cross_checks") or []:
            v = cc.get("variable")
            if v and v not in candidate_names:
                candidate_names.append(v)

    # Per-key matching: prefer matching the key name first; fall back to simulation_target slug
    # only when there's exactly one target key (single-key policies like ssf/3ia/aip where
    # the key is just "value").
    matched_per_key: dict[str, str | None] = {}
    for k in target_keys:
        # If key is generic "value", match against full simulation_target slug.
        hint = simulation_target if k == "value" else k
        matched_per_key[k] = _match_variable_for_key(candidate_names, slugify(hint))

    values_per_key: dict[str, list[float]] = {k: [] for k in target_keys}
    for scen in scenarios:
        cc_list = scen.get("cross_checks") or []
        for k in target_keys:
            var_name = matched_per_key[k]
            if var_name is None:
                continue
            agg = next(
                (cc.get("aggregated_value") for cc in cc_list if cc.get("variable") == var_name),
                None,
            )
            if agg is not None:
                try:
                    values_per_key[k].append(float(agg))
                except (TypeError, ValueError):
                    continue
    return values_per_key, matched_per_key


def compute_mape_per_key(
    values_per_key: dict[str, list[float]],
    gt: dict[str, float],
) -> dict[str, float | None]:
    """Per-key mean across scenarios → MAPE vs GT."""
    out: dict[str, float | None] = {}
    for k, vals in values_per_key.items():
        if not vals:
            out[k] = None
            continue
        mean = sum(vals) / len(vals)
        gv = gt.get(k)
        out[k] = (abs(mean - gv) / abs(gv) * 100.0) if (gv is not None and gv != 0) else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Condition (2) w/o Policy KG — ToC over (1)'s personas")
    ap.add_argument("--policy", required=True, help=f"Policy key. Available: {list(POLICY_MAPPING.keys())}")
    ap.add_argument("--baseline-run-dir", required=True, type=Path,
                    help="Path to a baseline_md run dir (output of Condition 1). Must contain policy_plan.json and simulation_personas.json.")
    ap.add_argument("--config-path", default="config/base.yaml")
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--scenarios", type=int, default=0,
                    help="If >0, limit to first N scenarios from the baseline run.")
    ap.add_argument("--disable-extra-chains", action="store_true",
                    help="Disable fwd_bwd_fwd_seq and fwd_bwd_fwd_bwd_seq chains for cheaper smoke runs.")
    args = ap.parse_args()

    if args.policy not in POLICY_MAPPING:
        print(f"ERROR: unknown policy '{args.policy}'", file=sys.stderr)
        return 2

    if not args.baseline_run_dir.is_dir():
        print(f"ERROR: --baseline-run-dir not found: {args.baseline_run_dir}", file=sys.stderr)
        return 2

    plan_path = args.baseline_run_dir / "policy_plan.json"
    personas_path = args.baseline_run_dir / "simulation_personas.json"
    if not plan_path.exists() or not personas_path.exists():
        print(f"ERROR: baseline run dir missing policy_plan.json or simulation_personas.json", file=sys.stderr)
        return 2

    configure_nemotron_env(args.policy)
    config = load_config(args.config_path)

    # Strip the experimental inner_schema_llm override unless the user set DEEPSEEK_V4_PRO.
    # The default config has `inner_schema_llm: {model: deepseek-v4-pro}` which would 404.
    inner_llm = (config.get("stage2_toc") or {}).get("inner_schema_llm") or {}
    if inner_llm.get("model") == "deepseek-v4-pro":
        config["stage2_toc"]["inner_schema_llm"] = {}
        print("  [config] cleared inner_schema_llm override (deepseek-v4-pro not present)")

    if args.disable_extra_chains:
        config["stage2_toc"]["fwd_bwd_fwd_seq"] = False
        config["stage2_toc"]["fwd_bwd_fwd_bwd_seq"] = False
        config["stage2_toc"]["final_forward_refinement"] = False
        print("  [config] disabled fwd_bwd_fwd_seq, fwd_bwd_fwd_bwd_seq, final_forward_refinement")

    model = config.get("llm", {}).get("model", "unknown")
    run_dir = make_run_dir("no_kg_toc", args.policy, args.run_tag)
    print(f"Run dir:           {run_dir}")
    print(f"Baseline source:   {args.baseline_run_dir}")

    # --- Step 1: Build KG-free PolicyContext (lazy import to defer LLM client setup) ---
    print("\n=== Building KG-free PolicyContext (build_policy_context with policy_graph_path=None) ===")
    from llm.client import USAGE_TRACKER
    from models.schemas import PolicyPlan
    from stage1.modules import build_policy_context

    # Snapshot for full-pipeline token accounting.
    pipeline_start = time.time()
    usage_before = USAGE_TRACKER.snapshot()

    with open(plan_path, "r", encoding="utf-8") as f:
        plan_dict = json.load(f)
    policy_plan = PolicyPlan(**plan_dict)
    t0 = time.time()
    policy_context = build_policy_context(
        policy=policy_plan,
        economic_context=None,
        config=config,
        policy_graph_path=None,
    )
    usage_after_pc = USAGE_TRACKER.snapshot()
    pc_usage = {k: usage_after_pc[k] - usage_before[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    pc_dict = policy_context.model_dump()
    save_json(run_dir / "policycontext.json", pc_dict)
    print(f"  done in {time.time()-t0:.1f}s; policy_name='{pc_dict.get('policy_name')}'")

    # --- Step 2: Load + augment personas ---
    print("\n=== Augmenting personas with synthetic ids (no KG semantics) ===")
    with open(personas_path, "r", encoding="utf-8") as f:
        sim_data = json.load(f)
    sim_data = augment_personas_with_ids(sim_data)
    if args.scenarios and args.scenarios > 0:
        sim_data["scenarios"] = sim_data["scenarios"][: args.scenarios]
        sim_data["scenario_count"] = len(sim_data["scenarios"])
    n_scen = len(sim_data.get("scenarios") or [])
    print(f"  scenarios: {n_scen}")
    save_json(run_dir / "simulation_personas.json", sim_data)

    # --- Step 3: Stage1Result adapter ---
    stage1_thin = build_thin_stage1_result(pc_dict)
    save_json(run_dir / "stage1_result.json", stage1_thin)

    # --- Step 4: Patch stage2_toc dedup so PERSONA_FIELDS-only participants get unique keys ---
    # _build_personas_from_stage1_personas dedups by `representative_type or persona_id or ""`.
    # We injected persona_id; that path works. The dedup fallback in _resolve_combo also
    # uses persona_id first → safe.

    # --- Step 5: Run stage2_toc ---
    print("\n=== Running stage2_toc.run_stage2_toc ===")
    targets_all = load_policy_targets()
    target_info = targets_all[args.policy]
    simulation_target = target_info["simulation_target"]
    gt = normalize_gt(target_info["gt"])
    print(f"  simulation_target: {simulation_target}")
    print(f"  gt: {gt}")

    from stage2_toc.pipeline import run_stage2_toc

    t0 = time.time()
    toc_result = run_stage2_toc(
        config=config,
        run_dir=run_dir,
        simulation_target=simulation_target,
        stage1_result_path=run_dir / "stage1_result.json",
        simulation_personas_path=run_dir / "simulation_personas.json",
    )
    elapsed = time.time() - t0
    print(f"  ToC done in {elapsed:.1f}s")

    # --- Step 6: Extract simulation_target value(s) across scenarios → MAPE ---
    toc_dict = toc_result.model_dump() if hasattr(toc_result, "model_dump") else toc_result
    scenarios = toc_dict.get("scenarios") or []
    target_keys = list(gt.keys())
    values_per_key, matched_per_key = extract_target_values_from_scenarios(
        scenarios, simulation_target, target_keys
    )
    for k in target_keys:
        print(f"\n  [{k}] matched variable: '{matched_per_key[k]}'  ({len(values_per_key[k])} per-scenario values)")
        print(f"  [{k}] values: {values_per_key[k]}")
    final_mape = compute_mape_per_key(values_per_key, gt)
    print(f"\n  final_mape: {final_mape}")

    # --- Step 7: Save results ---
    save_json(run_dir / "results.json", {
        "policy_key": args.policy,
        "condition": "no_kg_toc",
        "config_model": model,
        "config_path": args.config_path,
        "baseline_run_dir": str(args.baseline_run_dir),
        "simulation_target": simulation_target,
        "matched_variables": matched_per_key,
        "gt": gt,
        "n_scenarios_used": len(scenarios),
        "per_scenario_target_values": values_per_key,
        "mean_target_value": {
            k: (sum(vs) / len(vs)) if vs else None for k, vs in values_per_key.items()
        },
        "final_mape": final_mape,
        "aggregated_final_from_toc": toc_dict.get("aggregated_final"),
    })
    # Total pipeline tokens (policy_context + entire ToC stage including inner_schema).
    usage_end = USAGE_TRACKER.snapshot()
    pipeline_usage = {k: usage_end[k] - usage_before[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    toc_usage = {k: usage_end[k] - usage_after_pc[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    pipeline_elapsed = time.time() - pipeline_start

    # stage2_toc/pipeline.py wrote its own run_log.json into our run_dir — we overwrite
    # it with the ablation-level log (which includes the policy_context build phase).
    # The per-scenario detail and the original snapshot live in stage2_toc_result.json
    # + scenario_*.json + aggregated_quantitative.json.
    save_json(run_dir / "run_log.json", {
        "policy_key": args.policy,
        "condition": "no_kg_toc",
        "model": model,
        "baseline_run_dir": str(args.baseline_run_dir),
        "n_scenarios": len(scenarios),
        "final_mape": final_mape,
        "elapsed_seconds_total": pipeline_elapsed,
        "elapsed_seconds_toc": elapsed,
        "total_usage": pipeline_usage,
        "usage_breakdown": {
            "policy_context_build": pc_usage,
            "stage2_toc": toc_usage,
        },
    })
    print(f"\nDone. Run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
