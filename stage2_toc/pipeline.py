import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from llm.client import token_tracker
from models.schemas import (
    Stage1Result,
    ConcretePersona,
    ToCInnerSchema,
    ScenarioToCResult,
    ToCSimulationResult,
    StakeholderGroup,
    GroundedInputState,
)
from persona.simulation_schema import (
    SIMULATION_PERSONA_FIELDS,
    sanitize_simulation_persona,
)
# `construct_all_simulation_personas` (legacy LLM constructor) was previously
# imported here but never called — _build_personas_from_stage1_personas builds
# personas deterministically from Stage 1 output. Dead import removed.
from stage2_toc.runtime import set_org_kg_contexts
from stage2_toc.grounded_inputs import (
    apply_grounded_inputs_to_schema,
    build_grounded_input_state,
)
from stage2_toc.modules import (
    build_inner_toc_schema,
    build_simple_inner_schema,
    run_forward_pass,
    run_backward_pass,
    refine_forward_with_backward,
    cross_check_phases,
    aggregate_scenario,
    aggregate_across_scenarios,
    run_final_toc_reporter,
    run_phase_stakeholder_synthesizer,
    _aggregate_phase_round,
)


def save_json(data: dict | list, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _resolve_policy_graph_for_grounding(
    stage1_result_path: Path,
    grounded_config: dict,
) -> Path | None:
    explicit = grounded_config.get("policy_graph")
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    run_log_path = stage1_result_path.parent / "run_log.json"
    if not run_log_path.exists():
        return None
    run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
    value = run_log.get("policy_graph")
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def _extract_korean_name(professional_persona: str) -> str:
    """Pull the first whitespace-separated token from `professional_persona`.

    Nemotron-Personas-Korea convention: `<Name> 씨는 <description>` — so the
    first token is the Korean given/family name (e.g. '박결', '오정진'). When the
    text is empty or doesn't contain a clear leading token, return empty string;
    the caller then falls back to `representative_type`.
    """
    if not professional_persona:
        return ""
    parts = professional_persona.strip().split()
    return parts[0] if parts else ""


def _persona_display_name(record: dict) -> str:
    """Prefer explicit scenario names, then fall back to legacy persona text."""
    for key in ("name", "display_name"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _extract_korean_name(record.get("professional_persona") or "")


def _persona_field_whitelist() -> set[str]:
    """Return the fixed Stage 1-to-Stage 2 simulation-persona contract."""
    return set(SIMULATION_PERSONA_FIELDS)


def _build_personas_from_stage1_personas(sim_data: dict) -> list[ConcretePersona]:
    """Build `ConcretePersona` objects directly from Stage 1's persona projection.

    Personas appearing in multiple scenarios are deduped by canonical persona_id.
    """
    seen_persona_ids: set[str] = set()
    used_names: set[str] = set()
    personas: list[ConcretePersona] = []

    for scenario in sim_data.get("scenarios") or []:
        for record in scenario.get("participants") or []:
            compact = sanitize_simulation_persona(record)
            persona_id = str(compact.get("persona_id") or "")
            if not persona_id:
                raise ValueError("Every Stage 2 simulation persona must have persona_id")
            if persona_id in seen_persona_ids:
                continue
            seen_persona_ids.add(persona_id)

            base_name = _persona_display_name(compact) or persona_id
            name = base_name
            if name in used_names:
                # Disambiguate via short uuid hash (e.g. "오정진 (a1b2c3)").
                suffix = persona_id.split("__")[-1][:6] if "__" in persona_id else persona_id[:6]
                if suffix:
                    name = f"{base_name} ({suffix})"
                else:
                    i = 2
                    while f"{base_name}-{i}" in used_names:
                        i += 1
                    name = f"{base_name}-{i}"
            used_names.add(name)
            compact["name"] = name
            cp = ConcretePersona(**compact)
            personas.append(cp)
    return personas


def run_stage2_toc(
    config: dict,
    run_dir: Path,
    simulation_target: str,
    stage1_result_path: Path,
    simulation_personas_path: Optional[Path] = None,
    run_forward_flag: Optional[bool] = None,
    run_backward_flag: Optional[bool] = None,
    inner_schema_override: Optional[Path] = None,
    resume_from: Optional[Path] = None,
    scenario_range: Optional[tuple[int, int]] = None,
    simple_prediction: bool = False,
) -> ToCSimulationResult:
    cfg = config.get("stage2_toc", {})
    # Simple-prediction ablation: 1-phase Impact-only schema, forward only, no
    # backward / refinement / sequence chains. Force the relevant flags off so the
    # 2×2 ablation comparison stays clean (caller doesn't need to also pass
    # --no-backward etc).
    if simple_prediction:
        cfg = dict(cfg)
        cfg["final_forward_refinement"] = False
        cfg["fwd_bwd_fwd_seq"] = False
        cfg["fwd_bwd_fwd_bwd_seq"] = False
        cfg["simple_prediction"] = True
        config = {**config, "stage2_toc": cfg}
        run_backward_flag = False
        print("Simple-prediction mode: forcing backward=False, refinement=False, seq chains=False")
    do_forward = run_forward_flag if run_forward_flag is not None else cfg.get("run_forward", True)
    do_backward = run_backward_flag if run_backward_flag is not None else cfg.get("run_backward", True)
    if not do_forward and not do_backward:
        raise ValueError("At least one of run_forward / run_backward must be true.")

    # Bridge: persona constructor + scenario builder both read config["stage2"][...]
    bridged = dict(config)
    bridged.setdefault("stage2", {})
    bridged["stage2"].setdefault("sim_personas_per_entity", cfg.get("sim_personas_per_entity", 2))
    bridged["stage2"].setdefault("max_scenarios", cfg.get("max_scenarios", 20))
    bridged["stage2"].setdefault(
        "use_economic_context_in_simulation",
        cfg.get("use_economic_context_in_simulation", True),
    )

    if not stage1_result_path.exists():
        raise FileNotFoundError(f"Stage1 result not found: {stage1_result_path}")

    with open(stage1_result_path, "r", encoding="utf-8") as f:
        stage1_result = Stage1Result(**json.load(f))

    policy_context = stage1_result.policy_context
    if policy_context is None:
        raise ValueError(
            "stage1_result.policy_context is missing. Re-run Stage 1 with the current pipeline "
            "(policy_context is built KG-only and is the sole policy artifact Stage 2 consumes)."
        )
    prediction_results = stage1_result.predictions
    stakeholder_groups: list[StakeholderGroup] = stage1_result.stakeholder_groups
    resume_dir = resume_from or run_dir

    # ── Step 0: Extract document-grounded Inputs before schema construction ──
    grounded_inputs: GroundedInputState | None = None
    grounded_cfg = cfg.get("grounded_inputs") or {}
    if grounded_cfg.get("enabled", False) and not simple_prediction:
        grounded_path = run_dir / grounded_cfg.get("output_file", "grounded_input_state.json")
        resume_grounded = resume_dir / grounded_cfg.get("output_file", "grounded_input_state.json")
        override = grounded_cfg.get("state_file")
        if override and Path(override).exists():
            print(f"  Loading grounded Inputs override: {override}")
            grounded_inputs = GroundedInputState.model_validate(
                json.loads(Path(override).read_text(encoding="utf-8"))
            )
        elif resume_from and resume_grounded.exists():
            print(f"  [resume] Loading grounded Inputs from {resume_grounded}")
            grounded_inputs = GroundedInputState.model_validate(
                json.loads(resume_grounded.read_text(encoding="utf-8"))
            )
        else:
            print("Extracting document-grounded Inputs before ToC schema construction...")
            policy_graph_path = _resolve_policy_graph_for_grounding(
                stage1_result_path, grounded_cfg,
            )
            grounded_inputs = build_grounded_input_state(
                config, policy_context, policy_graph_path,
            )
        save_json(grounded_inputs.model_dump(mode="json"), grounded_path)
        print(f"  Grounded Input fields: {list(grounded_inputs.values)}")

    # ── Step 1: Build concrete personas directly from Stage 1's 14-field projection ──
    # No LLM call — we use Stage 1 personas as-is, only extracting `name` from
    # the Korean professional_persona's first word (e.g. "오정진 씨는 ..." → "오정진").
    if simulation_personas_path is None:
        simulation_personas_path = stage1_result_path.parent / "simulation_personas.json"
    if not simulation_personas_path.exists():
        raise FileNotFoundError(
            f"simulation_personas.json not found: {simulation_personas_path}. "
            f"Re-run stage1 to generate scenario-grouped personas."
        )
    with open(simulation_personas_path, "r", encoding="utf-8") as f:
        sim_data = json.load(f)

    # Optional scenario cut: keep only scenarios[lo-1:hi] (1-indexed inclusive).
    # Lets the caller run e.g. 1-50 of 300 pre-built scenarios per cut, with
    # the rest reserved for a later run.
    if scenario_range is not None:
        all_scenarios = sim_data.get("scenarios") or []
        lo, hi = scenario_range
        total = len(all_scenarios)
        if lo > total:
            raise ValueError(
                f"scenario_range {lo}-{hi} starts past available scenarios (have {total})."
            )
        hi_clamped = min(hi, total)
        sim_data = {**sim_data, "scenarios": all_scenarios[lo - 1 : hi_clamped]}
        sim_data["scenario_count"] = len(sim_data["scenarios"])
        sim_data["scenario_range"] = {"lo": lo, "hi": hi_clamped, "total_available": total}
        print(f"  scenario cut: {lo}-{hi_clamped} of {total} ({len(sim_data['scenarios'])} scenarios this run)")

    # Snapshot the input personas file into the run_dir so each run is self-contained.
    save_json(sim_data, run_dir / "simulation_personas.json")

    constructed_personas_path = run_dir / "constructed_personas.json"
    resume_personas_path = resume_dir / "constructed_personas.json"
    if resume_from and resume_personas_path.exists():
        print(f"  [resume] Loading constructed personas from {resume_personas_path}")
        with open(resume_personas_path, "r", encoding="utf-8") as f:
            constructed_personas = [ConcretePersona(**p) for p in json.load(f)]
        save_json([p.model_dump() for p in constructed_personas], constructed_personas_path)
    else:
        print("Building Stage 2 personas from Stage 1 simulation_personas.json (no LLM)...")
        constructed_personas = _build_personas_from_stage1_personas(sim_data)
        save_json([p.model_dump() for p in constructed_personas], constructed_personas_path)
    print(f"  Constructed: {[p.name for p in constructed_personas]}")

    # ── Step 2: Build inner ToC schema (cached) ───────────────────────────────
    schema_path = run_dir / cfg.get("inner_schema_file", "inner_schema.json")
    if simple_prediction:
        print("Building simple inner schema (1-phase Impact-only; no LLM call)...")
        inner_schema = build_simple_inner_schema(simulation_target)
        save_json(inner_schema.model_dump(), schema_path)
    elif inner_schema_override and inner_schema_override.exists():
        print(f"  Using inner-schema override: {inner_schema_override}")
        with open(inner_schema_override, "r", encoding="utf-8") as f:
            inner_schema = ToCInnerSchema(**json.load(f))
        save_json(inner_schema.model_dump(), schema_path)
    elif resume_from and (resume_dir / cfg.get("inner_schema_file", "inner_schema.json")).exists():
        rsp = resume_dir / cfg.get("inner_schema_file", "inner_schema.json")
        print(f"  [resume] Loading inner schema from {rsp}")
        with open(rsp, "r", encoding="utf-8") as f:
            inner_schema = ToCInnerSchema(**json.load(f))
        save_json(inner_schema.model_dump(), schema_path)
    else:
        print("Building inner ToC schema...")
        inner_schema = build_inner_toc_schema(policy_context, simulation_target, prediction_results, bridged)
    if grounded_inputs is not None:
        inner_schema = apply_grounded_inputs_to_schema(inner_schema, grounded_inputs)
    save_json(inner_schema.model_dump(), schema_path)
    print(f"  Inner schema phases: {[p.phase for p in inner_schema.phases]}")
    for p in inner_schema.phases:
        print(f"    {p.phase}: fields={[f.name for f in p.prediction_fields]}")

    # ── Step 3: Wire up scenario participants → ConcretePersona objects ───────
    # `sim_data` was already loaded in step 1; we just register the org context
    # and map each scenario's participant records to ConcretePersona instances.
    org_kg_contexts = (sim_data.get("shared_context") or {}).get("org_kg_contexts") or []
    set_org_kg_contexts(org_kg_contexts)
    print(f"  shared_context: {len(org_kg_contexts)} org_kg_context(s) registered for prompt injection")

    # persona_id is the canonical Stage 1-to-Stage 2 mapping key.
    persona_by_persona_id: dict[str, ConcretePersona] = {}
    for cp in constructed_personas:
        pid = getattr(cp, "persona_id", None) or cp.model_dump().get("persona_id")
        if pid:
            persona_by_persona_id[pid] = cp
    scenarios_spec = sim_data.get("scenarios") or []

    def _resolve_combo(participants_records: list[dict]) -> list[ConcretePersona]:
        """Given a scenario's participant records, return the matching ConcretePersona
        objects. Match strategy: persona_id (canonical) → unused persona of the
        same stakeholder_type for legacy bundles."""
        out = []
        used_ids: set[int] = set()
        by_stype: dict[str, list[ConcretePersona]] = {}
        for cp in constructed_personas:
            by_stype.setdefault(cp.stakeholder_type, []).append(cp)
        for rec in participants_records:
            cp = None
            # Primary lookup: persona_id (unique per persona)
            pid = rec.get("persona_id")
            if pid:
                cp = persona_by_persona_id.get(pid)
            # Legacy fallback: any unused persona of the same stakeholder_type.
            if cp is None:
                pool = by_stype.get(rec.get("stakeholder_type", ""), [])
                cp = next((p for p in pool if id(p) not in used_ids), None)
                if cp is None and pool:
                    cp = pool[0]
            if cp is not None:
                out.append(cp)
                used_ids.add(id(cp))
        return out

    # ── Load already-completed scenarios if resuming ──────────────────────────
    completed: dict[int, dict] = {}
    if resume_from:
        for path in resume_dir.glob("scenario_*.json"):
            try:
                idx = int(path.stem.split("_")[1])
                completed[idx] = json.load(open(path))
            except Exception:
                pass
        if completed:
            print(f"  [resume] {len(completed)} scenarios already done; copying to {run_dir}")
            for idx, data in completed.items():
                save_json(data, run_dir / f"scenario_{idx}.json")

    # ── Step 4: Run scenarios (parallelism via ThreadPool) ────────────────────
    token_tracker.enable()
    token_tracker.reset()
    simulation_start = time.time()

    run_log = {
        "model": config["llm"]["model"],
        "temperature": config["llm"].get("temperature", 0.7),
        "simulation_target": simulation_target,
        "start_time": datetime.now().isoformat(),
        "scenarios": [],
    }

    parallel_fb = bool(cfg.get("parallel_forward_backward", True))
    parallel_scn = bool(cfg.get("parallel_scenarios", True))

    def _run_one_scenario(scenario_idx: int, scenario_spec: dict) -> tuple[int, ScenarioToCResult, dict]:
        sid_base = scenario_spec.get("scenario_id") or f"scenario_{scenario_idx}"
        combo = _resolve_combo(scenario_spec.get("participants") or [])
        if not combo:
            print(f"  [{scenario_idx}] {sid_base} — skipped (no resolvable personas)")
            empty = ScenarioToCResult(
                scenario_id=sid_base, participants=[], forward_pass=None,
                backward_pass=None, cross_checks=[], scenario_impact={}, scenario_confidence=0.0,
            )
            return scenario_idx, empty, {"scenario_id": sid_base, "status": "no_personas"}

        scenario_id = f"S{scenario_idx}_" + "_".join(p.name.replace(" ", "-")[:12] for p in combo)
        if scenario_idx in completed:
            print(f"  [{scenario_idx}] {scenario_id} — skipped (resume)")
            return scenario_idx, ScenarioToCResult(**completed[scenario_idx]), {"scenario_id": scenario_id, "status": "resumed"}

        print(f"  [{scenario_idx}] {scenario_id} ({len(combo)} participants)")
        scenario_start = time.time()
        pre_snapshot = token_tracker.snapshot()

        forward_pass = None
        backward_pass = None
        fwd_bwd_fwd_pass = None
        fwd_bwd_fwd_bwd_pass = None

        scenario_path = run_dir / f"scenario_{scenario_idx}.json"

        def _checkpoint(stage: str):
            partial = aggregate_scenario(
                scenario_id, combo, forward_pass, backward_pass, [],
                fwd_bwd_fwd_pass=fwd_bwd_fwd_pass,
                fwd_bwd_fwd_bwd_pass=fwd_bwd_fwd_bwd_pass,
            )
            save_json(partial.model_dump(), scenario_path)
            print(f"    [{scenario_id}] [checkpoint] saved after {stage}")

        # Within-scenario parallelism: forward and backward are independent when
        # sequential_backward=false (the default and forced in our config).
        if parallel_fb and do_forward and do_backward and not cfg.get("sequential_backward", False):
            print(f"    [{scenario_id}] forward||backward in parallel")
            with ThreadPoolExecutor(max_workers=2) as inner_ex:
                fwd_future = inner_ex.submit(
                    run_forward_pass, combo, inner_schema, policy_context, simulation_target, bridged,
                    grounded_inputs=grounded_inputs,
                )
                bwd_future = inner_ex.submit(
                    run_backward_pass, combo, inner_schema, policy_context, simulation_target, bridged, None,
                    grounded_inputs=grounded_inputs,
                )
                forward_pass = fwd_future.result()
                backward_pass = bwd_future.result()
            _checkpoint("forward+backward")
        else:
            if do_forward:
                print(f"    [{scenario_id}] --- forward pass ---")
                forward_pass = run_forward_pass(
                    combo, inner_schema, policy_context, simulation_target, bridged,
                    grounded_inputs=grounded_inputs,
                )
                _checkpoint("forward")
            if do_backward:
                print(f"    [{scenario_id}] --- backward pass ---")
                backward_pass = run_backward_pass(
                    combo, inner_schema, policy_context, simulation_target, bridged,
                    forward_pass=forward_pass,
                    grounded_inputs=grounded_inputs,
                )
                _checkpoint("backward")

        if (forward_pass is not None and backward_pass is not None
                and cfg.get("final_forward_refinement", True)):
            print(f"    [{scenario_id}] --- final forward refinement (sees backward) ---")
            forward_pass = refine_forward_with_backward(
                forward_pass, backward_pass, combo, inner_schema, policy_context, bridged,
                simulation_target=simulation_target,
            )
            _checkpoint("final_forward_refinement")

        if (forward_pass is not None and backward_pass is not None
                and cfg.get("fwd_bwd_fwd_seq", True)):
            print(f"    [{scenario_id}] --- fwd_bwd_fwd_seq (5th chain) ---")
            bwd_inputs = next((r for r in backward_pass if r.phase == "Inputs"), None)
            if bwd_inputs is None:
                print(f"    [{scenario_id}] [WARN] backward Inputs phase missing; skipping 5th chain")
            else:
                bwd_inputs_agg = _aggregate_phase_round(bwd_inputs)
                inputs_seed = (
                    "[SEQUENTIAL MODE] Backward direction derived the following Inputs-phase "
                    "aggregated values (round-tripped from the Fwd→Bwd chain). Treat these as "
                    "a hypothesized upstream-conditions snapshot to forward-simulate from:\n"
                    + json.dumps(bwd_inputs_agg, ensure_ascii=False, indent=2)
                )
                print(f"    [{scenario_id}] [fwd_bwd_fwd_seq] seeded from backward Inputs: {bwd_inputs_agg}")
                fwd_bwd_fwd_pass = run_forward_pass(
                    combo, inner_schema, policy_context, simulation_target, bridged,
                    inputs_seed=inputs_seed,
                    backward_pass=backward_pass,
                    grounded_inputs=grounded_inputs,
                )
                _checkpoint("fwd_bwd_fwd_seq (5th)")

        if (forward_pass is not None and backward_pass is not None
                and fwd_bwd_fwd_pass is not None
                and cfg.get("fwd_bwd_fwd_bwd_seq", True)):
            print(f"    [{scenario_id}] --- fwd_bwd_fwd_bwd_seq (6th chain) ---")
            fwd_bwd_fwd_bwd_pass = run_backward_pass(
                combo, inner_schema, policy_context, simulation_target, bridged,
                forward_pass=fwd_bwd_fwd_pass,
                grounded_inputs=grounded_inputs,
            )
            _checkpoint("fwd_bwd_fwd_bwd_seq (6th)")

        cross_checks = cross_check_phases(forward_pass, backward_pass, inner_schema)
        scenario_result = aggregate_scenario(
            scenario_id, combo, forward_pass, backward_pass, cross_checks,
            fwd_bwd_fwd_pass=fwd_bwd_fwd_pass,
            fwd_bwd_fwd_bwd_pass=fwd_bwd_fwd_bwd_pass,
        )
        save_json(scenario_result.model_dump(), scenario_path)

        post_snapshot = token_tracker.snapshot()
        scenario_elapsed = time.time() - scenario_start
        log_entry = {
            "scenario_id": scenario_id,
            "participants": [p.name for p in combo],
            "llm_calls": post_snapshot["call_count"] - pre_snapshot["call_count"],
            "prompt_tokens": post_snapshot["prompt_tokens"] - pre_snapshot["prompt_tokens"],
            "completion_tokens": post_snapshot["completion_tokens"] - pre_snapshot["completion_tokens"],
            "total_tokens": post_snapshot["total_tokens"] - pre_snapshot["total_tokens"],
            "elapsed_sec": round(scenario_elapsed, 2),
        }
        print(f"    [{scenario_id}] +{log_entry['total_tokens']} tokens "
              f"({log_entry['llm_calls']} calls, {round(scenario_elapsed, 1)}s)")
        return scenario_idx, scenario_result, log_entry

    scenario_results: list[ScenarioToCResult] = [None] * len(scenarios_spec)
    log_entries: list[Optional[dict]] = [None] * len(scenarios_spec)

    if parallel_scn and len(scenarios_spec) > 1:
        print(f"  Running {len(scenarios_spec)} scenarios in parallel "
              f"(forward||backward inside each = up to {2*len(scenarios_spec)} in-flight LLM threads)")
        with ThreadPoolExecutor(max_workers=len(scenarios_spec)) as ex:
            futures = {
                ex.submit(_run_one_scenario, i + 1, spec): i
                for i, spec in enumerate(scenarios_spec)
            }
            for fut in as_completed(futures):
                idx_pos = futures[fut]
                scenario_idx, sresult, log_entry = fut.result()
                scenario_results[idx_pos] = sresult
                log_entries[idx_pos] = log_entry
    else:
        for i, spec in enumerate(scenarios_spec):
            scenario_idx, sresult, log_entry = _run_one_scenario(i + 1, spec)
            scenario_results[i] = sresult
            log_entries[i] = log_entry

    run_log["scenarios"] = [e for e in log_entries if e is not None]
    scenario_results = [r for r in scenario_results if r is not None]

    # ── Step 5: Aggregate + final reporter ────────────────────────────────────
    # The deterministic aggregation + run_log are persisted FIRST, before any
    # post-scenario LLM call (narrative reporter, per-stakeholder synthesizer).
    # That way a 400/Payment/timeout in those calls — which can blow up on
    # large cuts (50+ scenarios) because the prompts include every scenario's
    # cross_checks — never destroys the quantitative output of a run that
    # already cost real money to produce.
    print("Aggregating across scenarios...")
    aggregated_final, impl_gap_report = aggregate_across_scenarios(scenario_results, inner_schema)

    # Persist quantitative aggregation + run_log immediately (cheap, deterministic).
    save_json({
        "simulation_target": simulation_target,
        "scenario_count": len(scenario_results),
        "aggregated_final": aggregated_final,
        "implementation_gap_report": [c.model_dump() for c in impl_gap_report],
    }, run_dir / "aggregated_quantitative.json")

    total_elapsed = time.time() - simulation_start
    total_snapshot = token_tracker.snapshot()
    run_log["end_time"] = datetime.now().isoformat()
    run_log["total"] = {
        **total_snapshot,
        "wall_time_sec": round(total_elapsed, 2),
        "wall_time_min": round(total_elapsed / 60, 1),
    }
    save_json(run_log, run_dir / "run_log.json")
    print(f"  Quantitative aggregation + run_log saved (pre-LLM-aggregator checkpoint)")
    print(f"  Total: {total_snapshot['total_tokens']} tokens, "
          f"{total_snapshot['call_count']} calls, {round(total_elapsed / 60, 1)} min")

    # LLM-driven qualitative aggregators are OPT-IN. Default is quantitative-only
    # because (a) the per-(phase, stakeholder) synthesizer regularly hits 400
    # Bad Request on cuts ≥ 50 scenarios (prompt too large), (b) the narrative
    # reporter is rarely consumed by downstream analysis, and (c) both add
    # latency/cost without contributing to the headline quantitative results.
    # Re-enable with `stage2_toc.run_llm_aggregators: true` in the config.
    narrative: str = ""
    phase_stakeholder_summaries: dict = {}
    if cfg.get("run_llm_aggregators", False):
        try:
            print("Running final ToC reporter...")
            narrative = run_final_toc_reporter(
                simulation_target, scenario_results, aggregated_final, impl_gap_report, bridged,
            )
        except Exception as exc:
            print(f"  [warn] final ToC reporter failed: {exc!r} — narrative_synthesis will be empty.")

        try:
            print("Running per-(phase, stakeholder) argument synthesizer...")
            phase_stakeholder_summaries = run_phase_stakeholder_synthesizer(
                scenario_results, simulation_target, bridged,
            )
        except Exception as exc:
            print(f"  [warn] phase-stakeholder synthesizer failed: {exc!r} — phase_stakeholder_summaries will be empty.")
    else:
        print("LLM aggregators skipped (set stage2_toc.run_llm_aggregators=true to enable).")

    token_tracker.disable()

    final = ToCSimulationResult(
        simulation_target=simulation_target,
        inner_schema=inner_schema,
        constructed_personas=constructed_personas,
        scenarios=scenario_results,
        aggregated_final=aggregated_final,
        implementation_gap_report=impl_gap_report,
        narrative_synthesis=narrative,
        phase_stakeholder_summaries=phase_stakeholder_summaries,
    )
    out_path = run_dir / cfg.get("output_file", "stage2_toc_result.json")
    save_json(final.model_dump(), out_path)

    save_json({
        "simulation_target": simulation_target,
        "aggregated_final": aggregated_final,
        "implementation_gap_report": [c.model_dump() for c in impl_gap_report],
        "narrative_synthesis": narrative,
        "phase_stakeholder_summaries": phase_stakeholder_summaries,
    }, run_dir / "final_report.json")

    print(f"  aggregated_final={aggregated_final}")
    return final
