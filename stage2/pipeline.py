import json
from pathlib import Path

from models.schemas import Stage1Result, FinalReport, PolicyReportResult
from stage2.modules import construct_all_simulation_personas, run_concrete_scenario, run_final_reporter


def save_json(data: dict | list, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _build_graph_scenarios(
    constructed_personas: list,
    graph_data: dict,
) -> list[tuple[list, list[dict]]]:
    """
    Build diverse scenarios from the persona-only subgraph using multi-scale enumeration:

    Scale 1 — each unique pair of personas with a direct edge (focused 2-person debate)
    Scale 2 — each triangle (3 mutually connected personas)
    Scale 3 — each connected component (grand N-person debate)

    Each scenario carries all edges within its participant set as relationship context.
    Deduplicates by frozenset of participants across all scales.

    Returns list of (participants: list[ConcretePersona], relationships: list[dict]).
    """
    from itertools import combinations

    entity_to_persona = {cp.entity_name: cp for cp in constructed_personas if cp.entity_name}
    persona_names = set(entity_to_persona.keys())

    name_to_uuid = {n["name"]: n["uuid"] for n in graph_data.get("nodes", [])}
    uuid_to_name = {n["uuid"]: n["name"] for n in graph_data.get("nodes", [])}

    # Build persona-only edge list (deduplicated by name-pair)
    persona_edges: dict[frozenset, list[dict]] = {}
    for edge in graph_data.get("edges", []):
        src_name = uuid_to_name.get(edge["source_node_uuid"])
        tgt_name = uuid_to_name.get(edge["target_node_uuid"])
        if src_name in persona_names and tgt_name in persona_names and src_name != tgt_name:
            key = frozenset([src_name, tgt_name])
            persona_edges.setdefault(key, []).append({
                "source": src_name,
                "target": tgt_name,
                "relationship": edge["name"],
                "fact": edge["fact"],
            })

    # Build adjacency set (persona name → set of directly connected persona names)
    adjacency: dict[str, set] = {}
    for pair in persona_edges:
        a, b = tuple(pair)
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    def get_edges_for_group(group: set[str]) -> list[dict]:
        """Collect all edges between members of the group."""
        result = []
        for pair, edges in persona_edges.items():
            if pair.issubset(group):
                result.extend(edges)
        return result

    seen: set[frozenset] = set()
    scenarios = []

    def add_scenario(group: set[str]) -> None:
        key = frozenset(group)
        if key in seen or len(group) < 2:
            return
        seen.add(key)
        participants = [entity_to_persona[n] for n in group]
        rel_ctx = get_edges_for_group(group)
        scenarios.append((participants, rel_ctx))

    # Scale 1 — all direct pairs
    for pair in persona_edges:
        add_scenario(set(pair))

    # Scale 2 — all triangles
    connected_personas = list(adjacency.keys())
    for a, b, c in combinations(connected_personas, 3):
        if b in adjacency.get(a, set()) and c in adjacency.get(a, set()) and c in adjacency.get(b, set()):
            add_scenario({a, b, c})

    # Scale 3 — connected components (BFS)
    visited: set[str] = set()
    for start in connected_personas:
        if start in visited:
            continue
        component: set[str] = set()
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            queue.extend(adjacency.get(node, set()) - visited)
        if len(component) >= 2:
            add_scenario(component)

    return scenarios


def _build_cartesian_scenarios(
    constructed_personas: list,
    stakeholder_groups: list,
    max_scenarios: int | None = None,
) -> list[tuple[list, list[dict]]]:
    """
    Build scenarios as Cartesian product: one persona per stakeholder type.
    Each combination forms one scenario. No relationship context (empty list).
    If max_scenarios is set and the total exceeds it, randomly sample that many.
    """
    import random
    from itertools import product as cartesian_product

    # Group all concrete personas by stakeholder type
    groups: dict[str, list] = {}
    for cp in constructed_personas:
        stype = cp.stakeholder_type
        if stype:
            groups.setdefault(stype, []).append(cp)

    combos = list(cartesian_product(*groups.values()))
    if max_scenarios and len(combos) > max_scenarios:
        print(f"  Sampling {max_scenarios} of {len(combos)} possible Cartesian combinations")
        combos = random.sample(combos, max_scenarios)
    return [(list(combo), []) for combo in combos]


def run_stage2(config: dict, run_dir: Path, simulation_target: str,
               stage1_result_path: Path | None = None,
               graph_data_path: Path | None = None,
               use_graph: bool = False,
               use_initial_personas: bool = False,
               resume_from: Path | None = None) -> FinalReport:
    if stage1_result_path is None:
        stage1_result_path = run_dir / "stage1_result.json"
    if not stage1_result_path.exists():
        raise FileNotFoundError(f"Stage1 result not found: {stage1_result_path}")

    with open(stage1_result_path, "r", encoding="utf-8") as f:
        stage1_result = Stage1Result(**json.load(f))

    policy = stage1_result.policy_plan
    context = stage1_result.economic_context
    prediction_results = stage1_result.predictions

    if use_initial_personas:
        personas_initial_path = stage1_result_path.parent / "personas_initial.json"
        if not personas_initial_path.exists():
            raise FileNotFoundError(f"personas_initial.json not found at {personas_initial_path}")
        with open(personas_initial_path, "r", encoding="utf-8") as f:
            from models.schemas import StakeholderGroup
            stakeholder_groups = [StakeholderGroup(**g) for g in json.load(f)]
        print("  [ablation] Using initial (unrefined) personas from personas_initial.json")
    else:
        stakeholder_groups = stage1_result.stakeholder_groups

    # ── Step 1: Construct concrete simulation personas (1-to-1 per abstract type) ──
    print("Constructing simulation personas...")
    constructed_personas = construct_all_simulation_personas(
        stakeholder_groups, prediction_results, config
    )
    save_json([p.model_dump() for p in constructed_personas], run_dir / "constructed_personas.json")
    print(f"  Constructed: {[p.name for p in constructed_personas]}")

    # ── Step 2: Build scenarios ────────────────────────────────────────────────
    if use_graph:
        graph_data = None
        if graph_data_path is None:
            candidate = stage1_result_path.parent / "graph_data.json"
            if candidate.exists():
                graph_data_path = candidate
        if graph_data_path and graph_data_path.exists():
            with open(graph_data_path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
        if graph_data:
            graph_scenarios = _build_graph_scenarios(constructed_personas, graph_data)
            print(f"  {len(graph_scenarios)} graph-based scenarios (multi-scale edge enumeration)")
        else:
            graph_scenarios = [(constructed_personas, [])]
            print("  No graph data found — falling back to single scenario.")
    else:
        max_scenarios = config.get("stage2", {}).get("max_scenarios", None)
        graph_scenarios = _build_cartesian_scenarios(constructed_personas, stakeholder_groups, max_scenarios)
        print(f"  {len(graph_scenarios)} Cartesian scenarios (one per stakeholder type combination)")

    # ── Load already-completed scenarios if resuming ──────────────────────────
    completed: dict[int, dict] = {}
    resume_dir = resume_from or run_dir
    for path in resume_dir.glob("scenario_*.json"):
        try:
            idx = int(path.stem.split("_")[1])
            completed[idx] = json.load(open(path))
        except Exception:
            pass
    if completed:
        print(f"  Resuming: {len(completed)} scenarios already done, copying to {run_dir}")
        for idx, data in completed.items():
            save_json(data, run_dir / f"scenario_{idx}.json")

    from models.schemas import ConcreteScenarioResult
    print(f"Running {len(graph_scenarios)} scenarios...")
    scenario_results = []
    for i, (combo, rel_ctx) in enumerate(graph_scenarios):
        scenario_id = "_".join(p.name.replace(" ", "-")[:12] for p in combo)
        scenario_id = f"S{i+1}_{scenario_id}"

        if (i + 1) in completed:
            print(f"  [{i+1}/{len(graph_scenarios)}] {scenario_id} — skipped (already done)")
            scenario_results.append(ConcreteScenarioResult(**completed[i + 1]))
            continue

        print(f"  [{i+1}/{len(graph_scenarios)}] {scenario_id} ({len(combo)} participants, {len(rel_ctx)} relationships)")
        result = run_concrete_scenario(scenario_id, combo, policy, context, simulation_target, config, rel_ctx)
        save_json(result.model_dump(), run_dir / f"scenario_{i+1}.json")
        scenario_results.append(result)

    report = FinalReport(
        simulation_target=simulation_target,
        stakeholder_groups=stakeholder_groups,
        stage1_predictions=stage1_result.predictions,
        stage1_critic_results=stage1_result.critic_results,
        constructed_personas=constructed_personas,
        scenarios=scenario_results,
    )
    save_json(report.model_dump(), run_dir / "stage2_result.json")

    # ── Final Reporter ─────────────────────────────────────────────────────────
    print("Running final reporter...")
    ground_truth = config.get("stage2", {}).get("ground_truth", None)
    config["_simulation_target"] = simulation_target
    policy_report = run_final_reporter(scenario_results, policy, config, ground_truth=ground_truth)
    save_json(policy_report.model_dump(), run_dir / "final_report.json")
    print(f"  avg_prediction={policy_report.avg_prediction}, range={policy_report.prediction_range}"
          + (f", error={policy_report.error} ({policy_report.error_pct}%)" if policy_report.error is not None else ""))

    return report
