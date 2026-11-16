"""Precompute a complete parent-conditioned ToC pathway tree.

By default the policy Inputs round is fixed, matching the current interface, and
every later phase branches into enabling / baseline / constraining children.
Use ``--branch-inputs`` to branch Inputs from a policy-only root as well.

The output is checkpointed after every expanded parent. Re-running the same
command resumes missing branches without regenerating completed nodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.schemas import (  # noqa: E402
    ConcretePersona,
    PathwayBranchNode,
    PhaseRoundResult,
    PolicyContext,
    ToCInnerSchema,
)
from stage2_toc.branching import (  # noqa: E402
    BRANCH_TRANSITIONS,
    MECHANISM_COVERAGE_LEVELS,
    MECHANISM_PLANNER_MODES,
    generate_phase_children,
    mechanism_planner_mode,
    parent_context_hash,
)
from stage2_toc.runtime import set_org_kg_contexts  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs/branching/pathway_branch_tree.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def load_runtime(
    source_dir: Path,
    scenario_index: int,
    inner_schema_override: Path | None = None,
) -> dict[str, Any]:
    run_log = load_json(source_dir / "run_log.json")
    scenario = load_json(source_dir / f"scenario_{scenario_index}.json")

    def repo_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    config_path = repo_path(run_log.get("config_path") or "config/base.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["stage2_toc"] = {
        **(config.get("stage2_toc") or {}),
        **((run_log.get("config") or {}).get("stage2_toc") or {}),
    }

    simulation_personas = load_json(repo_path(run_log.get("simulation_personas") or str(repo_path(run_log["stage1_result"]).parent / "simulation_personas.json")))
    set_org_kg_contexts((simulation_personas.get("shared_context") or {}).get("org_kg_contexts"))
    stage1 = load_json(repo_path(run_log["stage1_result"]))
    schema_path = inner_schema_override or (source_dir / "inner_schema.json")
    return {
        "config": config,
        "scenario": scenario,
        "policy_context": PolicyContext.model_validate(stage1["policy_context"]),
        "personas": [ConcretePersona.model_validate(p) for p in scenario["participants"]],
        "inner_schema": ToCInnerSchema.model_validate(load_json(schema_path)),
        "simulation_target": run_log["simulation_target"],
    }


def new_tree(runtime: dict[str, Any], source_dir: Path, scenario_index: int, branch_inputs: bool) -> dict:
    if branch_inputs:
        root_result = None
        root_phase_index = -1
    else:
        root_result = PhaseRoundResult.model_validate(runtime["scenario"]["forward_pass"][0])
        root_phase_index = 0
    root = PathwayBranchNode(
        node_id="root",
        parent_id=None,
        transition_mode=None,
        phase_index=root_phase_index,
        parent_context_hash=parent_context_hash([]),
        phase_result=root_result,
    )
    return {
        "format_version": 1,
        "generation_mode": "phase_local_independent_children",
        "source_run_dir": str(source_dir),
        "scenario_index": scenario_index,
        "scenario_id": runtime["scenario"]["scenario_id"],
        "simulation_target": runtime["simulation_target"],
        "branch_inputs": branch_inputs,
        "transitions": list(BRANCH_TRANSITIONS),
        "phases": [spec.phase for spec in runtime["inner_schema"].phases],
        "nodes": {"root": root.model_dump(mode="json")},
        "completed_parent_ids": [],
    }


def history_for(tree: dict, node_id: str) -> list[PhaseRoundResult]:
    nodes = tree["nodes"]
    if node_id not in nodes:
        raise KeyError(f"unknown node: {node_id}")
    history = []
    node = nodes[node_id]
    while node is not None:
        if node.get("phase_result"):
            history.append(PhaseRoundResult.model_validate(node["phase_result"]))
        parent_id = node.get("parent_id")
        node = nodes.get(parent_id) if parent_id else None
    history.reverse()
    return history


def _is_baseline_spine_node(node_id: str) -> bool:
    return node_id == "root" or all(
        segment == "baseline" for segment in node_id.split("/")[1:]
    )


def pending_parent_ids(
    tree: dict,
    phase_count: int,
    expansion_strategy: str = "full_tree",
) -> deque[str]:
    completed = set(tree.get("completed_parent_ids") or [])
    candidates = sorted(
        tree["nodes"].values(),
        key=lambda node: (node["phase_index"], node["node_id"]),
    )
    return deque(
        node["node_id"] for node in candidates
        if node["phase_index"] < phase_count - 1 and node["node_id"] not in completed
        and (
            expansion_strategy == "full_tree"
            or _is_baseline_spine_node(node["node_id"])
        )
    )


def build_tree(
    tree: dict,
    runtime: dict[str, Any],
    output: Path,
    transitions: tuple[str, ...] = BRANCH_TRANSITIONS,
    parallel_transitions: bool = False,
    parallel_parents: int = 1,
    max_expansions: int | None = None,
    expansion_strategy: str = "full_tree",
    stop_after_phase: str | None = None,
) -> dict:
    phase_names = [phase.phase for phase in runtime["inner_schema"].phases]
    if stop_after_phase is not None:
        if stop_after_phase not in phase_names:
            raise ValueError(
                f"stop phase {stop_after_phase!r} not found in schema phases {phase_names}"
            )
        phase_count = phase_names.index(stop_after_phase) + 1
    else:
        phase_count = len(phase_names)
    queue = pending_parent_ids(tree, phase_count, expansion_strategy)
    expanded_this_run = 0
    planner_enabled = bool(
        ((runtime["config"].get("stage2_toc") or {}).get("branch_mechanism_planner") or {}).get(
            "enabled", False,
        )
    )

    def _coverage_levels_for(parent_id: str) -> dict[str, str] | None:
        if not planner_enabled:
            return None
        planner_cfg = (
            (runtime["config"].get("stage2_toc") or {}).get("branch_mechanism_planner") or {}
        )
        override = planner_cfg.get("coverage_level_override")
        if override:
            return {"enabling": override, "constraining": override}

        parent_phase_index = tree["nodes"][parent_id]["phase_index"]
        peer_ids = [
            node["node_id"] for node in tree["nodes"].values()
            if node["phase_index"] == parent_phase_index
            and node["phase_index"] < phase_count - 1
        ]
        levels = {}
        for transition in ("enabling", "constraining"):
            ordered = sorted(
                peer_ids,
                key=lambda node_id: hashlib.sha256(
                    f"{node_id}|{transition}".encode("utf-8")
                ).hexdigest(),
            )
            if len(ordered) == 1:
                levels[transition] = "material"
            else:
                levels[transition] = MECHANISM_COVERAGE_LEVELS[
                    ordered.index(parent_id) % len(MECHANISM_COVERAGE_LEVELS)
                ]
        return levels

    def _expand(parent_id: str) -> tuple[str, list[PathwayBranchNode]]:
        history = history_for(tree, parent_id)
        children = generate_phase_children(
            parent_id=parent_id,
            prior_phases=history,
            personas=runtime["personas"],
            inner_schema=runtime["inner_schema"],
            policy_context=runtime["policy_context"],
            simulation_target=runtime["simulation_target"],
            config=runtime["config"],
            transitions=transitions,
            parallel_transitions=parallel_transitions,
            coverage_levels=_coverage_levels_for(parent_id),
        )
        return parent_id, children

    while queue and (max_expansions is None or expanded_this_run < max_expansions):
        remaining = (
            len(queue)
            if max_expansions is None
            else min(len(queue), max_expansions - expanded_this_run)
        )
        batch_size = min(parallel_parents, remaining)
        parent_batch = [queue.popleft() for _ in range(batch_size)]

        if len(parent_batch) > 1:
            with ThreadPoolExecutor(max_workers=len(parent_batch)) as executor:
                expanded_batch = list(executor.map(_expand, parent_batch))
        else:
            expanded_batch = [_expand(parent_batch[0])]

        for parent_id, children in expanded_batch:
            for child in children:
                tree["nodes"][child.node_id] = child.model_dump(mode="json")
                if (
                    child.phase_index < phase_count - 1
                    and (
                        expansion_strategy == "full_tree"
                        or child.transition_mode == "baseline"
                    )
                ):
                    queue.append(child.node_id)
            tree["completed_parent_ids"].append(parent_id)
            expanded_this_run += 1
            atomic_save(tree, output)
            print(
                f"expanded {parent_id}: +{len(children)} children; "
                f"nodes={len(tree['nodes'])}, pending={len(queue)}"
            )

    return tree


def generation_plan(
    phase_count: int,
    persona_count: int,
    branch_inputs: bool,
    branch_factor: int = len(BRANCH_TRANSITIONS),
    interaction_mode: str = "independent",
    planner_calls_per_parent: int = 0,
    expansion_strategy: str = "full_tree",
) -> dict[str, int]:
    branching_depth = phase_count if branch_inputs else phase_count - 1
    if expansion_strategy == "baseline_spine":
        generated_nodes = branch_factor * branching_depth
        complete_paths = branch_factor
        expanded_parents = branching_depth
    else:
        generated_nodes = sum(branch_factor ** depth for depth in range(1, branching_depth + 1))
        complete_paths = branch_factor ** branching_depth
        expanded_parents = sum(branch_factor ** depth for depth in range(branching_depth))
    persona_calls = persona_count * (2 if interaction_mode == "independent_revision" else 1)
    llm_calls = generated_nodes * (persona_calls + 1)
    llm_calls += expanded_parents * planner_calls_per_parent
    return {
        "generated_nodes": generated_nodes,
        "complete_paths": complete_paths,
        "estimated_llm_calls": llm_calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--inner-schema",
        type=Path,
        default=None,
        help="Override the source run's inner schema (useful with an Inputs-only bootstrap run)",
    )
    parser.add_argument("--branch-inputs", action="store_true")
    parser.add_argument(
        "--transitions",
        default=",".join(BRANCH_TRANSITIONS),
        help="Comma-separated local transitions to generate: enabling,baseline,constraining",
    )
    parser.add_argument(
        "--interaction-mode",
        choices=("sequential", "independent", "independent_revision"),
        default=None,
        help="Override stage2_toc.phase_interaction_mode for this tree",
    )
    parser.add_argument(
        "--parallel-transitions",
        action="store_true",
        help="Generate sibling transition modes concurrently for each parent",
    )
    parser.add_argument(
        "--parallel-parents",
        type=int,
        default=1,
        help="Number of independent parent nodes to expand concurrently",
    )
    parser.add_argument(
        "--mechanism-planner",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the optional shared E/C mechanism planner",
    )
    parser.add_argument(
        "--coverage-level",
        choices=MECHANISM_COVERAGE_LEVELS,
        default=None,
        help="Force one planner coverage level for a controlled pilot",
    )
    parser.add_argument(
        "--planner-mode",
        choices=MECHANISM_PLANNER_MODES,
        default=None,
        help="per_transition uses one E/C call each; joint_siblings plans E/B/C together once per parent",
    )
    parser.add_argument(
        "--branch-prompt-experiment",
        choices=(
            "current", "intensity", "persona_constraint", "persona_mechanism_coverage",
            "role_constraint_activation",
        ),
        default=None,
        help="Apply one additive persona-prompt experiment without changing the base branch prompt",
    )
    parser.add_argument(
        "--expansion-strategy",
        choices=("full_tree", "baseline_spine"),
        default="full_tree",
        help="Expand every child or only the baseline child while retaining E/B/C siblings",
    )
    parser.add_argument("--restart", action="store_true", help="Ignore an existing checkpoint and start over")
    parser.add_argument("--max-expansions", type=int, default=None, help="Stop after N parents (smoke testing)")
    parser.add_argument(
        "--stop-after-phase",
        default=None,
        help="Checkpoint after this phase; omit later to resume through the remaining phases",
    )
    parser.add_argument("--plan-only", action="store_true", help="Print tree size and estimated call count without writing")
    args = parser.parse_args()
    if args.parallel_parents < 1:
        parser.error("--parallel-parents must be at least 1")

    transitions = tuple(part.strip() for part in args.transitions.split(",") if part.strip())
    invalid = set(transitions) - set(BRANCH_TRANSITIONS)
    if not transitions or invalid or len(set(transitions)) != len(transitions):
        parser.error(
            "--transitions must contain one or more unique values from "
            f"{list(BRANCH_TRANSITIONS)}; got {list(transitions)}"
        )

    runtime = load_runtime(args.source_dir, args.scenario, args.inner_schema)
    phase_names = [phase.phase for phase in runtime["inner_schema"].phases]
    if args.stop_after_phase and args.stop_after_phase not in phase_names:
        parser.error(
            f"--stop-after-phase must be one of {phase_names}; got {args.stop_after_phase!r}"
        )
    effective_phase_count = (
        phase_names.index(args.stop_after_phase) + 1
        if args.stop_after_phase
        else len(phase_names)
    )
    if args.interaction_mode:
        runtime["config"].setdefault("stage2_toc", {})["phase_interaction_mode"] = args.interaction_mode
    if args.branch_prompt_experiment:
        runtime["config"].setdefault("stage2_toc", {})[
            "branch_prompt_experiment"
        ] = args.branch_prompt_experiment
    interaction_mode = runtime["config"].get("stage2_toc", {}).get("phase_interaction_mode")
    branch_prompt_experiment = (
        runtime["config"].get("stage2_toc", {}).get("branch_prompt_experiment") or "current"
    )
    planner_cfg = runtime["config"].setdefault("stage2_toc", {}).setdefault(
        "branch_mechanism_planner", {},
    )
    if args.mechanism_planner is not None:
        planner_cfg["enabled"] = args.mechanism_planner
    if args.planner_mode:
        planner_cfg["mode"] = args.planner_mode
    if args.coverage_level:
        planner_cfg["coverage_level_override"] = args.coverage_level
    planner_enabled = bool(planner_cfg.get("enabled", False))
    planner_mode = mechanism_planner_mode(runtime["config"])
    if planner_enabled and planner_mode == "joint_siblings":
        planner_calls_per_parent = 1
    elif planner_enabled:
        planner_calls_per_parent = sum(
            transition in {"enabling", "constraining"} for transition in transitions
        )
    else:
        planner_calls_per_parent = 0
    plan = generation_plan(
        effective_phase_count,
        len(runtime["personas"]),
        args.branch_inputs,
        branch_factor=len(transitions),
        interaction_mode=interaction_mode,
        planner_calls_per_parent=planner_calls_per_parent,
        expansion_strategy=args.expansion_strategy,
    )
    print(
        f"plan: generated_nodes={plan['generated_nodes']}, "
        f"complete_paths={plan['complete_paths']}, "
        f"estimated_llm_calls={plan['estimated_llm_calls']}"
    )
    if args.plan_only:
        return
    if args.output.exists() and not args.restart:
        tree = load_json(args.output)
        expected = (str(args.source_dir), args.scenario, args.branch_inputs)
        actual = (tree.get("source_run_dir"), tree.get("scenario_index"), tree.get("branch_inputs"))
        if actual != expected:
            raise ValueError(f"checkpoint source/options mismatch: expected {expected}, found {actual}")
        print(f"resuming {args.output} with {len(tree['nodes'])} nodes")
    else:
        tree = new_tree(runtime, args.source_dir, args.scenario, args.branch_inputs)
        tree["transitions"] = list(transitions)
        tree["phase_interaction_mode"] = interaction_mode
        tree["mechanism_planner_enabled"] = planner_enabled
        tree["mechanism_planner_mode"] = planner_mode
        tree["branch_prompt_experiment"] = branch_prompt_experiment
        tree["expansion_strategy"] = args.expansion_strategy
        tree["coverage_level_override"] = planner_cfg.get("coverage_level_override")
        atomic_save(tree, args.output)

    if tuple(tree.get("transitions") or ()) != transitions:
        raise ValueError(
            "checkpoint transition mismatch: "
            f"expected {list(transitions)}, found {tree.get('transitions')}"
        )
    if tree.get("phase_interaction_mode") != interaction_mode:
        raise ValueError(
            "checkpoint interaction-mode mismatch: "
            f"expected {interaction_mode!r}, found {tree.get('phase_interaction_mode')!r}"
        )
    if bool(tree.get("mechanism_planner_enabled", False)) != planner_enabled:
        raise ValueError(
            "checkpoint mechanism-planner mismatch: "
            f"expected {planner_enabled}, found {tree.get('mechanism_planner_enabled')}"
        )
    if tree.get("mechanism_planner_mode", "per_transition") != planner_mode:
        raise ValueError(
            "checkpoint mechanism-planner mode mismatch: "
            f"expected {planner_mode!r}, found {tree.get('mechanism_planner_mode')!r}"
        )
    if tree.get("branch_prompt_experiment", "current") != branch_prompt_experiment:
        raise ValueError(
            "checkpoint branch-prompt experiment mismatch: "
            f"expected {branch_prompt_experiment!r}, "
            f"found {tree.get('branch_prompt_experiment', 'current')!r}"
        )
    if tree.get("expansion_strategy", "full_tree") != args.expansion_strategy:
        raise ValueError(
            "checkpoint expansion-strategy mismatch: "
            f"expected {args.expansion_strategy!r}, "
            f"found {tree.get('expansion_strategy', 'full_tree')!r}"
        )
    if tree.get("coverage_level_override") != planner_cfg.get("coverage_level_override"):
        raise ValueError(
            "checkpoint coverage-level mismatch: "
            f"expected {planner_cfg.get('coverage_level_override')!r}, "
            f"found {tree.get('coverage_level_override')!r}"
        )

    build_tree(
        tree,
        runtime,
        args.output,
        transitions=transitions,
        parallel_transitions=args.parallel_transitions,
        parallel_parents=args.parallel_parents,
        max_expansions=args.max_expansions,
        expansion_strategy=args.expansion_strategy,
        stop_after_phase=args.stop_after_phase,
    )
    leaves = sum(
        1 for node in tree["nodes"].values()
        if node["phase_index"] == effective_phase_count - 1
    )
    print(
        f"saved {args.output}: nodes={len(tree['nodes'])}, "
        f"paths_through_{phase_names[effective_phase_count - 1]}={leaves}"
    )


if __name__ == "__main__":
    main()
