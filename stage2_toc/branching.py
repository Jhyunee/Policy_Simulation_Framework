"""Phase-local branching for exploratory Theory of Change pathways.

The legacy pipeline applies one stance to an entire linear run. This module is
additive: it expands one frozen parent history into three independently generated
children whose only designed difference is the local transition directive.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from models.schemas import (
    BranchMechanismAnchor,
    ConcretePersona,
    PathwayBranchNode,
    PhaseRoundResult,
    PolicyContext,
    ToCInnerSchema,
)
from models.prompts import (
    BRANCH_MECHANISM_PLANNER_SYSTEM,
    BRANCH_MECHANISM_PLANNER_USER,
    JOINT_SIBLING_PLANNER_SYSTEM,
    JOINT_SIBLING_PLANNER_USER,
)
from llm.client import chat_completion
from stage2_toc.modules import _phase_round, resolve_configured_posting_order


# These names describe the local condition applied after a fixed parent state;
# they do not promise a globally good, average, or bad outcome.
BRANCH_TRANSITIONS = ("enabling", "baseline", "constraining")
MECHANISM_COVERAGE_LEVELS = ("incremental", "material", "boundary")
MECHANISM_PLANNER_MODES = ("per_transition", "joint_siblings")


def mechanism_planner_mode(config: dict) -> str:
    planner_cfg = (config.get("stage2_toc") or {}).get("branch_mechanism_planner") or {}
    mode = str(planner_cfg.get("mode") or "per_transition")
    if mode not in MECHANISM_PLANNER_MODES:
        raise ValueError(
            "branch_mechanism_planner.mode must be one of "
            f"{list(MECHANISM_PLANNER_MODES)}; got {mode!r}"
        )
    return mode


def parent_context_hash(prior_phases: Iterable[PhaseRoundResult]) -> str:
    payload = [phase.model_dump(mode="json") for phase in prior_phases]
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def mechanism_coverage_level(
    *,
    parent_id: str,
    parent_hash: str,
    phase_index: int,
    transition_mode: str,
    config: dict,
) -> str:
    planner_cfg = (config.get("stage2_toc") or {}).get("branch_mechanism_planner") or {}
    override = planner_cfg.get("coverage_level_override")
    if override:
        if override not in MECHANISM_COVERAGE_LEVELS:
            raise ValueError(
                "branch_mechanism_planner.coverage_level_override must be one of "
                f"{list(MECHANISM_COVERAGE_LEVELS)}; got {override!r}"
            )
        return override
    key = f"{parent_id}|{parent_hash}|{phase_index}|{transition_mode}"
    index = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % len(MECHANISM_COVERAGE_LEVELS)
    return MECHANISM_COVERAGE_LEVELS[index]


def _aggregate_parent_phase(phase: PhaseRoundResult) -> dict:
    if phase.state_type == "document_grounded":
        return dict(phase.grounded_values)
    if not phase.revised_posts:
        return {}
    values = {}
    for field in phase.revised_posts[0].prediction_values:
        candidates = [post.prediction_values.get(field) for post in phase.revised_posts]
        numeric = [
            value for value in candidates
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        values[field] = sum(numeric) / len(numeric) if len(numeric) == len(candidates) else candidates[0]
    return values


def plan_branch_mechanism(
    *,
    transition_mode: str,
    coverage_level: str,
    phase_spec,
    prior_phases: list[PhaseRoundResult],
    policy_context: PolicyContext,
    simulation_target: str,
    config: dict,
) -> BranchMechanismAnchor:
    if transition_mode not in {"enabling", "constraining"}:
        raise ValueError("mechanism anchors are only planned for enabling/constraining nodes")
    requirement = (
        "Select a condition that strengthens the inherited causal pathway."
        if transition_mode == "enabling"
        else "Select a bottleneck, adverse response, or feedback that weakens the inherited causal pathway."
    )
    parent_history = [
        {
            "phase": phase.phase,
            "aggregated_values": _aggregate_parent_phase(phase),
            "phase_summary": phase.phase_summary,
        }
        for phase in prior_phases
    ]
    phase_fields = [
        field.model_dump(mode="json")
        for field in phase_spec.prediction_fields
    ]
    system_prompt = BRANCH_MECHANISM_PLANNER_SYSTEM
    if (config.get("stage2_toc") or {}).get("prompt_variant") == "ko_natural":
        system_prompt += " Write all JSON string values in natural Korean."
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": BRANCH_MECHANISM_PLANNER_USER.format(
                transition_mode=transition_mode,
                coverage_level=coverage_level,
                phase=phase_spec.phase,
                simulation_target=simulation_target,
                transition_requirement=requirement,
                phase_fields=json.dumps(phase_fields, ensure_ascii=False, indent=2),
                parent_history=json.dumps(parent_history, ensure_ascii=False, indent=2),
                policy_context=json.dumps(policy_context.model_dump(), ensure_ascii=False, indent=2),
            ),
        },
    ]
    payload = json.loads(chat_completion(messages, config), strict=False)
    payload["transition_mode"] = transition_mode
    payload["coverage_level"] = coverage_level
    return BranchMechanismAnchor.model_validate(payload)


def plan_sibling_mechanisms(
    *,
    phase_spec,
    prior_phases: list[PhaseRoundResult],
    policy_context: PolicyContext,
    simulation_target: str,
    coverage_levels: dict[str, str],
    config: dict,
) -> dict[str, BranchMechanismAnchor]:
    """Plan a contrastive E/B/C condition set once for one frozen parent."""
    parent_history = [
        {
            "phase": phase.phase,
            "aggregated_values": _aggregate_parent_phase(phase),
            "phase_summary": phase.phase_summary,
        }
        for phase in prior_phases
    ]
    phase_fields = [field.model_dump(mode="json") for field in phase_spec.prediction_fields]
    requested_levels = {
        "enabling": coverage_levels.get("enabling", "material"),
        "baseline": "reference",
        "constraining": coverage_levels.get("constraining", "material"),
    }
    system_prompt = JOINT_SIBLING_PLANNER_SYSTEM
    if (config.get("stage2_toc") or {}).get("prompt_variant") == "ko_natural":
        system_prompt += " Write all JSON string values in natural Korean."
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": JOINT_SIBLING_PLANNER_USER.format(
                phase=phase_spec.phase,
                simulation_target=simulation_target,
                coverage_levels=json.dumps(requested_levels, ensure_ascii=False),
                phase_fields=json.dumps(phase_fields, ensure_ascii=False, indent=2),
                parent_history=json.dumps(parent_history, ensure_ascii=False, indent=2),
                policy_context=json.dumps(policy_context.model_dump(), ensure_ascii=False, indent=2),
            ),
        },
    ]
    payload = json.loads(chat_completion(messages, config), strict=False)
    branches = payload.get("branches") or {}
    missing = set(BRANCH_TRANSITIONS) - set(branches)
    if missing:
        raise ValueError(f"joint sibling planner omitted branches: {sorted(missing)}")
    common_conditions = payload.get("common_parent_conditions") or []
    sibling_contrast = str(payload.get("sibling_contrast") or "")
    if not common_conditions:
        raise ValueError("joint sibling planner must identify common_parent_conditions")
    if not sibling_contrast.strip():
        raise ValueError("joint sibling planner must explain sibling_contrast")
    expected_variables = {field.name for field in phase_spec.prediction_fields}
    anchors = {}
    for transition_mode in BRANCH_TRANSITIONS:
        branch_payload = dict(branches[transition_mode] or {})
        branch_payload.update({
            "transition_mode": transition_mode,
            "coverage_level": requested_levels[transition_mode],
            "common_parent_conditions": common_conditions,
            "sibling_contrast": sibling_contrast,
        })
        anchor = BranchMechanismAnchor.model_validate(branch_payload)
        effect_variables = [effect.variable for effect in anchor.expected_variable_effects]
        if len(effect_variables) != len(set(effect_variables)):
            raise ValueError(
                f"joint sibling planner duplicated variable effects for {transition_mode}: "
                f"{effect_variables}"
            )
        if set(effect_variables) != expected_variables:
            raise ValueError(
                f"joint sibling planner variable mismatch for {transition_mode}: "
                f"expected {sorted(expected_variables)}, found {sorted(effect_variables)}"
            )
        anchors[transition_mode] = anchor
    normalized_triggers = {
        re.sub(r"\s+", " ", anchor.trigger.strip().lower())
        for anchor in anchors.values()
    }
    if len(normalized_triggers) != len(BRANCH_TRANSITIONS):
        raise ValueError("joint sibling planner must provide distinct E/B/C triggers")
    return anchors


def generate_phase_children(
    *,
    parent_id: str,
    prior_phases: list[PhaseRoundResult],
    personas: list[ConcretePersona],
    inner_schema: ToCInnerSchema,
    policy_context: PolicyContext,
    simulation_target: str,
    config: dict,
    transitions: tuple[str, ...] = BRANCH_TRANSITIONS,
    parallel_transitions: bool = False,
    coverage_levels: dict[str, str] | None = None,
) -> list[PathwayBranchNode]:
    """Generate sibling phase nodes from one immutable parent history.

    Calls remain separate so each persona produces an independent estimate for
    each transition. Siblings receive deep-copied, byte-equivalent parent context
    and an identical persona posting order.
    """
    phase_index = len(prior_phases)
    if phase_index >= len(inner_schema.phases):
        raise ValueError("the parent pathway already contains every schema phase")
    invalid = set(transitions) - set(BRANCH_TRANSITIONS)
    if invalid:
        raise ValueError(f"unsupported transition modes: {sorted(invalid)}")

    phase_spec = inner_schema.phases[phase_index]
    context_hash = parent_context_hash(prior_phases)
    fixed_order = (
        resolve_configured_posting_order(personas, config)
        or [persona.name for persona in personas]
    )
    planner_enabled = bool(
        ((config.get("stage2_toc") or {}).get("branch_mechanism_planner") or {}).get("enabled", False)
    )
    planner_mode = mechanism_planner_mode(config)
    resolved_coverage_levels = {
        transition_mode: (
            (coverage_levels or {}).get(transition_mode)
            or mechanism_coverage_level(
                parent_id=parent_id,
                parent_hash=context_hash,
                phase_index=phase_index,
                transition_mode=transition_mode,
                config=config,
            )
        )
        for transition_mode in ("enabling", "constraining")
    }
    joint_anchors = None
    if planner_enabled and planner_mode == "joint_siblings":
        joint_anchors = plan_sibling_mechanisms(
            phase_spec=phase_spec,
            prior_phases=prior_phases,
            policy_context=policy_context,
            simulation_target=simulation_target,
            coverage_levels=resolved_coverage_levels,
            config=copy.deepcopy(config),
        )

    def _generate_child(transition_mode: str) -> PathwayBranchNode:
        mechanism_anchor = joint_anchors.get(transition_mode) if joint_anchors else None
        if (
            planner_enabled
            and planner_mode == "per_transition"
            and transition_mode in {"enabling", "constraining"}
        ):
            mechanism_anchor = plan_branch_mechanism(
                transition_mode=transition_mode,
                coverage_level=resolved_coverage_levels[transition_mode],
                phase_spec=phase_spec,
                prior_phases=prior_phases,
                policy_context=policy_context,
                simulation_target=simulation_target,
                config=copy.deepcopy(config),
            )
        result = _phase_round(
            direction="forward",
            phase=phase_spec.phase,
            phase_spec=phase_spec,
            personas=personas,
            policy_context=policy_context,
            simulation_target=simulation_target,
            prior_phases=copy.deepcopy(prior_phases),
            policy_goal_seed=None,
            config=copy.deepcopy(config),
            inner_schema=inner_schema,
            transition_mode=transition_mode,
            fixed_posting_order=fixed_order,
            mechanism_anchor=mechanism_anchor,
        )
        return PathwayBranchNode(
            node_id=f"{parent_id}/{transition_mode}",
            parent_id=parent_id,
            transition_mode=transition_mode,
            phase_index=phase_index,
            parent_context_hash=context_hash,
            mechanism_anchor=mechanism_anchor,
            phase_result=result,
        )

    if parallel_transitions and len(transitions) > 1:
        with ThreadPoolExecutor(max_workers=len(transitions)) as executor:
            children = list(executor.map(_generate_child, transitions))
    else:
        children = [_generate_child(transition) for transition in transitions]
    return children
