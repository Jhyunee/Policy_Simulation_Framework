import json
import random
import re
from typing import Optional

from llm.client import chat_completion
from stage2_toc.runtime import get_org_kg_contexts
from models.prompts import (
    INNER_SCHEMA_BUILDER_SYSTEM, INNER_SCHEMA_BUILDER_USER, INNER_SCHEMA_TARGET_ONLY_CLAUSE,
    FORWARD_PHASE_POST_SYSTEM, FORWARD_PHASE_POST_USER,
    FORWARD_PHASE_REVISE_SYSTEM, FORWARD_PHASE_REVISE_USER,
    BACKWARD_PHASE_POST_SYSTEM, BACKWARD_PHASE_POST_USER,
    BACKWARD_PHASE_REVISE_SYSTEM, BACKWARD_PHASE_REVISE_USER,
    REFINEMENT_JUDGE_SYSTEM, REFINEMENT_JUDGE_USER,
    FINAL_TOC_REPORTER_SYSTEM, FINAL_TOC_REPORTER_USER,
    PHASE_STAKEHOLDER_SYNTHESIZER_SYSTEM, PHASE_STAKEHOLDER_SYNTHESIZER_USER,
    SEQUENTIAL_FORWARD_POST_SYSTEM, SEQUENTIAL_FORWARD_POST_USER,
    SEQUENTIAL_BACKWARD_POST_SYSTEM, SEQUENTIAL_BACKWARD_POST_USER,
    PHASE_SUMMARY_SYSTEM, PHASE_SUMMARY_USER,
    SEQUENTIAL_FORWARD_POST_SYSTEM_KO_NATURAL, SEQUENTIAL_FORWARD_POST_USER_KO_NATURAL,
    PHASE_SUMMARY_SYSTEM_KO_NATURAL, PHASE_SUMMARY_USER_KO_NATURAL,
    PHASE_LOCAL_BRANCH_SYSTEM_KO_NATURAL, PHASE_LOCAL_BRANCH_USER_KO_NATURAL,
    PHASE_LOCAL_BRANCH_DIRECTIVES_KO, PHASE_LOCAL_BRANCH_EXPERIMENTS_KO,
    PHASE_SUMMARY_SYSTEM_EN_NATURAL, PHASE_SUMMARY_USER_EN_NATURAL,
    PHASE_LOCAL_BRANCH_SYSTEM_EN_NATURAL, PHASE_LOCAL_BRANCH_USER_EN_NATURAL,
    PHASE_LOCAL_BRANCH_DIRECTIVES_EN, PHASE_LOCAL_BRANCH_EXPERIMENTS_EN,
)
from models.schemas import (
    TOC_PHASES,
    PolicyContext,
    PredictionResult,
    ConcretePersona,
    PhaseVariableSpec,
    PhaseInnerSpec,
    ToCInnerSchema,
    PersonaPhaseEstimate,
    PhaseRoundResult,
    PhaseCrossCheck,
    ScenarioToCResult,
    BranchMechanismAnchor,
    GroundedInputState,
)


def _policy_context_text(config: dict, policy_context: PolicyContext) -> str:
    """Return policy_context as a string for prompt injection.

    By default, serializes the KG-derived PolicyContext as JSON. If the config
    has ``stage2_toc.policy_md_text`` set (raw markdown content), use that
    instead — KG-ablation variant for the policy_context section of the prompt.
    """
    override = (config.get("stage2_toc") or {}).get("policy_md_text")
    if override:
        return override
    return json.dumps(policy_context.model_dump(), ensure_ascii=False)


# Stance condition variable: an optional framing directive appended to the persona
# reasoning system prompt. The three steered conditions share one scaffold and differ
# only on the up/center/down axis (symmetric construction), so the only variable across
# conditions is semantic content — not sentence presence/length. ``baseline`` is the
# empty-directive escape hatch (reproduces default behavior). Stance changes only how
# genuine ambiguity is resolved; it does not touch the output JSON schema.
_STANCE_SCAFFOLD = (
    " ADDITIONAL FRAMING — {label} stance: when an estimate is genuinely uncertain, "
    "resolve the ambiguity toward {axis}. Do not fabricate — stay grounded in the "
    "persona and policy context; only the resolution of genuine ambiguity changes."
)
_STANCE_DIRECTIVES = {
    "baseline": "",  # empty → reproduces current behavior (sanity anchor, not in main study)
    "optimistic": _STANCE_SCAFFOLD.format(
        label="optimistic",
        axis="the upside — assume successful implementation, strong stakeholder "
             "uptake, and synergistic (additive) policy effects",
    ),
    "neutral": _STANCE_SCAFFOLD.format(
        label="neutral",
        axis="the single most likely, balanced outcome — weigh upside and downside "
             "symmetrically, neither optimistic nor pessimistic",
    ),
    "conservative": _STANCE_SCAFFOLD.format(
        label="conservative",
        axis="the downside — assume implementation friction, modest behavior change, "
             "partial uptake, and that goals must stretch existing resources",
    ),
}

_STANCE_SCAFFOLD_KO_NATURAL = (
    " ADDITIONAL FRAMING — {label} stance. This is not only a numeric bias; "
    "it is a causal interpretation lens. When multiple mechanisms are plausible, "
    "foreground {mechanism_lens}. Keep factual anchors and the simulation target's "
    "time window fixed. Do not invent evidence. Make the Korean narrative explain "
    "why this pathway unfolds differently from other plausible pathways."
)
_STANCE_DIRECTIVES_KO_NATURAL = {
    "baseline": "",
    "optimistic": _STANCE_SCAFFOLD_KO_NATURAL.format(
        label="optimistic",
        mechanism_lens=(
            "enabling conditions: smooth implementation, strong applicant uptake, "
            "usable funding, effective mentoring, institutional support, and positive spillovers"
        ),
    ),
    "neutral": _STANCE_SCAFFOLD_KO_NATURAL.format(
        label="neutral",
        mechanism_lens=(
            "the base-rate pathway: policy-stated quantities, cohort timing, ordinary uptake, "
            "typical productivity, and effects observable only within the target window"
        ),
    ),
    "conservative": _STANCE_SCAFFOLD_KO_NATURAL.format(
        label="conservative",
        mechanism_lens=(
            "bottlenecks and delays: eligibility constraints, lab-transition friction, "
            "administrative burden, uneven regional capacity, ramp-up time, and limited short-term observability"
        ),
    ),
}


def _prompt_variant(config: dict) -> str:
    return (config.get("stage2_toc") or {}).get("prompt_variant", "default")


def _stance_directive(config: dict) -> str:
    """Return the stance framing directive (or "" for baseline/unset) for the
    persona reasoning system prompt. Read from ``stage2_toc.stance``."""
    stance = (config.get("stage2_toc") or {}).get("stance", "baseline")
    if _prompt_variant(config) == "ko_natural":
        return _STANCE_DIRECTIVES_KO_NATURAL.get(stance, "")
    return _STANCE_DIRECTIVES.get(stance, "")


def _phase_local_prompt_set(config: dict) -> tuple[str, str, dict[str, str]]:
    """Return the language-matched prompts for phase-local EBC branching."""
    variant = _prompt_variant(config)
    if variant == "ko_natural":
        return (
            PHASE_LOCAL_BRANCH_SYSTEM_KO_NATURAL,
            PHASE_LOCAL_BRANCH_USER_KO_NATURAL,
            PHASE_LOCAL_BRANCH_DIRECTIVES_KO,
        )
    if variant == "en_natural":
        return (
            PHASE_LOCAL_BRANCH_SYSTEM_EN_NATURAL,
            PHASE_LOCAL_BRANCH_USER_EN_NATURAL,
            PHASE_LOCAL_BRANCH_DIRECTIVES_EN,
        )
    raise ValueError(
        "phase-local branching requires stage2_toc.prompt_variant to be "
        f"'ko_natural' or 'en_natural'; got {variant!r}"
    )


def _phase_local_transition_directive(config: dict, transition_mode: str) -> str:
    """Return the additive phase-local branch directive.

    This is separate from ``stage2_toc.stance`` so legacy whole-run stance
    experiments keep their original prompt and behavior.
    """
    variant = _prompt_variant(config)
    _, _, directives = _phase_local_prompt_set(config)
    try:
        base = directives[transition_mode]
    except KeyError as exc:
        raise ValueError(
            "transition_mode must be one of "
            f"{sorted(directives)}; got {transition_mode!r}"
        ) from exc
    experiment_name = str(
        (config.get("stage2_toc") or {}).get("branch_prompt_experiment") or "current"
    )
    experiments = (
        PHASE_LOCAL_BRANCH_EXPERIMENTS_KO
        if variant == "ko_natural"
        else PHASE_LOCAL_BRANCH_EXPERIMENTS_EN
    )
    if experiment_name not in experiments:
        raise ValueError(
            "stage2_toc.branch_prompt_experiment must be one of "
            f"{sorted(experiments)}; got {experiment_name!r}"
        )
    return base + experiments[experiment_name][transition_mode]


def _format_branch_mechanism_anchor(anchor: BranchMechanismAnchor) -> str:
    propagation = "\n".join(f"  - {item}" for item in anchor.expected_propagation)
    common_conditions = "\n".join(f"  - {item}" for item in anchor.common_parent_conditions)
    variable_effects = "\n".join(
        "  - "
        f"{effect.variable}: direction={effect.direction}, timing={effect.timing}, "
        f"strength={effect.strength}; rationale={effect.rationale}"
        for effect in anchor.expected_variable_effects
    )
    return f"""

SHARED NODE-LEVEL MECHANISM ANCHOR
- Common parent conditions inherited by every sibling:
{common_conditions or "  - (refer to the inherited parent history)"}
- Coverage level: {anchor.coverage_level}
- Mechanism family: {anchor.mechanism_family}
- Trigger: {anchor.trigger}
- Affected causal link: {anchor.affected_causal_link}
- Expected propagation:
{propagation or "  - (no downstream propagation specified)"}
- Parent support: {anchor.parent_support}
- Plausibility limit: {anchor.plausibility_limit}
- Expected current-variable effects:
{variable_effects or "  - (no variable-level effects specified)"}
- Contrast with sibling conditions: {anchor.sibling_contrast or "(not specified)"}

This anchor is shared by every stakeholder in this node.
Do not replace it with another primary mechanism. Interpret its role-specific
effects and estimate values that follow from the inherited parent state.
Treat common parent conditions as inherited evidence, not as the distinctive
reason this child differs from its siblings. Explain the branch-specific trigger
and affected causal link before repeating shared constraints.
The anchor defines a causal mechanism, not a numerical magnitude. Preserve every
prediction field's declared unit and plausible scale. Do not convert qualitative
terms such as "many" or "high participation" into a percentage or an absolute
count without a denominator grounded in the inherited parent or policy context.
"""


def _sequential_forward_prompts(config: dict) -> tuple[str, str]:
    if _prompt_variant(config) == "ko_natural":
        return SEQUENTIAL_FORWARD_POST_SYSTEM_KO_NATURAL, SEQUENTIAL_FORWARD_POST_USER_KO_NATURAL
    return SEQUENTIAL_FORWARD_POST_SYSTEM, SEQUENTIAL_FORWARD_POST_USER


def _phase_summary_prompts(config: dict) -> tuple[str, str]:
    variant = _prompt_variant(config)
    if variant == "ko_natural":
        return PHASE_SUMMARY_SYSTEM_KO_NATURAL, PHASE_SUMMARY_USER_KO_NATURAL
    if variant == "en_natural":
        return PHASE_SUMMARY_SYSTEM_EN_NATURAL, PHASE_SUMMARY_USER_EN_NATURAL
    return PHASE_SUMMARY_SYSTEM, PHASE_SUMMARY_USER


# ───────────────────────── Inner schema builder ─────────────────────────

def build_inner_toc_schema(
    policy_context: PolicyContext,
    simulation_target: str,
    prediction_results: list[PredictionResult],
    config: dict,
) -> ToCInnerSchema:
    """One LLM call → produce per-policy inner schema (5 phases × {fields, points}).

    After the LLM responds, deterministically inject a prediction_field at the
    Outcomes phase whose `description` is the simulation_target verbatim — this
    guarantees the headline aggregate variable matches the CLI input regardless
    of LLM drift or framing overlays.
    """
    target_only = config.get("stage2_toc", {}).get("quantitative_target_only", False)
    target_only_clause = INNER_SCHEMA_TARGET_ONLY_CLAUSE if target_only else ""
    messages = [
        {"role": "system", "content": INNER_SCHEMA_BUILDER_SYSTEM},
        {
            "role": "user",
            "content": INNER_SCHEMA_BUILDER_USER.format(
                policy_context=_policy_context_text(config, policy_context),
                simulation_target=simulation_target,
                stage1_predictions=json.dumps(
                    [pr.model_dump() for pr in prediction_results], ensure_ascii=False,
                ),
                target_only_clause=target_only_clause,
            ),
        },
    ]
    # Optional model override for the inner-schema call: lets us route the
    # schema-shaping decision through a stronger model (e.g. deepseek-v4-pro)
    # while keeping per-scenario rounds on the cheaper default.
    inner_llm_override = (config.get("stage2_toc") or {}).get("inner_schema_llm") or {}
    if inner_llm_override:
        inner_config = {**config, "llm": {**(config.get("llm") or {}), **inner_llm_override}}
        print(f"  [inner_schema] using LLM override: {inner_llm_override}")
    else:
        inner_config = config
    data = json.loads(chat_completion(messages, inner_config), strict=False)
    schema = ToCInnerSchema(**data)

    if len(schema.phases) != 5:
        raise ValueError(f"Inner schema must have 5 phases, got {len(schema.phases)}")
    for expected, actual in zip(TOC_PHASES, [p.phase for p in schema.phases]):
        if expected != actual:
            raise ValueError(
                f"Inner schema phase order mismatch: expected {TOC_PHASES}, got {[p.phase for p in schema.phases]}"
            )
    targets = [t.strip() for t in (simulation_target or "").split(",") if t.strip()]
    for tgt in targets:
        schema = inject_target_into_schema(schema, tgt)
    return schema


_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")


def _slugify_target(target: str) -> str:
    """Derive a snake_case identifier from the simulation_target string."""
    slug = _SLUG_STRIP_RE.sub("_", target.lower()).strip("_")
    return slug or "simulation_target"


def _infer_type_unit(target: str) -> tuple[str, str]:
    """Infer (type, unit) from the leading verb of the simulation_target.

    Returns one of {"int", "float", "str"} for type and a best-guess unit string.
    """
    t = target.strip().lower()
    # rate/average → float. Match natural-language prefixes AND slug-style suffixes
    # like `..._employment_rate`, `..._research_rate`.
    if any(t.startswith(prefix) for prefix in (
        "average ", "mean ", "rate of ", "per ", "ratio of ",
    )) or " per " in t or t.endswith(("_rate", " rate")):
        return "float", "%"
    # percentage → float (0-100)
    if t.startswith(("percentage", "%", "share of", "fraction of", "proportion of")) or " percentage" in t or "%" in t:
        return "float", "%"
    # aggregate count → int
    if t.startswith(("number of", "count of", "total ", "how many ", "amount of", "sum of")) or t.startswith("num "):
        return "int", "count"
    # default: aggregate int
    return "int", "count"


def build_simple_inner_schema(simulation_target: str) -> ToCInnerSchema:
    """1-phase inner schema for the Simple ablation (Stage 2 ToC theory off).

    Skips the LLM call entirely. The synthesized schema has exactly one phase
    named ``"Impact"`` so existing aggregators (which filter ``phase == "Impact"``
    in ``aggregate_scenario`` and ``aggregate_across_scenarios``) continue to work
    unchanged. ``prediction_fields`` contains only the target outcome variable —
    same slug as the IAD/IG runs (via ``_slugify_target``), so per-scenario MAPE
    comparisons across the 2×2 ablation are directly numeric.

    Supports multi-target: comma-separated ``simulation_target`` strings produce
    one PhaseVariableSpec per target, all under the single Impact phase.
    """
    target_text = (simulation_target or "").strip()
    if not target_text:
        raise ValueError("simulation_target required for simple-prediction mode")
    targets = [t.strip() for t in target_text.split(",") if t.strip()] or [target_text]
    fields: list[PhaseVariableSpec] = []
    seen: set[str] = set()
    for t in targets:
        inferred_type, inferred_unit = _infer_type_unit(t)
        slug = _slugify_target(t)
        if slug in seen:
            continue
        seen.add(slug)
        fields.append(PhaseVariableSpec(
            name=slug, type=inferred_type, unit=inferred_unit, description=t,
        ))
    phase = PhaseInnerSpec(
        phase="Impact",
        prediction_fields=fields,
        discussion_points=[
            f"Predict {target_text} directly based on your role, the policy "
            "context, and grounded evidence. Give one number per target with brief reasoning."
        ],
    )
    return ToCInnerSchema(phases=[phase])


def inject_target_into_schema(
    schema: ToCInnerSchema,
    simulation_target: str,
    target_phase: str = "Outcomes",
) -> ToCInnerSchema:
    """Guarantee the target_phase's prediction_fields contains a field whose
    `description` is the simulation_target verbatim.

    name / type / unit are derived deterministically from the simulation_target
    (see `_slugify_target` and `_infer_type_unit`). Idempotent: if a matching
    field already exists (case-folded description match), this is a no-op.
    """
    target_text = (simulation_target or "").strip()
    if not target_text:
        return schema
    inferred_type, inferred_unit = _infer_type_unit(target_text)
    new_field = PhaseVariableSpec(
        name=_slugify_target(target_text),
        type=inferred_type,
        unit=inferred_unit,
        description=target_text,
    )
    for ph in schema.phases:
        if ph.phase != target_phase:
            continue
        # Idempotent: skip injection if any field already mirrors the target.
        for existing in ph.prediction_fields:
            if (existing.description or "").strip().casefold() == target_text.casefold():
                return schema
            if (existing.name or "").strip().casefold() == new_field.name.casefold():
                return schema
        # Prepend so it's the FIRST (headline) variable at the target phase.
        ph.prediction_fields = [new_field] + list(ph.prediction_fields)
        break
    return schema


# ───────────────────────── Per-phase debate round ─────────────────────────

def _format_phase_fields(phase_spec: PhaseInnerSpec) -> str:
    if not phase_spec.prediction_fields:
        return ("  (NONE — this phase has no quantitative prediction_fields. "
                "Set prediction_values to {} (empty object) and focus your output entirely on "
                "the narrative explaining causal mechanisms at this phase.)")
    return "\n".join(
        f"  - {f.name} ({f.type}, unit={f.unit}): {f.description}"
        for f in phase_spec.prediction_fields
    )


def _format_prior_phases(
    prior_phases: list[PhaseRoundResult],
    inner_schema: ToCInnerSchema,
) -> str:
    """Render prior-phase context for the next phase prompt (used by refinement step).

    For each prior phase includes:
      - the discussion_points (the "main question" that phase was answering)
      - the aggregated prediction_values across personas
      - per-persona narrative + evidence (so phase t can reason from phase t-1's mechanisms,
        not just its numbers)
    """
    if not prior_phases:
        return "(none — this is the first phase in this direction)"

    spec_by_phase = {s.phase: s for s in inner_schema.phases}
    blocks = []
    for pr in prior_phases:
        spec = spec_by_phase.get(pr.phase)
        agg = _aggregate_phase_round(pr)

        lines = [f"### Phase {pr.phase}"]
        if spec:
            lines.append("Main questions this phase addressed:")
            lines.extend(f"  - {q}" for q in spec.discussion_points)
        if pr.state_type == "document_grounded":
            lines.append(f"Document-grounded values: {json.dumps(agg, ensure_ascii=False)}")
            if pr.grounded_evidence:
                lines.append("Source evidence: " + json.dumps(pr.grounded_evidence, ensure_ascii=False))
        else:
            lines.append(f"Aggregated values (mean across personas): {json.dumps(agg, ensure_ascii=False)}")
        lines.append("Per-persona reasoning:")
        for est in pr.revised_posts:
            lines.append(
                f"  - {est.persona_name} ({est.stakeholder_type}):"
                f" values={json.dumps(est.prediction_values, ensure_ascii=False)};"
                f" narrative={est.narrative}"
            )
            if est.evidence:
                lines.append(f"      evidence: {json.dumps(est.evidence, ensure_ascii=False)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_prior_phases_summarized(
    prior_phases: list[PhaseRoundResult],
    persona_name: str,
    inner_schema: ToCInnerSchema,
) -> str:
    """Render prior-phase context using holistic summaries + persona's own utterances.

    For each prior phase includes:
      - holistic phase summary (no per-persona attribution)
      - this persona's own post from that phase (full)
      - aggregated values (mean)
    """
    if not prior_phases:
        return "--- Prior Phases ---\n(none — this is the first phase in this direction)"

    spec_by_phase = {s.phase: s for s in inner_schema.phases}
    blocks = []
    for pr in prior_phases:
        spec = spec_by_phase.get(pr.phase)
        agg = _aggregate_phase_round(pr)

        lines = [f"### Phase {pr.phase}"]
        if spec:
            lines.append("Main questions this phase addressed:")
            lines.extend(f"  - {q}" for q in spec.discussion_points)
        if pr.state_type == "document_grounded":
            lines.append(f"Document-grounded values: {json.dumps(agg, ensure_ascii=False)}")
            if pr.grounded_evidence:
                lines.append(
                    "Source evidence: "
                    + json.dumps(pr.grounded_evidence, ensure_ascii=False)
                )
        else:
            lines.append(f"Aggregated values (mean across personas): {json.dumps(agg, ensure_ascii=False)}")

        # Holistic summary (no individual attribution)
        if pr.phase_summary:
            lines.append(f"Discussion summary: {pr.phase_summary}")

        # This persona's own post from this phase
        own_post = next((p for p in pr.revised_posts if p.persona_name == persona_name), None)
        if own_post:
            lines.append(f"Your own post at this phase: values={json.dumps(own_post.prediction_values, ensure_ascii=False)}; narrative={own_post.narrative}")
            if own_post.evidence:
                lines.append(f"  your evidence: {json.dumps(own_post.evidence, ensure_ascii=False)}")

        blocks.append("\n".join(lines))
    return "--- Prior Phases ---\n" + "\n\n".join(blocks)


def _format_sequential_context(prior_posts: list[PersonaPhaseEstimate]) -> str:
    """Format within-phase sequential context — posts from personas who already spoke."""
    if not prior_posts:
        return "--- Other Participants' Posts at This Phase ---\n(you are the first to post at this phase)"

    lines = []
    for i, post in enumerate(prior_posts, 1):
        lines.append(
            f"  [{i}] {post.persona_name} ({post.stakeholder_type}):"
            f" values={json.dumps(post.prediction_values, ensure_ascii=False)};"
            f" narrative={post.narrative}"
        )
    return "--- Other Participants' Posts at This Phase ---\n" + "\n".join(lines)


def _format_independent_context() -> str:
    return (
        "--- Other Participants' Posts at This Phase ---\n"
        "(independent-first mode: no other participant posts are shown before your first estimate. "
        "Make your own estimate from the persona, policy context, and prior phase summaries.)"
    )


def resolve_configured_posting_order(
    personas: list[ConcretePersona],
    config: dict,
) -> Optional[list[str]]:
    """Resolve the configured order to canonical persona names."""
    raw_order = (config.get("stage2_toc") or {}).get("fixed_posting_order")
    if raw_order is None:
        return None
    if not isinstance(raw_order, list) or not raw_order:
        raise ValueError("stage2_toc.fixed_posting_order must be a non-empty list")

    remaining = list(personas)
    resolved: list[str] = []
    for entry in raw_order:
        token = str(entry).strip()
        matches = [
            persona for persona in remaining
            if token in {
                persona.name,
                persona.persona_id,
                persona.role_prototype_id,
                persona.stakeholder_type,
            }
        ]
        if len(matches) != 1:
            raise ValueError(
                "Each stage2_toc.fixed_posting_order entry must uniquely match one "
                f"remaining persona; {token!r} matched {len(matches)}"
            )
        persona = matches[0]
        resolved.append(persona.name)
        remaining.remove(persona)

    if remaining:
        raise ValueError(
            "stage2_toc.fixed_posting_order must include every participant exactly once; "
            f"missing {[persona.name for persona in remaining]}"
        )
    return resolved


def _format_deliberation_context(
    initial_posts: list[PersonaPhaseEstimate],
    persona_name: str,
) -> str:
    """Format all same-phase independent posts for the revision step."""
    lines = [
        "--- Independent Posts from Other Participants at This Phase ---",
        "You are now revising after seeing the other independent estimates. Do not anchor on the first or majority value.",
        "If your final estimate is similar, explain the role-specific reason independently. If you differ, explain the tradeoff or missing condition.",
    ]
    others = [p for p in initial_posts if p.persona_name != persona_name]
    if not others:
        lines.append("(no other independent posts available)")
        return "\n".join(lines)
    for i, post in enumerate(others, 1):
        lines.append(
            f"  [{i}] {post.persona_name} ({post.stakeholder_type}):"
            f" values={json.dumps(post.prediction_values, ensure_ascii=False)};"
            f" narrative={post.narrative}"
        )
    return "\n".join(lines)


def _generate_phase_summary(
    phase: str,
    posts: list[PersonaPhaseEstimate],
    direction: str,
    simulation_target: str,
    phase_spec: PhaseInnerSpec,
    config: dict,
    transition_mode: Optional[str] = None,
    mechanism_anchor: Optional[BranchMechanismAnchor] = None,
) -> str:
    """LLM call to produce holistic summary of a phase's discussion."""
    fields_text = _format_phase_fields(phase_spec)
    all_posts_json = json.dumps([p.model_dump() for p in posts], ensure_ascii=False)

    summary_sys, summary_user = _phase_summary_prompts(config)
    if transition_mode:
        summary_sys += (
            " This phase is a child generated under the phase-local "
            f"{transition_mode} transition. Summarize the concrete transition mechanism "
            "from its inherited parent, not the label itself."
        )
    if mechanism_anchor:
        summary_sys += (
            " Preserve the shared node-level mechanism anchor as the primary causal "
            "explanation and summarize how the stakeholders instantiate it. "
            "The anchor is: "
            + json.dumps(mechanism_anchor.model_dump(mode="json"), ensure_ascii=False)
        )
    messages = [
        {"role": "system", "content": summary_sys},
        {
            "role": "user",
            "content": summary_user.format(
                phase=phase,
                direction=direction,
                simulation_target=simulation_target,
                phase_fields=fields_text,
                org_kg_contexts=get_org_kg_contexts(),
                all_posts=all_posts_json,
            ),
        },
    ]
    data = json.loads(chat_completion(messages, config), strict=False)
    return data.get("summary", "")


def _aggregate_phase_round(pr: PhaseRoundResult) -> dict:
    """Mean across personas for numeric fields; first-persona value for str fields."""
    if pr.state_type == "document_grounded":
        return dict(pr.grounded_values)
    if not pr.revised_posts:
        return {}
    keys = list(pr.revised_posts[0].prediction_values.keys())
    agg = {}
    for k in keys:
        vals = [p.prediction_values.get(k) for p in pr.revised_posts]
        numeric = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if numeric and len(numeric) == len(vals):
            agg[k] = sum(numeric) / len(numeric)
        else:
            agg[k] = vals[0]
    return agg


def _phase_round(
    direction: str,
    phase: str,
    phase_spec: PhaseInnerSpec,
    personas: list[ConcretePersona],
    policy_context: PolicyContext,
    simulation_target: str,
    prior_phases: list[PhaseRoundResult],
    policy_goal_seed: Optional[str],
    config: dict,
    inner_schema: ToCInnerSchema,
    transition_mode: Optional[str] = None,
    fixed_posting_order: Optional[list[str]] = None,
    mechanism_anchor: Optional[BranchMechanismAnchor] = None,
) -> PhaseRoundResult:
    """Run one phase of stakeholder estimates.

    Supported ``stage2_toc.phase_interaction_mode`` values:
      - sequential: each persona sees earlier same-phase posts (legacy/current).
      - independent: each persona posts without same-phase context.
      - independent_revision: independent posts first, then revision after seeing
        all other independent posts.

    In all modes, the next phase receives the holistic phase summary and each
    persona's own previous-phase final post via ``_format_prior_phases_summarized``.
    """
    fields_text = _format_phase_fields(phase_spec)
    discussion_text = "\n".join(f"  - {q}" for q in phase_spec.discussion_points)
    policy_context_json = _policy_context_text(config, policy_context)

    if transition_mode and direction != "forward":
        raise ValueError("phase-local transitions are supported only for forward rounds")

    if transition_mode:
        post_sys, post_user, _ = _phase_local_prompt_set(config)
        transition_directive = _phase_local_transition_directive(config, transition_mode)
        if mechanism_anchor:
            transition_directive += _format_branch_mechanism_anchor(mechanism_anchor)
    elif direction == "forward":
        post_sys, post_user = _sequential_forward_prompts(config)
        transition_directive = ""
    else:
        post_sys = SEQUENTIAL_BACKWARD_POST_SYSTEM
        post_user = SEQUENTIAL_BACKWARD_POST_USER
        transition_directive = ""
    # Preserve the legacy whole-run stance path unless this is an explicitly
    # phase-local branch expansion.
    if not transition_mode:
        post_sys = post_sys + _stance_directive(config)

    effective_posting_order = fixed_posting_order
    if effective_posting_order is None:
        effective_posting_order = resolve_configured_posting_order(personas, config)

    if effective_posting_order is None:
        shuffled_personas = list(personas)
        random.shuffle(shuffled_personas)
    else:
        persona_by_name = {p.name: p for p in personas}
        expected = set(persona_by_name)
        provided = set(effective_posting_order)
        if len(effective_posting_order) != len(personas) or provided != expected:
            raise ValueError("fixed_posting_order must contain every persona name exactly once")
        shuffled_personas = [persona_by_name[name] for name in effective_posting_order]
    posting_order = [p.name for p in shuffled_personas]

    def _post_for_persona(
        persona: ConcretePersona,
        same_phase_context: str,
        step_label: str,
        idx: int,
    ) -> PersonaPhaseEstimate:
        # Prior-phase context: holistic summaries + own utterances
        prior_text = _format_prior_phases_summarized(prior_phases, persona.name, inner_schema)

        post_kwargs = dict(
            phase=phase,
            phase_fields=fields_text,
            discussion_points=discussion_text,
            prior_phase_context=prior_text,
            sequential_context=same_phase_context,
            simulation_target=simulation_target,
            policy_context=policy_context_json,
            org_kg_contexts=get_org_kg_contexts(),
            persona=json.dumps(persona.model_dump(), ensure_ascii=False),
            transition_directive=transition_directive,
        )
        if direction == "backward":
            post_kwargs["backward_seed"] = policy_goal_seed or policy_context.theory_of_change_summary
        else:
            post_kwargs["forward_seed"] = policy_goal_seed or ""
        messages = [
            {"role": "system", "content": post_sys},
            {"role": "user", "content": post_user.format(**post_kwargs)},
        ]
        data = json.loads(chat_completion(messages, config), strict=False)
        estimate = PersonaPhaseEstimate(
            persona_name=persona.name,
            stakeholder_type=persona.stakeholder_type,
            prediction_values=data.get("prediction_values") or {},
            narrative=data.get("narrative") or "",
            evidence=data.get("evidence", []),
        )
        print(f"        {persona.name} {step_label} (#{idx}/{len(shuffled_personas)})")
        return estimate

    stage2_cfg = config.get("stage2_toc") or {}
    interaction_mode = (stage2_cfg.get("phase_interaction_mode") or "").strip()
    if not interaction_mode:
        # Backward compatibility for runs/configs created before the explicit
        # three-way mode flag.
        interaction_mode = (
            "independent_revision"
            if stage2_cfg.get("independent_first_deliberation", False)
            else "sequential"
        )
    valid_modes = {"sequential", "independent", "independent_revision"}
    if interaction_mode not in valid_modes:
        raise ValueError(
            f"stage2_toc.phase_interaction_mode must be one of {sorted(valid_modes)}; "
            f"got {interaction_mode!r}"
        )

    if interaction_mode in {"independent", "independent_revision"}:
        print(f"        mode: {interaction_mode}")
        initial_posts: list[PersonaPhaseEstimate] = []
        for idx, persona in enumerate(shuffled_personas, 1):
            initial_posts.append(
                _post_for_persona(
                    persona=persona,
                    same_phase_context=_format_independent_context(),
                    step_label="posted independently",
                    idx=idx,
                )
            )

        if interaction_mode == "independent_revision":
            revised_posts: list[PersonaPhaseEstimate] = []
            for idx, persona in enumerate(shuffled_personas, 1):
                revised_posts.append(
                    _post_for_persona(
                        persona=persona,
                        same_phase_context=_format_deliberation_context(initial_posts, persona.name),
                        step_label="revised after deliberation",
                        idx=idx,
                    )
                )
        else:
            revised_posts = initial_posts
    else:
        # ── Sequential posts ──
        revised_posts = []
        for idx, persona in enumerate(shuffled_personas, 1):
            # Within-phase context: posts from earlier speakers
            seq_text = _format_sequential_context(revised_posts)
            revised_posts.append(
                _post_for_persona(
                    persona=persona,
                    same_phase_context=seq_text,
                    step_label="posted",
                    idx=idx,
                )
            )
        initial_posts = revised_posts

    # ── Generate holistic phase summary ──
    print(f"        generating phase summary...")
    phase_summary = _generate_phase_summary(
        phase, revised_posts, direction, simulation_target, phase_spec, config,
        transition_mode=transition_mode,
        mechanism_anchor=mechanism_anchor,
    )

    return PhaseRoundResult(
        phase=phase,
        direction=direction,
        initial_posts=initial_posts,
        revised_posts=revised_posts,
        phase_summary=phase_summary,
        posting_order=posting_order,
    )


# ───────────────────────── Forward / backward passes ─────────────────────────

def run_forward_pass(
    personas: list[ConcretePersona],
    inner_schema: ToCInnerSchema,
    policy_context: PolicyContext,
    simulation_target: str,
    config: dict,
    inputs_seed: Optional[str] = None,
    backward_pass: Optional[list[PhaseRoundResult]] = None,
    grounded_inputs: Optional[GroundedInputState] = None,
) -> list[PhaseRoundResult]:
    """Iterate phases in TOC_PHASES order (Inputs → Impact).

    If ``inputs_seed`` is provided, it is injected into the Inputs phase prompt only
    (via {forward_seed} slot). Used by the Fwd→Bwd→Fwd_seq chain to round-trip
    backward's Inputs aggregate back into a fresh forward simulation.

    If ``backward_pass`` is provided AND ``copy_backward_inputs_to_5th_forward`` is set,
    the Inputs phase LLM call is skipped and backward's Inputs PhaseRoundResult is reused
    verbatim. (Used by 5th chain — Fwd→Bwd→Fwd_seq.)
    """
    copy_bwd_inputs = (
        backward_pass is not None
        and config.get("stage2_toc", {}).get("copy_backward_inputs_to_5th_forward", False)
    )
    spec_by_phase = {s.phase: s for s in inner_schema.phases}
    # Iterate phases as declared by the schema (lets Simple ablation mode run a
    # single "Impact"-only phase without monkey-patching TOC_PHASES).
    phases_iter = [s.phase for s in inner_schema.phases]
    results: list[PhaseRoundResult] = []
    for i, phase in enumerate(phases_iter):
        if phase == "Inputs" and grounded_inputs is not None:
            from stage2_toc.grounded_inputs import grounded_input_phase_result

            print("      [forward] phase: Inputs — DOCUMENT-GROUNDED (persona calls skipped)")
            results.append(grounded_input_phase_result(grounded_inputs))
            continue
        if i == 0 and copy_bwd_inputs:
            bwd_inputs = next((r for r in backward_pass if r.phase == "Inputs"), None)
            if bwd_inputs is not None:
                print(f"      [forward] phase: Inputs — COPIED from backward (LLM call skipped)")
                results.append(bwd_inputs)
                continue
        print(f"      [forward] phase: {phase}")
        seed = inputs_seed if i == 0 else None
        rr = _phase_round(
            direction="forward", phase=phase, phase_spec=spec_by_phase[phase],
            personas=personas, policy_context=policy_context, simulation_target=simulation_target,
            prior_phases=results, policy_goal_seed=seed, config=config,
            inner_schema=inner_schema,
        )
        results.append(rr)
    return results


def run_backward_pass(
    personas: list[ConcretePersona],
    inner_schema: ToCInnerSchema,
    policy_context: PolicyContext,
    simulation_target: str,
    config: dict,
    forward_pass: Optional[list[PhaseRoundResult]] = None,
    grounded_inputs: Optional[GroundedInputState] = None,
) -> list[PhaseRoundResult]:
    """Iterate phases in REVERSE TOC_PHASES order (Impact → Inputs).

    Seed selection at the Impact phase:
      - `sequential_backward: false` (default) → use `policy_context.theory_of_change_summary`
        (KG-grounded goal statement; parallel mode)
      - `sequential_backward: true` → use forward's aggregated Impact values as a JSON snapshot
        (causal-attribution mode; requires forward_pass to be provided).

    The returned list is kept in TOC_PHASES order (Inputs first) for parity with forward.
    """
    sequential = config.get("stage2_toc", {}).get("sequential_backward", False)
    seed: str = policy_context.theory_of_change_summary
    if sequential:
        if forward_pass is None:
            print("    [WARN] sequential_backward=True but no forward_pass provided; falling back to theory_of_change_summary")
        else:
            fwd_impact = next((r for r in forward_pass if r.phase == "Impact"), None)
            if fwd_impact is None:
                print("    [WARN] sequential_backward=True but forward Impact phase missing; falling back to theory_of_change_summary")
            else:
                impact_aggregated = _aggregate_phase_round(fwd_impact)
                seed = (
                    "[SEQUENTIAL MODE] Forward direction predicts the following Impact-phase values "
                    "(aggregated across personas). Your job is to back-derive what upstream conditions "
                    "must have held to produce these values:\n"
                    + json.dumps(impact_aggregated, ensure_ascii=False, indent=2)
                )
                print(f"    [sequential_backward] seeded from forward Impact: {impact_aggregated}")

    # Optional: copy forward Impact directly into backward Impact (skip LLM call for that phase).
    # Useful for ablation studies — prevents backward Impact from drifting (truncation/over-anchor)
    # and ensures backward reasoning starts from exactly the forward Impact anchor.
    copy_fwd_impact = (
        sequential
        and config.get("stage2_toc", {}).get("copy_forward_impact_to_backward", False)
        and forward_pass is not None
    )

    spec_by_phase = {s.phase: s for s in inner_schema.phases}
    results_reverse: list[PhaseRoundResult] = []
    for phase in reversed(TOC_PHASES):
        if phase == "Inputs" and grounded_inputs is not None:
            from stage2_toc.grounded_inputs import grounded_input_phase_result

            print("      [backward] phase: Inputs — DOCUMENT-GROUNDED (persona calls skipped)")
            results_reverse.append(grounded_input_phase_result(grounded_inputs))
            continue
        if phase == "Impact" and copy_fwd_impact:
            fwd_impact = next((r for r in forward_pass if r.phase == "Impact"), None)
            if fwd_impact is not None:
                print(f"      [backward] phase: Impact — COPIED from forward (LLM call skipped)")
                results_reverse.append(fwd_impact)
                continue
        print(f"      [backward] phase: {phase}")
        rr = _phase_round(
            direction="backward", phase=phase, phase_spec=spec_by_phase[phase],
            personas=personas, policy_context=policy_context, simulation_target=simulation_target,
            prior_phases=results_reverse, policy_goal_seed=seed, config=config,
            inner_schema=inner_schema,
        )
        results_reverse.append(rr)
    by_phase = {r.phase: r for r in results_reverse}
    return [by_phase[p] for p in TOC_PHASES]


# ───────────────────────── Final forward refinement (sees backward) ─────────────────────────

def _format_prior_refinement_context(prior_refinement_summaries: list[tuple[str, str]]) -> str:
    """Format prior refinement phase summaries for the judgment prompt.

    Args:
        prior_refinement_summaries: list of (phase_name, summary_text) pairs
    """
    if not prior_refinement_summaries:
        return "--- Prior Refinement Judgments ---\n(none — this is the first phase in refinement)"

    lines = ["--- Prior Refinement Judgments ---"]
    for phase_name, summary in prior_refinement_summaries:
        lines.append(f"### {phase_name} (refinement judgment)")
        lines.append(f"{summary}")
    return "\n\n".join(lines)


def refine_forward_with_backward(
    forward_pass: list[PhaseRoundResult],
    backward_pass: list[PhaseRoundResult],
    personas: list[ConcretePersona],
    inner_schema: ToCInnerSchema,
    policy_context: PolicyContext,
    config: dict,
    simulation_target: str = "",
) -> list[PhaseRoundResult]:
    """Judgment-based refinement: for each phase, personas see the corresponding forward
    and backward summaries + their own utterances from both directions, then produce a
    final estimate. Uses sequential posting within each phase. Chained: later phases
    see summaries from earlier refinement phases.

    Mutates forward_pass in place: sets refined_posts on each PhaseRoundResult.
    """
    spec_by_phase = {s.phase: s for s in inner_schema.phases}
    bwd_by_phase = {r.phase: r for r in backward_pass}
    fwd_by_phase = {r.phase: r for r in forward_pass}
    policy_context_json = _policy_context_text(config, policy_context)
    # Stance condition variable: append framing directive (no-op for baseline/unset).
    refine_sys = REFINEMENT_JUDGE_SYSTEM + _stance_directive(config)

    # Accumulate refinement summaries for chaining
    prior_refinement_summaries: list[tuple[str, str]] = []

    for phase in TOC_PHASES:
        fwd_round = fwd_by_phase.get(phase)
        bwd_round = bwd_by_phase.get(phase)
        if fwd_round is None or bwd_round is None:
            continue
        spec = spec_by_phase.get(phase)
        if spec is None:
            continue
        fields_text = _format_phase_fields(spec)
        discussion_text = "\n".join(f"  - {q}" for q in spec.discussion_points)

        # Build persona lookup for own utterances
        fwd_by_name = {p.persona_name: p for p in fwd_round.revised_posts}
        bwd_by_name = {p.persona_name: p for p in bwd_round.revised_posts}

        # Prior refinement context (from earlier phases)
        prior_refine_text = _format_prior_refinement_context(prior_refinement_summaries)

        # Randomize posting order
        shuffled_personas = list(personas)
        random.shuffle(shuffled_personas)

        print(f"      [refinement judge] phase: {phase} ({len(prior_refinement_summaries)} prior)")
        refined: list[PersonaPhaseEstimate] = []
        for persona in shuffled_personas:
            own_fwd = fwd_by_name.get(persona.name)
            own_bwd = bwd_by_name.get(persona.name)

            own_fwd_text = json.dumps(own_fwd.model_dump(), ensure_ascii=False) if own_fwd else "(not available)"
            own_bwd_text = json.dumps(own_bwd.model_dump(), ensure_ascii=False) if own_bwd else "(not available)"

            seq_text = _format_sequential_context(refined)

            messages = [
                {"role": "system", "content": refine_sys},
                {
                    "role": "user",
                    "content": REFINEMENT_JUDGE_USER.format(
                        phase=phase,
                        phase_fields=fields_text,
                        discussion_points=discussion_text,
                        simulation_target=simulation_target,
                        forward_summary=fwd_round.phase_summary or "(no summary available)",
                        backward_summary=bwd_round.phase_summary or "(no summary available)",
                        own_forward_post=own_fwd_text,
                        own_backward_post=own_bwd_text,
                        prior_refinement_context=prior_refine_text,
                        sequential_context=seq_text,
                        policy_context=policy_context_json,
                        org_kg_contexts=get_org_kg_contexts(),
                        persona=json.dumps(persona.model_dump(), ensure_ascii=False),
                    ),
                },
            ]
            data = json.loads(chat_completion(messages, config), strict=False)
            refined.append(PersonaPhaseEstimate(
                persona_name=persona.name,
                stakeholder_type=persona.stakeholder_type,
                prediction_values=data.get("prediction_values") or {},
                narrative=data.get("narrative", ""),
                evidence=data.get("evidence", []),
                judgment=data.get("judgment", ""),
            ))
            print(f"        {persona.name} judged (#{len(refined)}/{len(shuffled_personas)})")

        fwd_round.refined_posts = refined

        # Generate refinement summary for this phase (used by later phases)
        print(f"        generating refinement summary...")
        refine_summary = _generate_phase_summary(
            phase, refined, "refinement", simulation_target, spec, config,
        )
        prior_refinement_summaries.append((phase, refine_summary))

    return forward_pass


# ───────────────────────── Cross-check + aggregation ─────────────────────────

def _mean_for_field(posts: list[PersonaPhaseEstimate], var_name: str) -> Optional[float]:
    vals = []
    for p in posts:
        v = p.prediction_values.get(var_name)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _forward_posts_for_agg(round_result: PhaseRoundResult) -> list[PersonaPhaseEstimate]:
    """Return refined_posts if present (final-forward-refinement was run), else revised_posts."""
    return round_result.refined_posts or round_result.revised_posts


def cross_check_phases(
    forward_pass: Optional[list[PhaseRoundResult]],
    backward_pass: Optional[list[PhaseRoundResult]],
    inner_schema: ToCInnerSchema,
) -> list[PhaseCrossCheck]:
    """For each phase × numeric field, compute F_agg, B_agg, gap, rel_gap, consistency, aggregated_value."""
    checks: list[PhaseCrossCheck] = []
    for spec in inner_schema.phases:
        fwd_round = next((r for r in forward_pass if r.phase == spec.phase), None) if forward_pass else None
        bwd_round = next((r for r in backward_pass if r.phase == spec.phase), None) if backward_pass else None
        for field in spec.prediction_fields:
            if field.type not in ("int", "float"):
                continue
            f_agg = _mean_for_field(_forward_posts_for_agg(fwd_round), field.name) if fwd_round else None
            b_agg = _mean_for_field(bwd_round.revised_posts, field.name) if bwd_round else None

            gap = rel_gap = consistency = None
            if f_agg is not None and b_agg is not None:
                # gap framing: backward (goal-required) minus forward (predicted).
                # positive gap = policy goal demands more than predicted will be delivered = implementation gap.
                gap = b_agg - f_agg
                denom = max(abs(f_agg), abs(b_agg), 1.0)
                rel_gap = abs(gap) / denom
                consistency = max(0.0, 1.0 - rel_gap)
                # aggregated_value is the PREDICTION, not a midpoint.
                # forward = predictive estimator (Elmore-style backward mapping is prescriptive, not predictive).
                aggregated = f_agg
            elif f_agg is not None:
                aggregated = f_agg
            elif b_agg is not None:
                # backward-only ablation: fall back to backward (only signal available).
                aggregated = b_agg
            else:
                aggregated = None

            checks.append(PhaseCrossCheck(
                phase=spec.phase,
                variable=field.name,
                forward_agg=f_agg,
                backward_agg=b_agg,
                gap=gap,
                relative_gap=rel_gap,
                consistency=consistency,
                aggregated_value=aggregated,
            ))
    return checks


def aggregate_scenario(
    scenario_id: str,
    participants: list[ConcretePersona],
    forward_pass: Optional[list[PhaseRoundResult]],
    backward_pass: Optional[list[PhaseRoundResult]],
    cross_checks: list[PhaseCrossCheck],
    fwd_bwd_fwd_pass: Optional[list[PhaseRoundResult]] = None,
    fwd_bwd_fwd_bwd_pass: Optional[list[PhaseRoundResult]] = None,
) -> ScenarioToCResult:
    """Build the per-scenario result, including Impact-phase aggregates + confidences."""
    impact_checks = [c for c in cross_checks if c.phase == "Impact"]
    scenario_impact = {c.variable: c.aggregated_value for c in impact_checks}
    scenario_confidence = {c.variable: c.consistency for c in impact_checks}

    return ScenarioToCResult(
        scenario_id=scenario_id,
        participants=participants,
        forward_pass=forward_pass,
        backward_pass=backward_pass,
        fwd_bwd_fwd_pass=fwd_bwd_fwd_pass,
        fwd_bwd_fwd_bwd_pass=fwd_bwd_fwd_bwd_pass,
        cross_checks=cross_checks,
        scenario_impact=scenario_impact,
        scenario_confidence=scenario_confidence,
    )


# ───────────────────────── Final reporter ─────────────────────────

def aggregate_across_scenarios(
    scenarios: list[ScenarioToCResult],
    inner_schema: ToCInnerSchema,
) -> tuple[dict[str, Optional[float]], list[PhaseCrossCheck]]:
    """Confidence-weighted Impact aggregation + aggregated implementation-gap report."""
    impact_spec = next((s for s in inner_schema.phases if s.phase == "Impact"), None)
    aggregated_final: dict[str, Optional[float]] = {}
    if impact_spec:
        for field in impact_spec.prediction_fields:
            if field.type not in ("int", "float"):
                continue
            num = denom = 0.0
            unweighted_vals = []
            for s in scenarios:
                v = s.scenario_impact.get(field.name)
                w = s.scenario_confidence.get(field.name)
                if v is None:
                    continue
                unweighted_vals.append(v)
                if w is not None and w > 0:
                    num += v * w
                    denom += w
            if denom > 0:
                aggregated_final[field.name] = num / denom
            elif unweighted_vals:
                aggregated_final[field.name] = sum(unweighted_vals) / len(unweighted_vals)
            else:
                aggregated_final[field.name] = None

    # Aggregated gap report: mean across scenarios per (phase, variable)
    gap_acc: dict[tuple[str, str], list[PhaseCrossCheck]] = {}
    for s in scenarios:
        for c in s.cross_checks:
            gap_acc.setdefault((c.phase, c.variable), []).append(c)

    impl_gap_report: list[PhaseCrossCheck] = []
    for (phase, variable), checks in gap_acc.items():
        def _mean(attr):
            vals = [getattr(c, attr) for c in checks if getattr(c, attr) is not None]
            return sum(vals) / len(vals) if vals else None
        impl_gap_report.append(PhaseCrossCheck(
            phase=phase,
            variable=variable,
            forward_agg=_mean("forward_agg"),
            backward_agg=_mean("backward_agg"),
            gap=_mean("gap"),
            relative_gap=_mean("relative_gap"),
            consistency=_mean("consistency"),
            aggregated_value=_mean("aggregated_value"),
        ))
    return aggregated_final, impl_gap_report


def run_phase_stakeholder_synthesizer(
    scenarios: list[ScenarioToCResult],
    simulation_target: str,
    config: dict,
) -> dict[str, dict[str, dict[str, str]]]:
    """1 LLM call → produce per-(phase, stakeholder_type) summary across forward/backward/refined.

    Aggregates all scenarios: when multiple personas exist for the same stakeholder_type
    (across scenarios or within), we include all of their narratives in the records and let
    the LLM produce a consolidated summary.
    """
    # Build per-(phase, stakeholder_type) records from the scenarios
    records: dict[str, dict[str, list[dict]]] = {}
    for s in scenarios:
        # Index forward / backward / refined posts by phase
        fwd_by_phase = {r.phase: r for r in (s.forward_pass or [])}
        bwd_by_phase = {r.phase: r for r in (s.backward_pass or [])}
        for phase in TOC_PHASES:
            fwd_round = fwd_by_phase.get(phase)
            bwd_round = bwd_by_phase.get(phase)
            # Per-persona entries in this phase
            persona_index: dict[str, dict] = {}
            if fwd_round:
                for est in fwd_round.revised_posts:
                    persona_index.setdefault(est.persona_name, {
                        "persona_name": est.persona_name,
                        "stakeholder_type": est.stakeholder_type,
                        "scenario_id": s.scenario_id,
                    })["forward"] = {
                        "prediction_values": est.prediction_values,
                        "narrative": est.narrative,
                        "evidence": est.evidence,
                    }
                if fwd_round.refined_posts:
                    for est in fwd_round.refined_posts:
                        persona_index.setdefault(est.persona_name, {
                            "persona_name": est.persona_name,
                            "stakeholder_type": est.stakeholder_type,
                            "scenario_id": s.scenario_id,
                        })["refined"] = {
                            "prediction_values": est.prediction_values,
                            "narrative": est.narrative,
                            "evidence": est.evidence,
                        }
            if bwd_round:
                for est in bwd_round.revised_posts:
                    persona_index.setdefault(est.persona_name, {
                        "persona_name": est.persona_name,
                        "stakeholder_type": est.stakeholder_type,
                        "scenario_id": s.scenario_id,
                    })["backward"] = {
                        "prediction_values": est.prediction_values,
                        "narrative": est.narrative,
                        "evidence": est.evidence,
                    }
            for entry in persona_index.values():
                stype = entry["stakeholder_type"]
                records.setdefault(phase, {}).setdefault(stype, []).append(entry)

    messages = [
        {"role": "system", "content": PHASE_STAKEHOLDER_SYNTHESIZER_SYSTEM},
        {
            "role": "user",
            "content": PHASE_STAKEHOLDER_SYNTHESIZER_USER.format(
                simulation_target=simulation_target,
                org_kg_contexts=get_org_kg_contexts(),
                records=json.dumps(records, ensure_ascii=False, indent=2),
            ),
        },
    ]
    data = json.loads(chat_completion(messages, config), strict=False)
    summaries = data.get("phase_stakeholder_summaries", {})
    # Defensive: ensure all 5 phase keys exist with all 3 direction subkeys
    for phase in TOC_PHASES:
        summaries.setdefault(phase, {})
        for direction in ("forward", "backward", "refined"):
            summaries[phase].setdefault(direction, {})
    return summaries


def run_final_toc_reporter(
    simulation_target: str,
    scenarios: list[ScenarioToCResult],
    aggregated_final: dict[str, Optional[float]],
    implementation_gap_report: list[PhaseCrossCheck],
    config: dict,
) -> str:
    """LLM call → narrative_synthesis string."""
    scenario_summaries = []
    for s in scenarios:
        scenario_summaries.append({
            "scenario_id": s.scenario_id,
            "participants": [p.name for p in s.participants],
            "scenario_impact": s.scenario_impact,
            "scenario_confidence": s.scenario_confidence,
            "top_gaps": sorted(
                [c.model_dump() for c in s.cross_checks if c.relative_gap is not None],
                key=lambda c: c["relative_gap"], reverse=True,
            )[:3],
        })

    messages = [
        {"role": "system", "content": FINAL_TOC_REPORTER_SYSTEM},
        {
            "role": "user",
            "content": FINAL_TOC_REPORTER_USER.format(
                simulation_target=simulation_target,
                num_scenarios=len(scenarios),
                aggregated_final=json.dumps(aggregated_final, ensure_ascii=False),
                implementation_gap_report=json.dumps(
                    [c.model_dump() for c in implementation_gap_report], ensure_ascii=False,
                ),
                scenario_summaries=json.dumps(scenario_summaries, ensure_ascii=False),
            ),
        },
    ]
    data = json.loads(chat_completion(messages, config), strict=False)
    return data["narrative_synthesis"]
