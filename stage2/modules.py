import json

from llm.client import chat_completion


def _org_kg_contexts_payload() -> str:
    """Return the serialized org_kg_contexts injected into ToC round prompts."""
    try:
        from stage2_toc.runtime import get_org_kg_contexts
        return get_org_kg_contexts()
    except ImportError:
        return "(no organizations registered for this run)"
from models.prompts import (
    SIM_PERSONA_CONSTRUCTOR_SYSTEM, SIM_PERSONA_CONSTRUCTOR_USER,
    ROUND1_SYSTEM, ROUND1_USER,
    DYNAMIC_ROUND_SYSTEM, DYNAMIC_ROUND_USER,
    MANDATORY_ROUND_SYSTEM, MANDATORY_ROUND_USER,
    FINAL_ROUND_SYSTEM, FINAL_ROUND_USER,
    JUDGE_SYSTEM, JUDGE_USER,
    POLICY_REPORTER_SYSTEM, POLICY_REPORTER_USER,
)
from models.schemas import (
    EconomicContext,
    PolicyPlan,
    PersonaProfile,
    PredictionResult,
    StakeholderGroup,
    ConcretePersona,
    Round1Comment,
    DynamicRoundComment,
    FinalRoundComment,
    JudgeSummary,
    ConcreteScenarioResult,
    PolicyReportResult,
)


def construct_all_simulation_personas(
    stakeholder_groups: list[StakeholderGroup],
    prediction_results: list[PredictionResult],
    config: dict,
) -> list[ConcretePersona]:
    """Construct N concrete personas per stage1 persona type (N = sim_personas_per_entity in config)."""
    n = config.get("stage2", {}).get("sim_personas_per_entity", 1)
    pred_map = {pr.stakeholder_type: {p.representative_type: p for p in pr.predictions} for pr in prediction_results}
    result = []
    used_names: list[str] = []
    for group in stakeholder_groups:
        group_preds = pred_map.get(group.stakeholder_type, {})
        for persona in group.personas:
            prediction = group_preds.get(persona.representative_type)
            for _ in range(n):
                used_names_note = (
                    f"\n\nNames already assigned to other participants (do NOT use these): {', '.join(used_names)}"
                    if used_names else ""
                )
                messages = [
                    {"role": "system", "content": SIM_PERSONA_CONSTRUCTOR_SYSTEM},
                    {
                        "role": "user",
                        "content": SIM_PERSONA_CONSTRUCTOR_USER.format(
                            stakeholder_type=group.stakeholder_type,
                            org_kg_contexts=_org_kg_contexts_payload(),
                            personas_with_predictions=json.dumps(
                                [{
                                    "persona": {k: v for k, v in persona.model_dump().items()
                                                if k != "policy_outcome_magnitude"},
                                    "stage1_prediction": {k: v for k, v in (prediction.model_dump() if prediction else {}).items()
                                                          if k != "predicted_value"},
                                }],
                                ensure_ascii=False,
                            ),
                        ) + used_names_note,
                    },
                ]
                data = json.loads(chat_completion(messages, config))
                cp = ConcretePersona(**data)
                cp.entity_name = persona.representative_type
                used_names.append(cp.name)
                result.append(cp)
    return result


def _check_consensus(current_predictions: dict[str, float], threshold: float) -> bool:
    """Return True if all predictions fall within threshold relative range."""
    values = list(current_predictions.values())
    if len(values) < 2:
        return True
    mean = sum(values) / len(values)
    if mean == 0:
        return True
    return (max(values) - min(values)) / mean < threshold


def _format_debate_history(
    round1: list[Round1Comment],
    dynamic_rounds: list[list[DynamicRoundComment]],
    current_predictions: dict[str, float],
    max_recent_rounds: int = 3,
) -> str:
    """Format debate history as a readable string. Keeps Round 1 + last N dynamic rounds."""
    sections = []

    r1_lines = [
        f"  {c.representative_type} ({c.stakeholder_type}): predicted {c.predicted_value} — {c.comment}"
        for c in round1
    ]
    sections.append("Round 1 (Initial Predictions):\n" + "\n".join(r1_lines))

    # Only include the most recent rounds to avoid context overflow
    recent = dynamic_rounds[-max_recent_rounds:] if len(dynamic_rounds) > max_recent_rounds else dynamic_rounds
    offset = len(dynamic_rounds) - len(recent)
    if offset > 0:
        sections.append(f"[Rounds 2–{1 + offset} omitted for brevity — see current predictions below]")

    for i, dround in enumerate(recent, start=2 + offset):
        lines = []
        for c in dround:
            if c.spoke:
                lines.append(f"  {c.representative_type}: revised to {c.revised_predicted_value} — {c.comment}")
            else:
                lines.append(f"  {c.representative_type}: [passed — holding {current_predictions.get(c.representative_type, '?')}]")
        sections.append(f"Round {i}:\n" + "\n".join(lines))

    cur = "\n".join(f"  {name}: {val}" for name, val in current_predictions.items())
    sections.append(f"Current predictions:\n{cur}")

    return "\n\n".join(sections)


def run_concrete_scenario(
    scenario_id: str,
    concrete_personas: list[ConcretePersona],
    policy: PolicyPlan,
    context: EconomicContext,
    simulation_target: str,
    config: dict,
    relationship_context: list[dict] | None = None,
) -> ConcreteScenarioResult:
    """Run dynamic debate loop + final round + judge."""
    max_dynamic_rounds = config["stage2"].get("max_dynamic_rounds", 5)
    consensus_threshold = config["stage2"].get("consensus_range_ratio", 0.2)
    use_consensus = config["stage2"].get("use_consensus", True)
    mandatory_speaking = config["stage2"].get("mandatory_speaking", False)
    scenario = [(p.stakeholder_type, p) for p in concrete_personas]

    # Round 1 — mandatory
    round1 = _run_round1(scenario, policy, context, simulation_target, config, relationship_context)
    print(f"    Round 1 done: {[f'{c.representative_type}={c.predicted_value}' for c in round1]}")

    # Track current predictions (updated when a persona speaks)
    current_predictions = {c.representative_type: c.predicted_value for c in round1}

    # Dynamic rounds
    dynamic_rounds: list[list[DynamicRoundComment]] = []
    consensus_reached = use_consensus and _check_consensus(current_predictions, consensus_threshold)

    for round_num in range(2, 2 + max_dynamic_rounds):
        if consensus_reached:
            print(f"    Consensus reached before round {round_num}, skipping dynamic rounds.")
            break

        history_text = _format_debate_history(round1, dynamic_rounds, current_predictions)
        if mandatory_speaking:
            dround = _run_mandatory_round(scenario, policy, context, simulation_target, config, history_text, current_predictions)
        else:
            dround = _run_dynamic_round(scenario, policy, context, simulation_target, config, history_text, current_predictions)
        dynamic_rounds.append(dround)

        # Update current predictions for speakers
        for c in dround:
            if c.spoke and c.revised_predicted_value is not None:
                current_predictions[c.representative_type] = c.revised_predicted_value

        all_passed = all(not c.spoke for c in dround)
        consensus_reached = use_consensus and _check_consensus(current_predictions, consensus_threshold)
        speakers = [c.representative_type for c in dround if c.spoke]
        print(f"    Round {round_num}: speakers={speakers}, predictions={current_predictions}, consensus={consensus_reached}")

        if use_consensus and (consensus_reached or all_passed):
            break

    # Final round — mandatory
    history_text = _format_debate_history(round1, dynamic_rounds, current_predictions)
    final_round = _run_final_round(scenario, policy, context, simulation_target, config, history_text, current_predictions)

    # Update predictions from final round for judge
    final_predictions = {c.representative_type: c.final_predicted_value for c in final_round}

    judge = _run_judge(round1, dynamic_rounds, final_round, policy, context, simulation_target,
                       consensus_reached, config)

    return ConcreteScenarioResult(
        scenario_id=scenario_id,
        participants=concrete_personas,
        round1=round1,
        dynamic_rounds=dynamic_rounds,
        final_round=final_round,
        consensus_reached=consensus_reached,
        judge=judge,
    )


def _econ_suffix(context: EconomicContext, config: dict) -> str:
    """Return economic context section if enabled in config, else empty string."""
    if config["stage2"].get("use_economic_context_in_simulation", True):
        return f"\n\n--- Economic Context ---\n{json.dumps(context.model_dump(), ensure_ascii=False)}"
    return ""


def _run_round1(
    scenario: list[tuple[str, ConcretePersona]],
    policy: PolicyPlan,
    context: EconomicContext,
    simulation_target: str,
    config: dict,
    relationship_context: list[dict] | None = None,
) -> list[Round1Comment]:
    rel_text = ""
    if relationship_context:
        lines = [f"- [{r['relationship']}] {r['source']} → {r['target']}: {r['fact']}"
                 for r in relationship_context]
        rel_text = "\n\n--- Known Relationships Among Participants ---\n" + "\n".join(lines)

    econ_text = _econ_suffix(context, config)
    results = []
    for stakeholder_type, persona in scenario:
        messages = [
            {"role": "system", "content": ROUND1_SYSTEM},
            {
                "role": "user",
                "content": ROUND1_USER.format(
                    simulation_target=simulation_target,
                    policy_plan=json.dumps(policy.model_dump(), ensure_ascii=False),
                    persona=json.dumps(persona.model_dump(), ensure_ascii=False),
                ) + econ_text + rel_text,
            },
        ]
        data = json.loads(chat_completion(messages, config))
        results.append(Round1Comment(
            stakeholder_type=stakeholder_type,
            representative_type=persona.representative_type,
            predicted_value=data["predicted_value"],
            comment=data["comment"],
            evidence=data["evidence"],
        ))
    return results


def _run_dynamic_round(
    scenario: list[tuple[str, ConcretePersona]],
    policy: PolicyPlan,
    context: EconomicContext,
    simulation_target: str,
    config: dict,
    debate_history: str,
    current_predictions: dict[str, float],
) -> list[DynamicRoundComment]:
    econ_text = _econ_suffix(context, config)
    results = []
    for stakeholder_type, persona in scenario:
        cur_pred = current_predictions.get(persona.representative_type, "unknown")
        messages = [
            {"role": "system", "content": DYNAMIC_ROUND_SYSTEM},
            {
                "role": "user",
                "content": DYNAMIC_ROUND_USER.format(
                    simulation_target=simulation_target,
                    policy_plan=json.dumps(policy.model_dump(), ensure_ascii=False),
                    persona=json.dumps(persona.model_dump(), ensure_ascii=False),
                    debate_history=debate_history,
                    current_prediction=cur_pred,
                ) + econ_text,
            },
        ]
        data = json.loads(chat_completion(messages, config))
        results.append(DynamicRoundComment(
            stakeholder_type=stakeholder_type,
            representative_type=persona.representative_type,
            spoke=bool(data.get("spoke", False)),
            revised_predicted_value=data.get("revised_predicted_value"),
            comment=data.get("comment"),
        ))
    return results


def _run_mandatory_round(
    scenario: list[tuple[str, ConcretePersona]],
    policy: PolicyPlan,
    context: EconomicContext,
    simulation_target: str,
    config: dict,
    debate_history: str,
    current_predictions: dict[str, float],
) -> list[DynamicRoundComment]:
    """Everyone speaks — no pass option. Each persona must engage with opposing views."""
    econ_text = _econ_suffix(context, config)
    results = []
    for stakeholder_type, persona in scenario:
        cur_pred = current_predictions.get(persona.representative_type, "unknown")
        others = {name: val for name, val in current_predictions.items()
                  if name != persona.representative_type}
        other_pred_text = ", ".join(f"{name}: {val}" for name, val in others.items())
        messages = [
            {"role": "system", "content": MANDATORY_ROUND_SYSTEM},
            {
                "role": "user",
                "content": MANDATORY_ROUND_USER.format(
                    simulation_target=simulation_target,
                    policy_plan=json.dumps(policy.model_dump(), ensure_ascii=False),
                    persona=json.dumps(persona.model_dump(), ensure_ascii=False),
                    debate_history=debate_history,
                    current_prediction=cur_pred,
                    other_predictions=other_pred_text,
                ) + econ_text,
            },
        ]
        data = json.loads(chat_completion(messages, config))
        results.append(DynamicRoundComment(
            stakeholder_type=stakeholder_type,
            representative_type=persona.representative_type,
            spoke=True,
            revised_predicted_value=data.get("revised_predicted_value"),
            comment=data.get("comment"),
        ))
    return results


def _run_final_round(
    scenario: list[tuple[str, ConcretePersona]],
    policy: PolicyPlan,
    context: EconomicContext,
    simulation_target: str,
    config: dict,
    debate_history: str,
    current_predictions: dict[str, float],
) -> list[FinalRoundComment]:
    econ_text = _econ_suffix(context, config)
    results = []
    for stakeholder_type, persona in scenario:
        cur_pred = current_predictions.get(persona.representative_type, "unknown")
        messages = [
            {"role": "system", "content": FINAL_ROUND_SYSTEM},
            {
                "role": "user",
                "content": FINAL_ROUND_USER.format(
                    simulation_target=simulation_target,
                    policy_plan=json.dumps(policy.model_dump(), ensure_ascii=False),
                    persona=json.dumps(persona.model_dump(), ensure_ascii=False),
                    debate_history=debate_history,
                    current_prediction=cur_pred,
                ) + econ_text,
            },
        ]
        data = json.loads(chat_completion(messages, config))
        results.append(FinalRoundComment(
            stakeholder_type=stakeholder_type,
            representative_type=persona.representative_type,
            final_predicted_value=data["final_predicted_value"],
            final_comment=data["final_comment"],
            improvement_suggestions=data["improvement_suggestions"],
        ))
    return results


def _run_judge(
    round1: list[Round1Comment],
    dynamic_rounds: list[list[DynamicRoundComment]],
    final_round: list[FinalRoundComment],
    policy: PolicyPlan,
    context: EconomicContext,
    simulation_target: str,
    consensus_reached: bool,
    config: dict,
) -> JudgeSummary:
    econ_text = _econ_suffix(context, config)
    full_discussion = json.dumps({
        "round1": [c.model_dump() for c in round1],
        "dynamic_rounds": [[c.model_dump() for c in dr] for dr in dynamic_rounds],
        "final_round": [c.model_dump() for c in final_round],
    }, ensure_ascii=False)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_USER.format(
                simulation_target=simulation_target,
                policy_plan=json.dumps(policy.model_dump(), ensure_ascii=False),
                full_discussion=full_discussion,
                consensus_reached=consensus_reached,
                num_rounds=1 + len(dynamic_rounds) + 1,
            ) + econ_text,
        },
    ]
    data = json.loads(chat_completion(messages, config))
    # Compute aggregated_prediction in code — LLM arithmetic is unreliable
    final_vals = [c.final_predicted_value for c in final_round]
    data["aggregated_prediction"] = round(sum(final_vals) / len(final_vals), 1) if final_vals else 0.0
    return JudgeSummary(**data)


def run_final_reporter(
    scenarios: list[ConcreteScenarioResult],
    policy: PolicyPlan,
    config: dict,
    ground_truth: float | None = None,
) -> PolicyReportResult:
    """Aggregate all scenario results and synthesize a final policy report via LLM."""
    simulation_target = config.get("_simulation_target", "")

    # Compute avg prediction by averaging per-scenario judge aggregations (each scenario has equal weight)
    judge_preds = [s.judge.aggregated_prediction for s in scenarios]
    avg_prediction = round(sum(judge_preds) / len(judge_preds), 1) if judge_preds else 0.0
    all_final_preds = [c.final_predicted_value for s in scenarios for c in s.final_round]
    pred_min = round(min(all_final_preds), 0) if all_final_preds else 0
    pred_max = round(max(all_final_preds), 0) if all_final_preds else 0
    prediction_range = f"{int(pred_min)}-{int(pred_max)}"

    error = round(abs(avg_prediction - ground_truth), 1) if ground_truth is not None else None
    error_pct = round(abs(avg_prediction - ground_truth) / ground_truth * 100, 1) if ground_truth else None

    # Build compact scenario summaries for the LLM
    scenario_summaries = []
    for s in scenarios:
        scenario_summaries.append({
            "scenario_id": s.scenario_id,
            "judge": {
                "aggregated_prediction": s.judge.aggregated_prediction,
                "stance_summary": s.judge.stance_summary,
                "key_agreements": s.judge.key_agreements,
                "key_disagreements": s.judge.key_disagreements,
            },
            "final_round": [
                {
                    "representative_type": c.representative_type,
                    "stakeholder_type": c.stakeholder_type,
                    "final_predicted_value": c.final_predicted_value,
                    "final_comment": c.final_comment,
                    "improvement_suggestions": c.improvement_suggestions,
                }
                for c in s.final_round
            ],
        })

    messages = [
        {"role": "system", "content": POLICY_REPORTER_SYSTEM},
        {
            "role": "user",
            "content": POLICY_REPORTER_USER.format(
                simulation_target=simulation_target,
                num_scenarios=len(scenarios),
                policy_plan=json.dumps(policy.model_dump(), ensure_ascii=False),
                scenario_summaries=json.dumps(scenario_summaries, ensure_ascii=False),
            ),
        },
    ]
    data = json.loads(chat_completion(messages, config))
    return PolicyReportResult(
        simulation_target=simulation_target,
        avg_prediction=avg_prediction,
        prediction_range=prediction_range,
        ground_truth=ground_truth,
        error=error,
        error_pct=error_pct,
        common_evidence=data["common_evidence"],
        common_concerns=data["common_concerns"],
        common_recommendations=data["common_recommendations"],
        synthesis=data["synthesis"],
    )
