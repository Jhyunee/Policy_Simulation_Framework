"""
IAD/IG-aware ontology generation for stakeholder-centered policy simulation.
"""

import json
from typing import Any

from llm.client import chat_completion
from graph.normalizer import DEFAULT_IAD_RULE_BY_FAMILY


ONTOLOGY_SYSTEM_PROMPT = """You are an expert knowledge graph ontology designer for stakeholder-centered policy impact simulation.
Analyze the policy document content and simulation requirement, then define a policy KG ontology that represents the
policy's institutional structure, stakeholder positions, action situation, rules, information flows, and resource flows.
Use Institutional Analysis and Development (IAD) for node categories and Institutional Grammar (IG 2.0) for regulative
and constitutive edge rationale.

IMPORTANT: Output only valid JSON. Do not include markdown or code block formatting.
LANGUAGE REQUIREMENT: All ontology labels, descriptions, rationales, examples, attributes, and analysis text must be in
English, even when the source document is in another language. Translate source terms into concise English labels.

Core purpose:
- Persona-eligible actor or position entities become simulation actor anchors for Stage 1 persona generation.
- A persona-eligible type may represent an individual person, a social group, an organization, or an official position
  that can act, decide, implement, fund, evaluate, receive benefits, bear costs, or be affected in the policy.
- Persona-eligible actor anchors must remain methodologically distinct by agent modality and role family:
  individual, stakeholder group, host organization, committee/panel member, official position holder, policy administrator,
  mentor, evaluator, direct beneficiary, pipeline actor, and affected actor are not interchangeable.
- Non-actor policy objects explain what constrains, enables, funds, evaluates, informs, or measures those roles.
- The ontology must support direct node/edge instance extraction without Zep and must be generalizable across policies.
- The ontology should make later instance extraction compact and edge-ready. Avoid entity types that would mostly become
  isolated glossary nodes rather than actors, rules, resources, information, actions, or outcomes used in relations.

Use IAD as the node rationale layer:
- Participants: people, groups, or organizations that act, implement, fund, evaluate, receive benefits, bear costs, or are affected.
- Positions: formal policy/program role slots that participants can occupy, such as applicant, fellow, reviewer, mentor, host institution,
  program officer, principal investigator, or committee member.
- Actions: policy process phases or required activities, such as application, selection, agreement, execution, reporting,
  evaluation, payment, monitoring, or appeal.
- Information: documents, criteria, reports, plans, eligibility evidence, metrics, notices, records, or decision signals.
- Costs_and_Benefits: funding, grants, salaries, indirect costs, opportunity costs, compliance burdens, rewards, penalties,
  sanctions, or other material/non-material incentives.

Use IAD's seven Rules-in-Use as the analytical lens for every entity type. Assign exactly one iad_rule_type to each entity type
(use null only when the entity is purely descriptive context with no rule-bearing content, which should be rare):
- position: defines what positions/roles exist in the action arena (who can occupy a role).
  Typical entity types: PostdoctoralResearcher, PrincipalInvestigator, EvaluationPanel, ApprovalAuthority.
- boundary: governs entry to and exit from positions — eligibility, qualification, exclusion, target population.
  Typical entity types: EligibilityCriteria, ExclusionCriteria, TargetPopulation, ApplicantQualification.
- choice: governs what actions a position-holder may take — permissions, allowed alternatives, discretionary actions.
  Typical entity types: ApplicationProcess, ResearchExecution, MobilityOption, ResearchTopicChange.
- aggregation: governs how decisions get made across positions — voting, approval chains, authority hierarchy.
  Typical entity types: DecisionAuthority, ApprovalPanel, GoverningCommittee, OversightBody.
- information: governs what information must flow between which positions — disclosure, reporting, evidence requirements.
  Typical entity types: ResearchProposal, ResultReport, NotificationDocument, EvaluationCriteriaDocument.
- payoff: governs costs, benefits, rewards, sanctions assigned to positions and actions — material and non-material.
  Typical entity types: ResearchGrant, IndirectCost, Allowance, NoncompliancePenalty, Sanction.
- scope: governs the outcomes/range targeted by the policy — quantitative caps, target metrics, expected outcomes.
  Typical entity types: AnnualSelectionQuota, FundingLimit, TargetMetric, ExpectedOutcome.

Mapping guidance:
- node_family ActorParticipant / PositionRole → iad_rule_type=position.
- node_family EligibilityCondition / RestrictionConstraint → iad_rule_type=boundary.
- node_family PolicyActionProcess → iad_rule_type=choice (when describing actor-level discretionary actions) or aggregation (when describing decision processes such as approval, voting, selection).
- node_family AuthorityControl → iad_rule_type=aggregation.
- node_family InformationEvidence → iad_rule_type=information.
- node_family ResourceBenefitCost → iad_rule_type=payoff.
- node_family TemporalMilestone → iad_rule_type=scope (when defining outcome timeframe) or boundary (when defining eligibility timeframe).
- node_family EvaluationMetricOutcome → iad_rule_type=scope.
This mapping is a default; pick the rule that the entity type most directly governs in the policy's institutional logic.

Use node_family as the finer-grained policy-simulation layer. Every entity type must use exactly one of:
- ActorParticipant: beneficiaries, affected groups, implementers, funders, evaluators, host organizations, agencies.
- PositionRole: official role slots such as applicant, selected fellow, principal investigator, reviewer, mentor, officer.
- EligibilityCondition: eligibility pathways, required qualifications, demographic/statutory conditions.
- RestrictionConstraint: prohibitions, exclusions, limits, mandatory constraints, compliance burdens.
- PolicyActionProcess: application, review, selection, agreement, execution, reporting, monitoring, evaluation, notification.
- InformationEvidence: proposals, reports, criteria, evidence, notices, records, metrics, decision signals.
- ResourceBenefitCost: grants, salaries, direct costs, indirect costs, allowances, support services, sanctions, costs.
- AuthorityControl: decision bodies, approval authority, management authority, governance control points.
- TemporalMilestone: deadlines, years, phases, start dates, support periods, frequency limits.
- EvaluationMetricOutcome: evaluation criteria, selection ratios, expected outputs, intended outcomes, performance metrics.

Use IG 2.0 as the institutional statement rationale layer:
- A regulative statement has the shape [Actor] [DEONTIC] [Aim] [Object] IF [Condition] OR ELSE [Consequence].
- A constitutive statement defines institutional meaning: [Term/condition/action/object] counts-as, means, qualifies as,
  or is defined as [institutional category/status]. Use this for definition clauses and statutory qualification language.
- A structural/strategy statement records an observed process, resource flow, evaluation relation, or causal relation
  without a deontic or definition function.
- Map must/shall/required/has to/mandatory to "must". Treat should/recommended/endeavor-to language as soft unless
  the document clearly uses it as a mandatory rule.
- Map may/can/is allowed/eligible to "can".
- Map cannot/must not/prohibited/restricted/ineligible to "cannot".

Return JSON with this exact structure:
{
  "entity_types": [
    {
      "name": "EntityTypeName",
      "description": "Brief description under 100 characters",
      "iad_category": "Participants | Positions | Actions | Information | Costs_and_Benefits",
      "node_family": "ActorParticipant | PositionRole | EligibilityCondition | RestrictionConstraint | PolicyActionProcess | InformationEvidence | ResourceBenefitCost | AuthorityControl | TemporalMilestone | EvaluationMetricOutcome",
      "iad_rule_type": "position | boundary | choice | aggregation | information | payoff | scope",
      "is_actor": true,
      "persona_eligible": true,
      "attributes": [
        {"name": "attribute_name", "type": "text", "description": "Attribute description"}
      ],
      "examples": ["Example entity 1", "Example entity 2"],
      "rationale": "Why this type matters in IAD terms"
    }
  ],
  "edge_types": [
    {
      "name": "RELATIONSHIP_NAME",
      "description": "Brief description under 100 characters",
      "edge_category": "structural | regulative | constitutive",
      "edge_family": "EligibilityExclusion | ProcessSequence | AuthorityGovernance | ResourceIncentive | InformationEvidence | PermissionConstraint | EvaluationSelection | OutcomeCausalMechanism | DefinitionQualification",
      "ig_deontic": "must | can | cannot | null",
      "source_targets": [
        {"source": "SourceEntityType", "target": "TargetEntityType"}
      ],
      "attributes": [],
      "required_attributes": ["source_text"],
      "rationale": "Why this relationship matters in policy/IAD/IG terms"
    }
  ],
  "analysis_summary": "Brief summary of the policy ecosystem"
}

Entity type rules:
1. Output 12-20 entity types when the document supports that many. Use more only when the policy has clearly distinct
   actors, rules, resources, information objects, processes, or outcomes that are likely to appear in KG edges.
2. Keep stakeholder actors central, but include non-actor policy objects needed to explain actions, information, costs,
   benefits, and institutional rules.
3. Assign exactly one IAD category to every entity type.
4. Assign exactly one node_family to every entity type. The node_family should be more specific than the IAD category.
   Non-actor objects must not use ActorParticipant or PositionRole.
5. Mark persona_eligible=true broadly for actor, stakeholder group, organization, or official position types that can become
   simulation personas or simulation agents.
   Use persona_eligible=true for target populations, near-eligible applicants/candidates, students or pipeline groups,
   beneficiaries, implementers, evaluators, funders, intermediaries, host organizations, regulators, affected stakeholders,
   administrative officers, reviewers, mentors, and official role slots.
   Do not restrict persona_eligible to the main beneficiary only.
   Set persona_eligible=false only for non-actors such as documents, metrics, budgets, process phases, policy programs,
   resources, or purely passive objects.
6. Coverage requirements:
   - Participants: at least 3 types
   - Positions: at least 2 types
   - Actions: 2-4 types
   - Information: at least 4 types, including eligibility, evidence, criteria, or reports when supported
   - Costs_and_Benefits: at least 2 types when funding or cost details are described
   - At least one node_family candidate for EligibilityCondition, RestrictionConstraint, PolicyActionProcess,
     InformationEvidence, ResourceBenefitCost, AuthorityControl, TemporalMilestone, and EvaluationMetricOutcome when supported.
7. Disambiguation:
   - Government ministries and agencies are Participants, not Positions.
   - Positions are official role slots occupied by Participants. If the document describes a target group as a role in the
     program (e.g., applicant, fellow, selected researcher), represent that role explicitly.
   - Actions are process phases; create one type per phase, not one per mention.
   - Documents, plans, reports, proposals, eligibility evidence, and evaluation criteria are Information, not Costs_and_Benefits.
   - Costs_and_Benefits should be split by resource/reward/sanction type when the document distinguishes them.
   - Grants, funding limits, salaries, indirect costs, support services, and sanctions are Costs_and_Benefits.
   - Committees, panels, ministries, agencies, foundations, and institutions are actors only when they perform policy actions.
     Support programs, grants, reports, proposals, notifications, compliance requirements, and funding limits are not actors.
   - Metrics can be included only when they are used as Information for decisions or evaluation.
8. Do not use reserved attribute names: name, uuid, group_id, created_at, summary.
9. Avoid generic umbrella type names such as TargetPopulation, FundingAgencies, ResearchCosts, or ResearchBenefits when
   the document supports more specific policy-domain types. Derive type names from the current document's own roles,
   organizations, documents, resources, criteria, and decision bodies. Use examples only as naming patterns, not as
   reusable labels for unrelated policies.
   Also avoid generic placeholder types that duplicate IAD categories or node families.
   If the document says "young scientists" but then defines a statutory target role such as postdoctoral researchers,
   use the statutory role type as the direct-beneficiary actor and keep the broader phrase only in descriptions.
10. Specific programs or projects are usually policy objects, not funding agencies. Do not classify a program as a
   funding agency unless the document states it acts as an administrative organization.
11. Do not create a persona-eligible actor type named after the policy or program itself. For example, for a fellowship
    policy, create PostdoctoralResearcher or EarlyCareerResearcher as the beneficiary actor, not SejongScienceFellow
    or FellowshipProgram. The policy/program itself should be non-persona policy context.
12. For each persona-eligible type, include attributes that help later persona synthesis check policy-role consistency,
    when supported by the document:
    - agent_modality
    - role_family
    - eligibility_criteria
    - exclusion_criteria
    - role_scope
    - decision_authority
    - resources_accessed
    - reporting_or_compliance_duties
    Use only attributes that can be grounded in the document or clearly inferred from the role.
13. Preserve diversity in stakeholder roles. Do not collapse implementers, beneficiaries, applicants/candidates,
    pipeline groups, evaluators, funders, intermediaries, host organizations, and affected groups into a single generic
    stakeholder type.
14. If the policy defines a role through eligibility pathways (e.g., degree obtained, within N years, under a certain age,
    doctoral-track, applicant, selected participant), represent the pathway as Information/attributes or a Position/role
    node instead of using it as a reason to exclude adjacent stakeholder actors from persona generation.
15. All entity type names must be English PascalCase. Do not use source-language labels as ontology type names.
16. Do not collapse collective bodies into private persons. A committee, panel, ministry, foundation, or institution can
    be an actor anchor, but its persona use must be as an institutional profile, member, representative, or role-holder.
17. For policies with career pipelines, include adjacent/pipeline actors when the document supports them, separate from
    the direct beneficiary. Their role_family should be pipeline_actor or affected_actor, not direct_beneficiary.
18. The final KG validator will prune instance nodes that are not connected to supported edges. Design entity types so
    useful instances can participate in explicit actor-rule-resource-information-action relations.

Edge type rules:
1. Output 18-28 edge types when supported. Use fewer only if the document is genuinely sparse.
2. Assign exactly one edge_family to every edge type.
3. Include the following edge families when supported by the document:
   - EligibilityExclusion: ELIGIBLE_FOR, INELIGIBLE_FOR, REQUIRES, EXCLUDES.
   - ProcessSequence: APPLIES_TO, SUBMITS, REVIEWS, PRECEDES, TRIGGERS, NOTIFIES.
   - AuthorityGovernance: IMPLEMENTS, MANAGES, DECIDES, REPORTS_TO, REQUIRES_APPROVAL.
   - ResourceIncentive: FUNDS, RECEIVES, ALLOCATES, COVERS_COST, PROVIDES_SUPPORT, WITHHOLDS.
   - InformationEvidence: USES_INFORMATION, REQUIRES_DOCUMENT, PRODUCES_REPORT, USES_CRITERIA.
   - PermissionConstraint: CAN_APPLY, CAN_CHOOSE, CAN_MOVE, CAN_CHANGE, CANNOT_APPLY, CANNOT_RECEIVE,
     CANNOT_CONDUCT, LIMITED_TO.
   - EvaluationSelection: EVALUATES, SELECTS, APPROVES, SCORES, MONITORS.
   - OutcomeCausalMechanism: TARGETS, AIMS_TO_IMPROVE, MEASURES, EXPECTED_TO_INCREASE.
   - DefinitionQualification: DEFINED_AS, CONSTITUTES, COUNTS_AS, QUALIFIES_AS.
4. Include structural edges such as PRECEDES, FUNDS, RECEIVES, INFLUENCES, IMPLEMENTS, EVALUATES, SUBMITS,
   USES_INFORMATION when supported by the document.
5. Include regulative IG edges for deontic rules. Use specific names such as MUST_SUBMIT, CAN_CHOOSE,
   CANNOT_CHANGE, MUST_EVALUATE.
6. Regulative edge types must set edge_category="regulative", ig_deontic to must/can/cannot, and required_attributes
   to include deontic, activation_condition, execution_constraint, or_else_text, and source_text.
   Do not use condition_text in new ontology schemas; it is a deprecated legacy field that the builder only accepts
   for backward compatibility.
7. Constitutive definition edge types must set edge_category="constitutive", ig_deontic=null, and required_attributes
   to include source_text. Use them only for explicit definition, meaning, counts-as, or qualification statements.
8. source_targets must reference entity type names defined in entity_types.
9. Separate regulative rules from structural facts:
   - MUST_SUBMIT means the actor is institutionally required to submit something.
   - SUBMITS means the submission relation exists as part of the process.
   Both may appear when both the rule and process fact are supported.
10. Include eligibility and exclusion relationships when supported, such as ELIGIBLE_FOR, INELIGIBLE_FOR, REQUIRES,
   EXCLUDES, SELECTS, APPROVES, FUNDS, REPORTS_TO, or EVALUATES.
11. Do not infer a prohibition from a permission statement. Only create CANNOT_* when the source clearly states a
   restriction, exclusion, prohibition, ineligibility, or impossibility.
12. Each edge type should help explain policy impact simulation: who can act, who must act, who receives resources,
   who bears costs, who evaluates whom, what information is used, and which institutional conditions shape outcomes.
13. All edge type names must be English UPPER_SNAKE_CASE.
14. Regulative edge source_targets must preserve IG role scope: the source type is the actor that holds the obligation,
    permission, or prohibition. Beneficiary eligibility edges must not be typed as evaluator or organization constraints.
15. Resource allocation, funding decision, evaluation, approval, selection, and monitoring edges must use the actor with
    institutional authority as the source. Do not make a beneficiary the source of ALLOCATES, FUNDS, APPROVES, SELECTS,
    EVALUATES, or MONITORS unless the document explicitly gives that authority to the beneficiary.
"""


MAX_TEXT_LENGTH = 50000

_IAD_CATEGORIES = {
    "Participants",
    "Positions",
    "Actions",
    "Information",
    "Costs_and_Benefits",
}

_NODE_FAMILIES = {
    "ActorParticipant",
    "PositionRole",
    "EligibilityCondition",
    "RestrictionConstraint",
    "PolicyActionProcess",
    "InformationEvidence",
    "ResourceBenefitCost",
    "AuthorityControl",
    "TemporalMilestone",
    "EvaluationMetricOutcome",
}

_IAD_RULE_TYPES = {
    "position",
    "boundary",
    "choice",
    "aggregation",
    "information",
    "payoff",
    "scope",
}

# node_family -> iad_rule_type fallback comes from graph.normalizer.DEFAULT_IAD_RULE_BY_FAMILY,
# shared with validators and the local KG builder so all three modules agree on defaults.

_EDGE_FAMILIES = {
    "EligibilityExclusion",
    "ProcessSequence",
    "AuthorityGovernance",
    "ResourceIncentive",
    "InformationEvidence",
    "PermissionConstraint",
    "EvaluationSelection",
    "OutcomeCausalMechanism",
    "DefinitionQualification",
}

_DEFAULT_EDGE_TYPE_SPECS = [
    ("ELIGIBLE_FOR", "EligibilityExclusion", "structural", None, "Actor or role is eligible for a policy opportunity."),
    ("INELIGIBLE_FOR", "EligibilityExclusion", "regulative", "cannot", "Actor or role is excluded from a policy opportunity."),
    ("REQUIRES", "EligibilityExclusion", "regulative", "must", "Policy role or action requires a condition, document, or resource."),
    ("EXCLUDES", "EligibilityExclusion", "regulative", "cannot", "A condition or restriction excludes an actor, action, or benefit."),
    ("APPLIES_TO", "ProcessSequence", "structural", None, "Actor applies to a program, process, or opportunity."),
    ("SUBMITS", "InformationEvidence", "structural", None, "Actor submits information, evidence, proposal, or report."),
    ("REQUIRES_DOCUMENT", "InformationEvidence", "regulative", "must", "Actor or process requires a specific document or evidence."),
    ("USES_CRITERIA", "InformationEvidence", "structural", None, "Actor or process uses evaluation or eligibility criteria."),
    ("PRECEDES", "ProcessSequence", "structural", None, "One process phase precedes another phase."),
    ("TRIGGERS", "ProcessSequence", "structural", None, "One condition or event triggers a later process or decision."),
    ("NOTIFIES", "ProcessSequence", "structural", None, "Actor or authority notifies another actor of a decision or result."),
    ("IMPLEMENTS", "AuthorityGovernance", "structural", None, "Actor implements a policy process or action."),
    ("MANAGES", "AuthorityGovernance", "structural", None, "Actor manages a program, process, resource, or rule."),
    ("DECIDES", "AuthorityGovernance", "structural", None, "Actor or body makes a policy decision."),
    ("REQUIRES_APPROVAL", "AuthorityGovernance", "regulative", "must", "Action or change requires prior approval."),
    ("FUNDS", "ResourceIncentive", "structural", None, "Actor or program funds a person, organization, action, or resource."),
    ("RECEIVES", "ResourceIncentive", "structural", None, "Actor receives a resource, benefit, or support."),
    ("ALLOCATES", "ResourceIncentive", "structural", None, "Actor allocates funding, quota, or resources."),
    ("COVERS_COST", "ResourceIncentive", "structural", None, "Resource covers a specified cost category."),
    ("PROVIDES_SUPPORT", "ResourceIncentive", "structural", None, "Program or actor provides non-monetary support."),
    ("DEFINED_AS", "DefinitionQualification", "constitutive", None, "Term or role is explicitly defined as an institutional category."),
    ("CONSTITUTES", "DefinitionQualification", "constitutive", None, "Condition, status, or object constitutes another institutional category."),
    ("COUNTS_AS", "DefinitionQualification", "constitutive", None, "Condition, role, or status counts as another institutional status."),
    ("QUALIFIES_AS", "DefinitionQualification", "constitutive", None, "Condition or actor qualifies as an institutional role or status."),
    ("CAN_APPLY", "PermissionConstraint", "regulative", "can", "Actor is allowed to apply."),
    ("CAN_CHOOSE", "PermissionConstraint", "regulative", "can", "Actor is allowed to choose an option."),
    ("CAN_MOVE", "PermissionConstraint", "regulative", "can", "Actor is allowed to move or change location/institution."),
    ("CAN_CHANGE", "PermissionConstraint", "regulative", "can", "Actor is allowed to change a plan, topic, or institution."),
    ("CANNOT_APPLY", "PermissionConstraint", "regulative", "cannot", "Actor is prohibited or unable to apply."),
    ("CANNOT_CONDUCT", "PermissionConstraint", "regulative", "cannot", "Actor is prohibited from conducting an activity."),
    ("LIMITED_TO", "PermissionConstraint", "regulative", "must", "Action, resource, or permission is limited to a condition or frequency."),
    ("EVALUATES", "EvaluationSelection", "structural", None, "Actor evaluates an application, plan, report, or performance."),
    ("SELECTS", "EvaluationSelection", "structural", None, "Actor selects an application, actor, or project."),
    ("APPROVES", "EvaluationSelection", "structural", None, "Actor approves an application, change, or decision."),
    ("MONITORS", "EvaluationSelection", "structural", None, "Actor monitors implementation, compliance, or outcomes."),
    ("TARGETS", "OutcomeCausalMechanism", "structural", None, "Policy targets a population, issue, or outcome."),
    ("AIMS_TO_IMPROVE", "OutcomeCausalMechanism", "structural", None, "Policy action aims to improve an outcome."),
    ("MEASURES", "OutcomeCausalMechanism", "structural", None, "Metric measures an output, outcome, or impact."),
]


def generate_ontology(document_texts: list[str], simulation_requirement: str, config: dict) -> dict:
    """
    Generate an IAD/IG-aware ontology definition from policy document texts.
    """
    user_message = _build_user_message(document_texts, simulation_requirement)
    ontology_config = {
        **config,
        "llm": {**config["llm"], "temperature": 0.3},
    }
    messages = [
        {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    raw = chat_completion(messages, ontology_config)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Ontology LLM returned invalid JSON: {e}\nRaw response:\n{raw[:500]}")

    return _validate_and_process(result)


def _build_user_message(document_texts: list[str], simulation_requirement: str) -> str:
    combined_text = "\n\n---\n\n".join(document_texts)
    original_length = len(combined_text)

    if len(combined_text) > MAX_TEXT_LENGTH:
        combined_text = combined_text[:MAX_TEXT_LENGTH]
        combined_text += (
            f"\n\n...(original text was {original_length} characters; "
            f"truncated to first {MAX_TEXT_LENGTH} characters for ontology analysis)..."
        )

    return f"""## Simulation Requirement

{simulation_requirement}

## Document Content

{combined_text}

## HARD OUTPUT BUDGET (enforced — exceeding these caps invalidates the response)

- entity_types: AT MOST **10** items. Pick the most policy-essential types covering: (a) the key actor roles, (b) at least one Position role, (c) resource/funding, (d) eligibility/exclusion criteria, (e) evaluation/process, (f) outcome/report. If more than 10 candidates exist, MERGE near-duplicates and DROP peripheral types. Do NOT pad.
- edge_types: AT MOST **12** items. Cover one edge per IAD/IG family that the policy actually uses: EligibilityExclusion, ProcessSequence, AuthorityGovernance, ResourceIncentive, EvaluationSelection, InformationEvidence, DefinitionQualification. Prefer the most general edge per family (e.g. REQUIRES over a chain of variants) and DROP edges that won't repeatedly appear in the KG.

Return JSON with:
- entity_types: list of {{name, description, iad_category, node_family, is_actor, persona_eligible, attributes, examples, rationale}} — AT MOST 10
- edge_types: list of {{name, description, edge_category, edge_family, ig_deontic, source_targets, attributes, required_attributes, rationale}} — AT MOST 12
  Regulative edge required_attributes must use activation_condition and execution_constraint, not condition_text.
  Constitutive edge types should capture explicit definition, counts-as, and qualification clauses with ig_deontic=null.
- analysis_summary: brief summary of the policy ecosystem

Remember: all ontology output text must be English. Stakeholder actors remain central, but IAD/IG rationale must explain why each node and edge matters. Persona-eligible actor types should include agent_modality and role_family attributes when supported.
"""


def _validate_and_process(result: dict) -> dict:
    if "entity_types" not in result:
        result["entity_types"] = []
    if "edge_types" not in result:
        result["edge_types"] = []
    if "analysis_summary" not in result:
        result["analysis_summary"] = ""

    for entity in result["entity_types"]:
        if "attributes" not in entity or not isinstance(entity["attributes"], list):
            entity["attributes"] = []
        if "examples" not in entity or not isinstance(entity["examples"], list):
            entity["examples"] = []
        if len(entity.get("description", "")) > 100:
            entity["description"] = entity["description"][:97] + "..."

        category = entity.get("iad_category") or _infer_iad_category(entity)
        if category not in _IAD_CATEGORIES:
            category = _infer_iad_category(entity)
        category = _correct_iad_category(entity, category)
        entity["iad_category"] = category
        node_family = entity.get("node_family") or _infer_node_family(entity, category)
        if node_family not in _NODE_FAMILIES:
            node_family = _infer_node_family(entity, category)
        node_family = _correct_node_family(entity, node_family, category)
        entity["node_family"] = node_family
        rule_type = entity.get("iad_rule_type")
        if rule_type not in _IAD_RULE_TYPES:
            rule_type = DEFAULT_IAD_RULE_BY_FAMILY.get(node_family)
        entity["iad_rule_type"] = rule_type
        entity["is_actor"] = _as_bool(
            entity.get("is_actor"),
            default=category in {"Participants", "Positions"},
        )
        entity["persona_eligible"] = _as_bool(
            entity.get("persona_eligible"),
            default=entity["is_actor"] and category in {"Participants", "Positions"},
        )
        if _looks_like_policy_program(entity):
            entity["iad_category"] = (
                "Actions" if entity.get("node_family") == "PolicyActionProcess" else "Costs_and_Benefits"
            )
            if entity.get("node_family") in {"ActorParticipant", "PositionRole", ""}:
                entity["node_family"] = "ResourceBenefitCost"
            entity["is_actor"] = False
            entity["persona_eligible"] = False
        entity.setdefault("rationale", f"IAD {category}: explains stakeholder behavior or constraints.")

    for edge in result["edge_types"]:
        if "source_targets" not in edge or not isinstance(edge["source_targets"], list):
            edge["source_targets"] = []
        if "attributes" not in edge or not isinstance(edge["attributes"], list):
            edge["attributes"] = []
        edge["attributes"] = _normalize_edge_attributes(edge["attributes"])
        if len(edge.get("description", "")) > 100:
            edge["description"] = edge["description"][:97] + "..."

        edge_category = edge.get("edge_category") or _infer_edge_category(edge)
        if edge_category not in {"structural", "regulative", "constitutive"}:
            edge_category = _infer_edge_category(edge)
        edge["edge_category"] = edge_category
        edge_family = edge.get("edge_family") or _infer_edge_family(edge)
        if edge_family not in _EDGE_FAMILIES:
            edge_family = _infer_edge_family(edge)
        edge["edge_family"] = edge_family
        edge["ig_deontic"] = _infer_deontic(edge) if edge_category == "regulative" else None
        edge["required_attributes"] = _normalize_required_attributes(
            edge.get("required_attributes"),
            edge_category,
        )
        edge.setdefault(
            "rationale",
            _edge_rationale_default(edge_category),
        )

    _ensure_iad_coverage(result)
    _ensure_edge_family_coverage(result)

    if len(result["entity_types"]) > 24:
        result["entity_types"] = result["entity_types"][:24]
    if len(result["edge_types"]) > 34:
        result["edge_types"] = result["edge_types"][:34]

    return result


_REGULATIVE_REQUIRED_ATTRIBUTES = [
    "deontic",
    "activation_condition",
    "execution_constraint",
    "or_else_text",
    "source_text",
]


def _normalize_required_attributes(value: Any, edge_category: str) -> list[str]:
    existing = value if isinstance(value, list) else []
    required = _REGULATIVE_REQUIRED_ATTRIBUTES if edge_category == "regulative" else ["source_text"]
    normalized: list[str] = list(required)
    for item in existing:
        name = str(item or "").strip()
        if not name:
            continue
        if name == "condition_text":
            continue
        if name not in normalized:
            normalized.append(name)
    return normalized


def _normalize_edge_attributes(attributes: list[Any]) -> list[Any]:
    normalized: list[Any] = []
    inserted_condition_split = False
    for attribute in attributes:
        if not isinstance(attribute, dict):
            normalized.append(attribute)
            continue
        if attribute.get("name") != "condition_text":
            normalized.append(attribute)
            continue
        if not inserted_condition_split:
            normalized.extend([
                {
                    "name": "activation_condition",
                    "type": "text",
                    "description": "IG 2.0 Cac: when the rule applies or who is in scope.",
                },
                {
                    "name": "execution_constraint",
                    "type": "text",
                    "description": "IG 2.0 Cex: how the action must be performed once active.",
                },
            ])
            inserted_condition_split = True
    return normalized


def _edge_rationale_default(edge_category: str) -> str:
    if edge_category == "regulative":
        return "IG deontic rule."
    if edge_category == "constitutive":
        return "IG constitutive definition or qualification."
    return "Structural policy relationship."


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def _infer_iad_category(entity: dict) -> str:
    name_text = str(entity.get("name", "")).lower()
    text = f"{entity.get('name', '')} {entity.get('description', '')}".lower()
    if (
        any(k in name_text for k in ["program", "project", "scheme", "fellowship", "network"])
        and not any(k in name_text for k in ["officer", "manager", "agency", "ministry", "foundation", "institution"])
    ):
        return "Costs_and_Benefits"
    if any(k in text for k in ["proposal", "report", "criteria", "plan", "document", "information", "evidence"]):
        return "Information"
    if any(k in text for k in ["investigator", "panelist", "mentor", "chair", "manager", "reviewer", "officer"]):
        return "Positions"
    if any(k in text for k in ["ministry", "agency", "foundation", "institution", "university", "institute", "panel", "committee"]):
        return "Participants"
    if any(k in text for k in ["application", "selection", "evaluation", "execution", "review", "submission", "monitoring"]):
        return "Actions"
    if any(k in text for k in ["fund", "budget", "cost", "benefit", "grant", "subsidy", "penalty", "sanction"]):
        return "Costs_and_Benefits"
    if any(k in text for k in ["support", "allowance", "network", "mentorship", "program"]):
        return "Costs_and_Benefits"
    return "Participants"


def _infer_node_family(entity: dict, category: str | None = None) -> str:
    text = f"{entity.get('name', '')} {entity.get('description', '')} {' '.join(entity.get('examples', []))}".lower()
    if any(k in text for k in ["eligible", "eligibility", "qualification", "degree", "age", "nationality"]):
        return "EligibilityCondition"
    if any(k in text for k in ["exclude", "ineligible", "prohibit", "restriction", "limit", "not allowed", "cannot"]):
        return "RestrictionConstraint"
    if any(k in text for k in ["deadline", "year", "period", "phase", "duration", "milestone", "frequency"]):
        return "TemporalMilestone"
    if any(k in text for k in ["metric", "outcome", "performance", "criteria", "score", "ratio"]):
        return "EvaluationMetricOutcome"
    if any(k in text for k in ["proposal", "report", "plan", "document", "evidence", "notice", "record"]):
        return "InformationEvidence"
    if any(k in text for k in ["fund", "grant", "salary", "allowance", "cost", "benefit", "support", "sanction"]):
        return "ResourceBenefitCost"
    if any(k in text for k in ["application", "selection", "evaluation", "review", "submission", "execution", "monitoring"]):
        return "PolicyActionProcess"
    if any(k in text for k in ["ministry", "agency", "committee", "panel", "foundation", "authority", "approval", "office"]):
        return "AuthorityControl"
    if category == "Positions":
        return "PositionRole"
    if category == "Participants":
        return "ActorParticipant"
    if category == "Actions":
        return "PolicyActionProcess"
    if category == "Costs_and_Benefits":
        return "ResourceBenefitCost"
    return "InformationEvidence"


def _correct_iad_category(entity: dict, category: str) -> str:
    """Correct common ontology category confusions while preserving valid actor categories."""
    inferred = _infer_iad_category(entity)
    if _looks_like_actor_type(entity) and category in {"Costs_and_Benefits", "Information", "Actions"}:
        return "Participants"
    if inferred in {"Information", "Costs_and_Benefits", "Actions"} and category != inferred:
        return inferred
    if inferred == "Participants" and category == "Costs_and_Benefits":
        return inferred
    return category


def _correct_node_family(entity: dict, node_family: str, category: str) -> str:
    if category == "Participants" and _looks_like_actor_type(entity):
        return "ActorParticipant"
    if category == "Positions":
        return "PositionRole"
    return node_family


def _looks_like_actor_type(entity: dict) -> bool:
    attr_text = " ".join(
        str(item.get("name", "")) + " " + str(item.get("description", ""))
        for item in entity.get("attributes", [])
        if isinstance(item, dict)
    )
    text = f"{entity.get('name', '')} {entity.get('description', '')} {' '.join(entity.get('examples', []))} {attr_text}".lower()
    return any(
        marker in text
        for marker in (
            "researcher",
            "applicant",
            "candidate",
            "student",
            "mentor",
            "reviewer",
            "panel",
            "committee",
            "ministry",
            "agency",
            "foundation",
            "institution",
            "university",
            "officer",
            "administrator",
            "funder",
            "funding organization",
        )
    )


def _looks_like_policy_program(entity: dict) -> bool:
    name_text = str(entity.get("name", "")).lower()
    text = f"{entity.get('name', '')} {entity.get('description', '')} {' '.join(entity.get('examples', []))}".lower()
    program_markers = ["program", "project", "scheme", "fellowship", "network"]
    actor_markers = ["researcher", "student", "professor", "investigator", "panelist", "reviewer", "ministry", "agency", "officer"]
    if (
        any(marker in name_text for marker in program_markers)
        and not any(marker in name_text for marker in ["officer", "agency", "ministry", "foundation", "institution"])
    ):
        return True
    return any(marker in text for marker in program_markers) and not any(marker in text for marker in actor_markers)


def _infer_edge_category(edge: dict) -> str:
    name = edge.get("name", "").upper()
    if edge.get("edge_family") == "DefinitionQualification" or name in {
        "DEFINED_AS",
        "CONSTITUTES",
        "COUNTS_AS",
        "QUALIFIES_AS",
        "CLASSIFIED_AS",
        "REFERS_TO",
    }:
        return "constitutive"
    if name.startswith(("MUST_", "CAN_", "CANNOT_")):
        return "regulative"
    if edge.get("ig_deontic") in {"must", "can", "cannot"}:
        return "regulative"
    return "structural"


def _infer_edge_family(edge: dict) -> str:
    name = edge.get("name", "").upper()
    text = f"{name} {edge.get('description', '')}".lower()
    if any(k in name for k in ["DEFINED", "CONSTITUTES", "COUNTS_AS", "QUALIFIES", "CLASSIFIED", "REFERS_TO"]):
        return "DefinitionQualification"
    if any(k in text for k in ["defined as", "means", "refers to", "counts as", "qualifies as", "constitutes"]):
        return "DefinitionQualification"
    if any(k in name for k in ["ELIGIBLE", "INELIGIBLE", "REQUIRES", "EXCLUDES"]):
        return "EligibilityExclusion"
    if any(k in name for k in ["APPLIES", "SUBMITS", "PRECEDES", "TRIGGERS", "NOTIFIES", "REVIEWS"]):
        return "ProcessSequence"
    if any(k in name for k in ["IMPLEMENTS", "MANAGES", "DECIDES", "REPORTS_TO", "APPROVAL"]):
        return "AuthorityGovernance"
    if any(k in name for k in ["FUNDS", "RECEIVES", "ALLOCATES", "COVERS", "SUPPORT", "WITHHOLDS"]):
        return "ResourceIncentive"
    if any(k in name for k in ["INFORMATION", "DOCUMENT", "REPORT", "CRITERIA", "EVIDENCE"]):
        return "InformationEvidence"
    if name.startswith(("CAN_", "CANNOT_")) or any(k in name for k in ["LIMITED", "ALLOWED", "PERMITTED"]):
        return "PermissionConstraint"
    if any(k in name for k in ["EVALUATES", "SELECTS", "APPROVES", "SCORES", "MONITORS"]):
        return "EvaluationSelection"
    if any(k in name for k in ["TARGETS", "AIMS", "MEASURES", "EXPECTED", "OUTCOME", "IMPROVE"]):
        return "OutcomeCausalMechanism"
    if any(k in text for k in ["eligibility", "exclusion", "requirement"]):
        return "EligibilityExclusion"
    if any(k in text for k in ["funding", "resource", "cost", "benefit", "support"]):
        return "ResourceIncentive"
    if any(k in text for k in ["evaluate", "selection", "approval"]):
        return "EvaluationSelection"
    return "ProcessSequence"


def _infer_deontic(edge: dict) -> str | None:
    value = edge.get("ig_deontic")
    if value in {"must", "can", "cannot"}:
        return value
    name = edge.get("name", "").upper()
    if name.startswith("MUST_"):
        return "must"
    if name.startswith("CAN_"):
        return "can"
    if name.startswith("CANNOT_"):
        return "cannot"
    return None


def _ensure_iad_coverage(result: dict) -> None:
    present = {e.get("iad_category") for e in result["entity_types"]}
    existing_names = {e.get("name") for e in result["entity_types"]}
    templates = {
        "Participants": ("PolicyStakeholder", "Actor affected by or implementing the policy.", "ActorParticipant", True, True),
        "Positions": ("PolicyRole", "Official role slot in the policy process.", "PositionRole", True, True),
        "Actions": ("PolicyAction", "Policy process phase or required activity.", "PolicyActionProcess", False, False),
        "Information": ("PolicyInformation", "Document, criterion, or evidence used by actors.", "InformationEvidence", False, False),
        "Costs_and_Benefits": ("PolicyResource", "Resource, reward, cost, or sanction.", "ResourceBenefitCost", False, False),
    }
    for category, (name, description, node_family, is_actor, persona_eligible) in templates.items():
        if category in present or name in existing_names:
            continue
        result["entity_types"].append({
            "name": name,
            "description": description,
            "iad_category": category,
            "node_family": node_family,
            "iad_rule_type": DEFAULT_IAD_RULE_BY_FAMILY.get(node_family),
            "is_actor": is_actor,
            "persona_eligible": persona_eligible,
            "attributes": [],
            "examples": [],
            "rationale": f"Auto-recovered IAD {category} coverage because the LLM output omitted it.",
        })


def _ensure_edge_family_coverage(result: dict) -> None:
    existing_names = {edge.get("name") for edge in result["edge_types"]}
    existing_families = {edge.get("edge_family") for edge in result["edge_types"]}
    if len(result["edge_types"]) >= 28:
        return

    entity_types = result.get("entity_types", [])
    actors = [
        item["name"] for item in entity_types
        if item.get("is_actor") or item.get("iad_category") in {"Participants", "Positions"}
    ]
    actions = [item["name"] for item in entity_types if item.get("iad_category") == "Actions"]
    information = [item["name"] for item in entity_types if item.get("iad_category") == "Information"]
    resources = [item["name"] for item in entity_types if item.get("iad_category") == "Costs_and_Benefits"]
    positions = [item["name"] for item in entity_types if item.get("iad_category") == "Positions"]
    targets_by_family = {
        "EligibilityExclusion": information + actions + resources + positions,
        "ProcessSequence": actions + information,
        "AuthorityGovernance": actions + information + resources + actors,
        "ResourceIncentive": resources + actors,
        "InformationEvidence": information,
        "PermissionConstraint": actions + information + resources + actors,
        "EvaluationSelection": information + actions + actors,
        "OutcomeCausalMechanism": information + actions + resources + actors,
        "DefinitionQualification": positions + information + actions + resources + actors,
    }

    for name, family, category, deontic, description in _DEFAULT_EDGE_TYPE_SPECS:
        if len(result["edge_types"]) >= 28:
            break
        if name in existing_names:
            continue
        if family in existing_families:
            continue
        sources = actors or [item["name"] for item in entity_types[:1]]
        targets = targets_by_family.get(family) or [item["name"] for item in entity_types[:1]]
        source_targets = _source_target_pairs(sources, targets)
        if not source_targets:
            continue
        result["edge_types"].append({
            "name": name,
            "description": description,
            "edge_category": category,
            "edge_family": family,
            "ig_deontic": deontic,
            "source_targets": source_targets,
            "attributes": [],
            "required_attributes": (
                ["deontic", "activation_condition", "execution_constraint", "or_else_text", "source_text"]
                if category == "regulative"
                else ["source_text"]
            ),
            "rationale": f"Auto-added to preserve {family} coverage for IAD/IG policy simulation.",
        })
        existing_names.add(name)
        existing_families.add(family)


def _source_target_pairs(sources: list[str], targets: list[str], limit: int = 16) -> list[dict[str, str]]:
    pairs = []
    for source in sources:
        for target in targets:
            if source == target:
                continue
            pairs.append({"source": source, "target": target})
            if len(pairs) >= limit:
                return pairs
    return pairs
