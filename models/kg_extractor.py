# --- Local KG Extraction ---

KG_NODE_EXTRACTOR_SYSTEM = "You are an expert policy knowledge graph extractor for IAD/IG-based policy impact simulation. Respond only in valid JSON with no markdown formatting or code blocks. All generated node labels, summaries, attributes, warnings, and source_text evidence must be in English."

KG_NODE_EXTRACTOR_USER = """Extract policy KG node instances from the document chunk using the provided IAD/IG ontology.

Rules:
- Use only entity types defined in the ontology.
- LANGUAGE REQUIREMENT: all generated node names, summaries, attributes, warnings, and any other non-evidence strings
  must be in English.
- SOURCE EVIDENCE REQUIREMENT: source_text must be a short English evidence sentence translated or summarized from the
  smallest source fragment that supports the node. It must stay faithful to the document and must not add facts that are
  absent from the chunk.
- Extract both stakeholder actors and non-actor policy objects when they explain actor constraints, information, actions, costs, or benefits.
- Every authoritative node should be useful for at least one relation that changes simulation behavior. Do not extract
  isolated glossary terms, generic background concepts, or repeated labels that are unlikely to participate in an edge.
- Treat persona-eligible actor or position nodes as simulation actor anchors, not as individual biographies.
  A simulation actor anchor may be an individual role, stakeholder group, organization, or official position.
- For every persona-eligible node, set attributes.agent_modality to one of:
  individual, stakeholder_group_member, organization, organization_representative, official_position_holder, or committee_member.
- For every persona-eligible node, set attributes.role_family to one of:
  direct_beneficiary, evaluator, host_organization, policy_administrator, mentor, affected_actor, or pipeline_actor.
  This role_family controls which policy constraints are allowed to shape later personas.
- Keep collective bodies and their human speakers distinct. A panel, committee, ministry, foundation, or institution can be
  a KG actor anchor, but the later persona must be a member, representative, or role-holder unless the actor is kept as an
  institutional profile without Nemotron.
- Extract a role node only when the text supports that the role participates in, implements, evaluates, funds, receives,
  is constrained by, or is affected by the policy.
- Make persona_eligible true only for explicit policy actors, roles, organizations, positions, committees, or stakeholder
  groups that can reasonably receive at least one supported KG relation. Generic ontology placeholders are not persona anchors.
- Do not extract only the main beneficiary. Also extract supported applicants, candidates, pipeline groups, host organizations,
  administrators, evaluators, funders, reviewers, mentors, intermediaries, institutional governance bodies, and affected
  groups when the chunk supports them.
- If a role is near-eligible or adjacent to the target population, keep it as a simulation actor anchor when it could shape
  policy impact in Stage 2. Put the eligibility uncertainty in attributes rather than dropping the actor.
- Use node_family to guide coverage. Extract nodes from these families when supported:
  ActorParticipant, PositionRole, EligibilityCondition, RestrictionConstraint, PolicyActionProcess, InformationEvidence,
  ResourceBenefitCost, AuthorityControl, TemporalMilestone, and EvaluationMetricOutcome.
- For each extracted node, copy iad_rule_type directly from the matching ontology entity type. iad_rule_type is one of:
  position, boundary, choice, aggregation, information, payoff, scope. Override the ontology default ONLY when the
  source text makes the node govern a different IAD rule than its ontology type would suggest (e.g., a TemporalMilestone
  instance can be a boundary rule when it appears in eligibility windows but a scope rule when it appears in outcome
  timeframes). When in doubt, copy the ontology iad_rule_type unchanged.
- For Actions, extract process phases, not duplicate mentions.
- For Information, extract documents, criteria, plans, reports, eligibility evidence, notices, records, metrics, or
  decision signals. Proposals, reports, and plans are Information even when they influence funding.
- For Costs_and_Benefits, split distinct resources, rewards, costs, or sanctions when the text distinguishes them.
- Do not classify support programs, grants, reports, proposals, notifications, compliance requirements, or funding limits
  as ActorParticipant or EvaluationPanel. They are non-actor policy objects unless the document names a real organization,
  committee, or office that performs an action.
- Extract conditions, restrictions, process steps, dates/periods, decision authorities, evaluation criteria, resource
  components, and outcome metrics as separate node instances when they can affect simulation behavior.
- Keep names concise, English, and grounded in the source text. Do not simply reuse the ontology type as the instance name
  when the document gives a more specific actor, object, process, or rule label.
- Do not instantiate the policy/program title itself as a persona-eligible actor. If an ontology actor type represents
  fellowship recipients, selected researchers, applicants, or beneficiaries, the node instance must name the actor role
  (for example "Postdoctoral Researchers", "Selected Fellows", or "Applicants"), not the fellowship/program name.
- When a generic node and a more specific node would represent the same policy object, extract the more specific node only.
  Example: prefer "Basic Research Project Promotion Committee" over "ApprovalAuthority" when the committee is named.
- When the text uses a broad population label but defines a narrower statutory or operational role, use the narrower
  policy role as the direct-beneficiary actor. Keep the broad phrase in summary or attributes, not as a separate generic
  persona anchor.
- Fill attributes only with grounded information that helps later policy simulation or persona synthesis. Useful attributes
  include eligibility_criteria, exclusion_criteria, role_scope, decision_authority, support_period, funding_amount,
  reporting_duty, evaluation_criteria, target_quota, implementing_body, affected_population, agent_modality, and role_family.
- Do not create person-like nodes for generic examples unless the document identifies a real individual. Use role or
  organization nodes instead.
- Include source_text as a short English evidence sentence grounded in the document chunk.
- Do not invent missing eligibility criteria, actor motives, or demographic details.
- Target a compact, edge-ready node set. For dense chunks, 5-15 high-value node instances are usually better than many
  weakly connected concepts. Prefer preserving a specific institutional condition as a node over burying it only in attributes.

Return JSON with this exact structure:
{{
  "nodes": [
    {{
      "name": "node instance name",
      "type": "OntologyEntityTypeName",
      "iad_category": "Participants | Positions | Actions | Information | Costs_and_Benefits",
      "node_family": "ActorParticipant | PositionRole | EligibilityCondition | RestrictionConstraint | PolicyActionProcess | InformationEvidence | ResourceBenefitCost | AuthorityControl | TemporalMilestone | EvaluationMetricOutcome",
      "iad_rule_type": "position | boundary | choice | aggregation | information | payoff | scope",
      "summary": "brief grounded summary",
      "attributes": {{}},
      "source_text": "supporting source text",
      "confidence": 0.0
    }}
  ],
  "warnings": []
}}

--- WORKED EXAMPLE (methodological only; types must come from your actual ontology) ---

Do not reuse example labels unless the current ontology and document support them. The example shows extraction method,
not a domain vocabulary to carry into unrelated policy PDFs.

Suppose the chunk contains:
"세종과학펠로우십 지원 대상자는 박사학위를 취득한 후 7년 이내인 자로서 비전임 연구원에 한한다.
연구재단은 매년 100명 내외의 펠로우를 선정하며, 선정된 펠로우는 5년간 매년 1.3억원의 연구비를 지원받는다.
선정평가위원회는 연구계획서를 평가하여 선정 결과를 결정한다."

A high-quality node extraction yields:
[
  {{
    "name": "Postdoctoral Researcher",
    "type": "PostdoctoralResearcher",
    "iad_category": "Participants",
    "node_family": "ActorParticipant",
    "iad_rule_type": "position",
    "summary": "Direct beneficiary of the fellowship; non-tenure-track researcher within 7 years of PhD.",
    "attributes": {{
      "agent_modality": "individual",
      "role_family": "direct_beneficiary",
      "eligibility_criteria": "PhD obtained within 7 years; non-tenure-track",
      "exclusion_criteria": "tenure-track faculty are excluded"
    }},
    "source_text": "Non-tenure-track researchers who obtained a PhD within seven years are eligible to apply for support.",
    "confidence": 0.95
  }},
  {{
    "name": "Research Foundation",
    "type": "ResearchFoundation",
    "iad_category": "Participants",
    "node_family": "ActorParticipant",
    "iad_rule_type": "aggregation",
    "summary": "Implementing organization that selects fellows and disburses the grant.",
    "attributes": {{
      "agent_modality": "organization",
      "role_family": "policy_administrator"
    }},
    "source_text": "The research foundation selects about 100 fellows each year.",
    "confidence": 0.9
  }},
  {{
    "name": "Selection Evaluation Panel",
    "type": "EvaluationPanel",
    "iad_category": "Participants",
    "node_family": "ActorParticipant",
    "iad_rule_type": "aggregation",
    "summary": "Panel that evaluates research proposals and decides selection outcomes.",
    "attributes": {{
      "agent_modality": "committee_member",
      "role_family": "evaluator"
    }},
    "source_text": "The selection evaluation panel evaluates research proposals and decides selection results.",
    "confidence": 0.9
  }},
  {{
    "name": "Annual Fellowship Quota",
    "type": "FundingLimit",
    "iad_category": "Costs_and_Benefits",
    "node_family": "ResourceBenefitCost",
    "iad_rule_type": "scope",
    "summary": "Approximately 100 fellows are selected annually.",
    "attributes": {{"target_quota": "about 100 fellows per year"}},
    "source_text": "About 100 fellows are selected each year.",
    "confidence": 0.85
  }},
  {{
    "name": "Annual Research Grant",
    "type": "ResearchGrant",
    "iad_category": "Costs_and_Benefits",
    "node_family": "ResourceBenefitCost",
    "iad_rule_type": "payoff",
    "summary": "130 million KRW per year for 5 years per selected fellow.",
    "attributes": {{
      "funding_amount": "130 million KRW per year",
      "support_period": "5 years"
    }},
    "source_text": "Selected fellows receive 130 million KRW in research funding each year for five years.",
    "confidence": 0.95
  }},
  {{
    "name": "Research Proposal",
    "type": "ResearchProposal",
    "iad_category": "Information",
    "node_family": "InformationEvidence",
    "iad_rule_type": "information",
    "summary": "Proposal document used as the input to selection evaluation.",
    "attributes": {{}},
    "source_text": "Research proposals are evaluated during the selection process.",
    "confidence": 0.85
  }}
]

What this example demonstrates:
- The program label "세종과학펠로우십" is NOT a persona-eligible actor; the beneficiary actor is "Postdoctoral Researcher".
- The non-tenure-track restriction is preserved as both an eligibility/exclusion attribute and English source_text evidence.
- The evaluation panel and the research foundation are kept distinct (different role_family).
- Each node has a single, scoped source_text sentence that faithfully translates or summarizes the supporting source fragment.
- Numerical values (quota, grant amount, support period) live in attributes, not invented elsewhere.
- iad_rule_type makes the analytical lens explicit:
    - PostdoctoralResearcher → position (defines who occupies the beneficiary role)
    - Research Foundation and Selection Evaluation Panel → aggregation (they hold decision authority over selection)
    - Annual Fellowship Quota → scope (caps the policy outcome at ~100 selections per year)
    - Annual Research Grant → payoff (the material benefit assigned to the position)
    - Research Proposal → information (information that flows from applicant to evaluator)

--- Simulation Requirement ---
{simulation_requirement}

--- Ontology ---
{ontology}

--- Document Chunk {chunk_id} ---
{chunk_text}"""

KG_EDGE_EXTRACTOR_SYSTEM = """You are an expert policy knowledge graph edge extractor using IAD (Ostrom) and IG 2.0 (Institutional Grammar 2.0) for policy impact simulation.

First classify each candidate institutional statement:
  regulative = an actor must/can/cannot do an action to an object.
  constitutive = a term, condition, action, object, or status is defined as, means, counts as, constitutes, or qualifies as another institutional category.
  structural/strategy = a process, resource flow, evaluation relation, information relation, or causal relation without deontic or definition function.

For regulative statements, decompose using IG 2.0 ABDICO syntax:
  A = Attribute (the actor who holds the rule — maps to edge.source)
  D = Deontic (must / can / cannot — maps to edge.deontic)
  I = aIm (the action verb — maps to edge.type)
  B = oBject (the entity acted upon — maps to edge.target)
  C = Conditions, split into TWO IG 2.0 sub-fields:
        Cac (Activation Condition) — when does the rule apply / who is in scope (eligibility, target group,
              triggering event, time window). Maps to edge.activation_condition.
        Cex (Execution Constraint) — how must the action be performed once the rule fires (manner, deadline
              within the rule, format, prerequisite approval, frequency limit). Maps to edge.execution_constraint.
  O = Or-else (sanction or consequence on non-compliance — maps to edge.or_else_text)

For constitutive statements, do not force ABDICO. Extract the defined term or qualifying condition as source, the institutional category/status as target, set edge_category="constitutive", deontic=null, and use a DefinitionQualification edge type when the ontology provides one.

Respond only in valid JSON with no markdown formatting or code blocks. All generated facts, conditions, attributes, warnings, and source_text evidence must be in English."""

KG_EDGE_EXTRACTOR_USER = """Extract policy KG edge instances from the document chunk using the provided ontology and global node pool.

OUTPUT BUDGET (HARD LIMIT):
- Emit AT MOST {max_edges_per_chunk} edges from this chunk. If more than {max_edges_per_chunk} edges are plausible, KEEP only those with the strongest direct textual support and DROP weaker/redundant ones. Do NOT silently truncate; pick fewer, well-evidenced edges over many speculative ones. Adjacent chunks (via overlap) will cover edges you skip near the chunk boundary.

Rules:
- Use only edge types defined in the ontology.
- source and target must be exact node ids from the Global Node Pool.
- LANGUAGE REQUIREMENT: all generated facts, activation_condition, execution_constraint, or_else_text, attributes,
  warnings, and any other non-evidence strings must be in English.
- SOURCE EVIDENCE REQUIREMENT: source_text must be a short English evidence sentence translated or summarized from the
  smallest source fragment that supports the edge. It must directly entail the edge type, source, target, deontic relation,
  and conditions when present. Do not add facts that are absent from the chunk.
- Extract structural edges for process order, eligibility, selection, resource flow, implementation, evaluation, submission,
  reporting, monitoring, and information use when supported by the chunk.
- Extract constitutive IG edges for explicit definition, meaning, counts-as, constitutes, or qualification clauses when
  the ontology provides a DefinitionQualification edge type such as DEFINED_AS, CONSTITUTES, COUNTS_AS, or QUALIFIES_AS.
- Prefer edges that connect specific, source-supported nodes. If the Global Node Pool contains both a generic ontology-label
  node and a specific document-named node for the same actor/object, use the specific node.
- Use edge_family to guide coverage:
  EligibilityExclusion, ProcessSequence, AuthorityGovernance, ResourceIncentive, InformationEvidence,
  PermissionConstraint, EvaluationSelection, OutcomeCausalMechanism, and DefinitionQualification.
- Extract multiple specific institutional relations when they change simulation behavior: eligibility, exclusion,
  document requirements, application permissions, restrictions, frequency limits, approval requirements, selection,
  notification, funding components, support services, milestones, and outcome targets.
- Do not create edges to keep a weak or isolated node alive. The edge must be independently supported by source_text and
  must make the source or target meaningful for policy simulation.
- Resource allocation, funding, evaluation, approval, selection, and monitoring edges must use the actor with institutional
  authority as the source. Do not make a beneficiary the source of ALLOCATES, FUNDS, APPROVES, SELECTS, EVALUATES, or
  MONITORS unless the source text explicitly gives that authority to the beneficiary.
- Extract regulative IG edges for every clear institutional rule.
- Extract constitutive IG edges for every clear definition clause that changes who or what counts in the policy.

- IG 2.0 STATEMENT TYPE CLASSIFICATION:
    1. Regulative: A must/can/cannot do I to B under C, with optional O. Use ABDICO and edge_category="regulative".
    2. Constitutive: X means Y, X is defined as Y, X counts as Y, X constitutes Y, or X qualifies as Y. Use
       edge_category="constitutive", deontic=null, and put definition details in fact/attributes/source_text.
    3. Structural/strategy: A factual process, flow, evaluation, implementation, or causal relation. Use
       edge_category="structural", deontic=null.

- IG 2.0 ABDICO DECOMPOSITION — apply this six-step procedure to every candidate institutional statement before
  emitting an edge. If any of A / I / B is unclear from the chunk, do NOT emit the edge.
    1. A — Attribute: identify the actor, role, organization, or position that holds the obligation, permission,
         or prohibition. This is edge.source. The source must be a node id from the Global Node Pool.
         Do not assign a rule to the target merely because the target is affected by the rule.
         Example: "박사학위 후 7년 이내인 자는 신청할 수 있다" → A = the postdoctoral researcher (the rule-holder),
         not the fellowship program.
    2. D — Deontic: choose exactly one of {{must, can, cannot}} for regulative edges, or null for structural/constitutive facts.
         - must: 의무 / 하여야 한다 / required / shall / must / has to / required to / mandatory.
         - can: 권한 / 할 수 있다 / 가능 / may / can / is allowed / is eligible / is permitted.
         - cannot: 불가 / 제외 / ~할 수 없다 / 금지 / cannot / must not / prohibited / restricted / ineligible /
                   excluded / not allowed.
         Soft normative phrasing such as "노력해야 한다" / "권장한다" / "should" should NOT be promoted to "must".
         Place such soft-norm statements as attributes on a structural edge instead.
    3. I — aIm: the action verb that maps to edge.type. Pick from the ontology's edge types.
         Use MUST_SUBMIT / MUST_REPORT / MUST_EVALUATE for required actions; CAN_APPLY / CAN_CHOOSE / CAN_MOVE
         for permitted actions; CANNOT_APPLY / CANNOT_CONDUCT / CANNOT_CHANGE for prohibited actions.
    4. B — oBject: identify the policy object the action is directed at — a program, document, resource, role,
         decision, or process. This is edge.target. Use the most specific node id available in the Global Node Pool.
    5. C — Conditions: SPLIT every conditional clause into Cac vs Cex.
         - activation_condition (Cac): WHEN does the rule fire / WHO is in scope?
             eligibility windows ("박사학위 취득 후 7년 이내"), qualifications ("박사학위 소지자"),
             exclusion criteria ("전임 연구원은 제외"), target population, triggering events ("신청 후").
             Cac determines whether the rule applies to a given persona at all.
         - execution_constraint (Cex): HOW must the action be performed once the rule applies?
             deadlines within the rule ("30일 이내에", "회계연도 종료 후 1개월 내"),
             procedural prerequisites ("위원장 승인 후", "사전 심의를 거쳐"),
             format/manner ("온라인으로", "서면으로"),
             frequency limits ("연 1회", "분기마다").
             Cex determines the compliance burden of executing the action.
         If a single clause is purely about scope/eligibility, put it ONLY in activation_condition and leave
         execution_constraint null. If a single clause is purely about how-to-perform, do the reverse.
         If a sentence contains both kinds of clauses, split them across the two fields. Use null when absent.
         Do NOT duplicate the same clause into both fields.
    6. O — Or-else: extract any explicit sanction or consequence into or_else_text — penalty, recovery of funds,
         disqualification, suspension, etc. Use null when no sanction is explicit. Do NOT invent sanctions.

- After ABDICO decomposition, do not transfer direct-beneficiary eligibility constraints to evaluators, host
  organizations, administrators, mentors, committees, or funders unless the source text explicitly scopes that
  same constraint to them. Cross-role transfer is the most common ABDICO mistake.
- Extract CANNOT_* only when the source text clearly contains prohibition, restriction, ineligibility, impossibility, or
  negative language equivalent to "not allowed", "prohibited", "restricted", "excluded", or "ineligible".
- Do not convert permission statements equivalent to "can", "may", "allowed", or "possible" into cannot.
- Separate rules from process facts:
  - Use MUST_SUBMIT when the actor is required to submit a document.
  - Use SUBMITS when the submission occurs as a structural process relation.
  - Use CAN_CHOOSE when an actor has institutional permission to choose.
  - Use RECEIVES or FUNDS for resource flows, not for obligations.
- Prefer edges that preserve policy-role consistency for later persona synthesis: eligibility, exclusion, resource access,
  duties, permissions, evaluation, reporting, and actor-object-action relations.
- Every edge must include source_text. source_text must directly entail the edge type, source, target, and deontic relation.
  If the quote only provides broad background but does not support the relation, do not extract the edge.
- Validation will drop candidate edges whose source_text does not directly support the relation. Dropped candidates are
  QA/audit material only and must not be treated as authoritative KG facts.
- For regulative edges, also include activation_condition (Cac), execution_constraint (Cex), and or_else_text.
  Use null when a given field is not explicit in the source text.
- Do not output condition_text. It is a deprecated legacy field; split conditions into activation_condition and
  execution_constraint instead.
- source_text must be a short English evidence sentence grounded in the document chunk.
- Do not invent edges unsupported by the chunk.
- For structural (non-regulative) edges, conditional details go into attributes; activation_condition and
  execution_constraint are intended primarily for regulative edges but may also be set on structural edges
  when the chunk describes a clear scope or procedural constraint on the structural relation.
- If a candidate edge would create a role/persona inconsistency not supported by the document, do not extract it.
- Target 10-30 edge instances per chunk when the chunk is information-dense. Do not collapse distinct conditions,
  restrictions, support items, or process steps into one generic edge.

Return JSON with this exact structure:
{{
  "edges": [
    {{
      "type": "EDGE_TYPE_NAME",
      "edge_category": "structural | regulative | constitutive",
      "edge_family": "EligibilityExclusion | ProcessSequence | AuthorityGovernance | ResourceIncentive | InformationEvidence | PermissionConstraint | EvaluationSelection | OutcomeCausalMechanism | DefinitionQualification",
      "source": "ExactSourceNodeId",
      "target": "ExactTargetNodeId",
      "fact": "brief factual relation",
      "deontic": "must | can | cannot | null",
      "activation_condition": "IG 2.0 Cac — when/to whom the rule applies, or null",
      "execution_constraint": "IG 2.0 Cex — how the action must be performed, or null",
      "or_else_text": "sanction/consequence text or null",
      "source_text": "supporting source text",
      "attributes": {{}},
      "confidence": 0.0
    }}
  ],
  "warnings": []
}}

--- WORKED EXAMPLE (methodological only; edge types must come from your actual ontology) ---

Do not reuse example edge labels unless the current ontology defines them. The example shows IG decomposition method,
not a domain vocabulary to carry into unrelated policy PDFs.

Suppose the chunk contains:
"세종과학펠로우십 지원 대상자는 박사학위를 취득한 후 7년 이내인 자로서 비전임 연구원에 한한다.
연구재단은 매년 100명 내외의 펠로우를 선정하며, 선정된 펠로우는 5년간 매년 1.3억원의 연구비를 지원받는다.
선정평가위원회는 연구계획서를 평가하여 선정 결과를 결정한다."

And the Global Node Pool contains node ids:
  PostdoctoralResearcher:Postdoctoral Researcher
  ResearchFoundation:Research Foundation
  EvaluationPanel:Selection Evaluation Panel
  FundingLimit:Annual Fellowship Quota
  ResearchGrant:Annual Research Grant
  ResearchProposal:Research Proposal
  FellowshipProgram:Sejong Science Fellowship

A high-quality edge extraction yields:
[
  {{
    "type": "CAN_APPLY",
    "edge_category": "regulative",
    "edge_family": "PermissionConstraint",
    "source": "PostdoctoralResearcher:Postdoctoral Researcher",
    "target": "FellowshipProgram:Sejong Science Fellowship",
    "fact": "Postdoctoral researchers within 7 years of PhD may apply to the fellowship.",
    "deontic": "can",
    "activation_condition": "PhD obtained within 7 years and applicant is a non-tenure-track researcher",
    "execution_constraint": null,
    "or_else_text": null,
    "source_text": "The support target is limited to non-tenure-track researchers who obtained a PhD within seven years.",
    "attributes": {{}},
    "confidence": 0.95
  }},
  {{
    "type": "CANNOT_APPLY",
    "edge_category": "regulative",
    "edge_family": "PermissionConstraint",
    "source": "PostdoctoralResearcher:Postdoctoral Researcher",
    "target": "FellowshipProgram:Sejong Science Fellowship",
    "fact": "Tenure-track faculty are excluded from applying.",
    "deontic": "cannot",
    "activation_condition": "applicant is tenure-track (전임 연구원)",
    "execution_constraint": null,
    "or_else_text": null,
    "source_text": "Tenure-track researchers are excluded from applying because support is limited to non-tenure-track researchers.",
    "attributes": {{}},
    "confidence": 0.9
  }},
  {{
    "type": "SELECTS",
    "edge_category": "structural",
    "edge_family": "EvaluationSelection",
    "source": "ResearchFoundation:Research Foundation",
    "target": "PostdoctoralResearcher:Postdoctoral Researcher",
    "fact": "The Research Foundation selects about 100 fellows per year.",
    "deontic": null,
    "activation_condition": null,
    "execution_constraint": "annual selection cycle",
    "or_else_text": null,
    "source_text": "The research foundation selects about 100 fellows each year.",
    "attributes": {{"annual_quota": "about 100"}},
    "confidence": 0.9
  }},
  {{
    "type": "EVALUATES",
    "edge_category": "structural",
    "edge_family": "EvaluationSelection",
    "source": "EvaluationPanel:Selection Evaluation Panel",
    "target": "ResearchProposal:Research Proposal",
    "fact": "The Selection Evaluation Panel evaluates research proposals.",
    "deontic": null,
    "activation_condition": null,
    "execution_constraint": null,
    "or_else_text": null,
    "source_text": "The selection evaluation panel evaluates research proposals.",
    "attributes": {{}},
    "confidence": 0.9
  }},
  {{
    "type": "FUNDS",
    "edge_category": "structural",
    "edge_family": "ResourceIncentive",
    "source": "ResearchFoundation:Research Foundation",
    "target": "ResearchGrant:Annual Research Grant",
    "fact": "The Foundation funds the annual research grant.",
    "deontic": null,
    "activation_condition": "fellow has been selected",
    "execution_constraint": "disbursement spans 5 years at 130M KRW per year",
    "or_else_text": null,
    "source_text": "Selected fellows receive 130 million KRW in research funding each year for five years.",
    "attributes": {{"funding_amount": "130M KRW/yr", "support_period": "5 years"}},
    "confidence": 0.85
  }},
  {{
    "type": "RECEIVES",
    "edge_category": "structural",
    "edge_family": "ResourceIncentive",
    "source": "PostdoctoralResearcher:Postdoctoral Researcher",
    "target": "ResearchGrant:Annual Research Grant",
    "fact": "Selected postdoctoral researchers receive the annual grant.",
    "deontic": null,
    "activation_condition": "researcher has been selected as a fellow",
    "execution_constraint": null,
    "or_else_text": null,
    "source_text": "Selected fellows receive annual research funding for five years.",
    "attributes": {{}},
    "confidence": 0.85
  }},
  {{
    "type": "LIMITED_TO",
    "edge_category": "regulative",
    "edge_family": "PermissionConstraint",
    "source": "ResearchFoundation:Research Foundation",
    "target": "FundingLimit:Annual Fellowship Quota",
    "fact": "The annual selection is limited to approximately 100 fellows.",
    "deontic": "must",
    "activation_condition": null,
    "execution_constraint": "annual cohort size cap of about 100 fellows",
    "or_else_text": null,
    "source_text": "About 100 fellows are selected each year.",
    "attributes": {{}},
    "confidence": 0.8
  }}
]

What this example demonstrates:
- The same chunk produces both a permission edge (CAN_APPLY) and a prohibition edge (CANNOT_APPLY): the eligibility window is a permission for those inside it, the non-tenure-track restriction is an explicit exclusion. They are NOT collapsed into a single edge.
- Authority is correctly assigned: ResearchFoundation is the source of SELECTS and FUNDS; PostdoctoralResearcher is the source of CAN_APPLY (the rule-holder of the permission). The beneficiary is never the source of SELECTS/FUNDS.
- The committee is the source of EVALUATES (an evaluator), not an applicant.
- Structural process edges (SELECTS, EVALUATES, FUNDS, RECEIVES) carry deontic=null. Only true regulative norms (CAN_APPLY, CANNOT_APPLY, LIMITED_TO) carry must/can/cannot.
- IG 2.0 Cac vs Cex split: CAN_APPLY's eligibility ("PhD within 7 years") is purely WHO is in scope → activation_condition. LIMITED_TO's "annual cohort size cap" is HOW the action must be performed → execution_constraint. RECEIVES is gated on prior selection (Cac) but has no procedural constraint (Cex=null).
- source_text for each edge is a concise English evidence sentence for THAT edge — not the whole paragraph.

--- Simulation Requirement ---
{simulation_requirement}

--- Ontology ---
{ontology}

--- Global Node Pool ---
{node_pool}

--- Document Chunk {chunk_id} ---
{chunk_text}"""
