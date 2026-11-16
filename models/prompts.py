# All LLM prompts used across the pipeline

# --- Economy Constructor ---

ECONOMY_CONSTRUCTOR_SYSTEM = "You are an expert AI policy analyst. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

ECONOMY_CONSTRUCTOR_USER = """Analyze the following background documents and extract an economic context for AI talent development policy in Korea.

Return a JSON object with exactly these fields:
- summary: overall summary of the current AI and higher education policy landscape (string)
- key_ai_trends: list of key AI trends identified from the documents (list of strings)
- policy_landscape: description of current government policy roadmap and direction (string)

Documents:
{pdf_text}"""


# --- Policy Reader ---

POLICY_READER_SYSTEM = "You are an expert policy analyst. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

POLICY_READER_USER = """Read the following policy document and extract its key information.

Return a JSON object with exactly these fields:
- policy_name: official name of the policy (string)
- year_of_plan: year the policy was established or published (string)
- policy_duration: duration or implementation period of the policy (string)
- policy_goal: main goals and objectives of the policy, including any explicit quantitative targets (e.g., number of selected candidates, number of fellows, target headcount) if stated in the document (string)
- policy_content: detailed description of policy contents and measures (string)
- budget: total budget amount if an explicit monetary value is stated in the document, otherwise null

Policy document:
{pdf_text}"""


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


# --- Entity to Role Profile (actor-level base for later persona synthesis) ---

ENTITY_TO_PERSONA_SYSTEM = "You are an expert in IAD/IG-based stakeholder actor modeling for policy impact simulation. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

ENTITY_TO_PERSONA_USER = """Given the following actor, stakeholder group, organization, or position entity extracted from a policy knowledge graph, generate a policy actor profile for policy impact simulation.

Important framing:
- LANGUAGE REQUIREMENT: all generated persona-profile fields must be in English, even if the source KG evidence or policy
  document was originally written in another language.
- This is NOT an individual biography.
- One KG entity can later be instantiated by many different simulation personas or agents.
- The entity may represent an individual role, a stakeholder group, an organization, or an official position.
- Your task is to describe the policy actor's role, policy-rule constraints, incentives, and expected policy stance.
- Downstream persona synthesis may combine this actor profile with external demographic/persona seeds, but this actor profile
  must remain grounded in the KG and policy document.
- Classify the entity conceptually as one of these role families while writing objectives and constraints:
  direct_beneficiary, evaluator, host_organization, policy_administrator, mentor, affected_actor, or pipeline_actor.
  This classification is methodological, not a demographic claim.

Entity name: {entity_name}
Entity type: {entity_type}
Entity summary: {entity_summary}

Context extracted from the policy document:
{entity_context}

--- Policy Context ---
{policy_context}

Use the IAD/IG context carefully:
- IAD role rationale explains what kind of policy actor, participant group, or role slot this entity represents.
- IG rules describe what this role must do, can do, or cannot do.
- Information, actions, costs, and benefits linked to the role should shape its objectives and constraints.
- Eligibility criteria, exclusion criteria, resource access, reporting duties, and evaluation relations are especially important.
- The final KG context contains only validation-supported authoritative relations. If a validation audit is supplied
  elsewhere, treat it as QA material only and never convert dropped candidates into obligations, permissions, prohibitions,
  eligibility gates, resources, or hard constraints.
- Apply eligibility or exclusion criteria only to the role that actually holds them in the KG context.
  Direct-beneficiary constraints must not be copied into evaluator, organization, administrator, mentor, or funder profiles.
- For committees, panels, ministries, foundations, and institutions, model the policy actor as an institutional actor or
  representative role. Do not turn the collective body itself into a private individual.
- Do not add demographics, family background, hobbies, or individual life history.

Generate a JSON object with exactly these fields:
- representative_type: use "{entity_name}" exactly as-is (string)
- core_objectives: list of 2-4 actor-level objectives regarding the policy, grounded in the entity context above (list of strings)
- core_constraints: list of 2-4 policy-rule constraints, eligibility limits, compliance burdens, resource limits, or risks
  this actor faces, grounded in the entity context above (list of strings)

Important:
- All fields must be grounded in the entity context provided above, not invented from general knowledge.
- Do NOT describe a single individual. Do not include age, sex, occupation, education, family, or hobbies.
- core_objectives and core_constraints must reflect this actor's relationships in the policy ecosystem.
- If the context is too thin, keep the profile conservative."""


# --- KG + Nemotron Persona Synthesis (future Stage 1 persona construction prompt) ---

KG_NEMOTRON_PERSONA_SYNTHESIS_SYSTEM = "You are an expert in policy simulation persona construction. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

KG_NEMOTRON_PERSONA_SYNTHESIS_USER = """Create one Stage 2 simulation participant profile by combining:
1. a policy actor profile grounded in a policy KG, and
2. one country-specific Nemotron-Personas seed row used as realistic individual background material.

The KG actor profile is the authoritative policy role definition.
The Nemotron seed provides individual-level demographic, occupational, family, skill, and aspiration variation, but it
must not override the KG-defined policy role.

LANGUAGE REQUIREMENT:
- All output strings must be in English.
- If Nemotron seed fields are written in Korean or another language, translate and summarize them into English in the
  returned demographics, persona_profile, compatibility_assessment, policy_position, and iad_ig_context.
- Do not copy non-English seed text into the output. Preserve the meaning, not the original language.

Use the Nemotron seed fields deliberately:
- professional_persona: use for source occupation, work setting, professional habits, and domain experience.
- skills_and_expertise: use for capability, implementation capacity, research/technical strengths, and limits.
- career_goals_and_ambitions: use for individual objectives, incentives, and likely response to policy opportunities.
- family_persona: use for personal constraints, risk tolerance, mobility, stability needs, care burden, and time pressure.
- demographics such as age, sex, marital_status, military_status, family_type, education_level, bachelors_field,
  occupation, district, and province: preserve as source attributes and use only when they plausibly shape policy
  experience or uncertainty.
- hobbies_and_interests: use sparingly as human texture only when it helps make the persona realistic; do not let it
  drive policy stance.

Actor framing:
- The KG actor may be an individual role, a stakeholder group, an organization, or an official position.
- The KG actor profile contains role_family and constraint_scope. Treat those as authoritative for deciding which
  policy constraints apply to this simulation participant.
- If the KG actor is an individual role or target-population member, map the seed to that actor anchor for simulation.
  Do not assert real-world occupancy of the role unless the seed directly supports it.
- If the KG actor is a stakeholder group, instantiate the seed as one plausible member of that group.
- If the KG actor is an organization or official position, instantiate the seed as a human representative or role-holder
  speaking from that organization/position.
- Do not convert an organization itself into a private person. Preserve the distinction with agent_modality.
- If the KG actor is a panel or committee, the output participant is a panel/committee member or representative, not the
  committee as a private person.

Do not perform rejection:
- The seed has already passed deterministic sampling, broad occupation/education screening, and ranking before this prompt.
- Always return status="accepted".
- Your job is synthesis and conservative adaptation, not filtering.
- If the seed is imperfect or only broadly aligned, retain it and record the uncertainty in compatibility_assessment.
- Do not require an exact occupation label, exact policy-role title, exact degree timing, selection history, fellowship history,
  institution name, or research achievement unless the KG actor profile explicitly requires and the seed explicitly contradicts it.
- Preserve demographic diversity when it does not directly contradict the KG role.
- Do not treat bachelors_field, current occupation wording, region, sex, marital status, family type, hobbies, or age as
  automatic exclusions.
- If eligibility has alternative routes, do not treat one unmet or unknown route as disqualifying.
- Distinguish seed occupation from policy_role. The KG assigns the simulated policy role; the seed occupation is background.
- Do not invent PhD status, selection history, institution names, or achievements. When details are unknown, describe the
  persona as a plausible simulation participant with explicitly noted uncertainty.
- Do not erase mismatch by rewriting the seed's current occupation as the policy_role.
- In persona_profile, explicitly distinguish source background from simulated policy role.
- If eligibility or role occupancy is uncertain, write that the person is "mapped to" or "modeled under" the policy_role
  for simulation, rather than asserting current legal eligibility or current employment in that role.
- Never write that the person is currently the policy_role unless the seed itself says so. Avoid wording equivalent to
  "currently a {{policy_role}}" when the role comes only from the KG.
- Put age, occupation, degree-timing, or field uncertainty in compatibility_assessment.uncertainties instead of hiding it.

Use the IAD/IG context:
- obligations, permissions, prohibitions, resources, information requirements, actions, and evaluation relations should shape
  the persona's objectives, constraints, and expected reaction to the policy.
- The final KG context contains only validation-supported authoritative relations. Validation audit entries are QA-only
  dropped candidates and must not be used as obligations, permissions, prohibitions, eligibility gates, or resource entitlements.
- Eligibility or exclusion criteria must be respected only within their role scope.
- For role_family=direct_beneficiary, explicit statutory eligibility criteria are binding for simulation handoff when they
  are observable in the seed. State unobservable criteria as uncertainty rather than pretending they are verified.
- For role_family=evaluator, use expertise, evaluation criteria, fairness, independence, workload, and legitimacy constraints.
  Do not apply applicant or fellow eligibility conditions to evaluators.
- For role_family=host_organization, use hosting capacity, employment/contracting, research administration, fund management,
  compliance, and institutional incentives. Do not apply individual applicant age or degree-timing criteria to the institution.
- For role_family=policy_administrator, use budget, implementation, governance, accountability, selection, and monitoring constraints.
- For role_family=mentor, use expertise, mentoring capacity, network support, and career guidance constraints.
- For role_family=pipeline_actor or affected_actor, keep the person relevant to policy diffusion or future participation without
  claiming direct legal eligibility unless the KG context supports it.
- If the KG context states an explicit observable eligibility or exclusion rule for a direct-beneficiary role, respect it
  in the simulation handoff. If the Nemotron seed lacks the field needed to verify the rule, record the missing evidence
  as uncertainty rather than inventing eligibility. If a seed directly contradicts an observable direct-beneficiary rule,
  adapt it only as a near-target, affected, or pipeline participant when the KG role supports that interpretation.

Return JSON with exactly this structure:
{{
  "status": "accepted",
  "rejection_reason": null,
  "persona": {{
    "persona_id_hint": "string or null",
    "role_entity_id": "string",
    "policy_role": "string",
    "agent_modality": "individual | stakeholder_group_member | organization_representative | official_position_holder | committee_member",
    "role_family": "direct_beneficiary | evaluator | host_organization | policy_administrator | mentor | affected_actor | pipeline_actor",
    "constraint_scope": "short sentence explaining which policy constraints apply to this participant",
    "source": {{
      "kg": "policy_graph",
      "persona_seed": "nemotron-personas",
      "nemotron_uuid": "string"
    }},
    "demographics": {{
      "sex": "string or null",
      "age": "number or null",
      "education_level": "string or null",
      "bachelors_field": "string or null",
      "occupation": "string or null",
      "province": "string or null",
      "district": "string or null",
      "marital_status": "string or null",
      "family_type": "string or null"
    }},
    "translated_seed_profile": {{
      "persona": "English summary or null",
      "professional_persona": "English summary or null",
      "family_persona": "English summary or null",
      "skills_and_expertise": "English summary or null",
      "career_goals_and_ambitions": "English summary or null",
      "hobbies_and_interests": "English summary or null"
    }},
    "persona_profile": "one compact paragraph that separates source background from simulated policy_role mapping",
    "compatibility_assessment": {{
      "status": "direct_match | adapted_match | representative_match | uncertain_but_retained",
      "basis": ["short grounded reasons for retaining this seed"],
      "uncertainties": ["short notes about missing or imperfect alignment"]
    }},
    "policy_position": {{
      "objectives": ["2-4 role- and individual-grounded objectives"],
      "constraints": ["2-4 role- and individual-grounded constraints"]
    }},
    "iad_ig_context": {{
      "obligations": [],
      "permissions": [],
      "prohibitions": [],
      "resources": [],
      "information": [],
      "actions": [],
      "evaluation": []
    }}
  }}
}}

--- KG Actor Profile ---
{role_profile}

--- KG Context ---
{kg_context}

--- Policy Context ---
{policy_context}

--- Nemotron Seed Row ---
{nemotron_seed}"""


# --- Nemotron Value Profile Inference ---

NEMOTRON_VALUE_PROFILE_SYSTEM = "You are an expert policy-methodology assistant for adaptive persona sampling. Respond only in valid JSON with no markdown formatting or code blocks. All explanatory output string values must be in English."

NEMOTRON_VALUE_PROFILE_USER = """Infer soft Nemotron-Personas-Korea matching signals for one policy KG actor.

Purpose:
- Build a generalizable, policy-adaptive matching profile.
- Do NOT create role-specific hard-coded filters.
- The output will be used for broad first-stage candidate retrieval, 1-5 soft scoring, LLM reranking, and traceable
  sampling rationale.
- LANGUAGE REQUIREMENT: all reasons, summaries, rationales, guidance, constraints, and notes must be in English. Values
  copied from the observed Nemotron lists may remain in their exact original form because they are internal matching keys.

Methodological framing:
- IAD defines the policy actor, position, organization, beneficiary, implementer, evaluator, or affected group.
- IG/IAD KG context defines obligations, permissions, resources, information requirements, actions, and constraints.
- The KG actor context may include role_family and constraint_scope. Use them to decide whether the actor is a direct
  beneficiary, evaluator, host organization, policy administrator, mentor, affected actor, or pipeline actor.
- Nemotron provides real individual background fields. Its education_level, bachelors_field, and occupation values are
  not policy roles by themselves.
- A person can be relevant to a policy simulation even if they are not a perfect statutory beneficiary. For example,
  applicants, near-eligible candidates, students, early-career workers, institutional representatives, administrators,
  reviewers, and affected stakeholders can all be useful simulation participants if the KG actor role supports it.

Rules:
- Use only values that appear in the observed Nemotron value lists below.
- Include direct and adjacent signals separately.
- Include lower-degree or student signals when the policy role plausibly includes candidates, applicants, pipelines,
  trainees, prospective entrants, or affected near-eligible groups.
- Do not exclude bachelor-level or humanities/social-science fields unless the KG context explicitly says they are impossible
  for the actor role.
- Age should normally be treated as contextual metadata, not a hard filter. If the policy document includes an explicit
  observable age rule, express it as role-scoped context for scoring and uncertainty, not as a dataset-wide exclusion.
- Apply hard constraints only to the role that actually holds them. Direct-beneficiary eligibility should not shape
  evaluator, organization, administrator, mentor, or funder sampling.
- For direct-beneficiary roles, include observable statutory constraints as hard_constraints only when the policy document
  states them clearly and the Nemotron field can observe them. These constraints guide scoring/reranking; they should not
  become hard-coded Python filters. If the key field is absent, keep it as uncertainty.
- For evaluator or committee-member roles, prioritize expertise, field familiarity, evaluation/analysis experience, and
  legitimacy. Do not use beneficiary age or degree-timing as a gate.
- For host organizations or policy administrators, infer values for plausible representatives or leave individual matching
  broad. Organizations themselves should not be filtered as private people.
- For evaluator and policy-administrator representatives, education or bachelors_field alone is not enough. Prefer
  occupation or professional text signals that indicate research, technical expertise, public administration, evaluation,
  governance, program management, compliance, or institutional coordination.
- Prefer soft relevance scoring over filtering. If uncertainty exists, keep the value and explain the uncertainty.
- For organization or official-position actors, infer values for plausible human representatives or role holders, not for
  the organization as a private person.

Return JSON with exactly this structure:
{{
  "actor_role_summary": "one sentence",
  "sampling_rationale": "brief rationale grounded in IAD/IG and the observed Nemotron value distribution",
  "education_values": [
    {{"value": "exact observed education_level", "relevance": "direct | adjacent | broad | uncertain", "reason": "short reason"}}
  ],
  "bachelors_field_values": [
    {{"value": "exact observed bachelors_field", "relevance": "direct | adjacent | broad | uncertain", "reason": "short reason"}}
  ],
  "occupation_values": [
    {{"value": "exact observed occupation", "relevance": "direct | adjacent | broad | uncertain", "reason": "short reason"}}
  ],
  "occupation_keywords": ["soft source-language or English keywords useful for occupation/professional text scoring"],
  "negative_keywords": ["only explicit contradictions or clearly irrelevant occupation keywords"],
  "hard_constraints": [
    {{"field": "field name", "constraint": "constraint text", "source": "KG or policy reason"}}
  ],
  "sampling_guidance": {{
    "filtering_policy": "soft_scoring_only",
    "minimum_relevance": "minimal | broad | adjacent | direct",
    "preserve_diversity_notes": ["short notes"]
  }}
}}

--- KG Actor ---
{actor_context}

--- Policy Context ---
{policy_context}

--- Observed Nemotron Values ---
{value_profile}"""


# --- Nemotron Candidate Reranker ---

NEMOTRON_RERANKER_SYSTEM = "You are an expert policy simulation persona reranker. Respond only in valid JSON with no markdown formatting or code blocks. All explanatory string values must be in English."

NEMOTRON_RERANKER_USER = """Score Nemotron-Personas-Korea candidate rows for one policy KG actor.

Purpose:
- The deterministic stage has already produced a broad candidate set using observable columns such as occupation, age,
  education_level, and bachelors_field.
- Your job is to use policy-role context and richer Nemotron text fields to assign a role-fit score from 1 to 5.
- This is a reranking task, not a hard rejection task. Use the full 1-5 range and keep uncertainty visible.
- Return one ranked_candidates entry for every candidate uuid provided in Candidate Rows.

Scoring rubric:
- 5 = direct role fit. The seed strongly matches the KG actor role, constraints, domain, and likely policy experience.
- 4 = strong adjacent fit. The seed is close to the role and would be useful with minor uncertainty.
- 3 = plausible broad fit. The seed is relevant to the policy ecosystem but not a direct role match.
- 2 = weak but interpretable fit. The seed can represent an affected/adjacent perspective with clear uncertainty.
- 1 = minimal fit. Keep only if diversity or contrast is methodologically useful.

Use these fields deliberately:
- occupation, education_level, bachelors_field, and age as observable structured signals.
- professional_persona for actual work setting and domain practice.
- skills_and_expertise for capability and policy interaction capacity.
- career_goals_and_ambitions for incentives and likely response to the policy.
- family_persona only for constraints that plausibly affect policy participation.

Important:
- Do not apply direct-beneficiary eligibility to evaluators, organizations, administrators, mentors, or funders.
- Do not hard reject a seed because of age, field, sex, region, occupation wording, or degree wording. Penalize only when
  it creates clear role uncertainty.
- If a direct-beneficiary KG role contains explicit observable eligibility constraints, use them as strong role-fit
  evidence. If a seed directly contradicts them, assign a lower role-fit score and explain the uncertainty; do not hide
  the contradiction.
- matching_profile.observable_eligibility maps policy eligibility text to Nemotron columns. Treat fields marked
  observable=true as structured evidence, observable="indirect" as weak textual evidence, and observable=false as
  uncertainty. When eligibility logic has an OR route and one route is unobservable, do not reject solely on the
  observable route.
- For direct-beneficiary roles, a seed that visibly conflicts with an observable eligibility constraint should normally
  receive score 1-2 unless the candidate row itself provides strong alternative evidence that resolves the conflict.
- Distinguish source occupation from simulated policy_role. A seed can be mapped to a policy role for simulation without
  claiming that the person currently occupies that role.
- Prefer policy-role relevance over demographic tidiness.

Return JSON with exactly this structure:
{{
  "ranked_candidates": [
    {{
      "uuid": "candidate uuid",
      "role_fit_score": 1,
      "rationale": "short role-fit rationale",
      "uncertainties": ["short uncertainty notes"]
    }}
  ]
}}

--- KG Actor Context ---
{actor_context}

--- Policy Context ---
{policy_context}

--- Adaptive Matching Profile ---
{matching_profile}

--- Candidate Rows ---
{candidates}"""


# --- Policy Context Constructor ---

POLICY_CONTEXT_SYSTEM = "You are an expert policy analyst building shared scenario context for downstream policy simulation. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

POLICY_CONTEXT_USER = """Build a shared policy context object for Stage 2 simulation.

Purpose:
- This is global scenario context, not a persona.
- LANGUAGE REQUIREMENT: all generated policy context fields must be in English, even if the source policy document or KG
  evidence was originally written in another language.
- It should tell Stage 2 what field/domain the policy belongs to, what kind of policy instrument it is, what mechanism it uses,
  who it targets, who implements it, and what outcomes it is trying to produce.
- Use only the provided policy plan, economic context, KG ontology summary, KG actor nodes, and KG edge summaries.
- When KG edge summaries contain eligibility-related activation_condition, execution_constraint, or source_text, preserve
  those details in eligibility_logic. Keep observable constraints explicit, especially age thresholds, degree/timing
  windows, exclusions, and whether the logic is AND or OR.
- Do not collapse "PhD within N years OR under age M" into a single hard age filter. Write it as two routes so the
  Nemotron sampler can use age as observable evidence while leaving PhD award timing as unobserved uncertainty.
- Do not invent detailed facts that are not supported. If uncertain, use a conservative general label.

Return JSON with exactly this structure:
{{
  "policy_name": "string",
  "policy_domain": "broad policy domain, e.g. science and technology human resources / research funding",
  "policy_subdomains": ["2-5 subdomains"],
  "policy_type": "policy type, e.g. fellowship / research grant / talent development policy",
  "policy_instruments": ["funding", "eligibility criteria", "institutional choice", "evaluation", "..."],
  "target_population": ["target or near-target groups"],
  "primary_beneficiaries": ["direct beneficiaries"],
  "implementing_actors": ["ministries, research institutions, evaluators, program officers, etc."],
  "delivery_mechanism": "how the policy reaches actors",
  "eligibility_logic": "how eligibility or selection works at a high level",
  "resource_logic": "what resources/costs/benefits are mobilized",
  "expected_outcomes": ["2-5 expected outcomes"],
  "theory_of_change_summary": "one compact paragraph explaining inputs -> activities -> outputs -> outcomes",
  "simulation_relevance": "how this shared context should be used by Stage 2 personas"
}}

{sections}"""


# --- Personas from Text (no-graph ablation) ---

PERSONAS_FROM_TEXT_SYSTEM = "You are an expert in stakeholder analysis and policy impact assessment. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

PERSONAS_FROM_TEXT_USER = """You are given a past policy document. Identify the key stakeholder types involved in this policy ecosystem, then generate {n_personas} representative persona(s) per stakeholder type.

--- Past Policy Document ---
{past_policy_text}

--- New Policy Context ---
{policy_context}

Instructions:
- Identify distinct stakeholder types present in the past policy document (e.g. government agency, research institute, university, industry, beneficiary).
- For each stakeholder type, generate exactly {n_personas} persona(s) with a distinct representative name grounded in the past policy document.
- Each persona must be grounded in the past policy document — use real organizations, roles, or participant types mentioned there.
- Do NOT invent stakeholders not evidenced in the past policy document.
- Do NOT use specific numerical outcome values in any field.

Return a JSON object with a single key "stakeholder_groups" — a list of objects, each with:
- stakeholder_type: the stakeholder category label (string)
- personas: list of {n_personas} persona object(s), each with:
  - representative_type: a specific representative name grounded in the past policy document (string)
  - core_objectives: list of 2-4 concrete objectives this stakeholder has regarding the new policy (list of strings)
  - core_constraints: list of 2-4 real constraints or concerns this stakeholder faces (list of strings)"""


# --- Predictor ---

PREDICTOR_SYSTEM = "You are an expert policy impact analyst. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

PREDICTOR_USER = """You are predicting the outcome of the following policy from the perspective of each representative type in a stakeholder group.

Prediction target: {prediction_target}

IMPORTANT: The predicted_value must be expressed in the same unit and scale as the prediction target. For example:
- If the target is a rate or percentage (e.g. "employment rate"), predicted_value must be a percentage between 0 and 100.
- If the target is a count (e.g. "number of fellows selected"), predicted_value must be an integer count.
- Do NOT convert rates to counts or counts to rates.

Apply this persona's policy_outcome_magnitude as an adjustment to the baseline value implied by the prediction target. For example, if a baseline employment rate is ~75% and policy_outcome_magnitude is "+5-10% above target", the predicted_value should be in the range of 78-82.

For each representative type, provide:
- predicted_value: the numerical estimate in the same unit as the prediction target, adjusted by this persona's perspective (float)
- prediction: qualitative explanation of why this persona arrives at that estimate (string)
- evidence: concrete reasons supporting this estimate, referencing policy content and this persona's objectives/constraints (list of strings)

Return a JSON object with a single key "predictions" — a list of objects, one per representative type, each with:
- representative_type: the representative type label (string)
- predicted_value: numerical estimate in the same unit as the prediction target (float)
- prediction: qualitative explanation (string)
- evidence: list of supporting reasons (list of strings)

--- Policy ---
{policy_plan}

--- Stakeholder Group: {stakeholder_type} ---
{personas}"""


# --- Critic (single call across all stakeholder groups) ---

CRITIC_SYSTEM = "You are an expert policy critic and peer reviewer. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

CRITIC_USER = """You are evaluating whether each stakeholder persona's worldview — their core_objectives, core_constraints, stance, policy_outcome_expectation, and policy_outcome_magnitude — is realistically grounded given the actual policy outcome.

The actual recorded outcome for the prediction target is:
{past_policy_result}

Each entry in the input already contains a precomputed "error_score" and "direction" field. Use these directly — do not recompute them.

For each persona:
1. If error_score <= {threshold}: the worldview is valid — set refinement_suggestions to [].
2. If error_score > {threshold}: the worldview is misaligned. Follow these steps:
   a. Read the direction field — TOO OPTIMISTIC means the persona overestimated, TOO PESSIMISTIC means it underestimated.
   b. Identify what this persona fundamentally misunderstands about the policy's real-world impact. Focus on core_objectives, core_constraints, stance, and policy_outcome_magnitude — NOT on organizational actions (e.g., do NOT suggest "do more outreach").
      - If TOO PESSIMISTIC: the persona overstated barriers or underestimated the policy's reach and magnitude of success. Identify which constraints are unrealistically restrictive and whether policy_outcome_magnitude is too conservative.
      - If TOO OPTIMISTIC: the persona ignored structural limits or overestimated the policy's magnitude of success. Identify which objectives are inflated and whether policy_outcome_magnitude is too aggressive.
   c. Write refinement_suggestions as a list of plain strings describing specific worldview corrections — including whether to revise the magnitude estimate upward or downward. Do NOT include the actual numerical outcome value in suggestions. Express magnitude corrections qualitatively (e.g. "the policy exceeded its target by a much larger margin than estimated", "the magnitude estimate should be revised significantly upward").

Also consider cross-stakeholder consistency — if the actual outcome implies something about how one group is affected, check whether other groups' worldviews account for that reality.

Return a JSON object with a single key "critic_results" — a list of objects, one per stakeholder group, each with:
- stakeholder_type: use the exact stakeholder_type string from the input (string)
- results: list of objects, one per representative type, each with:
  - representative_type: the representative type label (string)
  - error_score: use the precomputed error_score from the input (float)
  - refinement_suggestions: list of worldview correction strings (empty list if valid)

--- Policy ---
{policy_plan}

--- All Stakeholder Personas and Predictions ---
{all_predictions}"""


# --- Persona Refiner ---

PERSONA_REFINER_SYSTEM = "You are an expert in stakeholder analysis and policy impact assessment. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

PERSONA_REFINER_USER = """A persona's worldview has been flagged as misaligned with the actual policy outcome. Refine the persona's profile based on the critic's suggestions. You do NOT have access to the actual numerical outcome — reason purely from the critic's qualitative corrections and this persona's updated worldview.

You must NOT change the representative_type. Apply the corrections to update all fields. The result should read as a realistic, grounded perspective.

Return a JSON object with exactly these fields:
- core_objectives: updated list of core objectives (list of strings)
- core_constraints: updated list of constraints and concerns (list of strings)
- stance: updated overall stance toward the policy (string)
- policy_outcome_expectation: updated qualitative expectation (e.g. "significantly exceed target"). Do NOT use specific numerical values.
- policy_outcome_evidence: updated reasoning string. Do NOT use specific numerical values.
- policy_outcome_magnitude: updated percentage range estimate based on the corrected worldview (e.g. "+40-60% above annual target"). Derive this from your understanding of the policy's reach and the critic's corrections — do NOT reverse-engineer from any actual value.

--- Policy ---
{policy_plan}

--- Original Persona ---
{persona}

--- Critic's Worldview Corrections ---
{refinement_suggestions}"""


# --- Stage 2: Simulation Persona Constructor ---

SIM_PERSONA_CONSTRUCTOR_SYSTEM = "You are an expert in stakeholder analysis and policy simulation design. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

SIM_PERSONA_CONSTRUCTOR_USER = """You are constructing a specific, concrete simulation persona for the {stakeholder_type} stakeholder group in a policy simulation.

You are given a set of abstract persona profiles for this stakeholder group, along with each persona's Stage 1 prediction result (their numerical estimate and reasoning for the prediction target).

Your task: synthesize ONE vivid, specific individual who embodies the key characteristics of this stakeholder group. This person will participate directly in a policy simulation debate. Ground their profile in the abstract personas — carry forward the worldview, stance, and qualitative expectations most representative of this group. Do NOT carry forward any numerical magnitude estimates.

Return a JSON object with exactly these fields:
- stakeholder_type: "{stakeholder_type}" (string)
- name: a plausible full name, e.g. "Dr. Ji-ho Kim" (string)
- institution: the specific institution or organization they belong to (string)
- role: their specific job title or role (string)
- background: 2-3 sentences describing their background and how they relate to this policy (string)
- goals_in_simulation: list of 2-3 concrete goals this person brings to the simulation discussion (list of strings)
- concerns: list of 2-3 specific concerns or objections they have about the policy (list of strings)
- stance: their overall stance toward the policy (string)
- policy_outcome_expectation: their qualitative expectation of how the policy will perform — focus on mechanisms and conditions, not numerical estimates (string)

--- Policy-Implementing Organizations (KG context) ---
{org_kg_contexts}

--- Abstract Personas and Stage 1 Prediction Results for {stakeholder_type} ---
{personas_with_predictions}"""


# --- Stage 2: Round 1 (Initial Comment) ---

ROUND1_SYSTEM = "You are a stakeholder participant in a policy simulation. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

ROUND1_USER = """You are participating in a policy simulation as the following persona. Give your initial assessment of the expected outcome of this policy for the simulation target.

Simulation target: {simulation_target}

Note: your persona's worldview was calibrated against actual policy implementation data. This gives you an evidence-based foundation for your perspective. Engage genuinely with the discussion, but don't abandon your grounded view without good reason.

Predict the TOTAL system-level outcome for the simulation target based on your persona's worldview — your background, demographics, professional experience, and skills. Reason from the mechanisms you understand as this stakeholder, not from any numerical anchor. Be specific and grounded in your perspective.

Return a JSON object with exactly these fields:
- predicted_value: your numerical estimate of the simulation target (float)
- comment: your qualitative assessment of why the policy will produce this outcome (string)
- evidence: list of concrete reasons supporting your estimate, grounded in your persona's worldview (list of strings)

--- Policy ---
{policy_plan}

--- Your Persona ---
{persona}"""


# --- Stage 2: Dynamic Round (Speak or Pass) ---

DYNAMIC_ROUND_SYSTEM = "You are a stakeholder participant in a policy simulation. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

DYNAMIC_ROUND_USER = """You are in an ongoing policy simulation debate. Review the full discussion history below and decide whether to speak or pass this round.

Simulation target: {simulation_target}

Rules:
- SPEAK if you have new arguments, if others raised points that change your view, or if you disagree with the current direction.
  If you speak, you must revise your predicted value — speaking means you are updating your position.
- PASS if the discussion has not changed your view and your current prediction still stands.
  Passing means you maintain your last predicted value: {current_prediction}.

Return a JSON object with exactly these fields:
- spoke: true if you choose to speak, false if you pass (boolean)
- revised_predicted_value: your updated numerical estimate — required if spoke is true, omit if false (float or null)
- comment: your reasoning for speaking or passing — required if spoke is true, omit if false (string or null)

--- Policy ---
{policy_plan}

--- Your Persona ---
{persona}

--- Full Debate History ---
{debate_history}"""


# --- Stage 2: Mandatory Round (Everyone Speaks) ---

MANDATORY_ROUND_SYSTEM = "You are a stakeholder participant in a policy simulation. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

MANDATORY_ROUND_USER = """You have been listening to an ongoing policy simulation debate. Review what others have said and revise your view in light of their perspectives.

Simulation target: {simulation_target}

Your current prediction: {current_prediction}
Other participants' current predictions: {other_predictions}

Note: your persona's worldview was calibrated against actual policy implementation data — engage genuinely with others' perspectives and update when they raise valid points, but don't drift from your grounded view simply due to social pressure.

Read the full debate history below. Then:
- If another participant raised a point that meaningfully shifts your understanding, revise your prediction toward or away from theirs accordingly.
- If you are not moved by the arguments, explain specifically why — and consider whether your own prior estimate needs any refinement.
- Either way, your revised_predicted_value should reflect honest engagement with what you heard, not just a restatement.

Return a JSON object with exactly these fields:
- revised_predicted_value: your updated numerical estimate after hearing others (float)
- comment: what you took from the debate, what you agree or disagree with, and why your estimate is what it is now (string)

--- Policy ---
{policy_plan}

--- Your Persona ---
{persona}

--- Full Debate History ---
{debate_history}"""


# --- Stage 2: Final Round (Final Prediction + Improvement) ---

FINAL_ROUND_SYSTEM = "You are a stakeholder participant in a policy simulation. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

FINAL_ROUND_USER = """The debate has concluded. Give your final prediction and policy improvement recommendations.

Simulation target: {simulation_target}

Your current prediction going into this final round: {current_prediction}

Based on the full discussion, give your definitive position. Your final_predicted_value should reflect everything you heard — revise it if the debate changed your view, or hold it if not.

Return a JSON object with exactly these fields:
- final_predicted_value: your final numerical estimate for the simulation target (float)
- final_comment: a concise statement of your final position and what shaped it (string)
- improvement_suggestions: concrete recommendations for improving this policy from your stakeholder perspective (string)

--- Policy ---
{policy_plan}

--- Your Persona ---
{persona}

--- Full Debate History ---
{debate_history}"""


# --- Stage 2: Judge ---

JUDGE_SYSTEM = "You are an expert policy analyst and simulation judge. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

POLICY_REPORTER_SYSTEM = "You are an expert policy analyst synthesizing the results of a multi-scenario stakeholder simulation. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

POLICY_REPORTER_USER = """You are writing a final synthesis report from a multi-scenario policy simulation.

Simulation target: {simulation_target}

You are given the results of {num_scenarios} simulation scenarios. Each scenario involved a different combination of stakeholders debating the expected policy outcome.

Your task: synthesize the collective intelligence from all scenarios into a structured report.

Return a JSON object with exactly these fields:
- common_evidence: list of 3-5 evidence themes that appeared across multiple scenarios — why stakeholders believe the policy will or won't hit its target (list of strings)
- common_concerns: list of 3-5 concern themes that were repeatedly raised — structural, operational, or contextual barriers (list of strings)
- common_recommendations: list of 3-5 improvement recommendations that appeared consistently across participants (list of strings)
- synthesis: a concise 3-5 sentence narrative summarizing the overall stakeholder view — what the simulation collectively suggests about the policy's expected outcome and the key forces driving it (string)

Focus on patterns that recur across scenarios, not individual outlier views. Do NOT include any numerical values in your response.

--- Policy ---
{policy_plan}

--- Scenario Results ---
{scenario_summaries}"""


# ─────────────── Free-form KG extraction (IAD/IG ablation) ───────────────
# Same extraction structure as the IAD/IG prompts above but with all IAD/IG
# typed slots removed (no iad_category/node_family/iad_rule_type, no deontic/
# edge_category/activation_condition/execution_constraint/or_else_text). Used
# by the 2×2 ablation experiment to isolate the contribution of IAD/IG
# institutional theory to downstream simulation accuracy.

KG_NODE_EXTRACTOR_SYSTEM_FREEFORM = (
    "You are an expert policy knowledge graph extractor. "
    "Respond only in valid JSON with no markdown formatting or code blocks. "
    "All generated node labels, summaries, attributes, warnings, and source_text "
    "evidence must be in English."
)

KG_NODE_EXTRACTOR_USER_FREEFORM = """Extract policy KG node instances from the document chunk.

Rules:
- LANGUAGE REQUIREMENT: all generated node names, summaries, attributes, warnings, and any other non-evidence strings
  must be in English.
- SOURCE EVIDENCE REQUIREMENT: source_text must be a short English evidence sentence translated or summarized from the
  smallest source fragment that supports the node. It must stay faithful to the document and must not add facts that are
  absent from the chunk.
- Extract both stakeholder actors and non-actor policy objects (rules, resources, processes, documents, time windows,
  evaluation criteria) when they explain actor constraints, information, actions, costs, or benefits.
- Every authoritative node should be useful for at least one relation that changes simulation behavior. Do not extract
  isolated glossary terms, generic background concepts, or repeated labels that are unlikely to participate in an edge.
- Treat persona-eligible actor or position nodes as simulation actor anchors, not as individual biographies.
  A simulation actor anchor may be an individual role, stakeholder group, organization, or official position.
- For each persona-eligible node (any actor, role, group, organization, or position that could act inside the policy),
  set persona_eligible=true and is_actor=true. Set persona_eligible=false and is_actor=false for non-actor policy
  objects (documents, resources, rules, time windows, evaluation criteria).
- Keep collective bodies and their human speakers distinct. A panel, committee, ministry, foundation, or institution can be
  a KG actor anchor, but the later persona must be a member, representative, or role-holder unless the actor is kept as an
  institutional profile without Nemotron.
- Extract a role node only when the text supports that the role participates in, implements, evaluates, funds, receives,
  is constrained by, or is affected by the policy.
- Make persona_eligible true only for explicit policy actors, roles, organizations, positions, committees, or stakeholder
  groups that can reasonably receive at least one supported KG relation. Generic placeholders are not persona anchors.
- Do not extract only the main beneficiary. Also extract supported applicants, candidates, pipeline groups, host organizations,
  administrators, evaluators, funders, reviewers, mentors, intermediaries, institutional governance bodies, and affected
  groups when the chunk supports them.
- If a role is near-eligible or adjacent to the target population, keep it as a simulation actor anchor when it could shape
  policy impact in Stage 2. Put the eligibility uncertainty in attributes rather than dropping the actor.
- Use `type` as a free-form short English label that names the node concretely (e.g. "Postdoctoral Researcher",
  "Research Foundation", "Evaluation Panel", "Annual Research Grant", "Research Proposal"). Do NOT force any controlled
  vocabulary on the type.
- For action/process nodes, extract process phases, not duplicate mentions.
- For document/information nodes, extract documents, criteria, plans, reports, eligibility evidence, notices, records,
  metrics, or decision signals. Proposals, reports, and plans are documents/information even when they influence funding.
- For resource/benefit/cost/sanction nodes, split distinct resources, rewards, costs, or sanctions when the text
  distinguishes them.
- Do not classify support programs, grants, reports, proposals, notifications, compliance requirements, or funding limits
  as actors. They are non-actor policy objects unless the document names a real organization, committee, or office that
  performs an action.
- Extract conditions, restrictions, process steps, dates/periods, decision authorities, evaluation criteria, resource
  components, and outcome metrics as separate node instances when they can affect simulation behavior.
- Keep names concise, English, and grounded in the source text. Do not simply reuse a generic category label as the
  instance name when the document gives a more specific actor, object, process, or rule label.
- Do not instantiate the policy/program title itself as a persona-eligible actor. If the policy supports fellowship
  recipients, selected researchers, applicants, or beneficiaries, the node instance must name the actor role
  (for example "Postdoctoral Researchers", "Selected Fellows", or "Applicants"), not the fellowship/program name.
- When a generic node and a more specific node would represent the same policy object, extract the more specific node only.
  Example: prefer "Basic Research Project Promotion Committee" over "ApprovalAuthority" when the committee is named.
- When the text uses a broad population label but defines a narrower statutory or operational role, use the narrower
  policy role as the direct-beneficiary actor. Keep the broad phrase in summary or attributes, not as a separate generic
  persona anchor.
- Fill attributes only with grounded information that helps later policy simulation or persona synthesis. Useful attribute
  keys include eligibility_criteria, exclusion_criteria, role_scope, decision_authority, support_period, funding_amount,
  reporting_duty, evaluation_criteria, target_quota, implementing_body, and affected_population. Add other keys when the
  source text supports them. Only emit grounded facts.
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
      "type": "freeform short English label",
      "summary": "brief grounded summary",
      "attributes": {{}},
      "is_actor": true,
      "persona_eligible": true,
      "source_text": "supporting source text",
      "confidence": 0.0
    }}
  ],
  "warnings": []
}}

--- WORKED EXAMPLE (methodological only; type labels are illustrative, not vocabulary to reuse) ---

Suppose the chunk contains:
"세종과학펠로우십 지원 대상자는 박사학위를 취득한 후 7년 이내인 자로서 비전임 연구원에 한한다.
연구재단은 매년 100명 내외의 펠로우를 선정하며, 선정된 펠로우는 5년간 매년 1.3억원의 연구비를 지원받는다.
선정평가위원회는 연구계획서를 평가하여 선정 결과를 결정한다."

A high-quality node extraction yields:
[
  {{
    "name": "Postdoctoral Researcher",
    "type": "Postdoctoral Researcher",
    "summary": "Direct beneficiary of the fellowship; non-tenure-track researcher within 7 years of PhD.",
    "attributes": {{
      "eligibility_criteria": "PhD obtained within 7 years; non-tenure-track",
      "exclusion_criteria": "tenure-track faculty are excluded"
    }},
    "is_actor": true,
    "persona_eligible": true,
    "source_text": "Non-tenure-track researchers who obtained a PhD within seven years are eligible to apply for support.",
    "confidence": 0.95
  }},
  {{
    "name": "Research Foundation",
    "type": "Research Foundation",
    "summary": "Implementing organization that selects fellows and disburses the grant.",
    "attributes": {{"implementing_body": "national research funding agency"}},
    "is_actor": true,
    "persona_eligible": true,
    "source_text": "The research foundation selects about 100 fellows each year.",
    "confidence": 0.9
  }},
  {{
    "name": "Selection Evaluation Panel",
    "type": "Evaluation Panel",
    "summary": "Panel that evaluates research proposals and decides selection outcomes.",
    "attributes": {{"decision_authority": "approves selection outcomes"}},
    "is_actor": true,
    "persona_eligible": true,
    "source_text": "The selection evaluation panel evaluates research proposals and decides selection results.",
    "confidence": 0.9
  }},
  {{
    "name": "Annual Fellowship Quota",
    "type": "Funding Limit",
    "summary": "Approximately 100 fellows are selected annually.",
    "attributes": {{"target_quota": "about 100 fellows per year"}},
    "is_actor": false,
    "persona_eligible": false,
    "source_text": "About 100 fellows are selected each year.",
    "confidence": 0.85
  }},
  {{
    "name": "Annual Research Grant",
    "type": "Research Grant",
    "summary": "130 million KRW per year for 5 years per selected fellow.",
    "attributes": {{
      "funding_amount": "130 million KRW per year",
      "support_period": "5 years"
    }},
    "is_actor": false,
    "persona_eligible": false,
    "source_text": "Selected fellows receive 130 million KRW in research funding each year for five years.",
    "confidence": 0.95
  }},
  {{
    "name": "Research Proposal",
    "type": "Research Proposal",
    "summary": "Proposal document used as the input to selection evaluation.",
    "attributes": {{}},
    "is_actor": false,
    "persona_eligible": false,
    "source_text": "Research proposals are evaluated during the selection process.",
    "confidence": 0.85
  }}
]

What this example demonstrates:
- The program label "세종과학펠로우십" is NOT a persona-eligible actor; the beneficiary actor is "Postdoctoral Researcher".
- The non-tenure-track restriction is preserved as both an eligibility/exclusion attribute and English source_text evidence.
- The evaluation panel and the research foundation are kept distinct.
- Each node has a single, scoped source_text sentence that faithfully translates or summarizes the supporting source fragment.
- Numerical values (quota, grant amount, support period) live in attributes, not invented elsewhere.
- Non-actor policy objects (Quota, Grant, Proposal) have is_actor=false and persona_eligible=false.

--- Simulation Requirement ---
{simulation_requirement}

--- Document Chunk {chunk_id} ---
{chunk_text}"""

KG_EDGE_EXTRACTOR_SYSTEM_FREEFORM = (
    "You are an expert policy knowledge graph edge extractor. "
    "Respond only in valid JSON with no markdown formatting or code blocks. "
    "All generated facts, attributes, warnings, and source_text evidence must be in English."
)

KG_EDGE_EXTRACTOR_USER_FREEFORM = """Extract policy KG edge instances from the document chunk against the provided global node pool.

OUTPUT BUDGET (HARD LIMIT):
- Emit AT MOST {max_edges_per_chunk} edges from this chunk. If more than {max_edges_per_chunk} edges are plausible, KEEP only those with the strongest direct textual support and DROP weaker/redundant ones. Do NOT silently truncate; pick fewer, well-evidenced edges over many speculative ones. Adjacent chunks (via overlap) will cover edges you skip near the chunk boundary.

Rules:
- source and target must be exact node ids from the Global Node Pool.
- LANGUAGE REQUIREMENT: all generated facts, attributes, warnings, and any other non-evidence strings must be in English.
- SOURCE EVIDENCE REQUIREMENT: source_text must be a short English evidence sentence translated or summarized from the
  smallest source fragment that supports the edge. It must directly entail the edge type, source, target, and any other
  fact when present. Do not add facts that are absent from the chunk.
- Use `type` as a free-form short VERB_PHRASE in CAPS_WITH_UNDERSCORES that names the relation concretely
  (e.g. SELECTS, EVALUATES, RECEIVES, APPLIES_TO, IS_ELIGIBLE_FOR, FUNDS, MONITORS, REPORTS_TO, REQUIRES_DOCUMENT,
  ALLOCATES, NOTIFIES, MENTORS). Do NOT force any controlled vocabulary on the type.
- Extract edges for process order, eligibility, selection, resource flow, implementation, evaluation, submission,
  reporting, monitoring, information use, and definitions when supported by the chunk.
- Prefer edges that connect specific, source-supported nodes. If the Global Node Pool contains both a generic category-label
  node and a specific document-named node for the same actor/object, use the specific node.
- Extract multiple specific institutional relations when they change simulation behavior: eligibility, exclusion, document
  requirements, application permissions, restrictions, frequency limits, approval requirements, selection, notification,
  funding components, support services, milestones, and outcome targets.
- Do not create edges to keep a weak or isolated node alive. The edge must be independently supported by source_text and
  must make the source or target meaningful for policy simulation.
- Resource allocation, funding, evaluation, approval, selection, and monitoring edges must use the actor with institutional
  authority as the source. Do not make a beneficiary the source of ALLOCATES, FUNDS, APPROVES, SELECTS, EVALUATES, or
  MONITORS unless the source text explicitly gives that authority to the beneficiary.
- Prefer edges that preserve policy-role consistency for later persona synthesis: eligibility, exclusion, resource access,
  duties, permissions, evaluation, reporting, and actor-object-action relations.
- Every edge must include source_text. source_text must directly entail the edge type, source, and target.
  If the quote only provides broad background but does not support the relation, do not extract the edge.
- Validation will drop candidate edges whose source_text does not directly support the relation. Dropped candidates are
  QA/audit material only and must not be treated as authoritative KG facts.
- Do not invent edges unsupported by the chunk.
- If a candidate edge would create a role/persona inconsistency not supported by the document, do not extract it.
- Target 10-30 edge instances per chunk when the chunk is information-dense. Do not collapse distinct conditions,
  restrictions, support items, or process steps into one generic edge.

Return JSON with this exact structure:
{{
  "edges": [
    {{
      "type": "VERB_PHRASE",
      "source": "ExactSourceNodeId",
      "target": "ExactTargetNodeId",
      "fact": "brief factual relation",
      "source_text": "supporting source text",
      "attributes": {{}},
      "confidence": 0.0
    }}
  ],
  "warnings": []
}}

--- WORKED EXAMPLE (methodological only; verb-phrase types are illustrative, not vocabulary to reuse) ---

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

A high-quality edge extraction yields:
[
  {{
    "type": "IS_ELIGIBLE_FOR",
    "source": "PostdoctoralResearcher:Postdoctoral Researcher",
    "target": "ResearchGrant:Annual Research Grant",
    "fact": "Postdoctoral researchers within 7 years of PhD and non-tenure-track may receive the fellowship grant.",
    "source_text": "Non-tenure-track researchers who obtained a PhD within seven years are eligible to apply for support.",
    "attributes": {{"eligibility_window": "within 7 years of PhD"}},
    "confidence": 0.9
  }},
  {{
    "type": "SELECTS",
    "source": "ResearchFoundation:Research Foundation",
    "target": "PostdoctoralResearcher:Postdoctoral Researcher",
    "fact": "The research foundation selects about 100 postdoctoral researchers each year as fellows.",
    "source_text": "The research foundation selects about 100 fellows each year.",
    "attributes": {{"frequency": "annual", "quota": "about 100 per year"}},
    "confidence": 0.95
  }},
  {{
    "type": "EVALUATES",
    "source": "EvaluationPanel:Selection Evaluation Panel",
    "target": "ResearchProposal:Research Proposal",
    "fact": "The selection evaluation panel evaluates the research proposal to decide selection outcomes.",
    "source_text": "The selection evaluation panel evaluates research proposals and decides selection results.",
    "attributes": {{}},
    "confidence": 0.95
  }},
  {{
    "type": "ALLOCATES",
    "source": "ResearchFoundation:Research Foundation",
    "target": "ResearchGrant:Annual Research Grant",
    "fact": "The research foundation allocates 130 million KRW per year for 5 years to each selected fellow.",
    "source_text": "Selected fellows receive 130 million KRW in research funding each year for five years.",
    "attributes": {{"amount": "130 million KRW/year", "duration": "5 years"}},
    "confidence": 0.95
  }},
  {{
    "type": "REQUIRES_DOCUMENT",
    "source": "ResearchFoundation:Research Foundation",
    "target": "ResearchProposal:Research Proposal",
    "fact": "Selection by the research foundation requires submission of a research proposal for panel evaluation.",
    "source_text": "Research proposals are evaluated during the selection process.",
    "attributes": {{}},
    "confidence": 0.85
  }}
]

What this example demonstrates:
- The actor with institutional authority is the SOURCE of resource/selection/evaluation edges (SELECTS, ALLOCATES, EVALUATES).
- The beneficiary is the SOURCE of IS_ELIGIBLE_FOR / APPLIES_TO style edges.
- Numerical scope (quota, amount, duration) goes into edge attributes, not invented elsewhere.
- Every edge has source_text that directly supports the relation; nothing is paraphrased beyond the document.

--- Global Node Pool ---
{node_pool}

--- Document Chunk {chunk_id} ---
{chunk_text}"""


# ───────────────────────────── Stage 2 ToC ─────────────────────────────

# --- ToC Inner Schema Builder ---

INNER_SCHEMA_BUILDER_SYSTEM = "You are an expert in Theory of Change and policy logic-model design. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

INNER_SCHEMA_BUILDER_USER = """You are designing the per-phase variable schema for a Theory of Change simulation of a specific policy.

The 5 ToC phases are FIXED in this order: Inputs, Activities, Outputs, Outcomes, Impact. You do NOT choose the phase names — they are constants.

For each of the 5 phases, design:
- prediction_fields: 2–4 concrete, measurable variables that personas should estimate at this phase. Each variable must have:
    - name: snake_case machine key (string)
    - unit: measurement unit, e.g. "KRW", "count", "person", "%", "publications", "patents" (string)
    - type: one of "int" | "float" | "str" (string)
    - description: short human-readable explanation (string)
- discussion_points: 2–4 questions or causal mechanisms personas should address when filling those fields at this phase (list of strings)

{target_only_clause}

Critical guidance:
- The variables at the Outcomes phase MUST include at least one numeric variable that aligns with the simulation_target below (so forward and backward can be cross-checked on it).
- Inputs = resources committed (budget, slots, infrastructure). Activities = what the program does with those inputs. Outputs = direct measurable products of activities. Outcomes = short/medium-term changes attributable to outputs. Impact = long-term systemic change.
- Variables at each phase should be units that BOTH a forward predictor and a backward "what's-required" reasoner can fill in. Avoid variables that only make sense in one direction.
- Do NOT propose more than 4 variables or 4 discussion points per phase — keep it focused.

Return a JSON object with exactly this shape:
{{
  "phases": [
    {{"phase": "Inputs",     "prediction_fields": [...], "discussion_points": [...]}},
    {{"phase": "Activities", "prediction_fields": [...], "discussion_points": [...]}},
    {{"phase": "Outputs",    "prediction_fields": [...], "discussion_points": [...]}},
    {{"phase": "Outcomes",   "prediction_fields": [...], "discussion_points": [...]}},
    {{"phase": "Impact",     "prediction_fields": [...], "discussion_points": [...]}}
  ]
}}

The 5 phase names must appear in this exact order.

--- Policy Context ---
{policy_context}

--- Simulation Target ---
{simulation_target}

--- Stage 1 Predictions (for unit/scale inference) ---
{stage1_predictions}"""


INNER_SCHEMA_TARGET_ONLY_CLAUSE = """\
TARGET-ONLY MODE — IMPORTANT:
- The simulation_target maps to ONE specific phase (typically Outcomes for outcome-style targets, or Impact for long-term-effect targets). For SSF "Number of SCI(E) publications" → Outcomes phase.
- ONLY that phase should have prediction_fields. Make it ONE field — the variable corresponding to the simulation_target. Keep its name, unit, type, description focused on the simulation_target itself.
- All OTHER phases must have prediction_fields = [] (an empty list). They get richer discussion_points (3–5 substantive causal questions) instead, since their output will be narrative-only.
- The narrative reasoning chain still flows through all 5 phases — only the quantitative anchor moves to the target phase exclusively."""


# --- Repeated ToC Inner Schema Builder (fixed Outcomes) ---

INNER_SCHEMA_ENSEMBLE_SYSTEM = """You are an expert in Theory of Change and policy logic-model design. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."""

INNER_SCHEMA_ENSEMBLE_USER = """Design one independent candidate schema for the four non-fixed phases of a policy Theory of Change.

The Outcomes phase is fixed by the researcher and is shown only as the causal endpoint that the other phases should support. Do not return, revise, rename, add to, or remove any Outcome field.

For each phase in this exact order — Inputs, Activities, Outputs, Impact — return:
- prediction_fields: 2–4 concrete, measurable variables. Each field has name (snake_case), unit, type (int | float | str), and description.
- discussion_points: 2–4 questions about mechanisms, constraints, implementation variation, or causal links relevant to that phase.

Requirements:
- Inputs are committed resources or capacities; Activities are actions using them; Outputs are direct products or reach; Impact is longer-term systemic change beyond the fixed short/medium-term Outcomes.
- Every Impact field must specify an explicit measurement horizon at least 3 years after policy implementation (for example, a 3-year rate, 5-year cumulative cost, or 10-year equity gap). A field measured only in the policy year or described merely as "post-policy" is not an Impact.
- Do not restate any fixed Outcome as an Impact by changing its subgroup, unit, aggregation, or wording. In particular, immediate out-of-pocket cost, refill frequency, and medication-coverage/adherence measures remain Outcomes when they are already fixed there.
- Within Impact, propose distinct systemic dimensions where the policy context supports them, such as downstream health events, multi-year public spending, and longer-term equity. Do not return two operationalizations of the same construct.
- Ground fields in the policy context. Preserve policy-stated quantities and constraints exactly when referenced.
- It is valid to infer a new mechanism or quantity for simulation when the policy document does not state it. Clearly describe what the field measures; do not present an inferred quantity as a policy-stated fact.
- Seek a useful alternative operationalization in this independent run, but do not add irrelevant variables merely to appear different.
- Avoid synonyms or trivial reformulations within this run.

Return exactly:
{{
  "phases": [
    {{"phase": "Inputs", "prediction_fields": [...], "discussion_points": [...]}},
    {{"phase": "Activities", "prediction_fields": [...], "discussion_points": [...]}},
    {{"phase": "Outputs", "prediction_fields": [...], "discussion_points": [...]}},
    {{"phase": "Impact", "prediction_fields": [...], "discussion_points": [...]}}
  ]
}}

--- Policy Context ---
{policy_context}

--- Fixed Outcomes (immutable causal endpoint; not part of your response) ---
{fixed_outcomes}

--- Stage 1 Predictions (context for scale only; do not copy unsupported values as policy facts) ---
{stage1_predictions}

Independent candidate run: {run_index} of {run_count}
"""


INNER_SCHEMA_DEDUP_SYSTEM = """You consolidate policy-simulation schema candidates. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."""

INNER_SCHEMA_DEDUP_USER = """Consolidate semantically duplicate candidate fields and discussion points within ONE Theory of Change phase.

Merge items only when they operationalize the same construct at the same causal level. Shared subject matter, correlation, or a possible causal relationship is not sufficient. In particular:
- Never merge fields with different units or data types.
- Keep resources, actions, reach/products, outcomes, and long-term impacts distinct.
- Keep totals, rates, proportions, per-capita measures, and subgroup-specific measures distinct unless they are genuinely interchangeable.
- Prefer a canonical definition that preserves the most specific compatible meaning without inventing information.
- Every candidate ID must appear in exactly one returned group.

Return exactly:
{{
  "field_groups": [
    {{
      "source_ids": ["..."],
      "canonical_field": {{"name": "snake_case", "unit": "...", "type": "int|float|str", "description": "..."}},
      "rationale": "why these are or are not a semantic duplicate"
    }}
  ],
  "discussion_groups": [
    {{
      "source_ids": ["..."],
      "canonical_text": "...",
      "rationale": "why these are or are not a semantic duplicate"
    }}
  ]
}}

Phase: {phase}

Field candidates:
{field_candidates}

Discussion-point candidates:
{discussion_candidates}
"""


INNER_SCHEMA_COMPACT_SYSTEM = """You select a compact, causally coherent Theory of Change schema. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."""

INNER_SCHEMA_COMPACT_USER = """Select a compact schema from existing consolidated candidates for a policy simulation.

Select exactly 3 prediction fields and exactly 3 discussion points for each non-fixed phase: Inputs, Activities, Outputs, and Impact. Evaluate all four phases jointly so that the same or nearly identical construct does not appear in multiple phases.
Every candidate ID may appear at most once in its phase list. Before returning, verify that each selected_field_ids list and each selected_discussion_ids list contains exactly 3 distinct IDs.

Selection principles:
- Inputs: resources or implementation capacities available before action.
- Activities: actions performed using the inputs.
- Outputs: direct products, delivery, coverage, or reach produced by activities.
- Impact: longer-term or systemic consequences downstream of the fixed Outcomes. Select only candidates marked impact_horizon_eligible=true; each must contain explicit evidence of a multi-year, lifetime, or intergenerational horizon.
- The fixed Outcomes already define the policy's short- and medium-term endpoints. Do not select Impact candidates that restate those fixed Outcome constructs in another unit, time label, or wording.
- Prefer fields that form a coherent causal chain and are well supported across independent runs, but support count alone must not override phase validity or construct distinctness.
- Use only the supplied candidate IDs. Do not rename, rewrite, or invent fields.

Return exactly:
{{
  "phases": [
    {{
      "phase": "Inputs",
      "selected_field_ids": ["...", "...", "..."],
      "selected_discussion_ids": ["...", "...", "..."],
      "rationale": "..."
    }},
    {{"phase": "Activities", "selected_field_ids": ["...", "...", "..."], "selected_discussion_ids": ["...", "...", "..."], "rationale": "..."}},
    {{"phase": "Outputs", "selected_field_ids": ["...", "...", "..."], "selected_discussion_ids": ["...", "...", "..."], "rationale": "..."}},
    {{"phase": "Impact", "selected_field_ids": ["...", "...", "..."], "selected_discussion_ids": ["...", "...", "..."], "rationale": "..."}}
  ],
  "cross_phase_rationale": "Explain how construct overlap and phase leakage were avoided."
}}

--- Policy Context ---
{policy_context}

--- Fixed Outcomes (immutable and excluded from selection) ---
{fixed_outcomes}

--- Consolidated Candidates ---
{candidates}
"""


# --- ToC Forward Phase: Initial Post ---

FORWARD_PHASE_POST_SYSTEM = "You are a stakeholder participant in a Theory of Change policy simulation. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

FORWARD_PHASE_POST_USER = """You are reasoning FORWARD through a Theory of Change for a specific policy. The forward direction asks: given the policy's inputs and the prior phases of the causal chain, WHAT WILL HAPPEN at the current phase?

Current phase: {phase}

This phase's variables to estimate (the "prediction_fields"):
{phase_fields}

Discussion points to address in your reasoning:
{discussion_points}

Your prior phases in the forward chain (already estimated upstream by this same group):
{prior_phases}

Simulation target: {simulation_target}

As your persona, predict numerical/textual values for each prediction_field of the current phase. Reason from your persona's worldview — your background, demographics, professional experience, skills, and personal context — and from the upstream phases above. Be concrete and grounded.

CRITICAL — factual anchor:
- Treat any quantitative values explicitly stated in policy_context (resource_logic, expected_outcomes, eligibility_logic, theory_of_change_summary) as GROUND TRUTH. Do NOT re-estimate facts the policy itself has stated.
- If a stated value's scope is ambiguous (e.g., "130M KRW" — per fellow vs total annual), reason from policy_context (resource_logic, theory_of_change_summary, eligibility_logic) to determine the most plausible interpretation, then commit to it consistently across all your variable estimates.
- Only estimate variables that the policy does NOT directly state.

FOCUS — only the simulation_target is quantitatively evaluated:
- The variable that maps to the simulation_target above is the ONLY one whose numeric value will be aggregated, cross-checked, and reported as a prediction. For SSF that is a publications-count variable at the Outcomes phase.
- For all OTHER variables: your numeric estimate is just rough context for downstream reasoning. Don't agonize over precision. A reasonable order-of-magnitude estimate is fine.
- Invest your effort in the **narrative** for non-target variables: explain the causal mechanism, not the exact number. The narrative is the actual product at non-target phases.

Return a JSON object with exactly these fields:
- prediction_values: an object whose keys are the prediction_field names listed above and whose values match the field type (int/float/str). Include EVERY field. For non-target variables, a rough estimate is fine. (object)
- narrative: a concise causal narrative explaining how the prior phases lead to these values at this phase, from your persona's perspective. THIS IS THE PRIMARY OUTPUT for non-target phases. (string)
- evidence: list of concrete reasons supporting your estimates, grounded in your persona's worldview and the prior phases (list of strings)

--- Policy Context ---
{policy_context}

--- Your Persona ---
{persona}"""


# --- ToC Forward Phase: Revise after seeing peers ---

FORWARD_PHASE_REVISE_SYSTEM = "You are a stakeholder participant in a Theory of Change policy simulation. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

FORWARD_PHASE_REVISE_USER = """You posted an initial forward-direction estimate for the current phase. Now you have seen the other participants' initial estimates. Revise (or hold) your numbers and reasoning.

Current phase: {phase}
This phase's prediction_fields: {phase_fields}

Your initial post:
{your_initial_post}

Other participants' initial posts at this phase:
{peer_posts}

Update your estimate honestly — engage with peers' points where they shift your view, hold your ground where they don't. Speaking does NOT require agreement.

CRITICAL — discussion should refine, not amplify:
- If your initial value already clustered with most peers, refine modestly within that cluster — do NOT drift toward an outlier (high or low) just because one peer posted an extreme number.
- A big revision (>20% from your initial) requires NEW evidence or a NEW mechanism, not just peer pressure.
- Within-direction debate should narrow disagreement and converge toward a realistic estimate, not push numbers more extreme.

FOCUS — only the simulation_target is quantitatively evaluated:
- For variables that DON'T map to the simulation_target: don't over-engineer the revise step. If peers cluster around your initial estimate, hold. If they diverge widely on a non-target variable, pick a reasonable middle without lengthy justification — the precise number doesn't matter, the causal narrative does.
- For the simulation_target variable specifically: the revise step matters most here. Engage seriously with peer reasoning about this variable.

Return a JSON object with exactly these fields:
- prediction_values: revised values for every prediction_field (object)
- narrative: revised causal narrative for this phase, including what (if anything) shifted you. Stay focused on mechanism — for non-target variables, the narrative is what we evaluate. (string)
- evidence: list of supporting reasons for the revised estimate (list of strings)

--- Policy Context ---
{policy_context}

--- Your Persona ---
{persona}"""


# --- ToC Backward Phase: Initial Post ---

BACKWARD_PHASE_POST_SYSTEM = "You are a stakeholder participant in a Theory of Change policy simulation. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

BACKWARD_PHASE_POST_USER = """You are reasoning BACKWARD through a Theory of Change for a specific policy.

The backward seed (provided below as `backward_seed`) determines your mode:
- If `backward_seed` is the policy's free-text **goal/aspiration** → PARALLEL/REQUIREMENT-TRACING mode. Question: "what MUST be true at each upstream phase for that aspirational goal to actually hold?"
- If `backward_seed` is a **JSON snapshot of forward-direction Impact values** (the simulation's predicted Impact) → SEQUENTIAL/CAUSAL-ATTRIBUTION mode. Question: "given that forward predicts these Impact values, what upstream conditions must have held to produce this predicted Impact?" In this mode, do NOT re-derive aspirational targets — your job is to trace what produced the predicted outcome, anchored to forward's prediction.

Either way: this is NOT a fresh prediction. It is reasoning backward from a fixed downstream anchor.

Current phase: {phase}

This phase's variables to estimate (the "prediction_fields"):
{phase_fields}

Discussion points to address:
{discussion_points}

Already estimated downstream of this phase in the backward chain (i.e. the LATER ToC phases, which were filled in first by this same group):
{prior_phases}

Backward seed (only used at the Impact phase — read carefully to determine mode):
{backward_seed}

Simulation target: {simulation_target}

As your persona, fill in each prediction_field with the value that MUST hold at this phase for the downstream estimates above (and ultimately the backward seed) to be achieved. Reason from your persona's worldview about what conditions, scale, or activities are NECESSARY — not what you'd freshly predict.

CRITICAL — factual anchor:
- Treat any quantitative values explicitly stated in policy_context (resource_logic, expected_outcomes, eligibility_logic, theory_of_change_summary) as GROUND TRUTH. Do NOT re-estimate facts the policy itself has stated.
- If a stated value's scope is ambiguous (e.g., "130M KRW" — per fellow vs total annual), reason from policy_context (resource_logic, theory_of_change_summary, eligibility_logic) to determine the most plausible interpretation, then commit to it consistently across all your variable estimates.
- Only estimate (or back-derive a requirement for) variables that the policy does NOT directly state.

CRITICAL — bounded requirement reasoning:
- The policy commits specific resources (see resource_logic and eligibility_logic in policy_context). Anchor your required values to what is plausibly achievable within at most ~1.5–3x the committed policy resources. Do NOT scale numbers indefinitely.
- If the policy goal is genuinely unachievable even within that stretch, set your prediction_values at this realistic ceiling and EXPLICITLY call out the resulting shortfall in your narrative (e.g. "the goal would require X but only Y is feasible under stated resources").
- The point of backward reasoning is to surface a credible implementation gap, not to compute an unbounded ideal. A 5x or 10x divergence from committed scope usually means you are over-scaling.

FOCUS — only the simulation_target is quantitatively evaluated:
- The variable that maps to the simulation_target is the ONLY one whose backward goal-required value gets reported as a meaningful diagnostic. For SSF that's the publications-count variable at Outcomes.
- For OTHER variables in this backward chain: your numeric requirement is rough context. Don't agonize over precision. Order-of-magnitude is fine.
- Invest your effort in the **narrative** for non-target variables: what mechanism would need to hold for the goal? The narrative is the product at non-target phases.

Return a JSON object with exactly these fields:
- prediction_values: an object whose keys are the prediction_field names and whose values match the field type. Every field included. For non-target variables, a rough estimate is fine. (object)
- narrative: a concise causal narrative explaining what is REQUIRED at this phase to enable the downstream phases / policy goal. THIS IS THE PRIMARY OUTPUT for non-target phases. (string)
- evidence: list of concrete reasons supporting these required values (list of strings)

--- Policy Context ---
{policy_context}

--- Your Persona ---
{persona}"""


# --- ToC Backward Phase: Revise after seeing peers ---

BACKWARD_PHASE_REVISE_SYSTEM = "You are a stakeholder participant in a Theory of Change policy simulation. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

BACKWARD_PHASE_REVISE_USER = """You posted an initial backward-direction (requirement-tracing) estimate for the current phase. You have now seen the other participants' initial backward estimates for this same phase. Revise (or hold) your required values.

Current phase: {phase}
This phase's prediction_fields: {phase_fields}

Your initial post:
{your_initial_post}

Other participants' initial posts at this phase:
{peer_posts}

Remember: you are reasoning about what MUST be true at this phase for the downstream goal to hold, not what you predict. Update where peers raise valid requirements you missed; hold where their requirements seem unrealistic or off-target.

CRITICAL — bounded requirement reasoning still applies:
- Required values must remain anchored to ~1.5–3x the policy's committed resources at most. If a peer proposed a value much larger than this, treat it skeptically and revise downward unless they have specific feasibility justification.
- Where the goal is genuinely unachievable within the realistic ceiling, hold your value at the ceiling and use the narrative to surface the shortfall.

CRITICAL — discussion should refine, not amplify:
- If your initial value already fell within the realistic ceiling, do NOT drift further upward during revise — only revise upward if a peer raised a feasibility consideration you genuinely missed.
- A big upward revision (>20% from your initial) requires NEW evidence about resource feasibility, not just a peer posting a higher number.
- Within-direction debate should narrow disagreement, not push required values more extreme.

FOCUS — only the simulation_target is quantitatively evaluated:
- For variables that DON'T map to the simulation_target: don't agonize over precision in this revise step. Pick a reasonable middle of the cluster and move on.
- For the simulation_target variable: engage seriously with peer reasoning about its required value.

Return a JSON object with exactly these fields:
- prediction_values: revised required values for every prediction_field (object)
- narrative: revised requirement narrative, including what (if anything) shifted you. For non-target variables, focus narrative on mechanism. (string)
- evidence: list of supporting reasons for the revised requirement (list of strings)

--- Policy Context ---
{policy_context}

--- Your Persona ---
{persona}"""


# --- ToC Final Forward Refinement (per-phase, after seeing backward) ---

FORWARD_FINAL_REFINE_SYSTEM = "You are a forward-direction predictor in a Theory of Change policy simulation, refining your prediction after reviewing backward (goal-required) analysis. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

FORWARD_FINAL_REFINE_USER = """You previously produced a FORWARD-direction prediction (what will plausibly happen) for this phase. A separate backward-direction analysis (independently estimating what would be REQUIRED for the policy goal to hold) has now produced its own estimates for this same phase. You will see both, plus the FULL chains of forward and backward across all phases for context, plus your group's already-refined predictions for upstream phases.

Your task: refine your forward prediction at the current phase, with full causal-chain awareness. The backward analysis is a DIAGNOSTIC — it tells you what the goal demands, NOT what will actually happen. Use it (and the chain context) to surface considerations you may have missed, but DO NOT move toward backward unless you see a concrete mechanism in the policy/world that would push the actual outcome higher (or lower) than your original forward prediction.

Current phase: {phase}

This phase's prediction_fields:
{phase_fields}

--- Refined chain so far (your group's already-refined estimates for upstream phases — USE THIS as the chain context, since these may have corrected upstream values your original forward chain didn't have) ---
{refined_chain_so_far}

--- Full forward chain (revised, all 5 phases — original forward predictions across the chain, for full causal-flow context) ---
{forward_chain_full}

--- Full backward chain (revised, all 5 phases — original backward goal-required analysis across the chain, for full requirement-tracing context) ---
{backward_chain_full}

--- THIS PHASE: Forward consensus (revised) — the prediction baseline you are refining ---
{forward_revised}

--- THIS PHASE: Backward consensus (revised) — the goal-required analysis ---
{backward_revised}

Refinement guidelines:
- Default behavior: HOLD your forward estimate. Backward demanding more does NOT mean reality will deliver more.
- Only revise IF you can identify a specific mechanism, capacity, or constraint surfaced by backward that you genuinely missed — not just because backward's number is bigger.
- If you do revise upward, the revision should be modest and grounded in the new mechanism, not a compromise toward backward's number.
- If backward's analysis confirms your prediction or just reflects an unrealistic goal, hold and explicitly note the implementation gap in your narrative.

FOCUS — only the simulation_target is quantitatively evaluated:
- For non-target variables, refinement is mostly cosmetic — these aren't the predictions we evaluate. Hold unless you spot a clear narrative-level insight worth capturing.
- For the simulation_target variable specifically: this is where refinement matters. If backward surfaced a real mechanism affecting this target value, engage with it.

CHAIN-AWARE REASONING (new): with the full forward and backward chains plus your refined-chain-so-far visible, look for:
- Chain-level inconsistencies (e.g., "forward Activities of 200 projects can't plausibly produce backward's required Outcomes of 3000 publications — the chain breaks somewhere")
- Upstream corrections you've already made (in refined_chain_so_far) that should propagate into this phase's reasoning
- Causal mechanisms backward surfaced across the chain that forward missed (e.g., "backward identifies an X→Y bottleneck at upstream phases that forward overlooked")
- If you make a chain-aware revision, reference the specific upstream phase + variable that motivated it in your narrative.

Return a JSON object with exactly these fields:
- prediction_values: refined values for every prediction_field (object). Many or all may be unchanged.
- narrative: a short narrative explaining what (if anything) you refined and why — explicitly cite which backward observation prompted any change, or explicitly state "held — backward reflects unrealistic goal" if no change. (string)
- evidence: list of supporting reasons for the refined estimate (list of strings)

--- Policy Context ---
{policy_context}

--- Your Persona ---
{persona}"""


# --- ToC Per-(Phase × Stakeholder) Argument Synthesizer ---

PHASE_STAKEHOLDER_SYNTHESIZER_SYSTEM = "You are an expert policy analyst summarizing per-stakeholder arguments at each phase of a Theory of Change simulation. Respond only in valid JSON with no markdown formatting or code blocks. All output string values must be in English."

PHASE_STAKEHOLDER_SYNTHESIZER_USER = """You are producing a structured summary of how each stakeholder group argued at each ToC phase, broken out by direction.

Simulation target: {simulation_target}

The 5 ToC phases (in order): Inputs, Activities, Outputs, Outcomes, Impact.
The 3 directions:
- "forward":  what the stakeholder PREDICTED would happen at this phase (revised post)
- "backward": what the stakeholder said the policy goal would REQUIRE at this phase (revised post)
- "refined":  the stakeholder's FINAL forward position after reviewing backward's analysis (held vs. moved, and why)

For EACH (phase, direction, stakeholder_type) cell, write a 1–3 sentence summary capturing that specific direction's argument for that stakeholder. Be specific — reference concrete numbers and named variables. If a direction was not run for a stakeholder, set its value to an empty string "".

--- Policy-Implementing Organizations (KG context) ---
{org_kg_contexts}

Per-(phase, stakeholder) records — each entry has the persona's forward post + backward post + refined post (when present):
{records}

Return a JSON object with EXACTLY this nested shape:
{{
  "phase_stakeholder_summaries": {{
    "Inputs": {{
      "forward":  {{"<stakeholder_type_1>": "...", "<stakeholder_type_2>": "..."}},
      "backward": {{"<stakeholder_type_1>": "...", "<stakeholder_type_2>": "..."}},
      "refined":  {{"<stakeholder_type_1>": "...", "<stakeholder_type_2>": "..."}}
    }},
    "Activities": {{ "forward": {{...}}, "backward": {{...}}, "refined": {{...}} }},
    "Outputs":    {{ "forward": {{...}}, "backward": {{...}}, "refined": {{...}} }},
    "Outcomes":   {{ "forward": {{...}}, "backward": {{...}}, "refined": {{...}} }},
    "Impact":     {{ "forward": {{...}}, "backward": {{...}}, "refined": {{...}} }}
  }}
}}

All 5 phase keys are required, in that exact order. Each phase must contain all 3 direction keys (forward, backward, refined). Each direction's dict must contain one entry per stakeholder_type that appears in the records (use empty string "" when that stakeholder has no data for that direction)."""


# --- ToC Final Reporter ---

FINAL_TOC_REPORTER_SYSTEM = "You are a senior policy analyst writing a forward-looking trajectory assessment of a specific policy for a policymaker audience. Respond only in valid JSON with no markdown formatting wrappers (the value of narrative_synthesis is itself a markdown-flavored prose string). All output string values, including narrative_synthesis, must be in English."

FINAL_TOC_REPORTER_USER = """You are writing a forward-looking POLICY ANALYSIS based on a Theory of Change simulation. Your audience is a policymaker who needs to understand:
1. How this policy will unfold from inputs to impact (the trajectory).
2. What the policy is doing well at each phase.
3. What it is falling short on at each phase.
4. What concrete actions could close the gaps.

This is NOT a summary of what the simulated personas argued. The personas were the analytical tool. Your output is the policymaker-facing assessment.

Simulation target: {simulation_target}

KEY METHODOLOGICAL FRAMING:
- **Forward** = the predictive estimate of what the policy WILL plausibly produce, given committed resources. This is what's actually likely to happen.
- **Backward** = the goal-implied requirement of what would need to hold for the stated policy goal to be achieved. This is the prescriptive benchmark.
- **Implementation gap** = backward − forward at any given phase × variable. Positive gap = the policy goal demands more than the committed resources will deliver.
- Treat forward as REALITY, backward as the GOAL-IMPLIED YARDSTICK.
- When only one direction was run for a phase, work with what you have and note the missing perspective rather than fabricating it.

Return a JSON object with exactly this field:
- narrative_synthesis (string): a multi-paragraph markdown-flavored policy analysis. Use the section headings below VERBATIM as bolded markdown headers (e.g. "**Overall trajectory**"). Write in prose, not bullet lists, except where the section header explicitly says "bullets". Reference concrete numbers, named phase variables, and named phases.

Required sections inside `narrative_synthesis`, IN THIS ORDER:

**Overall trajectory** — 3–5 sentences telling the story of how the policy unfolds from Inputs through Impact based on forward predictions. What kind of program is this likely to be in practice? What's the headline outcome on the simulation_target?

**Per-phase analysis** — five short paragraphs, one per phase (Inputs, Activities, Outputs, Outcomes, Impact). Each paragraph: (a) what the policy is doing at this phase per forward, with concrete numbers; (b) what the goal would require here per backward, with numbers (or "not analyzed" if backward wasn't run); (c) where the policy is strong at this phase; (d) where it's falling short; (e) downstream implication for the next phase.

**What's working** — 3–5 bullet points of cross-phase strengths. Be specific: name the phase, the variable, the number.

**What's falling short** — 3–5 bullet points of cross-phase weaknesses. Be specific: phase, variable, magnitude of gap.

**Recommendations** — 3–5 bullet points of concrete actions a policymaker could take to close the gaps. Each recommendation should target a specific weakness identified above and either (i) scale resources to close the gap, or (ii) recalibrate the goal to match feasible capacity. Avoid generic advice.

--- Aggregated Final Impact Values (forward predictions, confidence-weighted) ---
{aggregated_final}

--- Implementation Gap Report (aggregated across scenarios) ---
{implementation_gap_report}

--- Per-Scenario Summaries ---
{scenario_summaries}"""


JUDGE_USER = """You are summarizing the outcome of a multi-stakeholder policy simulation debate.

Simulation target: {simulation_target}
Consensus reached: {consensus_reached}
Rounds of debate: {num_rounds}

Synthesize the full debate into a structured summary. Use participants' FINAL round predictions for aggregated_prediction and prediction_range.

Return a JSON object with exactly these fields:
- prediction_range: the range of final predicted values as a string (e.g. "500-680")
- stance_summary: a concise summary of how stances evolved across the debate (string)
- key_agreements: list of points where participants broadly agreed (list of strings)
- key_disagreements: list of points where participants disagreed or held different views (list of strings)

--- Policy ---
{policy_plan}

--- Full Simulation Debate ---
{full_discussion}"""


# --- PersonaScopeDefiner Stage 1: demographic scope (age + education_level) ---

SCOPE_DEFINER_DEMOGRAPHIC_SYSTEM = "You are an expert policy-methodology assistant who selects demographic scopes (age range and education level) for one policy KG actor. You produce ONLY age and education_level tier values; occupation will be selected in a separate stage. Respond only in valid JSON with no markdown formatting or code blocks. All explanatory string values must be in English."

SCOPE_DEFINER_DEMOGRAPHIC_USER = """Define the tiered demographic scope (age_range and education_level only) for one policy KG actor.

Purpose:
- This is Stage 1 of a two-stage candidate-persona scope pipeline.
- A downstream SQL query will use your output to narrow Nemotron-Personas-Korea rows by (age, education_level).
- The narrowed pool is then handed to Stage 2 (occupation selection) — DO NOT pick occupations here.

Tier meaning (rubric):
- "direct"    = best-fit demographic envelope for the actor role.
- "plausible" = additional widening that captures plausible adjacent demographics.
- "weak"      = broadest fallback envelope.

Cumulative semantics (CRITICAL):
- Each tier's effective filter is the UNION of all values up to and including it.
- "plausible.education_level" lists NEW additions that union with direct's set.
- "weak.education_level" lists NEW additions that union with direct + plausible.
- Empty list means "no new values at this tier" -> the cumulative set inherits the previous tier unchanged.
- age_range must be monotonically widening: weak ⊇ plausible ⊇ direct. Each tier's [min, max] must be a SUPERSET of the previous tier's range.

Hard rules:
- age_range is a [min, max] integer pair within [{age_min}, {age_max}].
- education_level values must come EXACTLY from EDUCATION_LIST below.
- LANGUAGE: rationale must be in English. Korean education values stay Korean.

Edge cases:
- If the actor is an organization or institutional position, focus on demographics typical of HUMAN representatives (administrators, officers, faculty). Don't filter as if the org itself were a private individual.
- If the policy text caps age (e.g., "39 or under"), encode it in direct.age_range; widen modestly at plausible/weak.
- If the role doesn't constrain age at all, set direct.age_range to a sensible career-active band; weak may extend toward [{age_min}, {age_max}].
- If education isn't constrained, you may include the full set of EDUCATION_LIST across tiers as appropriate.

Return JSON with exactly this structure:
{{
  "direct": {{
    "education_level": ["<one or more values copied verbatim from EDUCATION_LIST>"],
    "age_range": [<int>, <int>]
  }},
  "plausible": {{
    "education_level": ["<NEW values, may be empty>"],
    "age_range": [<int>, <int>]
  }},
  "weak": {{
    "education_level": ["<NEW values, may be empty>"],
    "age_range": [<int>, <int>]
  }},
  "rationale": "Concise English explanation of how the demographic scope reflects the role's eligibility, exclusions, and policy intent."
}}

--- KG Actor Context ---
{actor_context}

--- Policy Context ---
{policy_context}

--- EDUCATION_LIST ({education_count} values) ---
{education_list}

--- AGE BOUNDS ---
{age_min} <= age <= {age_max}"""


# --- PersonaScopeDefiner Stage 2: occupation selection from demographic-narrowed pools ---

SCOPE_DEFINER_OCCUPATION_SYSTEM = "You are an expert policy-methodology assistant who selects role-fit occupations from candidate pools that are already narrowed to the demographic envelope chosen in Stage 1. Respond only in valid JSON with no markdown formatting or code blocks. All explanatory string values must be in English."

SCOPE_DEFINER_OCCUPATION_USER = """Pick the role-fit occupations for each tier (direct / plausible / weak) of one policy KG actor.

Context:
- This is Stage 2. A previous step locked in age_range and education_level per tier.
- For each tier, you receive a CANDIDATE POOL of occupations that ALREADY co-occur with the tier's age + education_level filter in Nemotron-Personas-Korea. The number after each occupation is the row count within that demographic — a feasibility signal, not a relevance signal.
- The downstream SQL filter will tag each Nemotron row with the most-specific tier whose AND-conjunction (occupation, education_level, age_range) it satisfies.

Tier meaning:
- "direct"    = best-fit occupations for this role.
- "plausible" = additional plausible-adjacent occupations.
- "weak"      = additional fallback occupations.

Cumulative semantics (CRITICAL):
- Each tier emits NEW occupations that broaden the previous tier.
- Don't repeat any occupation across tiers; each occupation appears in at most ONE tier.
- An empty list means "no new occupations at this tier" -> the cumulative occupation set inherits the previous tier unchanged.

Hard rules:
- Each tier may include AT MOST {max_per_tier} NEW occupation values.
- Use only values that appear in the corresponding tier's CANDIDATE POOL below. Copy them EXACTLY.
- count is a feasibility signal: prefer non-trivial counts when possible, but role-fit MEANING is the primary criterion. A 50-row occupation that names the role precisely beats a 1500-row occupation that's loosely related.
- LANGUAGE: rationale in English. Korean occupation values stay Korean.

Selection guidance:
- If no candidate is a true direct fit, you MAY leave direct empty. The widening loop will fill from plausible/weak. But if a candidate is a clear closest-semantic-match (e.g., "X 연구원" for a research role even though no exact "박사후연구원" entry exists), include it in direct.
- Scan the candidate pool for semantic anchors implied by the role: research roles -> entries containing 연구원/연구/기술자/과학; evaluator roles -> 교수/평가/전문가; teaching roles -> 교사/강사; administrative roles -> 행정/공무원/사무.
- For evaluator/committee roles, prefer occupations indicating senior expertise. For direct beneficiaries, prefer occupations that match the policy's target population.

Return JSON with exactly this structure:
{{
  "direct":    ["<= {max_per_tier} occupation strings copied verbatim from DIRECT POOL"],
  "plausible": ["<= {max_per_tier} NEW occupation strings copied verbatim from PLAUSIBLE POOL (none in direct)"],
  "weak":      ["<= {max_per_tier} NEW occupation strings copied verbatim from WEAK POOL (none in direct or plausible)"],
  "rationale": "Concise English explanation of why these occupations match each tier."
}}

--- KG Actor Context ---
{actor_context}

--- Demographic Scope (locked in Stage 1) ---
{demographic_scope}

--- DIRECT POOL ({direct_count} candidates, format: count  occupation) ---
{direct_pool}

--- PLAUSIBLE POOL ({plausible_count} candidates) ---
{plausible_pool}

--- WEAK POOL ({weak_count} candidates) ---
{weak_pool}"""


# --- Policy role actions synthesizer (one call per persona-eligible entity) ---

ROLE_ACTIONS_SYNTHESIZER_SYSTEM = "You synthesize concise English action statements for one policy KG actor, drawn from the related edges in the knowledge graph. Respond only in valid JSON with no markdown formatting or code blocks."

ROLE_ACTIONS_SYNTHESIZER_USER = """Synthesize 3-7 concise English action statements describing what this policy KG actor does, must do, may do, and cannot do, in the context of the simulated policy.

Rules:
- Use ONLY information present in the supplied related_edges and related_nodes. Do NOT invent obligations.
- One statement per distinct action; merge near-duplicate edges. Each statement should be a complete sentence ending with a period.
- Annotate normative force inline using the existing `deontic` field:
    - deontic="must" -> suffix "(required)" at end of sentence
    - deontic="cannot" / "prohibited" -> suffix "(prohibited)"
    - deontic="can" -> suffix "(allowed)"
    - deontic missing or "none" -> no suffix
- Phrase from the actor's perspective in third person ("Postdoctoral researchers submit ...", "The Evaluation Panel evaluates ...").
- Cap at 7 statements. If fewer than 3 distinct actions exist, return all that exist.
- LANGUAGE: all output in English. Korean source_text may be paraphrased for readability but must not change meaning.

Return JSON with exactly this structure:
{{
  "actions": ["<sentence>", "<sentence>", ...]
}}

--- KG Actor Context ---
{actor_context}

--- Deterministic raw action list (for reference; you may merge/polish) ---
{raw_actions}"""


# --- Persona refiner (single call: refined professional_persona + skills_and_expertise + rationale) ---

PERSONA_REFINER_SYSTEM = "You refine one persona's three narrative text fields (`professional_persona`, `skills_and_expertise`, `career_goals_and_ambitions`) by INTEGRATING the institutional role, ECOSYSTEM POSITION (who funds / oversees / collaborates with the focal actor), and norms (must / can / cannot) from a policy knowledge graph into a SINGLE coherent identity. First select ONE integration strategy (A=direct substitution, B=skill transfer, C=coexistence/life-stage, D=incoherent) per persona; then apply it across the three fields so that the persona DEMONSTRATES AWARENESS of their position in the ecosystem (not as a name-drop, but as a worldview shaped by the policy structure). Output language MUST match the original persona text (typically Korean) — do not translate. Preserve personality, values, emotional tone, and individual quirks. Never modify any demographic field (sex, age, education_level, occupation, district, province). Respond only in valid JSON with no markdown formatting or code blocks."

PERSONA_REFINER_USER = """Refine ONE persona's three narrative fields by INTEGRATING the policy-KG-defined role and the must / can / cannot norms INTO the existing narrative as one coherent identity.

TARGET FIELDS (refine these ONLY):
  1) professional_persona
  2) skills_and_expertise
  3) career_goals_and_ambitions

DO-NOT-TOUCH FIELDS (demographics — never modify):
  sex, age, education_level, occupation, district, province
  (and any other anchor in `persona_demographics`)

STEP 1 — SELECT INTEGRATION STRATEGY (one per persona)

Choose ONE primary strategy based on how the persona's Nemotron-given background relates to the policy role. Optionally pair with a secondary if hybrid.

(A) Direct substitution — Nemotron occupation matches the policy role.
    Persona's current identity IS the role; just contextualize within the policy setting.
    Effect: the role appears as the persona's primary professional identity early in each field.

(B) Skill / experience transfer — different occupation, but capabilities transfer.
    Persona's Nemotron background becomes ASSETS they bring to the role (analytical, communication, leadership, technical, etc.).
    Effect: the original occupation experience is reframed as competency leveraged inside the role.

(C) Coexistence / life-stage — persona could plausibly be the Nemotron occupation AND hold the role simultaneously (working professional pursuing study; civic participant in parallel program).
    Effect: the role is one chapter of the persona's life, woven alongside the Nemotron occupation as parallel engagement.

(D) Incoherent — no plausible bridge.
    Use ONLY when occupation field is fundamentally incompatible AND no skill transfers AND coexistence is implausible.
    Effect: leave all three fields verbatim; rationale="no_change_needed:incoherent_fit".

STEP 2 — APPLY THE CHOSEN STRATEGY TO ALL THREE FIELDS

The chosen strategy must drive integration in PP, SE, and CG with one consistent identity. Each field may emphasize different aspects (role identity in PP, capabilities in SE, goals in CG) but the underlying strategy must remain the same.

If a specific field genuinely cannot be touched without breaking coherence (rare — usually the persona-level strategy works across all three), set that field's rationale to "no_change_needed" while keeping the persona-level strategy unchanged.

HARD CONSTRAINT — POSITION RULE (non-negotiable, structural)
The policy role mention MUST appear inside the FIRST sentence of each refined field, as a clause or descriptor. The role MUST NOT appear only in a later sentence, and MUST NOT appear in a new trailing sentence added after the original narrative. This rule is structural — apply it regardless of the original sentence count or length.

If the role cannot be inserted into the first sentence without distorting the persona's original meaning, the strategy is D (incoherent) — do NOT fall back to appending a role-declaring sentence elsewhere.

ALSO INVALID — STACKED TWIN-IDENTITY in the first sentence
If the only way to satisfy the position rule is to glue two identities into the first sentence ("X is both <ROLE> AND <original occupation>"), that is NOT acceptable integration — it is the same two-identity problem displaced into one sentence. Escalate to strategy D instead.

VALID structures (role woven into the first sentence as ONE identity):
- "[Persona] is/works as <ROLE>, who [original first-sentence content...]"
- "[Persona], drawing on [background], serves in <ROLE> by [original first-sentence content...]"

INVALID structures:
- "[Original narrative.] Additionally, [Persona] participates in <ROLE>..." (trailing)
- "[Original narrative unchanged.] As <ROLE>, [Persona] also..." (trailing)
- "[Original narrative.] [New sentence introducing <ROLE>.]" (trailing)
- "[Persona] is <ROLE>, and is also a/an <original occupation>..." (twin labels stacked — escalate to D)
- "[Persona], an <original occupation>, is also a <ROLE>..." (twin labels stacked — escalate to D)

D ESCALATION IS EXPECTED AND NORMAL
Strategy D should be used freely whenever a persona's background does not naturally support strategies A, B, or C. Across a batch or run, a substantial fraction of personas (often 30–60% depending on KG-narrative overlap) should resolve to D. Do NOT bias toward refining every persona — forcing weak integrations defeats the purpose. When in doubt between B and D, prefer D.

EXECUTION RULES
- ONE coherent identity per persona — never produce two parallel selves stitched with a closing role-declaration sentence.
- Output language must match the original persona text (do not translate).
- Preserve personality, values, emotional tone, and quirks — do NOT erase or homogenize.
- Preserve all demographic anchors as factual constants — REFRAME in context but underlying facts do not change.
- Sentence count and total length may shift ±30%. Tighter > bloated.
- Anchor policy-specific wording to `fact` / `Constraint` text (system names, procedures). Do NOT abstract them away.
- Do NOT invent regulations, deadlines, or organizations absent from the listed norms.

SELF-CHECK before output
Read your full draft. Does it read as ONE person with ONE identity (strategy A/B/C applied coherently), or as two parallel selves with a tacked-on role declaration? If the latter, change strategy or rewrite — typically B (skill transfer) or D (incoherent) is the honest fix.

NORM-FIELD MAPPING (suggested, not mandatory)
- professional_persona ← role identity + acknowledgment of key obligations (M*)
- skills_and_expertise ← competencies needed to fulfill the role, including capacity to exercise permissions (C*)
- career_goals_and_ambitions ← goals situated within the activity space the norms define

NORM SELECTION
Under the length-preservation constraint you cannot reflect every norm. Select norms most relevant to this specific persona and field, and record which IDs you used in `kg_evidence`. If a target field already reflects the institutional context, leave it verbatim and set its rationale to "no_change_needed".

OUTPUT FORMAT — return JSON exactly:
{{
  "strategy": "<A|B|C|D>",
  "strategy_secondary": "<A|B|C|null>",
  "strategy_rationale": "<one short sentence: why this strategy fits this persona-role pair>",
  "professional_persona":        "<refined full text in same language as original, or original verbatim if D>",
  "skills_and_expertise":         "<refined full text in same language as original, or original verbatim if D>",
  "career_goals_and_ambitions":  "<refined full text in same language as original, or original verbatim if D>",
  "rationale": {{
    "professional_persona":        "<one short sentence explaining how the chosen strategy was applied here, or 'no_change_needed'>",
    "skills_and_expertise":         "<one short sentence explaining how the chosen strategy was applied here, or 'no_change_needed'>",
    "career_goals_and_ambitions":  "<one short sentence explaining how the chosen strategy was applied here, or 'no_change_needed'>"
  }},
  "kg_evidence": [
    {{"id": "M3", "fact": "<exact fact text from the listed norm>"}},
    {{"id": "C1", "fact": "<exact fact text from the listed norm>"}}
  ]
}}

If strategy == "D", set all three fields to the original verbatim, all rationales to "no_change_needed:incoherent_fit", and kg_evidence to [].

--- POLICY ECOSYSTEM (the policy actors; positions the focal actor in context) ---
The policy involves the following actors. The persona's role is FIXED to the ★ focal actor — do NOT switch them to another actor. Use the ecosystem to understand WHERE the focal actor sits: who funds them, who oversees them, who they collaborate with, what lifecycle phase they operate in.

{policy_ecosystem}

ECOSYSTEM ROLE INTEGRATION (MANDATORY for all non-D personas)

The refined persona MUST demonstrate awareness of their position within the policy ecosystem — not as a name-drop, but as a worldview shaped by who funds, oversees, supports, or collaborates with the focal actor. This shapes how the persona understands their daily work, capabilities, and goals.

Specifically, the refined narrative reflects awareness of (at least ONE of):
- Who AUTHORIZES / OVERSEES the focal actor's activity (e.g., subject to a regulator, governing body, contractual counterparty, or review authority named in the KG).
- Where the focal actor sits OPERATIONALLY (e.g., within service delivery, administration, payment, compliance, communication, or dispute-resolution processes supported by the KG).
- The POLICY LIFECYCLE phase the focal actor operates in (e.g., implementation, service use, monitoring, enforcement, payment, evaluation, or remedy).
- COLLABORATIVE or adversarial ties to other ecosystem actors, stated only when supported by the KG relations.

FIELD DISTRIBUTION (which ecosystem aspect to weave into each field)
- professional_persona ← position in the ecosystem (who oversees them, where they sit).
- skills_and_expertise ← capabilities for navigating the actual ecosystem interactions in the KG (implementation, communication, compliance, service delivery, payment, review, or dispute resolution).
- career_goals_and_ambitions ← goals positioned within the policy lifecycle (where in the ecosystem they aim to grow).

GENERIC EXAMPLE (the refined output must be in the SAME LANGUAGE as the original — the example below is in English only to illustrate the integration pattern)
- Original first sentence: "She manages complex operational cases."
- After ecosystem integration: "As the focal policy-role representative, she manages complex operational cases under the oversight and process constraints stated in the KG."
The persona's focal role is retained; ecosystem awareness (oversight, operational position, lifecycle, and actor relations) is woven into the same sentence as ONE worldview. Apply the equivalent integration in the original language of each persona.

STRATEGY DECISION with ecosystem
- B may apply when the persona's background brings useful capabilities for a KG-supported policy activity (e.g., case analysis → compliance review; project management → multi-actor implementation coordination).
- D when the persona's background is fundamentally incompatible with the focal role AND no ecosystem-navigation capability transfers either.

RULES
- The persona's stakeholder_type is IMMUTABLE. Ecosystem awareness ≠ role reassignment.
- One coherent identity per persona — do NOT blend two roles in the first sentence ("X is <FOCAL> and also a <other actor>") — twin-identity anti-pattern; escalate to D.
- `kg_evidence` MUST cite FOCAL actor norm IDs (M*/C*/X*) only. Ecosystem actor refs go into refined text or strategy_rationale, NEVER into kg_evidence.
- Do NOT invent organizations, deadlines, or regulations absent from the ecosystem or focal-actor norms below.
- ±30% length constraint still applies. Aim for INTEGRATED ecosystem awareness in the persona's original language, not appended name-drops.

--- KG Actor Context (focal actor; role + norms — use these IDs for kg_evidence citations) ---
Each norm has a stable ID (M*=must, C*=can, X*=cannot). Only norms where this actor is the SUBJECT are listed. `activation_condition` is intentionally omitted — focus on role + norm, not eligibility.

{actor_context}

--- Persona professional_persona (verbatim) ---
{professional_persona}

--- Persona skills_and_expertise (verbatim) ---
{skills_and_expertise}

--- Persona career_goals_and_ambitions (verbatim) ---
{career_goals_and_ambitions}

--- Persona key demographics (read-only; never modify) ---
{persona_demographics}"""


# --- Batched variant — same instructions but processes N personas sharing one
#     actor_context. Saves prompt tokens (actor_context sent once) and round-trips.

PERSONA_REFINER_BATCH_SYSTEM = "You refine multiple personas' three narrative text fields (`professional_persona`, `skills_and_expertise`, `career_goals_and_ambitions`) by INTEGRATING the institutional role, ECOSYSTEM POSITION (who funds / oversees / collaborates with the focal actor), and norms (must / can / cannot) from a policy knowledge graph into a SINGLE coherent identity per persona. All personas share the SAME KG actor context but SELECT THE STRATEGY (A=direct substitution, B=skill transfer, C=coexistence/life-stage, D=incoherent) PER PERSONA based on each persona's Nemotron background — different personas in the batch may pick different strategies. Every non-D persona's refined narrative must DEMONSTRATE AWARENESS of their position in the ecosystem (not as a name-drop, but as a worldview shaped by the policy structure). Output language MUST match each persona's original text (do not translate). Preserve personality, values, emotional tone, and quirks per persona. Never modify any demographic field. Respond only in valid JSON with no markdown formatting or code blocks."

PERSONA_REFINER_BATCH_USER = """Refine {n_personas} personas. All share the same policy-KG-defined role and the must / can / cannot norms listed below; tailor the refinement to each persona individually.

TARGET FIELDS (refine these ONLY, per persona):
  1) professional_persona
  2) skills_and_expertise
  3) career_goals_and_ambitions

DO-NOT-TOUCH FIELDS (demographics — never modify):
  sex, age, education_level, occupation, district, province
  (and any other anchor in each persona's demographics)

STEP 1 — SELECT INTEGRATION STRATEGY PER PERSONA (independent per persona):

(A) Direct substitution — Nemotron occupation matches the policy role.
(B) Skill / experience transfer — different occupation, but capabilities transfer to the role.
(C) Coexistence / life-stage — persona can plausibly be both Nemotron-occupation AND role-holder in parallel.
(D) Incoherent — no plausible bridge; leave all three fields verbatim.

STEP 2 — APPLY THE CHOSEN STRATEGY TO ALL THREE FIELDS OF THAT PERSONA

One consistent identity per persona. Different personas in this batch may use different strategies — do NOT homogenize.

HARD CONSTRAINT — POSITION RULE (non-negotiable, applies per persona, per field)
The policy role mention MUST appear inside the FIRST sentence of each refined field, as a clause or descriptor. The role MUST NOT appear only in a later sentence, and MUST NOT appear in a new trailing sentence added after the original narrative.

If the role cannot be inserted into the first sentence without distorting the persona's original meaning, the strategy for that persona is D (incoherent) — do NOT fall back to appending a role-declaring sentence elsewhere.

ALSO INVALID — STACKED TWIN-IDENTITY in the first sentence
If satisfying the position rule requires gluing two identities into the first sentence ("X is both <ROLE> AND <original occupation>"), that is NOT acceptable integration. Escalate to strategy D for that persona.

VALID structures: role woven as a clause inside the first sentence forming ONE coherent identity (role-as-identity, role-via-skill-transfer, or role-as-life-stage forms).
INVALID structures:
- Role appears only after the first sentence (trailing append).
- "Additionally, ...", "Also, ...", or any new sentence that introduces the role only at the end.
- Twin labels stacked in the first sentence ("X is <ROLE> and also a/an <original occupation>...").
These outputs must be replaced by strategy D instead.

D ESCALATION IS EXPECTED AND NORMAL
Across the batch, a substantial fraction of personas (often 30–60% depending on KG-narrative overlap) should resolve to D. Do NOT bias toward refining every persona — forcing weak integrations defeats the purpose. When in doubt between B and D, prefer D.

EXECUTION RULES (per persona)
- ONE coherent identity per persona — never two parallel selves stitched together with a closing role-declaration sentence.
- Output language must match each persona's original text (do not translate).
- Preserve personality, values, emotional tone, and quirks per persona — do NOT erase or homogenize across the batch.
- Preserve all demographic anchors as factual constants — REFRAME in context but underlying facts do not change.
- Sentence count and total length may shift ±30% per persona. Tighter > bloated.
- Anchor policy-specific wording to `fact` / `Constraint` text. Do NOT abstract policy-specific terms away.
- Do NOT invent regulations, deadlines, or organizations not present in the listed norms.
- Tailor each persona individually — different strategies, different norm citations, different phrasings.

NORM-FIELD MAPPING (suggested, not mandatory)
- professional_persona ← role identity + acknowledgment of key obligations (M*)
- skills_and_expertise ← competencies needed to fulfill the role, including capacity to exercise permissions (C*)
- career_goals_and_ambitions ← goals situated within the activity space the norms define

NORM SELECTION
Per persona, select 1-3 norms most relevant and cite their IDs in `kg_evidence`. Different personas SHOULD typically cite different norms — healthy diversity. If a single field cannot be touched without breaking coherence, set that field's rationale to "no_change_needed" while keeping the persona-level strategy.

SELF-CHECK before output
For each persona, ask: "Does this read as ONE identity, or two parallel selves with a tacked-on role declaration?" If the latter, change strategy (often B or D) or rewrite.

NORM SELECTION
Per persona, select 1-3 norms most relevant and cite their IDs in `kg_evidence`. Different personas SHOULD typically cite different norms — healthy diversity. If a single field cannot be touched without breaking coherence, set that field's rationale to "no_change_needed" while keeping the persona-level strategy.

SELF-CHECK before output
For each persona, ask: "Does this read as ONE identity, or two parallel selves with a tacked-on role declaration?" If the latter, change strategy (often B or D) or rewrite.

OUTPUT FORMAT — return JSON exactly:
{{
  "results": [
    {{
      "persona_id": "<echo the input persona_id exactly>",
      "strategy": "<A|B|C|D>",
      "strategy_secondary": "<A|B|C|null>",
      "strategy_rationale": "<one short sentence: why this strategy fits this persona-role pair>",
      "professional_persona":        "<refined full text in same language as original, or original verbatim if strategy=D>",
      "skills_and_expertise":         "<refined full text in same language as original, or original verbatim if strategy=D>",
      "career_goals_and_ambitions":  "<refined full text in same language as original, or original verbatim if strategy=D>",
      "rationale": {{
        "professional_persona":        "<one short sentence: how the chosen strategy was applied here, or 'no_change_needed'>",
        "skills_and_expertise":         "<one short sentence: how the chosen strategy was applied here, or 'no_change_needed'>",
        "career_goals_and_ambitions":  "<one short sentence: how the chosen strategy was applied here, or 'no_change_needed'>"
      }},
      "kg_evidence": [
        {{"id": "M3", "fact": "<exact fact text from the listed norm>"}}
      ]
    }}
  ]
}}

CRITICAL: `results` MUST contain exactly {n_personas} entries, in the SAME ORDER as the input personas. Each `persona_id` in the output MUST exactly echo the matching input `persona_id`.

--- POLICY ECOSYSTEM (the policy actors; positions the focal actor in context) ---
All {n_personas} personas share the same focal actor (★ below). The ecosystem lists the other actors in the policy. The focal actor is FIXED — do NOT reassign any persona to another actor. Use the ecosystem to understand WHERE the focal actor sits: who funds them, who oversees them, who they collaborate with, what lifecycle phase they operate in.

{policy_ecosystem}

ECOSYSTEM ROLE INTEGRATION (MANDATORY for all non-D personas; applies per persona)

Each non-D persona's refined narrative MUST demonstrate awareness of their position within the policy ecosystem — not as a name-drop, but as a worldview shaped by who funds, oversees, supports, or collaborates with the focal actor.

For each non-D persona, the refined narrative reflects awareness of (at least ONE of):
- Who AUTHORIZES / OVERSEES the focal actor's activity, using only regulators, governing bodies, contractual counterparties, or review authorities present in the KG.
- Where the focal actor sits OPERATIONALLY within service delivery, administration, payment, compliance, communication, or dispute-resolution processes.
- The POLICY LIFECYCLE phase the focal actor operates in, such as implementation, service use, monitoring, enforcement, payment, evaluation, or remedy.
- COLLABORATIVE or adversarial ties to other ecosystem actors, stated only when supported by the KG relations.

Different personas SHOULD typically weave in DIFFERENT ecosystem aspects — do NOT homogenize the batch into the same ecosystem phrase. Each persona's ecosystem hook should fit their individual background.

FIELD DISTRIBUTION (which ecosystem aspect to weave into each field)
- professional_persona ← position in the ecosystem (who oversees them, where they sit).
- skills_and_expertise ← capabilities for navigating the actual ecosystem interactions in the KG (implementation, communication, compliance, service delivery, payment, review, or dispute resolution).
- career_goals_and_ambitions ← goals positioned within the policy lifecycle (where in the ecosystem they aim to grow).

GENERIC EXAMPLE (the refined output must be in the SAME LANGUAGE as each persona's original — the example below is in English only to illustrate the integration pattern)
- Original first sentence: "She manages complex operational cases."
- After ecosystem integration: "As the focal policy-role representative, she manages complex operational cases under the oversight and process constraints stated in the KG."
The persona's focal role is retained; ecosystem awareness (oversight, operational position, lifecycle, and actor relations) is woven into the same sentence as ONE worldview. Apply the equivalent integration in the original language of each persona.

STRATEGY DECISION with ecosystem
- B may apply when a persona's background brings useful capabilities for a KG-supported policy activity (e.g., case analysis → compliance review; project management → multi-actor implementation coordination).
- D when a persona's background is fundamentally incompatible with the focal role AND no ecosystem-navigation capability transfers either.

RULES (per persona)
- Every persona's stakeholder_type is IMMUTABLE. Ecosystem awareness ≠ role reassignment.
- One coherent identity per persona — do NOT blend two roles in the first sentence ("X is <FOCAL> and also a <other actor>") — twin-identity anti-pattern; escalate that persona to D.
- `kg_evidence` MUST cite FOCAL actor norm IDs (M*/C*/X*) only. Ecosystem actor refs go into refined text or strategy_rationale, NEVER into kg_evidence.
- Do NOT invent organizations, deadlines, or regulations absent from the ecosystem or focal-actor norms below.
- ±30% length constraint still applies per persona. Aim for INTEGRATED ecosystem awareness in each persona's original language, not appended name-drops.

--- KG Actor Context (shared by all {n_personas} personas; focal actor — cite these IDs in kg_evidence) ---
Each norm has a stable ID (M*=must, C*=can, X*=cannot). Only norms where this actor is the SUBJECT are listed. `activation_condition` is intentionally omitted — focus on role + norm, not eligibility.

{actor_context}

--- Personas to refine ({n_personas} total) ---
{personas_block}"""


# --- Persona refiner EXTENDED (R2) was removed. R2 produced goals_in_simulation
#     and stance, which are no longer persona fields. Use R1 (PERSONA_REFINER_USER)
#     for text-only refinement. ---


# --- Legacy single-stage Definer prompt (kept for fallback / ablation) ---

SCOPE_DEFINER_SYSTEM = "You are an expert policy-methodology assistant who selects Nemotron-Personas-Korea filter scopes for one policy KG actor. Respond only in valid JSON with no markdown formatting or code blocks. All explanatory string values must be in English."

SCOPE_DEFINER_USER = """Define a tiered candidate-persona scope for one policy KG actor.

Purpose:
- A downstream DuckDB query will hard-filter the Nemotron-Personas-Korea dataset using your output as IN-list / BETWEEN clauses, then walk tiers (direct -> plausible -> weak) until a minimum pool size is met.
- Your job is to pick the values that should appear in each tier across three axes: occupation, education_level, age_range.
- Each tier is an AND-conjunction (`occupation IN ... AND education_level IN ... AND age BETWEEN ...`).

Tier meaning (rubric):
- "direct"    = best-fit values that strongly match the actor role's responsibilities, eligibility, and domain.
- "plausible" = additional values that broaden the pool to plausible adjacent fits (different but reasonable matches).
- "weak"      = additional fallback values that only apply when direct + plausible together are too small.

Cumulative semantics (CRITICAL):
- The values you list at "plausible" are the NEW additions; they will be unioned with "direct" before SQL evaluation.
- The values at "weak" are NEW additions that union with direct + plausible.
- Do NOT repeat values across tiers. Each value should appear at most once.
- An empty list at a tier means "no new values for this axis at this tier" -> the cumulative set inherits the previous tier unchanged.

Hard rules:
- Each tier may include AT MOST 3 NEW occupation values. Pick the most useful ones; do not pad.
- Use only values that appear in the provided OCCUPATION_LIST and EDUCATION_LIST. Copy them EXACTLY.
- age_range must be a [min, max] integer pair with min <= max, both within [{age_min}, {age_max}]. Each tier's range must be a SUPERSET of the previous tier's range (i.e. monotonically widening).
- If the role does not constrain a particular axis at all, emit the broadest reasonable scope at "direct" and you may leave "plausible"/"weak" empty for that axis (they will inherit).
- LANGUAGE: rationale must be in English. Korean occupation/education values must stay in their original Korean form.

Edge cases:
- If the actor is an organization, official position, or evaluator (not an individual beneficiary), pick occupations and education levels that fit plausible HUMAN representatives (administrators, officers, reviewers, faculty, technical staff). Do not filter as if they were a private statutory beneficiary.
- For evaluator/committee roles, prioritize education_level "대학원" or "4년제 대학교" and occupations indicating research, evaluation, or domain expertise. Age ranges should reflect career-experienced adults (typically 30-65).
- For direct beneficiary roles with explicit age caps in the policy text, encode the age cap directly in "direct" age_range; widen modestly at "plausible" and "weak".

Return JSON with exactly this structure:
{{
  "direct": {{
    "occupation": ["<<=3 Korean occupation strings copied verbatim from OCCUPATION_LIST>"],
    "education_level": ["<one or more values copied verbatim from EDUCATION_LIST>"],
    "age_range": [<int>, <int>]
  }},
  "plausible": {{
    "occupation": ["<<=3 Korean occupation strings, NEW (not in direct)>"],
    "education_level": ["<NEW education_level values, may be empty>"],
    "age_range": [<int>, <int>]
  }},
  "weak": {{
    "occupation": ["<<=3 Korean occupation strings, NEW>"],
    "education_level": ["<NEW values, may be empty>"],
    "age_range": [<int>, <int>]
  }},
  "rationale": "Concise English explanation grounded in the KG actor context and tier semantics."
}}

--- KG Actor Context ---
{actor_context}

--- Policy Context ---
{policy_context}

--- OCCUPATION_LIST ({occupation_count} values, frequency-sorted) ---
{occupation_list}

--- EDUCATION_LIST ({education_count} values) ---
{education_list}

--- AGE BOUNDS ---
{age_min} <= age <= {age_max}"""


# --- ToC Refinement Judge (with org_kg_contexts) ---

REFINEMENT_JUDGE_SYSTEM = "You are a stakeholder participant in a Theory of Change policy simulation, now acting as a judge to produce your final estimate after seeing both forward and backward analyses. Respond only in valid JSON with no markdown formatting or code blocks."

REFINEMENT_JUDGE_USER = """You have two independent analyses of the same phase, produced by the same group from different angles:

- **Analysis A (Bottom-up)**: Started from committed resources and estimated upward through the causal chain — "given what we have, what will we produce?"
- **Analysis B (Top-down)**: Started from the policy goal and traced backward what each phase needs to deliver — "given what we want, what does this phase need to achieve?"

Neither analysis is inherently more realistic than the other. Your job: judge which analysis — or what synthesis — produces the most reasonable estimate for THIS phase.

Current phase: {phase}
Simulation target: {simulation_target}

This phase's prediction_fields:
{phase_fields}

Discussion points to address in your judgment:
{discussion_points}

--- Analysis A (Bottom-up estimate) ---
Group discussion summary:
{forward_summary}

Your own bottom-up post:
{own_forward_post}

--- Analysis B (Top-down requirement) ---
Group discussion summary:
{backward_summary}

Your own top-down post:
{own_backward_post}

{prior_refinement_context}

{sequential_context}

Judgment guidelines:
- BOTH analyses are legitimate. Do NOT default to either one.
- For each variable, ask: "What does a realistic, well-executed version of this program actually achieve at this phase?" — use your persona's domain expertise to judge.
- If the two analyses are close (within 10-15%), that's strong convergence — pick within that range confidently.
- If they diverge significantly, reason about WHY they diverge and which assumptions are more defensible.
- If prior participants at this phase have already posted their judgments (shown above), engage with their reasoning.

CRITICAL — factual anchor:
- Treat any quantitative values explicitly stated in policy_context (resource_logic, expected_outcomes, eligibility_logic) as GROUND TRUTH.
- Only judge variables that the policy does NOT directly state.

Return a JSON object with exactly these fields:
- prediction_values: your final values for every prediction_field (object)
- judgment: your reasoning about which analysis is more realistic and why — reference specific mechanisms or evidence from your persona's expertise (string)
- narrative: your final reasoning for these values at this phase (string)
- evidence: list of supporting reasons (list of strings)

--- Policy Context ---
{policy_context}

--- Policy-Implementing Organizations (KG context) ---
{org_kg_contexts}

--- Your Persona ---
{persona}"""


# --- ToC Sequential Forward Post (with org_kg_contexts) ---

SEQUENTIAL_FORWARD_POST_SYSTEM = "You are a stakeholder participant in a Theory of Change policy simulation. Respond only in valid JSON with no markdown formatting or code blocks."

SEQUENTIAL_FORWARD_POST_USER = """You are reasoning FORWARD through a Theory of Change for a specific policy. The forward direction asks: given the policy's inputs and the prior phases of the causal chain, WHAT WILL HAPPEN at the current phase?

Current phase: {phase}

This phase's variables to estimate (the "prediction_fields"):
{phase_fields}

Discussion points to address in your reasoning:
{discussion_points}

{prior_phase_context}

{sequential_context}

{forward_seed}

Simulation target: {simulation_target}

As your persona, predict numerical/textual values for each prediction_field of the current phase. Reason from your persona's worldview — your background, demographics, professional experience, skills, and personal context — and from the upstream phases above. Be concrete and grounded.

INDEPENDENCE IS ESSENTIAL:
1. FIRST: Form your OWN estimate based solely on your persona's worldview, expertise, and the prior-phase context. Your stakeholder type gives you a UNIQUE perspective — a postdoc sees productivity differently than a ministry official or evaluator.
2. THEN: If other participants have posted (shown above), engage with their reasoning. But do NOT adopt their numbers just because they posted first. If your independent reasoning yields a different estimate, STATE IT and explain why your perspective leads to a different conclusion.
3. DISAGREEMENT IS VALUABLE: Different stakeholders legitimately estimate differently. A researcher may assume 3-4 publications per fellow based on personal experience; an evaluator may assume 1.5 based on historical program averages. Both are valid — commit to YOUR number with YOUR reasoning.

CRITICAL — factual anchor:
- Treat any quantitative values explicitly stated in policy_context (resource_logic, expected_outcomes, eligibility_logic, theory_of_change_summary) as GROUND TRUTH. Do NOT re-estimate facts the policy itself has stated.
- If a stated value's scope is ambiguous (e.g., "130M KRW" — per fellow vs total annual), reason from policy_context (resource_logic, theory_of_change_summary, eligibility_logic) to determine the most plausible interpretation, then commit to it consistently across all your variable estimates.
- Only estimate variables that the policy does NOT directly state.

CRITICAL — TIME WINDOW must match `simulation_target`:
- The simulation_target ABOVE specifies the exact time window the simulation cares about (e.g. "2021-2022" = a 2-year window, NOT the full multi-year program lifetime).
- All your prediction_field values MUST be expressed in THAT SAME TIME WINDOW. If a variable name encodes a window (e.g. `sci_e_publications_2021_2022`, `number_of_sci_e_publications_in_2021_2022`), respect it literally — do NOT report a 5-year or full-program cumulative value for a 2-year-windowed field.
- COHORT/ACTIVE-POOL math: when the policy describes a multi-year cohort flow (e.g. "200 fellows per year over 5 years = 1,000 total"), the active cohort during the simulation_target's window is NOT the full 1,000. Only the cohorts that started early enough to be active inside the window contribute. For a "2021-2022" target with a program starting in 2020 at 200 fellows/year, the active pool is roughly the first 1-2 cohorts (≈200–400 fellows × ≈1–2 active years each = ≈400–700 fellow-years), NOT 1,000 fellows × 1.8 years = 1,800 fellow-years.
- If you find yourself multiplying a per-recipient rate by a denominator that includes recipients NOT YET ACTIVE in the target window, STOP and rescale to only the active sub-cohort.
- Phase-as-lifetime-stage framing (Impact = long-term end state) does NOT override the simulation_target's time window. If the target says "2021-2022", every variable, even those at the Impact phase, is computed for the 2021-2022 window — not the program's eventual long-term outcome.
- Express any uncertainty about the time window in `narrative` and `evidence`, but do NOT silently inflate values to the program-lifetime scale.

FOCUS — only the simulation_target is quantitatively evaluated:
- The variable that maps to the simulation_target above is the ONLY one whose numeric value will be aggregated, cross-checked, and reported as a prediction.
- For all OTHER variables: your numeric estimate is just rough context for downstream reasoning. Don't agonize over precision.
- Invest your effort in the **narrative** for non-target variables.

Return a JSON object with exactly these fields:
- prediction_values: an object whose keys are the prediction_field names listed above and whose values match the field type (int/float/str). Include EVERY field. (object)
- narrative: a concise causal narrative explaining how the prior phases lead to these values at this phase, from your persona's perspective. If prior speakers at this phase raised points, address them. (string)
- evidence: list of 3-4 concrete reasons supporting your estimates. Each entry MUST be 1-2 sentences only — keep it concise. (list of strings, length 3-4)

The output MUST follow EXACTLY this JSON shape (all three top-level keys are MANDATORY — do not omit `narrative` or `evidence`):
{{
  "prediction_values": {{"field_name_1": value_1, "field_name_2": value_2}},
  "narrative": "...",
  "evidence": ["reason 1 (1-2 sentences)", "reason 2 (1-2 sentences)", "reason 3 (1-2 sentences)"]
}}

--- Policy Context ---
{policy_context}

--- Policy-Implementing Organizations (KG context) ---
{org_kg_contexts}

--- Your Persona ---
{persona}"""


# --- ToC Sequential Backward Post (with org_kg_contexts) ---

SEQUENTIAL_BACKWARD_POST_SYSTEM = "You are a stakeholder participant in a Theory of Change policy simulation. Respond only in valid JSON with no markdown formatting or code blocks."

SEQUENTIAL_BACKWARD_POST_USER = """You are reasoning BACKWARD through a Theory of Change for a specific policy.

The backward seed (provided below as `backward_seed`) determines your mode:
- If `backward_seed` is the policy's free-text **goal/aspiration** → PARALLEL/REQUIREMENT-TRACING mode. Question: "what MUST be true at each upstream phase for that aspirational goal to actually hold?"
- If `backward_seed` is a **JSON snapshot of forward-direction Impact values** → SEQUENTIAL/CAUSAL-ATTRIBUTION mode. Question: "given that forward predicts these Impact values, what upstream conditions must have held to produce this predicted Impact?"

Either way: this is NOT a fresh prediction. It is reasoning backward from a fixed downstream anchor.

Current phase: {phase}

This phase's variables to estimate (the "prediction_fields"):
{phase_fields}

CRITICAL — phase identity preservation:
- The keys of `prediction_values` in your response MUST be EXACTLY the field names listed in `prediction_fields` above for the CURRENT phase ({phase}).
- DO NOT include any fields from prior phases (e.g., do not echo Outcomes-phase fields like publication counts when the current phase is Outputs).
- DO NOT re-emit fields you already produced in earlier phases of this backward iteration.
- Generate values ONLY for THIS phase's fields. Treat prior_phase_context below as background information, not as a field template.

Discussion points to address:
{discussion_points}

{prior_phase_context}

{sequential_context}

Backward seed (only used at the Impact phase — read carefully to determine mode):
{backward_seed}

Simulation target: {simulation_target}

As your persona, fill in each prediction_field with the value that MUST hold at this phase for the downstream estimates to be achieved. Reason from your persona's worldview about what conditions, scale, or activities are NECESSARY.

INDEPENDENCE IS ESSENTIAL:
1. FIRST: Form your OWN requirement estimate based solely on your persona's worldview and expertise. Your stakeholder type gives you a UNIQUE perspective on what is truly required — a researcher knows what productivity demands look like; a ministry official knows what resource constraints are realistic.
2. THEN: If other participants have posted (shown above), engage with their reasoning. But do NOT adopt their numbers just because they posted first. If your independent reasoning yields a different requirement, STATE IT and explain why.
3. DISAGREEMENT IS VALUABLE: Different stakeholders legitimately disagree on what is required. Commit to YOUR requirement with YOUR reasoning.

CRITICAL — factual anchor:
- Treat any quantitative values explicitly stated in policy_context as GROUND TRUTH. Do NOT re-estimate facts the policy itself has stated.
- Only estimate (or back-derive a requirement for) variables that the policy does NOT directly state.

CRITICAL — bounded requirement reasoning:
- Anchor your required values to what is plausibly achievable within at most ~1.5–3x the committed policy resources. Do NOT scale numbers indefinitely.
- The point of backward reasoning is to surface a credible implementation gap, not to compute an unbounded ideal.

CRITICAL — TIME WINDOW must match `simulation_target`:
- The simulation_target ABOVE specifies the exact time window the simulation cares about (e.g. "2021-2022" = a 2-year window, NOT the full multi-year program lifetime).
- All your prediction_field values MUST be expressed in THAT SAME TIME WINDOW. If a variable name encodes a window (e.g. `sci_e_publications_2021_2022`, `fellows_in_2022`), respect it literally — do NOT report a 5-year cumulative value for a 2-year-windowed field.
- The Impact / Outcomes phases of a Theory of Change are normally about long-term aspirations. When a prior phase's backward reasoning was framed at the FULL program lifetime (e.g. "300 fellows become independent researchers" over 5 years), you MUST rescope the cohort denominator to the simulation_target's window BEFORE back-calculating. Example: if the program runs 5 years × 200 fellows/year = 1,000 total fellows, but the simulation_target is "2021-2022", then the active cohort for THIS time window is roughly the first 1–2 cohorts (≈200–400 fellows), NOT 1,000.
- If you find yourself dividing by the FULL program total to back-calculate a windowed variable, STOP and rescale.
- Express any uncertainty about the time window explicitly in `narrative` and `evidence`, but do NOT silently inflate values to the program-lifetime scale.

FOCUS — only the simulation_target is quantitatively evaluated:
- For OTHER variables: your numeric requirement is rough context. Order-of-magnitude is fine.
- Invest your effort in the **narrative** for non-target variables.

Return a JSON object with exactly these fields:
- prediction_values: an object whose keys are the prediction_field names and whose values match the field type. Every field included. (object)
- narrative: a concise causal narrative explaining what is REQUIRED at this phase to enable the downstream phases / policy goal. If prior speakers at this phase raised points, address them. (string)
- evidence: list of 3-4 concrete reasons supporting these required values. Each entry MUST be 1-2 sentences only — keep it concise. (list of strings, length 3-4)

The output MUST follow EXACTLY this JSON shape (all three top-level keys are MANDATORY — do not omit `narrative` or `evidence`):
{{
  "prediction_values": {{"field_name_1": value_1, "field_name_2": value_2}},
  "narrative": "...",
  "evidence": ["reason 1 (1-2 sentences)", "reason 2 (1-2 sentences)", "reason 3 (1-2 sentences)"]
}}

--- Policy Context ---
{policy_context}

--- Policy-Implementing Organizations (KG context) ---
{org_kg_contexts}

--- Your Persona ---
{persona}"""


# --- ToC Phase Holistic Summarizer (with org_kg_contexts) ---

PHASE_SUMMARY_SYSTEM = "You are an expert policy analyst producing concise holistic summaries of stakeholder deliberations. Respond only in valid JSON with no markdown formatting or code blocks."

PHASE_SUMMARY_USER = """Summarize the following stakeholder deliberation at one phase of a Theory of Change simulation.

Phase: {phase}
Direction: {direction}
Simulation target: {simulation_target}

Variables estimated at this phase:
{phase_fields}

--- Policy-Implementing Organizations (KG context) ---
{org_kg_contexts}

All participant posts (in posting order):
{all_posts}

Produce a HOLISTIC summary — do NOT attribute arguments to individual personas. Instead capture:
1. Key themes and causal mechanisms the group discussed
2. Points of consensus (where estimates and reasoning aligned)
3. Points of disagreement (where estimates or reasoning diverged)
4. The range and central tendency of numeric predictions

Keep it concise: 3-5 sentences. Include specific numbers (prediction ranges, key values) but no persona names.

Return a JSON object with exactly this field:
- summary: the holistic summary string (string)"""


# --- ToC Sequential Forward Post: Korean natural reviewer variant ---
#
# Kept separate from SEQUENTIAL_FORWARD_POST_* so A/B comparisons can switch
# prompt style through config without changing the legacy prompt text.

SEQUENTIAL_FORWARD_POST_SYSTEM_KO_NATURAL = (
    "You are a stakeholder reviewer contributing to a Theory of Change policy assessment. "
    "Respond only in valid JSON with no markdown formatting or code blocks. "
    "JSON keys must remain exactly as requested in English. "
    "All user-facing string values in narrative and evidence must be written in natural Korean."
)


# --- ToC phase-local pathway branching ---
#
# This prompt is intentionally additive. The legacy global-stance prompts above
# remain available for reproducing the original three full-run conditions.
BRANCH_MECHANISM_PLANNER_SYSTEM = """You are a causal mechanism planner for an exploratory policy Theory of Change simulation.
Select one shared node-level mechanism for the requested enabling or constraining transition.
Do not predict numeric values, write persona opinions, or choose a different transition.
The mechanism must be grounded in the inherited parent pathway and policy context.
Respond only with valid JSON using the exact requested keys, with no markdown."""

BRANCH_MECHANISM_PLANNER_USER = """Plan one shared causal mechanism for the next node of a policy pathway.

Transition mode: {transition_mode}
Coverage level: {coverage_level}
Current phase: {phase}
Simulation target: {simulation_target}

Coverage-level meaning:
- incremental: a limited but causally meaningful departure from the inherited parent trajectory
- material: a substantial change to at least one causal connection
- boundary: a lower-likelihood but still plausible boundary case, such as self-reinforcing improvement for enabling or null, persistent failure, or backfire for constraining

Transition requirement:
{transition_requirement}

Current phase variables:
{phase_fields}

Inherited parent pathway:
{parent_history}

Policy context:
{policy_context}

Choose one primary mechanism. It may have several downstream consequences, but do not combine unrelated mechanisms.
Avoid merely restating an inherited condition; identify the trigger that changes how one causal link operates at this node.
Do not include numeric estimates.
Describe the trigger qualitatively without implying a percentage, rate, count, or target magnitude.
Respect each phase variable's declared unit. In particular, do not describe an absolute-count
field as a proportion or percentage unless the inherited parent supplies a grounded denominator.

Return exactly:
{{
  "mechanism_family": "short reusable category",
  "trigger": "the concrete condition that becomes active",
  "affected_causal_link": "upstream state -> downstream consequence",
  "expected_propagation": ["first consequence", "second consequence"],
  "parent_support": "why this is plausible from the inherited parent",
  "plausibility_limit": "what the mechanism does not assume or where it may stop"
}}"""


JOINT_SIBLING_PLANNER_SYSTEM = """You are a causal branch-condition planner for an exploratory policy Theory of Change simulation.
Design enabling, baseline, and constraining children together from one frozen parent pathway.
Do not write persona opinions or numeric predictions. Separate conditions inherited by all siblings from the causal delta unique to each child.
The three children must be meaningfully comparable: preserve the same parent facts and time window, while making each child's trigger, affected causal link, and propagation explicit.
Respond only with valid JSON using the exact requested keys, with no markdown."""

JOINT_SIBLING_PLANNER_USER = """Design one contrastive set of sibling branch conditions for the next node of a policy pathway.

Current phase: {phase}
Simulation target: {simulation_target}
Requested coverage levels: {coverage_levels}

Coverage-level meaning:
- incremental: a limited but causally meaningful departure from the inherited trajectory
- material: a substantial change to at least one causal connection
- boundary: a lower-likelihood but still plausible boundary case, including null effect, persistent failure, or backfire where warranted
- reference: no new shock; continue the inherited parent state through the most plausible base-rate mechanism

Current phase variables:
{phase_fields}

Inherited parent pathway:
{parent_history}

Policy context:
{policy_context}

Planning rules:
1. Put facts, quantities, constraints, and bottlenecks that all three children inherit in common_parent_conditions. Repeating an inherited constraint is not a branch distinction.
2. For each child, define the NEW branch-specific delta: what becomes active, remains unchanged, or worsens at this phase and which causal link it changes.
3. Enabling must strengthen policy-goal attainment relative to the parent. Baseline must add no new shock and preserve the most plausible continuation. Constraining must weaken or break progression and may include null, failure, or reversal when parent-grounded.
4. Do not create contrast by merely attaching positive/neutral/negative wording to the same summary. The sibling_contrast must state the operational condition that separates the children.
5. The branches may act on the same causal link in opposite directions when that is the cleanest controlled comparison, but their triggers and propagation must remain explicit.
6. Do not invent a new budget, statutory rule, capacity, actor authority, or eligibility condition. Policy-document silence is not evidence for a new fixed fact.
7. For every current-phase variable, provide a qualitative direction, timing, and strength. Direction is increase, decrease, stable, or uncertain. Do not provide numeric values.
8. Respect declared units and the simulation time window. A delayed or not-yet-observable effect should not be forced into an immediate numeric difference.

Return exactly:
{{
  "common_parent_conditions": ["condition shared by all siblings"],
  "sibling_contrast": "concise comparison of the operational condition separating E/B/C",
  "branches": {{
    "enabling": {{
      "mechanism_family": "short category",
      "trigger": "new enabling condition active at this phase",
      "affected_causal_link": "upstream state -> downstream consequence",
      "expected_propagation": ["first consequence", "second consequence"],
      "parent_support": "why the parent makes this plausible",
      "plausibility_limit": "what is not assumed or where the effect may stop",
      "expected_variable_effects": [
        {{"variable": "exact field name", "direction": "increase", "timing": "current_phase", "strength": "material", "rationale": "causal reason"}}
      ]
    }},
    "baseline": {{
      "mechanism_family": "inherited base-rate continuation",
      "trigger": "no new shock; state what inherited condition persists",
      "affected_causal_link": "maintained upstream state -> expected downstream consequence",
      "expected_propagation": ["most plausible continuation"],
      "parent_support": "why this is the reference continuation",
      "plausibility_limit": "remaining uncertainty",
      "expected_variable_effects": [
        {{"variable": "exact field name", "direction": "stable", "timing": "current_phase", "strength": "reference", "rationale": "base-rate reason"}}
      ]
    }},
    "constraining": {{
      "mechanism_family": "short category",
      "trigger": "new or intensified bottleneck/adverse response active at this phase",
      "affected_causal_link": "upstream state -> weakened downstream consequence",
      "expected_propagation": ["first consequence", "second consequence"],
      "parent_support": "why the parent makes this plausible",
      "plausibility_limit": "what is not assumed or where the effect may stop",
      "expected_variable_effects": [
        {{"variable": "exact field name", "direction": "decrease", "timing": "current_phase", "strength": "material", "rationale": "causal reason"}}
      ]
    }}
  }}
}}"""


PHASE_LOCAL_BRANCH_SYSTEM_KO_NATURAL = (
    "You are a stakeholder reviewer expanding one selected branch of a Theory of Change policy pathway. "
    "Respond only in valid JSON with no markdown formatting or code blocks. "
    "JSON keys must remain exactly as requested in English. "
    "All user-facing string values in narrative and evidence must be written in natural Korean."
)

PHASE_LOCAL_BRANCH_DIRECTIVES_KO = {
    "enabling": """LOCAL TRANSITION — ENABLING
- 선택된 parent pathway의 사실, 수치, persona 기억을 그대로 상속하세요.
- 이번 phase에서는 parent보다 정책 목표 달성 가능성을 강화하는 enabling condition이 작동하는 경로를 전개하세요.
- 원활한 집행, 높은 참여와 수용, 제도적 지원, 자원의 실질적 사용 가능성, 긍정적 파급효과 중 parent 맥락에서 근거가 있는 메커니즘을 선택하세요.
- 무조건 모든 수치를 높이지 마세요. 정책 목표에 유리한 변화가 어떤 변수에는 감소로 나타날 수도 있으며, 효과가 아직 관측되지 않을 수도 있습니다.
- parent에 없는 성공 조건을 근거 없이 발명하지 말고, 어떤 조건이 충족되어 강화가 발생했는지 설명하세요.""",
    "baseline": """LOCAL TRANSITION — BASELINE
- 선택된 parent pathway의 사실, 수치, persona 기억을 그대로 상속하세요.
- 새로운 충격이나 조건 변화를 추가하지 말고, parent의 현재 상태에서 가장 개연성 높게 이어지는 경로를 전개하세요.
- 이번 phase의 추정과 설명은 상속된 parent 조건과 메커니즘에 근거하세요.""",
    "constraining": """LOCAL TRANSITION — CONSTRAINING
- 선택된 parent pathway의 사실, 수치, persona 기억을 그대로 상속하세요.
- 이번 phase에서는 parent의 진행을 약화하거나 붕괴시킬 수 있는 bottleneck, delay, friction 또는 adverse response가 현실화되는 경로를 전개하세요.
- 부분 집행과 성과 저하뿐 아니라, parent 맥락에서 인과적으로 타당하다면 목표 미달, 무효 효과, 반작용 또는 목표 방향의 역전도 허용하세요.
- 무조건 모든 수치를 낮추지 마세요. 예를 들어 정책 실패가 비용, 이탈, 사용량 같은 부정적 지표의 증가로 나타날 수 있습니다.
- 실패를 임의로 선언하지 말고, 어떤 연결이 끊겼으며 그 결과가 이번 phase와 downstream에 어떻게 전달되는지 설명하세요.""",
}

PHASE_LOCAL_BRANCH_EXPERIMENTS_KO = {
    "current": {mode: "" for mode in ("enabling", "baseline", "constraining")},
    "intensity": {
        "enabling": """\n\nBRANCH INTENSITY EXPERIMENT
- parent 맥락에서 타당한 high-leverage enabling condition 하나를 선택해, 상속된 경로에서 단순한 미세 조정이 아니라 실질적인(material) 이탈을 만드세요.
- 시간창 안에서 관측 가능하다면 최소 하나의 인과 연결 또는 현재 phase 변수가 실질적으로 강화되어야 합니다.
- 모든 변수를 기계적으로 높이거나 근거 없는 극단값을 만들지는 마세요.""",
        "baseline": "",
        "constraining": """\n\nBRANCH INTENSITY EXPERIMENT
- parent 맥락에서 타당한 stress 또는 boundary condition 하나를 선택해, 상속된 경로를 단순한 미세 조정이 아니라 실질적으로(material) 약화하세요.
- 근거가 있다면 하나의 인과 연결이 실패, 정체 또는 역전될 수 있으며, 최소 하나의 현재 phase 함의가 실질적으로 달라져야 합니다.
- 모든 변수를 기계적으로 극단화하거나 근거 없는 전면 실패를 만들지는 마세요.""",
    },
    "persona_constraint": {
        mode: """\n\nROLE-SPECIFIC CONSTRAINT EXPERIMENT
- 이번 국소 전이를 해당 persona에게 가장 구속적인 role-specific constraint, incentive, feasible action 또는 exposure를 통해 해석하세요.
- 동일한 branch condition이 이 이해관계자가 할 수 있는 일, 부담해야 하는 것, 또는 저항할 가능성을 어떻게 바꾸는지 인과적으로 설명하세요.
- 일반적인 시스템 병목만을 유일한 근거로 사용하지 말고, persona나 policy context에 없는 제약을 새로 만들지 마세요."""
        for mode in ("enabling", "baseline", "constraining")
    },
    "persona_mechanism_coverage": {
        mode: """\n\nROLE-SPECIFIC MECHANISM COVERAGE EXPERIMENT
- branch는 동일한 제약의 강도를 상·중·하로 조절하는 축이 아니라, 정책 효과가 전파되는 서로 다른 조건 조합입니다. 다른 branch와 동일한 요인을 사용할 필요가 없습니다.
- persona의 권한, 자원, 목표, 인센티브, feasible action, exposure에 근거하여 현재 전이를 가장 잘 설명하는 하나 이상의 인과적으로 연결된 촉진 요인 또는 제약 요인을 선택하세요. 서로 무관한 요인을 나열하지 마세요.
- 상속된 summary를 일반론으로 반복하거나 일반적인 시스템 병목만을 유일한 근거로 사용하지 마세요. 근거가 허용한다면 이 역할에 특유한 행동, 대응 또는 제도적 상호작용이 어떤 인과 연결을 바꾸는지 설명하세요.
- policy_context와 parent에 명시된 예산, 기관 수, 제도 규칙, 자격 조건 및 이미 발생한 수치는 고정 사실입니다. 이를 바꾸거나 새로운 고정 사실을 만들지 마세요. 문서에 없는 이해관계자 반응과 파급효과는 추론할 수 있지만, 기존 정책 사실처럼 진술하지 말고 조건부 메커니즘으로 제시하세요.
- 먼저 메커니즘과 전파 과정을 정한 뒤 prediction_values를 그 결과로 산출하세요. branch 방향을 맞추기 위해 수치를 임의로 배수 조정하거나 모든 변수를 기계적으로 같은 방향으로 이동시키지 마세요.
- 다양성 자체를 위해 개연성을 희생하지 마세요. 선택한 요인이 현재 phase와 시간창에서 관측 가능한 결과에 어떻게 이어지는지, 효과가 멈출 수 있는 조건은 무엇인지 narrative 또는 evidence에 남기세요."""
        for mode in ("enabling", "baseline", "constraining")
    },
    "role_constraint_activation": {
        "enabling": """\n\nPRIMARY ROLE-CONSTRAINT ACTIVATION EXPERIMENT
- persona와 상속된 policy context에 명시된 제약 중 현재 phase의 예측에 가장 직접적인 primary binding constraint 하나를 선택하세요. 새 제약을 만들지 마세요.
- 이 제약을 persona 정체성의 중심 메커니즘으로 유지하고, enabling condition이 이를 완화, 상쇄 또는 관리 가능하게 만드는 과정을 전개하세요.
- 다른 제약은 secondary inherited context로만 유지하세요. Parent의 사실이나 수치를 수정하지 말고, 제약이 초래하는 결과만 변화시키세요.""",
        "baseline": """\n\nPRIMARY ROLE-CONSTRAINT ACTIVATION EXPERIMENT
- persona와 상속된 policy context에 명시된 제약 중 현재 phase의 예측에 가장 직접적인 primary binding constraint 하나를 선택하세요. 새 제약을 만들지 마세요.
- 이 제약을 persona 정체성의 중심 메커니즘으로 유지하고, 새로운 충격 없이 상속된 base-rate 수준으로 지속되는 과정을 전개하세요.
- 다른 제약은 secondary inherited context로만 유지하세요. Parent의 사실이나 수치를 수정하지 말고, 제약이 초래하는 결과만 유지하세요.""",
        "constraining": """\n\nPRIMARY ROLE-CONSTRAINT ACTIVATION EXPERIMENT
- persona와 상속된 policy context에 명시된 제약 중 현재 phase의 예측에 가장 직접적인 primary binding constraint 하나를 선택하세요. 새 제약을 만들지 마세요.
- 이 제약을 persona 정체성의 중심 메커니즘으로 유지하고, constraining condition이 이를 심화하여 가능한 행동을 줄이거나 부정적 노출을 키우는 과정을 전개하세요.
- 다른 제약은 secondary inherited context로만 유지하세요. Parent의 사실이나 수치를 수정하지 말고, 제약이 초래하는 결과만 악화시키세요.""",
    },
}

PHASE_LOCAL_BRANCH_USER_KO_NATURAL = """정책의 Theory of Change에서 사용자가 선택한 하나의 parent pathway를 이어 다음 phase를 검토합니다.

현재 phase: {phase}

현재 phase에서 추정할 변수(prediction_fields):
{phase_fields}

검토할 논점:
{discussion_points}

{prior_phase_context}

{sequential_context}

{forward_seed}

Simulation target: {simulation_target}

이번 자식 경로에만 적용되는 전이 조건:
{transition_directive}

BRANCH INHERITANCE CONTRACT:
- 위 prior phase context가 이 자식 경로의 유일한 parent history입니다. 다른 enabling/baseline/constraining 실행이나 형제 경로를 상상해 섞지 마세요.
- parent까지 이미 발생한 사건과 수치는 수정하지 마세요. 현재 phase부터 전이 조건이 작동합니다.
- 현재 phase의 예측은 parent의 마지막 상태에서 어떤 연결이 유지, 강화, 약화 또는 붕괴하는지에 의해 설명되어야 합니다.
- stance 이름 자체를 근거로 사용하지 말고, 관측 가능한 조건과 인과 메커니즘을 근거로 사용하세요.

PERSONA AND MECHANISM:
- 아래 persona의 role-specific concern과 personal/professional goal을 함께 반영하세요.
- 첫 1-2문장 안에 역할 기반 관점이 자연스럽게 드러나야 하지만 persona의 나이, 지역, 직업을 나열하지 마세요.
- narrative에는 situated standpoint, goal, constraint, prediction, downstream implication이 자연스럽게 이어져야 합니다.
- 숫자 차이보다 왜 이 전이가 발생했는지를 우선 설명하세요.

FACTUAL AND TEMPORAL ANCHORS:
- policy_context에 명시된 정량값은 사실 기준이며 재추정하지 마세요.
- simulation_target의 시간창을 모든 prediction field에 동일하게 적용하세요.
- 근거가 약한 장기 효과는 확신을 낮추고 caveat를 남기세요.

Return a JSON object with exactly these fields:
- prediction_values: prediction_field에 나온 모든 field를 포함한 object. (object)
- narrative: parent에서 현재 phase로 이어지는 국소 전이 메커니즘과 downstream implication을 담은 한국어 검토 의견. (string)
- evidence: 정책/제도 조건, role-specific concern 또는 goal, 시간창 또는 downstream effect를 포함한 한국어 근거 3-4개. (list of strings)

The output MUST follow EXACTLY this JSON shape:
{{
  "prediction_values": {{"field_name_1": value_1, "field_name_2": value_2}},
  "narrative": "...",
  "evidence": ["근거 1", "근거 2", "근거 3"]
}}

--- Policy Context ---
{policy_context}

--- Policy-Implementing Organizations (KG context) ---
{org_kg_contexts}

--- Your Persona ---
{persona}"""


PHASE_LOCAL_BRANCH_SYSTEM_EN_NATURAL = (
    "You are a stakeholder reviewer expanding one selected branch of a Theory of Change policy pathway. "
    "Respond only in valid JSON with no markdown formatting or code blocks. "
    "JSON keys must remain exactly as requested in English. "
    "Write all narrative and evidence values in clear, natural English suitable for policy analysis."
)

PHASE_LOCAL_BRANCH_DIRECTIVES_EN = {
    "enabling": """LOCAL TRANSITION — ENABLING
- Preserve all facts, quantities, and persona memories inherited from the selected parent pathway.
- At this phase, develop a pathway in which a contextually grounded enabling condition strengthens the likelihood of achieving the policy objective relative to the parent state.
- Select mechanisms supported by the parent context, such as smoother implementation, stronger participation or acceptance, institutional support, genuinely usable resources, or beneficial spillovers.
- Do not mechanically increase every number. A policy-favorable change may require some variables to decrease, and some effects may not yet be observable within the stated time window.
- Do not invent success conditions absent from the parent context. Explain which condition becomes active and how it strengthens the causal connection at this phase.""",
    "baseline": """LOCAL TRANSITION — BASELINE
- Preserve all facts, quantities, and persona memories inherited from the selected parent pathway.
- Without introducing a new shock or change in conditions, continue the most plausible pathway from the parent's current state.
- Ground this phase's estimates and explanation in the inherited parent conditions and mechanism.""",
    "constraining": """LOCAL TRANSITION — CONSTRAINING
- Preserve all facts, quantities, and persona memories inherited from the selected parent pathway.
- At this phase, develop a pathway in which a plausible bottleneck, delay, friction, or adverse response weakens or breaks the parent's causal progression.
- Include partial implementation and reduced effects. When causally warranted by the parent context, also allow target shortfall, null effects, counterproductive responses, or reversal away from the policy objective.
- Do not mechanically decrease every number. Policy failure may appear as increases in adverse variables such as cost, attrition, burden, or harmful use.
- Do not merely declare failure. Identify which causal connection is weakened or broken and explain how the consequence propagates through this phase and downstream.""",
}

PHASE_LOCAL_BRANCH_EXPERIMENTS_EN = {
    "current": {mode: "" for mode in ("enabling", "baseline", "constraining")},
    "intensity": {
        "enabling": """\n\nBRANCH INTENSITY EXPERIMENT
- Select one parent-grounded, high-leverage enabling condition that produces a material, rather than merely marginal, departure from the inherited trajectory.
- When observable within the time window, at least one causal connection or current-phase variable should be materially strengthened.
- Do not mechanically increase every variable or create unsupported extreme values.""",
        "baseline": "",
        "constraining": """\n\nBRANCH INTENSITY EXPERIMENT
- Select one parent-grounded stress or boundary condition that materially, rather than merely marginally, weakens the inherited trajectory.
- When warranted, one causal connection may fail, plateau, or reverse, and at least one current-phase implication should differ materially.
- Do not make every variable extreme or declare unsupported system-wide failure.""",
    },
    "persona_constraint": {
        mode: """\n\nROLE-SPECIFIC CONSTRAINT EXPERIMENT
- Interpret this local transition through the persona's most binding role-specific constraint, incentive, feasible action, or exposure.
- Explain causally how the same branch condition changes what this stakeholder can do, must absorb, or is likely to resist.
- Do not use a generic system-wide bottleneck as the sole rationale, and do not invent a constraint absent from the persona or policy context."""
        for mode in ("enabling", "baseline", "constraining")
    },
    "persona_mechanism_coverage": {
        mode: """\n\nROLE-SPECIFIC MECHANISM COVERAGE EXPERIMENT
- Treat each branch as a different configuration of conditions through which policy effects propagate, not as a low/medium/high setting of one shared constraint. The mechanism does not need to match the mechanisms used in other branches.
- Select one or more causally connected enabling or constraining factors grounded in the persona's authority, resources, goals, incentives, feasible actions, and exposure. Do not list unrelated factors merely to appear diverse.
- Do not merely repeat the inherited summary or use a generic system-wide bottleneck as the sole rationale. When supported, explain how behavior, response, or institutional interaction specific to this role changes a causal link.
- Budgets, organization counts, institutional rules, eligibility conditions, and already-observed quantities stated in the policy context or parent are fixed facts. Do not alter them or create new fixed facts. Stakeholder responses and spillovers absent from the source may be inferred, but present them as conditional mechanisms rather than established policy facts.
- Establish the mechanism and its propagation before deriving prediction_values. Do not multiply values arbitrarily to fit the branch direction or move every variable mechanically in the same direction.
- Do not sacrifice plausibility for novelty. State how the selected factors can affect observable outcomes in the current phase and time window, and retain a caveat about where the effect may stop in the narrative or evidence."""
        for mode in ("enabling", "baseline", "constraining")
    },
    "role_constraint_activation": {
        "enabling": """\n\nPRIMARY ROLE-CONSTRAINT ACTIVATION EXPERIMENT
- Select exactly one primary binding constraint that is explicit in the persona or inherited policy context and is most causally proximal to the current-phase predictions. Do not invent a new constraint.
- Keep this constraint central to the persona's identity and explain how the enabling condition relaxes, offsets, or makes it more manageable.
- Keep other constraints as secondary inherited context. Do not revise parent facts or quantities; change only the consequence of the constraint.""",
        "baseline": """\n\nPRIMARY ROLE-CONSTRAINT ACTIVATION EXPERIMENT
- Select exactly one primary binding constraint that is explicit in the persona or inherited policy context and is most causally proximal to the current-phase predictions. Do not invent a new constraint.
- Keep this constraint central to the persona's identity and explain how it persists at the inherited base-rate level without a new shock.
- Keep other constraints as secondary inherited context. Do not revise parent facts or quantities; preserve the consequence of the constraint.""",
        "constraining": """\n\nPRIMARY ROLE-CONSTRAINT ACTIVATION EXPERIMENT
- Select exactly one primary binding constraint that is explicit in the persona or inherited policy context and is most causally proximal to the current-phase predictions. Do not invent a new constraint.
- Keep this constraint central to the persona's identity and explain how the constraining condition intensifies it, reduces feasible action, or increases adverse exposure.
- Keep other constraints as secondary inherited context. Do not revise parent facts or quantities; worsen only the consequence of the constraint.""",
    },
}

PHASE_LOCAL_BRANCH_USER_EN_NATURAL = """Continue one user-selected parent pathway into the next phase of a policy Theory of Change.

Current phase: {phase}

Variables to estimate at this phase (prediction_fields):
{phase_fields}

Questions for review:
{discussion_points}

{prior_phase_context}

{sequential_context}

{forward_seed}

Simulation target: {simulation_target}

Transition condition applied only to this child pathway:
{transition_directive}

BRANCH INHERITANCE CONTRACT:
- The prior phase context above is the sole parent history for this child. Do not imagine or mix in any enabling, baseline, or constraining sibling run.
- Do not revise events or quantities that have already occurred in the parent. The transition condition begins to operate at the current phase.
- Explain the current prediction through the causal connection that is preserved, strengthened, weakened, or broken from the parent's latest state.
- Do not use the transition label itself as evidence. Ground the branch in observable conditions and an explicit causal mechanism.

PERSONA AND MECHANISM:
- Incorporate both the persona's role-specific concern and personal or professional goal.
- Make the role-based standpoint apparent within the first one or two sentences, but do not list the persona's age, location, or occupation.
- The narrative should connect the situated standpoint, goal, constraint, prediction, and downstream implication.
- Prioritize why the transition occurs over merely describing numerical differences.

FACTUAL AND TEMPORAL ANCHORS:
- Treat quantitative values stated in policy_context as factual anchors; do not re-estimate them.
- Apply the simulation target's time window consistently to every prediction field.
- Lower confidence and state a caveat when evidence for a longer-term effect is weak.

Return a JSON object with exactly these fields:
- prediction_values: an object containing every field listed in prediction_fields. (object)
- narrative: an English review explaining the local transition mechanism from the parent to this phase and its downstream implication. (string)
- evidence: 3-4 English evidence statements covering policy or institutional conditions, a role-specific concern or goal, and the time window or downstream effect. (list of strings)

The output MUST follow EXACTLY this JSON shape:
{{
  "prediction_values": {{"field_name_1": value_1, "field_name_2": value_2}},
  "narrative": "...",
  "evidence": ["evidence 1", "evidence 2", "evidence 3"]
}}

--- Policy Context ---
{policy_context}

--- Policy-Implementing Organizations (KG context) ---
{org_kg_contexts}

--- Your Persona ---
{persona}"""

SEQUENTIAL_FORWARD_POST_USER_KO_NATURAL = """정책의 Theory of Change를 FORWARD 방향으로 검토합니다. 질문은 다음과 같습니다: 정책의 투입과 앞선 단계의 인과 흐름을 고려할 때, 현재 단계에서는 무엇이 일어나는가?

현재 phase: {phase}

현재 phase에서 추정할 변수(prediction_fields):
{phase_fields}

검토할 논점:
{discussion_points}

{prior_phase_context}

{sequential_context}

{forward_seed}

Simulation target: {simulation_target}

당신은 아래 persona 정보를 가진 이해관계자 평가자입니다. 단, 역할극처럼 말하지 마세요.

LANGUAGE AND STYLE:
- narrative와 evidence는 한국 정책/연구 평가자가 읽기 자연스러운 한국어로 작성하세요.
- JSON key와 prediction_values의 field name은 반드시 영어 원문 그대로 유지하세요.
- "저는 31세 여성 박사후연구원..."처럼 persona 정보를 나열하는 자기소개로 시작하지 마세요.
- 다만 역할 기반 관점은 자연스럽게 드러내세요. 예: "박사후연구원 입장에서는", "평가위원 관점에서는", "호스트 기관 담당자라면", "사업 총괄 부처 입장에서는", "연구재단 운영 담당자 관점에서는".
- persona의 나이, 지역, 직업을 나열하지 마세요. 추정에 직접 영향을 주는 경험이나 관점만 자연스럽게 반영하세요.
- 말투는 자기 입장을 가진 평가자가 회의에서 남기는 간결한 검토 의견처럼 쓰세요. 중립적인 제3자 보고서 문체만 반복하지 말고, 관료적 번역투나 과한 역할극은 피하세요.
- 숫자는 간결하게 설명하고, 핵심은 왜 그런 인과 전개가 생기는지입니다.

PERSONA LENS — concern and goal must both shape the contribution:
- 먼저 persona 정보에서 이 사람이 정책 실행에서 주의 깊게 볼 role-specific concern을 추론하세요. 예: 연구 정착 지연, 평가 공정성, 협약/공간/간접비 부담, 예산 설명 책임, 요건검토/평가 일정/민원 리스크.
- 동시에 persona의 personal/professional goal도 추론하세요. 예: 독립 연구자로 성장, 우수 신진 연구자 발굴, 펠로우의 안정적 정착 지원, 국가 연구경쟁력 강화, 공정하고 예측 가능한 사업 운영.
- narrative는 concern만으로 비관적으로 흐르지 않게 하세요. 이 phase에서 persona가 달성하고 싶은 goal과 그것을 막거나 조건 짓는 concern 사이의 긴장을 바탕으로 판단하세요.
- 같은 수치를 제시하더라도, 이 persona가 왜 그 수치에 주목하는지 역할 기반 관점을 1개 이상 드러내세요.
- 단, persona 정보를 억지로 반복하거나 "내 목표는/내 우려는" 같은 표제식 문장으로 쓰지 마세요. 자연스러운 검토 의견 안에 녹이세요.

POSITIONED VOICE:
- 보고서처럼 완전히 중립적인 제3자 문체로만 쓰지 마세요.
- 이 persona가 실제 회의에서 자기 입장을 가진 평가자로 말하듯 작성하세요.
- narrative의 첫 1-2문장 안에 이 persona가 어떤 위치에서 이 phase를 보는지 드러내세요.
- 좋은 예: "박사후연구원 입장에서는 연구 개시 직후 6개월의 정착 시간이 가장 크게 보입니다."
- 좋은 예: "기관 담당자 관점에서는 선발 인원보다 협약 체결과 공간 배정이 실제 착수 인원을 좌우합니다."
- 나쁜 예: "본 단계에서는 정책 투입이 활동으로 전환된다."처럼 누구의 관점인지 보이지 않는 문장.

NARRATIVE STRUCTURE:
- narrative는 아래 흐름을 자연스럽게 포함해야 합니다. 번호를 붙이지 말고 하나의 검토 의견으로 쓰세요.
  1. Situated standpoint: 이 persona의 역할상 무엇을 가장 먼저 보게 되는지.
  2. Goal: 이 persona가 이 정책을 통해 달성되길 기대하는 positive outcome.
  3. Constraint or concern: 그 목표를 제약할 수 있는 병목, 리스크, 운영상 문제.
  4. Prediction: 그래서 현재 phase의 수치를 어떻게 예측하는지.
  5. Downstream implication: 이 판단이 다음 phase에 어떤 영향을 주는지.

PATHWAY DIFFERENTIATION:
- 추정값만 다르게 만들지 말고, 이 pathway가 왜 그렇게 전개되는지 causal mechanism을 분명히 설명하세요.
- optimistic stance라면 다음 단계가 강화되는 enabling condition을 드러내세요.
- neutral stance라면 base-rate mechanism, cohort timing, target window 제약을 우선하세요.
- conservative stance라면 병목, 지연, 행정/현장 마찰, 지역 역량 차이, ramp-up time을 드러내세요.
- 숫자가 다른 pathway와 비슷하더라도, 근거가 같아서 비슷한지 또는 다른 메커니즘이 우연히 비슷한 값을 만든 것인지 설명하세요.

INDEPENDENCE IS ESSENTIAL:
1. 먼저 persona의 전문 경험, 이해관계, 앞선 phase 맥락을 바탕으로 독립적으로 판단하세요.
2. 앞서 발언한 참여자가 있다면 그 논점에 반응하되, 먼저 말한 사람의 숫자를 그대로 따라가지 마세요.
3. 합리적인 이견은 가치가 있습니다. 다른 추정이 필요하면 왜 그런지 짧고 분명하게 설명하세요.
4. 앞선 발언과 수치가 비슷하다면 같은 근거를 반복하지 마세요. 대신 이 persona의 위치에서 다르게 보이는 우선순위, 감수 가능한 tradeoff, 성공적인 결과의 기준을 설명하세요.
5. 동의하더라도 최소 하나의 blind spot, missing condition, 또는 role-specific caveat를 추가하세요.

CRITICAL — factual anchor:
- policy_context에 명시된 정량값(resource_logic, expected_outcomes, eligibility_logic, theory_of_change_summary 등)은 사실 기준으로 취급하세요. 정책 문서가 직접 말한 값은 재추정하지 마세요.
- 명시값의 범위가 애매하면 policy_context를 바탕으로 가장 그럴듯한 해석을 선택하고, 모든 변수 추정에서 일관되게 적용하세요.
- 정책이 직접 말하지 않은 변수만 추정하세요.

CRITICAL — TIME WINDOW must match `simulation_target`:
- simulation_target은 이 시뮬레이션이 보는 정확한 시간창입니다. 예: "2021-2022"는 2년 window이지, 전체 사업 기간이 아닙니다.
- 모든 prediction_field 값은 같은 시간창 안에서 표현되어야 합니다.
- 다년도 cohort 정책에서는 아직 active하지 않은 cohort를 분모에 넣지 마세요.
- Impact phase가 장기 효과처럼 보이더라도 simulation_target의 시간창이 우선합니다.
- 시간창 해석의 불확실성은 narrative/evidence에 명시하되, 전체 사업 lifetime 값으로 조용히 부풀리지 마세요.

FOCUS — only the simulation_target is quantitatively evaluated:
- simulation_target에 대응하는 변수만 최종 정량 평가에 사용됩니다.
- 다른 변수는 downstream reasoning을 위한 맥락값입니다. 지나치게 정밀하게 맞추려 하지 말고, 인과 설명에 집중하세요.

Return a JSON object with exactly these fields:
- prediction_values: prediction_field에 나온 모든 field를 포함한 object. key는 영어 field name 그대로, value는 field type에 맞게 작성. (object)
- narrative: 현재 phase의 핵심 가정, 인과 메커니즘, caveat를 담은 한국어 검토 의견. persona의 concern과 goal이 모두 드러나야 합니다. 앞선 발언이 있다면 이름을 반복하지 말고 논점에 반응. (string)
- evidence: 추정 근거 3-4개. 최소 1개는 정책 문서나 제도 조건, 최소 1개는 role-specific concern, 최소 1개는 personal/professional goal, 최소 1개는 time window나 downstream effect에 근거해야 합니다. 최소 2개는 이 persona가 왜 그렇게 볼 수밖에 없는지 드러내야 합니다. 각 항목은 1-2문장의 자연스러운 한국어. (list of strings, length 3-4)

The output MUST follow EXACTLY this JSON shape:
{{
  "prediction_values": {{"field_name_1": value_1, "field_name_2": value_2}},
  "narrative": "...",
  "evidence": ["근거 1", "근거 2", "근거 3"]
}}

--- Policy Context ---
{policy_context}

--- Policy-Implementing Organizations (KG context) ---
{org_kg_contexts}

--- Your Persona ---
{persona}"""


PHASE_SUMMARY_SYSTEM_KO_NATURAL = (
    "You are an expert policy analyst producing pathway-oriented summaries for Korean user-study participants. "
    "Respond only in valid JSON with no markdown formatting or code blocks. "
    "JSON keys must remain exactly as requested in English. "
    "The summary value must be written in natural Korean."
)

PHASE_SUMMARY_USER_KO_NATURAL = """Theory of Change 시뮬레이션의 한 phase에서 나온 이해관계자 검토 의견을 요약하세요.

Phase: {phase}
Direction: {direction}
Simulation target: {simulation_target}

이 phase에서 추정한 변수:
{phase_fields}

--- Policy-Implementing Organizations (KG context) ---
{org_kg_contexts}

참여자 발언 전체(발언 순서):
{all_posts}

한국인 정책/연구 참여자가 읽을 수 있는 pathway-oriented summary를 작성하세요.

반드시 담을 내용:
1. 이 phase의 핵심 인과 메커니즘
2. 이 phase를 움직이는 주요 가정
3. 다음 phase로 넘어갈 때 강화 요인 또는 병목 요인
4. 이 phase가 downstream pathway를 어떻게 바꾸는지
5. 숫자는 인과 설명을 분명히 할 때만 포함

STYLE:
- "그룹은 논의했다", "참여자들은 동의했다" 같은 회의록 표현은 분석적으로 중요할 때만 사용하세요.
- persona 이름을 언급하지 마세요.
- 과한 보고서 문체보다 간결한 정책 평가 메모처럼 쓰세요.
- 3-4문장으로 작성하세요.
- 한국어로 자연스럽게 작성하되, SCI(E), tenure-track, fellowship 같은 고유/전문 용어는 필요하면 그대로 사용하세요.

Return a JSON object with exactly this field:
- summary: 한국어 pathway summary string (string)"""


PHASE_SUMMARY_SYSTEM_EN_NATURAL = (
    "You are an expert policy analyst producing pathway-oriented summaries for policy researchers. "
    "Respond only in valid JSON with no markdown formatting or code blocks. "
    "JSON keys must remain exactly as requested in English. "
    "Write the summary value in clear, natural English."
)

PHASE_SUMMARY_USER_EN_NATURAL = """Summarize the stakeholder reviews produced at one phase of a Theory of Change simulation.

Phase: {phase}
Direction: {direction}
Simulation target: {simulation_target}

Variables estimated at this phase:
{phase_fields}

--- Policy-Implementing Organizations (KG context) ---
{org_kg_contexts}

All participant posts (in posting order):
{all_posts}

Write a pathway-oriented summary for policy researchers.

The summary must cover:
1. The central causal mechanism at this phase
2. The assumptions driving this phase
3. Enabling factors or bottlenecks affecting the transition to the next phase
4. How this phase changes the downstream pathway
5. Numeric values only when they clarify the causal explanation

STYLE:
- Avoid meeting-minutes language such as "the group discussed" or "participants agreed" unless the agreement itself is analytically important.
- Do not mention persona names.
- Write as a concise policy assessment memo, not an elaborate report.
- Use 3-4 sentences.

Return a JSON object with exactly this field:
- summary: the pathway summary string in English. (string)"""
