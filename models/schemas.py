from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict, Any, Optional, Literal, Union


class EconomicContext(BaseModel):
    summary: str
    key_ai_trends: List[str]
    policy_landscape: str


class PolicyPlan(BaseModel):
    policy_name: str
    year_of_plan: str
    policy_duration: str
    policy_goal: str
    policy_content: str
    budget: Optional[Any] = None


class PolicyContext(BaseModel):
    policy_name: str
    policy_domain: str
    policy_subdomains: List[str] = Field(default_factory=list)
    policy_type: str
    policy_instruments: List[str] = Field(default_factory=list)
    target_population: List[str] = Field(default_factory=list)
    primary_beneficiaries: List[str] = Field(default_factory=list)
    implementing_actors: List[str] = Field(default_factory=list)
    delivery_mechanism: str
    eligibility_logic: str
    resource_logic: str
    expected_outcomes: List[str] = Field(default_factory=list)
    theory_of_change_summary: str
    simulation_relevance: str


class OntologyEntityType(BaseModel):
    name: str
    description: str = ""
    iad_category: str = "Participants"
    node_family: str = ""
    iad_rule_type: Optional[
        Literal["position", "boundary", "choice", "aggregation", "information", "payoff", "scope"]
    ] = None
    is_actor: bool = True
    persona_eligible: bool = True
    attributes: List[Dict[str, Any]] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)
    rationale: str = ""


class OntologyEdgeSourceTarget(BaseModel):
    source: str
    target: str


class OntologyEdgeType(BaseModel):
    name: str
    description: str = ""
    edge_category: str = "structural"
    edge_family: str = ""
    ig_deontic: Optional[Literal["must", "can", "cannot"]] = None
    source_targets: List[OntologyEdgeSourceTarget] = Field(default_factory=list)
    attributes: List[Dict[str, Any]] = Field(default_factory=list)
    required_attributes: List[str] = Field(default_factory=list)
    rationale: str = ""


class GraphOntology(BaseModel):
    entity_types: List[OntologyEntityType]
    edge_types: List[OntologyEdgeType]
    analysis_summary: str = ""


class SourceSpan(BaseModel):
    chunk_id: Optional[str] = None
    text: str = ""


class PolicyGraphNode(BaseModel):
    id: str
    name: str
    type: str
    iad_category: str
    node_family: str = ""
    iad_rule_type: Optional[
        Literal["position", "boundary", "choice", "aggregation", "information", "payoff", "scope"]
    ] = None
    is_actor: bool = True
    persona_eligible: bool = True
    summary: str = ""
    attributes: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    source_spans: List[SourceSpan] = Field(default_factory=list)


class PolicyGraphEdge(BaseModel):
    id: str
    type: str
    edge_category: str = "structural"
    edge_family: str = ""
    source: str
    target: str
    fact: str = ""
    deontic: Optional[Literal["must", "can", "cannot"]] = None
    activation_condition: Optional[str] = None
    execution_constraint: Optional[str] = None
    condition_text: Optional[str] = None  # deprecated; use activation_condition/execution_constraint
    or_else_text: Optional[str] = None
    source_text: str = ""
    attributes: Dict[str, Any] = Field(default_factory=dict)


class PolicyGraph(BaseModel):
    graph_id: str
    ontology: Optional[GraphOntology] = None
    nodes: List[PolicyGraphNode]
    edges: List[PolicyGraphEdge]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    validation_warnings: List[str] = Field(default_factory=list)


class PersonaProfile(BaseModel):
    representative_type: str
    core_objectives: List[str]
    core_constraints: List[str]
    role_entity_id: Optional[str] = None
    iad_category: Optional[str] = None
    obligations: List[Dict[str, Any]] = Field(default_factory=list)
    eligible_actions: List[Dict[str, Any]] = Field(default_factory=list)
    prohibited_actions: List[Dict[str, Any]] = Field(default_factory=list)
    kg_context: Dict[str, Any] = Field(default_factory=dict)
    demographics: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None


class RolePrototype(BaseModel):
    """Policy-grounded functional role used to construct a mechanism panel.

    This is an additive overlay: source KG ``is_actor`` and
    ``persona_eligible`` values remain unchanged for legacy reproducibility.
    """

    prototype_id: str
    label: str
    domain_role: str
    mechanism_functions: List[str] = Field(default_factory=list)
    role_family: str = "affected_actor"
    agent_modality: str = "individual"
    required: bool = True
    anchor_entity_id: str
    source_entity_ids: List[str] = Field(default_factory=list)
    source_node_types: List[str] = Field(default_factory=list)
    toc_phase_relevance: List[str] = Field(default_factory=list)
    objectives: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    causal_leverage_points: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    classification_basis: List[str] = Field(default_factory=list)


class StakeholderGroup(BaseModel):
    stakeholder_type: str          # University | Market | Beneficiary
    personas: List[PersonaProfile]


class PersonaPrediction(BaseModel):
    representative_type: str
    predicted_value: float         # numerical estimate for the prediction target
    prediction: str                # qualitative explanation
    evidence: List[str]


class PredictionResult(BaseModel):
    stakeholder_type: str
    predictions: List[PersonaPrediction]


class PersonaCriticResult(BaseModel):
    representative_type: str
    valid: bool                        # abs(predicted_value - actual_value) <= threshold
    error_score: float                 # absolute error: |predicted_value - actual_value|
    direction: str = ""               # "TOO OPTIMISTIC (overestimate)" or "TOO PESSIMISTIC (underestimate)"
    previous_predicted_value: float = 0.0
    refinement_suggestions: List[str]


class CriticResult(BaseModel):
    stakeholder_type: str
    results: List[PersonaCriticResult]


class Stage1Result(BaseModel):
    economic_context: Optional[EconomicContext] = None
    policy_plan: Optional[PolicyPlan] = None
    policy_context: Optional[PolicyContext] = None
    stakeholder_groups: List[StakeholderGroup]   # Backward-compatible alias for simulation_participants
    persona_pool: List[StakeholderGroup] = Field(default_factory=list)
    simulation_participants: List[StakeholderGroup] = Field(default_factory=list)
    predictions: List[PredictionResult] = Field(default_factory=list)
    critic_results: List[CriticResult] = Field(default_factory=list)
    handoff_metadata: Dict[str, Any] = Field(default_factory=dict)


class Round1Comment(BaseModel):
    stakeholder_type: str
    representative_type: str
    predicted_value: float
    comment: str
    evidence: List[str]


class DynamicRoundComment(BaseModel):
    stakeholder_type: str
    representative_type: str
    spoke: bool                              # False = passed, holding current prediction
    revised_predicted_value: Optional[float] = None
    comment: Optional[str] = None


class FinalRoundComment(BaseModel):
    stakeholder_type: str
    representative_type: str
    final_predicted_value: float
    final_comment: str
    improvement_suggestions: str


class JudgeSummary(BaseModel):
    aggregated_prediction: float
    prediction_range: str
    stance_summary: str
    key_agreements: List[str]
    key_disagreements: List[str]


class ConcretePersona(BaseModel):
    """Fixed Stage 2 persona payload injected into simulation prompts."""
    model_config = ConfigDict(extra="ignore")

    persona_id: str = ""
    name: str
    stakeholder_type: str
    role_prototype_id: str = ""
    role_anchor_entity_id: str = ""
    panel_position: Optional[int] = None
    agent_modality: str = ""

    age: Optional[int] = None
    sex: str = ""
    marital_status: str = ""
    education_level: str = ""
    occupation: str = ""
    province: str = ""
    district: str = ""

    professional_persona: str = ""
    skills_and_expertise: str = ""
    career_goals_and_ambitions: str = ""
    policy_role_context: Dict[str, Any] = Field(default_factory=dict)

    @property
    def representative_type(self) -> str:
        """Alias used by simulation round functions."""
        return self.name


class ConcreteScenarioResult(BaseModel):
    scenario_id: str
    participants: List[ConcretePersona]
    round1: List[Round1Comment]
    dynamic_rounds: List[List[DynamicRoundComment]]  # variable number of rounds
    final_round: List[FinalRoundComment]
    consensus_reached: bool
    judge: JudgeSummary


class FinalReport(BaseModel):
    simulation_target: str
    stakeholder_groups: List[StakeholderGroup] = []       # Stage 1 abstract personas
    stage1_predictions: List[PredictionResult] = []       # Stage 1 predicted values
    stage1_critic_results: List[CriticResult] = []        # Stage 1 validity per persona
    constructed_personas: List[ConcretePersona] = []
    scenarios: List[ConcreteScenarioResult]


class PolicyReportResult(BaseModel):
    simulation_target: str
    avg_prediction: float
    prediction_range: str                   # e.g. "580-750"
    ground_truth: Optional[float] = None
    error: Optional[float] = None
    error_pct: Optional[float] = None
    common_evidence: List[str]             # evidence themes shared across scenarios
    common_concerns: List[str]             # concern themes shared across scenarios
    common_recommendations: List[str]      # recommendation themes shared across scenarios
    synthesis: str                         # overall narrative summary


# ───────────────────────────── Stage 2 ToC ─────────────────────────────

# Outer ToC structure — fixed, never generated
TOC_PHASES = ("Inputs", "Activities", "Outputs", "Outcomes", "Impact")


class PhaseVariableSpec(BaseModel):
    name: str                                    # snake_case machine key
    unit: str                                    # e.g. "KRW", "count", "%"
    type: Literal["int", "float", "str"]
    description: str


class PhaseInnerSpec(BaseModel):
    phase: Literal["Inputs", "Activities", "Outputs", "Outcomes", "Impact"]
    prediction_fields: List[PhaseVariableSpec]   # 2–4 per phase
    discussion_points: List[str]                 # 2–4 per phase


class ToCInnerSchema(BaseModel):
    phases: List[PhaseInnerSpec]                 # always len 5, in TOC_PHASES order


class GroundedEvidence(BaseModel):
    source: str
    page: Optional[int] = None
    evidence_quote: str


class GroundedInputField(BaseModel):
    name: str
    value: Union[float, int, str]
    unit: str
    type: Literal["int", "float", "str"]
    description: str
    effective_date: Optional[str] = None
    scope: str = ""
    causal_relevance: str
    kg_links: List[str] = Field(default_factory=list)
    evidence: List[GroundedEvidence] = Field(default_factory=list)
    validation_status: Literal["validated", "rejected", "unresolved"] = "unresolved"
    validation_notes: List[str] = Field(default_factory=list)


class GroundedPolicyParameter(BaseModel):
    name: str
    value: Union[bool, float, int, str]
    unit: str
    description: str
    effective_date: Optional[str] = None
    evidence: List[GroundedEvidence] = Field(default_factory=list)
    validation_status: Literal["validated", "rejected", "unresolved"] = "unresolved"
    validation_notes: List[str] = Field(default_factory=list)


class GroundedInputState(BaseModel):
    policy_name: str
    state_type: Literal["document_grounded"] = "document_grounded"
    fields: List[GroundedInputField] = Field(default_factory=list)
    policy_parameters: List[GroundedPolicyParameter] = Field(default_factory=list)
    summary: str
    source_documents: List[str] = Field(default_factory=list)
    excluded_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    unresolved_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def values(self) -> Dict[str, Union[float, int, str]]:
        return {
            field.name: field.value
            for field in self.fields
            if field.validation_status == "validated"
        }


class PersonaPhaseEstimate(BaseModel):
    persona_name: str
    stakeholder_type: str
    prediction_values: Dict[str, Union[float, int, str]]   # keys = prediction_field names
    narrative: str
    evidence: List[str]
    judgment: Optional[str] = None  # refinement only: which direction the persona leaned toward and why


class PhaseRoundResult(BaseModel):
    phase: str
    direction: Literal["forward", "backward"]
    initial_posts: List[PersonaPhaseEstimate]
    revised_posts: List[PersonaPhaseEstimate]    # within-direction final per-persona estimates
    refined_posts: Optional[List[PersonaPhaseEstimate]] = None  # forward-only: refined after seeing backward
    phase_summary: Optional[str] = None           # holistic LLM summary of this phase's discussion
    posting_order: Optional[List[str]] = None     # persona names in posting order (for reproducibility)
    state_type: Literal["simulated", "document_grounded"] = "simulated"
    grounded_values: Dict[str, Union[float, int, str]] = Field(default_factory=dict)
    grounded_evidence: List[Dict[str, Any]] = Field(default_factory=list)


class BranchVariableEffect(BaseModel):
    """Qualitative variable-level implication fixed before persona estimates."""

    variable: str
    direction: Literal["increase", "decrease", "stable", "uncertain"]
    timing: Literal["current_phase", "delayed", "not_yet_observable"] = "current_phase"
    strength: Literal["incremental", "material", "boundary", "reference"]
    rationale: str


class BranchMechanismAnchor(BaseModel):
    """Shared causal condition applied to every persona in one branch node."""

    transition_mode: Literal["enabling", "baseline", "constraining"]
    coverage_level: Literal["incremental", "material", "boundary", "reference"]
    mechanism_family: str
    trigger: str
    affected_causal_link: str
    expected_propagation: List[str] = Field(default_factory=list)
    parent_support: str
    plausibility_limit: str
    common_parent_conditions: List[str] = Field(default_factory=list)
    sibling_contrast: str = ""
    expected_variable_effects: List[BranchVariableEffect] = Field(default_factory=list)


class PathwayBranchNode(BaseModel):
    node_id: str
    parent_id: Optional[str] = None
    # Legacy whole-run labels remain readable in archived output; newly
    # generated phase-local branches use condition-oriented labels below.
    transition_mode: Optional[Literal[
        "enabling", "baseline", "constraining",
        "optimistic", "neutral", "conservative",
    ]] = None
    phase_index: int
    parent_context_hash: str
    mechanism_anchor: Optional[BranchMechanismAnchor] = None
    phase_result: Optional[PhaseRoundResult] = None


class PhaseCrossCheck(BaseModel):
    phase: str
    variable: str
    forward_agg: Optional[float] = None
    backward_agg: Optional[float] = None
    gap: Optional[float] = None                  # forward_agg - backward_agg
    relative_gap: Optional[float] = None         # |gap| / max(|F|,|B|,1)
    consistency: Optional[float] = None          # max(0, 1 - relative_gap)
    aggregated_value: Optional[float] = None     # (F+B)/2 if both; else whichever present


class ScenarioToCResult(BaseModel):
    scenario_id: str
    participants: List[ConcretePersona]
    forward_pass: Optional[List[PhaseRoundResult]] = None    # len 5 or None if ablated
    backward_pass: Optional[List[PhaseRoundResult]] = None   # len 5 or None if ablated
    fwd_bwd_fwd_pass: Optional[List[PhaseRoundResult]] = None  # 5th chain: forward re-run seeded from backward Inputs
    fwd_bwd_fwd_bwd_pass: Optional[List[PhaseRoundResult]] = None  # 6th chain: backward re-run seeded from 5th chain's Impact
    cross_checks: List[PhaseCrossCheck]                      # per (phase, variable)
    scenario_impact: Dict[str, Optional[float]]              # aggregated values for Impact phase
    scenario_confidence: Dict[str, Optional[float]]          # consistency per Impact variable


class ToCSimulationResult(BaseModel):
    simulation_target: str
    inner_schema: ToCInnerSchema
    constructed_personas: List[ConcretePersona]
    scenarios: List[ScenarioToCResult]
    aggregated_final: Dict[str, Optional[float]]             # confidence-weighted across scenarios
    implementation_gap_report: List[PhaseCrossCheck]         # aggregated gaps across scenarios
    narrative_synthesis: str
    phase_stakeholder_summaries: Dict[str, Dict[str, Dict[str, str]]] = {}  # phase → direction (forward|backward|refined) → stakeholder_type → summary
