from models.schemas import (
    GroundedEvidence,
    GroundedInputField,
    GroundedInputState,
    GroundedPolicyParameter,
    PhaseInnerSpec,
    PhaseVariableSpec,
    ToCInnerSchema,
)
from stage2_toc.grounded_inputs import (
    _numeric_supported,
    _quote_supported,
    _select_complementary_fields,
    _validated_parameter,
    apply_grounded_inputs_to_schema,
    grounded_input_phase_result,
)


def test_numeric_support_handles_korean_age_prefix_and_scaled_counts():
    assert _numeric_supported(35, "일반(만35세 이상) 20%")
    assert _numeric_supported(19, "청년(만19~34세) 30%")
    assert _numeric_supported(800000, "약 80만명 전환 완료")


def test_numeric_support_handles_german_decimal_comma_and_grouping():
    assert _numeric_supported(49_825_000, "zusätzlicher Erfüllungsaufwand von 49,825 Mio. Euro")
    assert _numeric_supported(97_907_000, "Umstellungsaufwand in Höhe von 97,907 Mio. Euro")
    assert _numeric_supported(689_000, "Davon entfallen 689.000 Euro auf Informationspflichten")


def test_complementary_selection_deduplicates_geographic_hierarchy():
    def field(name: str, value: int, unit: str) -> GroundedInputField:
        return GroundedInputField(
            name=name,
            value=value,
            unit=unit,
            type="int",
            description=name,
            causal_relevance="implementation capacity",
            validation_status="validated",
        )

    candidates = [
        field("participating_cities_provinces", 17, "cities/provinces"),
        field("participating_districts", 189, "cities/counties/districts"),
        field("participating_card_issuers", 10, "card issuers"),
        field("converted_users", 800000, "people"),
    ]
    selected, dropped = _select_complementary_fields(candidates, 3)

    assert [item.name for item in selected] == [
        "participating_districts",
        "participating_card_issuers",
        "converted_users",
    ]
    assert [item.name for item in dropped] == ["participating_cities_provinces"]


def test_parameter_normalizes_explicit_fraction_to_percent():
    parameter = _validated_parameter(
        {
            "name": "refund_rate",
            "value": 0.2,
            "unit": "proportion",
            "description": "Refund rate.",
            "evidence": [{
                "source": "policy.pdf",
                "page": 1,
                "evidence_quote": "일반인 20%를 환급한다.",
            }],
        },
        "일반인 20%를 환급한다.",
        {"policy.pdf"},
    )

    assert parameter.validation_status == "validated"
    assert parameter.value == 20
    assert parameter.unit == "percent"
from stage2_toc.modules import _aggregate_phase_round, _format_prior_phases_summarized


def _state() -> GroundedInputState:
    evidence = GroundedEvidence(
        source="policy.pdf",
        page=1,
        evidence_quote="우선 10개 카드사의 누리집을 통해 전용 카드를 발급받는다.",
    )
    return GroundedInputState(
        policy_name="Example",
        fields=[GroundedInputField(
            name="card_issuer_partners_count",
            value=10,
            unit="card_issuers",
            type="int",
            description="Documented card issuer partners at launch.",
            causal_relevance="Conditions enrollment and refund delivery capacity.",
            evidence=[evidence],
            validation_status="validated",
        )],
        policy_parameters=[GroundedPolicyParameter(
            name="minimum_monthly_rides",
            value=15,
            unit="rides_per_month",
            description="Minimum monthly use threshold.",
            evidence=[GroundedEvidence(
                source="policy.pdf", page=1,
                evidence_quote="월 15회 이상 이용해야 한다.",
            )],
            validation_status="validated",
        )],
        summary="Document-grounded initial state.",
        source_documents=["policy.pdf"],
    )


def test_numeric_validation_supports_korean_scale_and_normalized_quotes() -> None:
    assert _numeric_supported(800_000, "4.28일 기준 약 80만명 전환 완료")
    assert _numeric_supported(53.3, "저소득층 53.3%")
    assert not _numeric_supported(12, "우선 10개 카드사")
    assert _quote_supported("우선 10개 카드사", "문서에는\n우선   10개 카드사가 있다")


def test_grounded_inputs_replace_only_input_schema() -> None:
    schema = ToCInnerSchema(phases=[
        PhaseInnerSpec(
            phase="Inputs",
            prediction_fields=[PhaseVariableSpec(
                name="estimated_budget", unit="KRW", type="float", description="Old field",
            )],
            discussion_points=["Old question"],
        ),
        *[
            PhaseInnerSpec(phase=phase, prediction_fields=[], discussion_points=[])
            for phase in ("Activities", "Outputs", "Outcomes", "Impact")
        ],
    ])
    grounded = apply_grounded_inputs_to_schema(schema, _state())
    assert grounded.phases[0].prediction_fields[0].name == "card_issuer_partners_count"
    assert grounded.phases[1:] == schema.phases[1:]


def test_grounded_phase_skips_personas_and_is_inherited_verbatim() -> None:
    state = _state()
    phase = grounded_input_phase_result(state)
    assert phase.revised_posts == []
    assert phase.state_type == "document_grounded"
    assert _aggregate_phase_round(phase) == {"card_issuer_partners_count": 10}

    schema = ToCInnerSchema(phases=[
        PhaseInnerSpec(
            phase="Inputs",
            prediction_fields=[PhaseVariableSpec(
                name="card_issuer_partners_count",
                unit="card_issuers",
                type="int",
                description="Documented partners.",
            )],
            discussion_points=["How does partner capacity affect implementation?"],
        ),
        *[
            PhaseInnerSpec(phase=name, prediction_fields=[], discussion_points=[])
            for name in ("Activities", "Outputs", "Outcomes", "Impact")
        ],
    ])
    rendered = _format_prior_phases_summarized([phase], "Persona A", schema)
    assert "Document-grounded values" in rendered
    assert "card_issuer_partners_count" in rendered
    assert "Your own post" not in rendered
