"""Document-grounded initialization for Theory of Change simulations."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from llm.client import chat_completion
from llm.pdf_loader import load_pdfs
from models.schemas import (
    GroundedEvidence,
    GroundedInputField,
    GroundedInputState,
    GroundedPolicyParameter,
    PhaseInnerSpec,
    PhaseRoundResult,
    PhaseVariableSpec,
    PolicyContext,
    ToCInnerSchema,
)


GROUNDING_SYSTEM = """You extract document-grounded initial conditions for a policy Theory of Change.
Return valid JSON only. Never invent or estimate a value. Every returned value must be explicitly stated in the supplied policy document and accompanied by a verbatim evidence quote and page number.

Distinguish:
- grounded_fields: numeric pre-implementation resources, counts of participating institutions or regions, installed capacity, numeric pre-launch population/state, or explicitly stated ex-ante implementation resource requirements and compliance burdens that causally condition implementation;
- policy_parameters: fixed eligibility rules, rates, thresholds, caps, dates, or benefit formulas;
- excluded_candidates: outcomes, post-launch achievements, examples, duplicate scope measures, or facts that are not useful implementation inputs.

Dates are always policy_parameters, never grounded_fields. An explicitly stated count remains a valid quantitative field even when the document also lists the counted entities.
Projected implementation resource requirements and transition/compliance burdens may be Inputs when explicitly stated. Do not treat projected outcomes or implementation achievements as Inputs. Do not fill a requested quota when the document lacks enough supported fields."""


GROUNDING_USER = """Extract candidate document-grounded initial conditions for this policy simulation.

Simulation scope:
{scope_instruction}

Selection rules:
- Return at most {candidate_limit} grounded_fields, ranked by causal relevance to downstream Activities. The system will select {max_fields} complementary fields after evidence validation.
- Prefer complementary fields over redundant measures of the same scope.
- Treat proportional variants of one rule (for example, one-, two-, and three-period
  versions of the same cap) as one mechanism family. Return the base or most causally
  informative variant as a grounded field and preserve the remaining variants as
  policy parameters when they are useful context.
- Prefer an explicitly documented operational capacity, deadline duration, or
  compliance burden over a second numeric variant of an already represented rule.
- Do not put counts of implementation partners or participating jurisdictions in policy_parameters; they are grounded_fields.
- When a parent jurisdiction count and its more granular participating-area count describe the same geographic coverage, include only the more granular count.
- grounded_fields must have numeric int or float values. Put dates, categorical rules, and formulas in policy_parameters.
- Field names must be snake_case and values must preserve the document's scale.
- Count fields must use type=int and unscaled integer values (for example, 80만 people -> 800000).
- Explicit counts of participating organizations or jurisdictions are valid grounded_fields, including when their names are also listed.
- Include fixed policy parameters separately; they are context, not predicted Inputs.
- evidence_quote must be copied verbatim and completely from the supplied document; do not omit words or parenthetical text inside the quoted sentence.
- source must be the source filename shown in the document text.
- kg_links may contain only IDs shown in the KG context.

Return exactly:
{{
  "policy_name": "...",
  "grounded_fields": [
    {{
      "name": "...", "value": 0, "unit": "...", "type": "int|float|str",
      "description": "...", "effective_date": null, "scope": "...",
      "causal_relevance": "...", "kg_links": ["..."],
      "evidence": [{{"source": "file.pdf", "page": 1, "evidence_quote": "verbatim quote"}}]
    }}
  ],
  "policy_parameters": [
    {{
      "name": "...", "value": 0, "unit": "...", "description": "...",
      "effective_date": null,
      "evidence": [{{"source": "file.pdf", "page": 1, "evidence_quote": "verbatim quote"}}]
    }}
  ],
  "excluded_candidates": [{{"name": "...", "reason": "..."}}],
  "unresolved_candidates": [{{"name": "...", "reason": "..."}}]
}}

--- Policy context ---
{policy_context}

--- Policy KG context ---
{kg_context}

--- Policy documents ---
{policy_documents}
"""


_SPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(
    r"(?<![\d.,])(?P<number>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<scale>trillion|billion|million|thousand|조|억|만|천)?",
    re.IGNORECASE,
)
_SCALE = {
    "trillion": 1_000_000_000_000,
    "billion": 1_000_000_000,
    "million": 1_000_000,
    "thousand": 1_000,
    "조": 1_000_000_000_000,
    "억": 100_000_000,
    "만": 10_000,
    "천": 1_000,
}

_GERMAN_NUMBER_RE = re.compile(
    r"(?<![\d.,])(?P<number>(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?)|(?:\d+(?:,\d+)?))\s*"
    r"(?P<scale>Mio\.?|Million(?:en)?|Mrd\.?|Milliard(?:e|en)?|Tsd\.?)?",
    re.IGNORECASE,
)
_GERMAN_SCALE = {
    "mio": 1_000_000,
    "million": 1_000_000,
    "millionen": 1_000_000,
    "mrd": 1_000_000_000,
    "milliarde": 1_000_000_000,
    "milliarden": 1_000_000_000,
    "tsd": 1_000,
}


def _normalize(text: str) -> str:
    return _SPACE_RE.sub(" ", text or "").strip()


def _document_sources(policy_dir: Path) -> list[str]:
    paths = sorted(policy_dir.glob("*.pdf"))
    if not paths:
        paths = sorted(policy_dir.glob("*.md")) + sorted(policy_dir.glob("*.markdown"))
    return [path.name for path in paths]


def _compact_kg(policy_graph_path: Path | None) -> tuple[str, set[str]]:
    if policy_graph_path is None or not policy_graph_path.exists():
        return "{}", set()
    graph = json.loads(policy_graph_path.read_text(encoding="utf-8"))
    nodes = [
        {
            "id": node.get("id"),
            "name": node.get("name"),
            "type": node.get("type"),
            "summary": node.get("summary"),
            "attributes": node.get("attributes") or {},
        }
        for node in graph.get("nodes") or []
    ]
    edges = [
        {
            "id": edge.get("id"),
            "type": edge.get("type"),
            "source": edge.get("source"),
            "target": edge.get("target"),
            "fact": edge.get("fact"),
            "activation_condition": edge.get("activation_condition"),
            "execution_constraint": edge.get("execution_constraint"),
        }
        for edge in graph.get("edges") or []
    ]
    ids = {
        str(item.get("id"))
        for item in [*nodes, *edges]
        if item.get("id")
    }
    return json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False), ids


def _numeric_mentions(text: str) -> list[float]:
    values = []
    for match in _NUMBER_RE.finditer(text or ""):
        number = float(match.group("number").replace(",", ""))
        scale = (match.group("scale") or "").lower()
        values.append(number * _SCALE.get(scale, 1))
    for match in _GERMAN_NUMBER_RE.finditer(text or ""):
        raw = match.group("number")
        if "." in raw and "," in raw:
            normalized = raw.replace(".", "").replace(",", ".")
        elif "," in raw:
            normalized = raw.replace(",", ".")
        elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", raw):
            normalized = raw.replace(".", "")
        else:
            normalized = raw
        scale = (match.group("scale") or "").lower().rstrip(".")
        values.append(float(normalized) * _GERMAN_SCALE.get(scale, 1))
    return values


def _numeric_supported(value: int | float | str, quote: str) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return True
    for candidate in _numeric_mentions(quote):
        tolerance = max(1e-6, abs(float(value)) * 1e-6)
        if math.isclose(float(value), candidate, rel_tol=0, abs_tol=tolerance):
            return True
    return False


def _quote_supported(quote: str, policy_text: str) -> bool:
    normalized_quote = _normalize(quote)
    return bool(normalized_quote) and normalized_quote in _normalize(policy_text)


def _validate_evidence(
    value: int | float | str,
    evidence_rows: list[dict[str, Any]],
    policy_text: str,
    source_documents: set[str],
) -> tuple[list[GroundedEvidence], list[str], bool]:
    evidence = []
    notes = []
    valid = False
    for raw in evidence_rows:
        item = GroundedEvidence.model_validate(raw)
        source_name = Path(item.source).name
        quote_ok = _quote_supported(item.evidence_quote, policy_text)
        value_ok = _numeric_supported(value, item.evidence_quote)
        source_ok = source_name in source_documents
        if not source_ok:
            notes.append(f"unknown source: {item.source}")
        if not quote_ok:
            notes.append(f"evidence quote not found verbatim after whitespace normalization: {item.evidence_quote}")
        if not value_ok:
            notes.append(f"value {value!r} is not explicitly supported by quote: {item.evidence_quote}")
        evidence.append(item)
        valid = valid or (source_ok and quote_ok and value_ok)
    if not evidence:
        notes.append("missing evidence")
    return evidence, notes, valid


def _validated_field(
    raw: dict[str, Any],
    policy_text: str,
    source_documents: set[str],
    kg_ids: set[str],
) -> GroundedInputField:
    evidence, notes, valid = _validate_evidence(
        raw.get("value"), raw.get("evidence") or [], policy_text, source_documents,
    )
    field = GroundedInputField(
        **{
            key: value for key, value in raw.items()
            if key not in {"evidence", "kg_links", "validation_status", "validation_notes"}
        },
        evidence=evidence,
        kg_links=[link for link in (raw.get("kg_links") or []) if link in kg_ids],
        validation_status="validated" if valid else "rejected",
        validation_notes=notes,
    )
    if field.type == "int" and isinstance(field.value, float) and field.value.is_integer():
        field.value = int(field.value)
    if field.type == "int" and not isinstance(field.value, int):
        field.validation_status = "rejected"
        field.validation_notes.append("int field does not contain an integer value")
    if field.type == "str" or not isinstance(field.value, (int, float)):
        field.validation_status = "rejected"
        field.validation_notes.append(
            "grounded Input fields must be numeric; dates and categorical values belong in policy_parameters"
        )
    return field


def _validated_parameter(
    raw: dict[str, Any],
    policy_text: str,
    source_documents: set[str],
) -> GroundedPolicyParameter:
    raw = dict(raw)
    raw_value = raw.get("value")
    quotes = " ".join(
        str(item.get("evidence_quote") or "") for item in (raw.get("evidence") or [])
    )
    if (
        isinstance(raw_value, (int, float))
        and not isinstance(raw_value, bool)
        and 0 < float(raw_value) <= 1
        and "%" in quotes
        and _numeric_supported(float(raw_value) * 100, quotes)
    ):
        raw["value"] = round(float(raw_value) * 100, 10)
        raw["unit"] = "percent"
    evidence, notes, valid = _validate_evidence(
        raw.get("value"), raw.get("evidence") or [], policy_text, source_documents,
    )
    parameter = GroundedPolicyParameter(
        **{key: value for key, value in raw.items() if key not in {"evidence", "validation_status", "validation_notes"}},
        evidence=evidence,
        validation_status="validated" if valid else "rejected",
        validation_notes=notes,
    )
    if parameter.unit.lower() in {"bool", "boolean"} and parameter.value in {0, 1, 0.0, 1.0}:
        parameter.value = bool(parameter.value)
    return parameter


def _field_family(field: GroundedInputField) -> str:
    text = " ".join((field.name, field.unit, field.description)).lower()
    families = {
        "geographic_coverage": (
            "jurisdiction", "municipalit", "local government", "province", "region",
            "district", "city", "cities", "county", "counties", "시·도", "시·군·구",
            "지자체",
        ),
        "delivery_partners": (
            "issuer", "provider", "operator", "partner organization", "card compan",
            "카드사", "운영기관",
        ),
        "benefit_cap": (
            "cost sharing cap", "cost-sharing cap", "copay cap", "payment cap",
            "monthly cap", "supply cap", "상한", "한도",
        ),
        "administrative_timing": (
            "reimbursement deadline", "processing deadline", "calendar days",
            "processing time", "transition duration", "처리 기한", "환급 기한",
        ),
        "existing_users": (
            "user", "member", "enrol", "beneficiar", "converted", "가입자", "이용자",
            "회원",
        ),
        "budget_capacity": ("budget", "funding", "appropriation", "예산", "재원"),
        "staff_capacity": ("staff", "workforce", "personnel", "인력"),
        "installed_capacity": ("capacity", "facility", "site", "equipment", "설비", "시설"),
    }
    for family, terms in families.items():
        if any(term in text for term in terms):
            return family
    return f"other:{field.name}"


def _family_specificity(field: GroundedInputField, family: str) -> int:
    if family != "geographic_coverage":
        return 0
    text = " ".join((field.name, field.unit, field.description)).lower()
    granular = ("municipalit", "district", "county", "counties", "city", "cities", "local government", "시·군·구", "지자체")
    broad = ("province", "state", "시·도")
    granular_score = sum(term in text for term in granular)
    broad_score = sum(term in text for term in broad)
    return (2 * granular_score) - broad_score


def _select_complementary_fields(
    fields: list[GroundedInputField],
    max_fields: int,
) -> tuple[list[GroundedInputField], list[GroundedInputField]]:
    by_family: dict[str, tuple[int, GroundedInputField]] = {}
    family_order: list[str] = []
    for field in fields:
        family = _field_family(field)
        score = _family_specificity(field, family)
        if family not in by_family:
            family_order.append(family)
            by_family[family] = (score, field)
        elif score > by_family[family][0]:
            by_family[family] = (score, field)
    selected = [by_family[family][1] for family in family_order][:max_fields]
    selected_ids = {id(field) for field in selected}
    dropped = [field for field in fields if id(field) not in selected_ids]
    return selected, dropped


def _grounded_summary(
    fields: list[GroundedInputField],
    parameters: list[GroundedPolicyParameter],
) -> str:
    field_text = "; ".join(
        f"{field.name}={field.value} {field.unit}" for field in fields
        if field.validation_status == "validated"
    ) or "no validated quantitative input fields"
    parameter_text = "; ".join(
        f"{parameter.name}={parameter.value} {parameter.unit}" for parameter in parameters
        if parameter.validation_status == "validated"
    ) or "no validated fixed policy parameters"
    return (
        "Document-grounded initial state. Inputs: " + field_text
        + ". Fixed policy parameters: " + parameter_text + "."
    )


def build_grounded_input_state(
    config: dict[str, Any],
    policy_context: PolicyContext,
    policy_graph_path: Path | None = None,
) -> GroundedInputState:
    grounding_cfg = (config.get("stage2_toc") or {}).get("grounded_inputs") or {}
    policy_dir = Path(config["paths"]["input_policy_dir"])
    policy_text = load_pdfs(str(policy_dir), config)
    source_documents = _document_sources(policy_dir)
    kg_context, kg_ids = _compact_kg(policy_graph_path)
    max_fields = int(grounding_cfg.get("max_fields", 3))
    candidate_limit = int(grounding_cfg.get("candidate_limit", max_fields * 2))
    messages = [
        {"role": "system", "content": GROUNDING_SYSTEM},
        {
            "role": "user",
            "content": GROUNDING_USER.format(
                max_fields=max_fields,
                candidate_limit=candidate_limit,
                scope_instruction=str(
                    grounding_cfg.get("scope_instruction")
                    or "Use the full supplied policy scope."
                ),
                policy_context=json.dumps(policy_context.model_dump(), ensure_ascii=False),
                kg_context=kg_context,
                policy_documents=policy_text,
            ),
        },
    ]
    llm_override = grounding_cfg.get("llm") or {}
    extraction_config = (
        {**config, "llm": {**(config.get("llm") or {}), **llm_override}}
        if llm_override else config
    )
    payload = json.loads(chat_completion(messages, extraction_config), strict=False)
    source_set = set(source_documents)
    fields = [
        _validated_field(raw, policy_text, source_set, kg_ids)
        for raw in (payload.get("grounded_fields") or [])[:candidate_limit]
    ]
    parameters = [
        _validated_parameter(raw, policy_text, source_set)
        for raw in (payload.get("policy_parameters") or [])
    ]
    validated_candidates = [field for field in fields if field.validation_status == "validated"]
    validated_fields, duplicate_fields = _select_complementary_fields(
        validated_candidates, max_fields,
    )
    rejected_fields = [field for field in fields if field.validation_status != "validated"]
    validated_parameters = [
        parameter for parameter in parameters
        if parameter.validation_status == "validated"
    ]
    rejected_parameters = [
        parameter for parameter in parameters
        if parameter.validation_status != "validated"
    ]
    unresolved = list(payload.get("unresolved_candidates") or [])
    unresolved.extend(
        {"name": field.name, "reason": "; ".join(field.validation_notes)}
        for field in rejected_fields
    )
    excluded = list(payload.get("excluded_candidates") or [])
    excluded.extend(
        {
            "name": field.name,
            "reason": f"not selected: overlaps the {_field_family(field)} mechanism family or falls below the {max_fields}-field limit",
        }
        for field in duplicate_fields
    )
    unresolved.extend(
        {
            "name": parameter.name,
            "reason": "policy parameter rejected: " + "; ".join(parameter.validation_notes),
        }
        for parameter in rejected_parameters
    )
    strict = bool(grounding_cfg.get("strict", True))
    if strict and not validated_fields:
        raise ValueError("Grounded Input Extraction produced no source-validated fields")
    return GroundedInputState(
        policy_name=str(payload.get("policy_name") or policy_context.policy_name),
        fields=validated_fields,
        policy_parameters=validated_parameters,
        summary=_grounded_summary(validated_fields, validated_parameters),
        source_documents=source_documents,
        excluded_candidates=excluded,
        unresolved_candidates=unresolved,
        extraction_metadata={
            "model": extraction_config.get("llm", {}).get("model"),
            "max_fields": max_fields,
            "candidate_limit": candidate_limit,
            "candidate_field_count": len(fields),
            "validated_candidate_count": len(validated_candidates),
            "validated_field_count": len(validated_fields),
            "deduplicated_field_count": len(duplicate_fields),
            "rejected_field_count": len(rejected_fields),
            "candidate_parameter_count": len(parameters),
            "validated_parameter_count": sum(
                parameter.validation_status == "validated" for parameter in parameters
            ),
            "policy_graph": str(policy_graph_path) if policy_graph_path else None,
        },
    )


def apply_grounded_inputs_to_schema(
    schema: ToCInnerSchema,
    state: GroundedInputState,
) -> ToCInnerSchema:
    fields = [
        PhaseVariableSpec(
            name=field.name,
            unit=field.unit,
            type=field.type,
            description=field.description,
        )
        for field in state.fields
        if field.validation_status == "validated"
    ]
    points = [
        f"How does the documented input '{field.name}' condition downstream implementation activities? {field.causal_relevance}"
        for field in state.fields
        if field.validation_status == "validated"
    ]
    phases = []
    for phase in schema.phases:
        if phase.phase == "Inputs":
            phases.append(PhaseInnerSpec(
                phase="Inputs",
                prediction_fields=fields,
                discussion_points=points,
            ))
        else:
            phases.append(phase.model_copy(deep=True))
    return ToCInnerSchema(phases=phases)


def grounded_input_phase_result(state: GroundedInputState) -> PhaseRoundResult:
    evidence = [
        {
            "field": field.name,
            "source": item.source,
            "page": item.page,
            "evidence_quote": item.evidence_quote,
        }
        for field in state.fields
        for item in field.evidence
    ]
    return PhaseRoundResult(
        phase="Inputs",
        direction="forward",
        initial_posts=[],
        revised_posts=[],
        phase_summary=state.summary,
        posting_order=[],
        state_type="document_grounded",
        grounded_values=state.values,
        grounded_evidence=evidence,
    )
