"""Per-persona role+norm refiner — single LLM call grounded in KG institutional context.

Refines three narrative fields in place on a persona's `demographics` dict:
  - professional_persona
  - skills_and_expertise
  - career_goals_and_ambitions

Originals are preserved under `*_pre_refine` keys so reviewers can diff. Demographic
fields (sex, age, education_level, occupation, district, province) are never touched.

KG context is filtered to **regulative** edges with `deontic in {must, can, cannot}`
attached to the actor node, grouped by deontic. `activation_condition` is
intentionally dropped — focus is role + norm, not eligibility.
"""

from __future__ import annotations

import json
from typing import Any

from llm.client import chat_completion
from models.prompts import (
    PERSONA_REFINER_SYSTEM,
    PERSONA_REFINER_USER,
    PERSONA_REFINER_BATCH_SYSTEM,
    PERSONA_REFINER_BATCH_USER,
)

from persona.scope_sql_filter import PERSONA_OUTPUT_FIELDS as _PERSONA_OUTPUT_FIELDS

# Long free-text fields are passed separately; demographics block is the
# small structured anchors (sex, age, district, etc.).
_NON_DEMO_FIELDS = {
    "professional_persona",
    "skills_and_expertise",
    "family_persona",
    "career_goals_and_ambitions",
}
_DEMO_KEYS_FOR_CONTEXT = tuple(f for f in _PERSONA_OUTPUT_FIELDS if f not in _NON_DEMO_FIELDS)

_VALID_DEONTICS = ("must", "can", "cannot")


def refine_persona(
    persona: Any,
    kg_context: dict[str, Any],
    config: dict[str, Any],
    policy_globals: dict[str, list[dict[str, Any]]] | None = None,  # kept for CLI signature compat; unused
    shared_context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Refine three narrative fields of one persona. Returns the LLM payload, or None on failure.

    Mutates `persona.demographics` in place: stores originals under `*_pre_refine`,
    replaces live values with refined text, and attaches rationale + KG evidence.
    """
    demographics = persona.demographics or {}
    original_pp = demographics.get("professional_persona") or ""
    original_se = demographics.get("skills_and_expertise") or ""
    original_cg = demographics.get("career_goals_and_ambitions") or ""
    if not (original_pp or original_se or original_cg):
        return None

    actor_context = _slim_actor_context(kg_context)
    persona_demos = {key: demographics.get(key) for key in _DEMO_KEYS_FOR_CONTEXT if demographics.get(key)}

    messages = build_messages(
        actor_context=actor_context,
        professional_persona=original_pp,
        skills_and_expertise=original_se,
        career_goals_and_ambitions=original_cg,
        persona_demographics=persona_demos,
        shared_context=shared_context,
    )
    try:
        raw = chat_completion(messages, config)
        payload = json.loads(raw)
    except Exception as exc:
        print(f"  [refiner] failed for persona {getattr(persona, 'representative_type', '?')}: {exc}")
        return None

    refined_pp = str(payload.get("professional_persona") or original_pp).strip() or original_pp
    refined_se = str(payload.get("skills_and_expertise") or original_se).strip() or original_se
    refined_cg = str(payload.get("career_goals_and_ambitions") or original_cg).strip() or original_cg
    rationale = payload.get("rationale") or {}
    evidence = _normalize_evidence(payload.get("kg_evidence") or [])

    new_demographics = dict(demographics)
    new_demographics["professional_persona_pre_refine"] = original_pp
    new_demographics["skills_and_expertise_pre_refine"] = original_se
    new_demographics["career_goals_and_ambitions_pre_refine"] = original_cg
    new_demographics["professional_persona"] = refined_pp
    new_demographics["skills_and_expertise"] = refined_se
    new_demographics["career_goals_and_ambitions"] = refined_cg
    new_demographics["refiner_strategy"]            = str(payload.get("strategy") or "")
    new_demographics["refiner_strategy_secondary"]  = str(payload.get("strategy_secondary") or "") if payload.get("strategy_secondary") else None
    new_demographics["refiner_strategy_rationale"]  = str(payload.get("strategy_rationale") or "")
    new_demographics["refiner_rationale"] = {
        "professional_persona":         str(rationale.get("professional_persona") or "no_change_needed"),
        "skills_and_expertise":          str(rationale.get("skills_and_expertise") or "no_change_needed"),
        "career_goals_and_ambitions":    str(rationale.get("career_goals_and_ambitions") or "no_change_needed"),
    }
    new_demographics["refiner_kg_evidence"] = evidence

    persona.demographics = new_demographics
    return payload


def refine_persona_batch(
    personas: list[Any],
    kg_context: dict[str, Any],
    config: dict[str, Any],
    shared_context: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]] | None:
    """Refine N personas sharing the same KG actor context in one LLM call.

    `personas` is a list of objects with `.demographics` dict and
    `.representative_type` attribute (PersonaProfile or _StubPersona). Each
    persona's demographics is mutated in place on success.

    Returns a dict {persona_id: payload_per_persona} on success, or None on
    failure. On batch-level failure the caller is expected to fall back to
    per-persona refinement.
    """
    if not personas:
        return {}

    inputs: list[dict[str, Any]] = []
    by_pid: dict[str, Any] = {}
    for p in personas:
        demos = p.demographics or {}
        pid = str(demos.get("persona_id") or getattr(p, "representative_type", "") or "")
        if not pid:
            continue
        inputs.append({
            "persona_id": pid,
            "professional_persona":       demos.get("professional_persona") or "",
            "skills_and_expertise":       demos.get("skills_and_expertise") or "",
            "career_goals_and_ambitions": demos.get("career_goals_and_ambitions") or "",
            "demographics": demos,
        })
        by_pid[pid] = p
    if not inputs:
        return {}

    actor_context = _slim_actor_context(kg_context)
    messages = build_batch_messages(actor_context, inputs, shared_context=shared_context)

    try:
        raw = chat_completion(messages, config)
        payload = json.loads(raw)
    except Exception as exc:
        print(f"  [batch-refiner] failed ({len(inputs)} personas): {exc}")
        return None

    results = payload.get("results") or []
    if not isinstance(results, list) or not results:
        print(f"  [batch-refiner] no `results` array in response (n={len(inputs)})")
        return None

    out: dict[str, dict[str, Any]] = {}
    for entry in results:
        if not isinstance(entry, dict):
            continue
        pid = str(entry.get("persona_id") or "").strip()
        target = by_pid.get(pid)
        if target is None:
            continue
        demos = target.demographics or {}
        orig_pp = demos.get("professional_persona") or ""
        orig_se = demos.get("skills_and_expertise") or ""
        orig_cg = demos.get("career_goals_and_ambitions") or ""
        refined_pp = str(entry.get("professional_persona") or orig_pp).strip() or orig_pp
        refined_se = str(entry.get("skills_and_expertise") or orig_se).strip() or orig_se
        refined_cg = str(entry.get("career_goals_and_ambitions") or orig_cg).strip() or orig_cg
        rationale = entry.get("rationale") or {}
        evidence = _normalize_evidence(entry.get("kg_evidence") or [])

        new_demos = dict(demos)
        new_demos["professional_persona_pre_refine"] = orig_pp
        new_demos["skills_and_expertise_pre_refine"] = orig_se
        new_demos["career_goals_and_ambitions_pre_refine"] = orig_cg
        new_demos["professional_persona"] = refined_pp
        new_demos["skills_and_expertise"] = refined_se
        new_demos["career_goals_and_ambitions"] = refined_cg
        new_demos["refiner_strategy"]            = str(entry.get("strategy") or "")
        new_demos["refiner_strategy_secondary"]  = str(entry.get("strategy_secondary") or "") if entry.get("strategy_secondary") else None
        new_demos["refiner_strategy_rationale"]  = str(entry.get("strategy_rationale") or "")
        new_demos["refiner_rationale"] = {
            "professional_persona":         str(rationale.get("professional_persona") or "no_change_needed"),
            "skills_and_expertise":          str(rationale.get("skills_and_expertise") or "no_change_needed"),
            "career_goals_and_ambitions":    str(rationale.get("career_goals_and_ambitions") or "no_change_needed"),
        }
        new_demos["refiner_kg_evidence"] = evidence
        target.demographics = new_demos
        out[pid] = entry

    # Detect missing personas — caller may want to fall back per-persona for them.
    missing = [pid for pid in by_pid if pid not in out]
    if missing:
        print(f"  [batch-refiner] {len(missing)}/{len(by_pid)} personas missing from response")
    return out


def build_messages(
    actor_context: dict[str, Any],
    professional_persona: str,
    skills_and_expertise: str,
    career_goals_and_ambitions: str,
    persona_demographics: dict[str, Any],
    shared_context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build the messages list passed to chat_completion.

    Exposed as a standalone function so the dry-run test can render the same
    prompt without invoking the LLM.
    """
    focal_name = actor_context.get("name") if isinstance(actor_context, dict) else None
    focal_etype = actor_context.get("entity_type") if isinstance(actor_context, dict) else None
    ecosystem_block = _render_policy_ecosystem_block(shared_context, focal_name, focal_etype)
    return [
        {"role": "system", "content": PERSONA_REFINER_SYSTEM},
        {
            "role": "user",
            "content": PERSONA_REFINER_USER.format(
                policy_ecosystem=ecosystem_block or "(no ecosystem context provided)",
                actor_context=_render_actor_context_block(actor_context),
                professional_persona=professional_persona,
                skills_and_expertise=skills_and_expertise,
                career_goals_and_ambitions=career_goals_and_ambitions,
                persona_demographics=json.dumps(persona_demographics, ensure_ascii=False, indent=2),
            ),
        },
    ]


def build_batch_messages(
    actor_context: dict[str, Any],
    personas: list[dict[str, Any]],
    shared_context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build messages for a batched refine call.

    `personas` is a list of dicts each containing at minimum: persona_id,
    professional_persona, skills_and_expertise, career_goals_and_ambitions,
    demographics (dict). The actor_context is shared.
    """
    blocks: list[str] = []
    for i, p in enumerate(personas, start=1):
        demos = {
            k: p["demographics"].get(k)
            for k in _DEMO_KEYS_FOR_CONTEXT
            if p.get("demographics", {}).get(k)
        }
        blocks.append(
            f"=== PERSONA {i} (persona_id: {p.get('persona_id', '')}) ===\n"
            f"professional_persona: {p.get('professional_persona') or ''}\n"
            f"skills_and_expertise: {p.get('skills_and_expertise') or ''}\n"
            f"career_goals_and_ambitions: {p.get('career_goals_and_ambitions') or ''}\n"
            f"demographics: {json.dumps(demos, ensure_ascii=False)}"
        )
    personas_block = "\n\n".join(blocks)

    focal_name = actor_context.get("name") if isinstance(actor_context, dict) else None
    focal_etype = actor_context.get("entity_type") if isinstance(actor_context, dict) else None
    ecosystem_block = _render_policy_ecosystem_block(shared_context, focal_name, focal_etype)

    return [
        {"role": "system", "content": PERSONA_REFINER_BATCH_SYSTEM},
        {
            "role": "user",
            "content": PERSONA_REFINER_BATCH_USER.format(
                n_personas=len(personas),
                policy_ecosystem=ecosystem_block or "(no ecosystem context provided)",
                actor_context=_render_actor_context_block(actor_context),
                personas_block=personas_block,
            ),
        },
    ]


def _render_policy_ecosystem_block(
    shared_context: dict[str, Any] | None,
    focal_name: str | None,
    focal_etype: str | None,
) -> str:
    """Render `shared_context.org_kg_contexts` as a labeled actor list.

    Each actor lists name, entity_type, role_scope (if present),
    source_spans_text and edge_source. The focal actor (matched on name OR
    entity_type) is marked with ★ and labeled FOCAL. Returns empty string when
    shared_context is missing or empty.
    """
    if not isinstance(shared_context, dict):
        return ""
    contexts = shared_context.get("org_kg_contexts") or []
    if not isinstance(contexts, list) or not contexts:
        return ""

    lines: list[str] = []
    for c in contexts:
        if not isinstance(c, dict):
            continue
        name = (c.get("name") or "").strip() or "?"
        etype = (c.get("entity_type") or "").strip() or "?"
        is_focal = (focal_name and name == focal_name) or (focal_etype and etype == focal_etype)
        marker = "★" if is_focal else "·"
        suffix = " — FOCAL (this persona's role; full norms appear in actor context below)" if is_focal else ""
        lines.append(f"{marker} {name} ({etype}){suffix}")
        rs = (c.get("role_scope") or "").strip()
        if rs:
            lines.append(f"    role_scope: {rs}")
        sps = (c.get("source_spans_text") or "").strip()
        if sps:
            lines.append(f"    policy passages: {sps}")
        es = (c.get("edge_source") or "").strip()
        if es:
            lines.append(f"    KG edges: {es}")
        lines.append("")
    return "\n".join(lines).rstrip()


_DEONTIC_PREFIX = {"must": "M", "can": "C", "cannot": "X"}


def _normalize_evidence(raw: Any) -> list[dict[str, str]]:
    """Coerce kg_evidence output to a list of {"id": str, "fact": str} dicts.

    Accepts the new structured form ({"id": "M3", "fact": "..."}) and the
    legacy string form ("M3: <fact>"). Drops blanks.
    """
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        eid = ""
        efact = ""
        if isinstance(item, dict):
            eid = str(item.get("id") or "").strip()
            efact = str(item.get("fact") or "").strip()
        elif isinstance(item, str):
            s = item.strip()
            if ":" in s:
                head, _, tail = s.partition(":")
                eid, efact = head.strip(), tail.strip()
            else:
                efact = s
        if eid or efact:
            out.append({"id": eid, "fact": efact})
    return out


def _slim_actor_context(kg_context: dict[str, Any]) -> dict[str, Any]:
    """Trim KG context to role + must/can/cannot regulative norms.

    Filtering rules:
      - `edge_category == "regulative"` only
      - `deontic in {must, can, cannot}` only
      - **`direction == "outgoing"` only** — the persona's group must be the
        SUBJECT (edge.source). Excludes duties of other actors directed at the
        group (e.g. "NRF must evaluate research groups").
      - `activation_condition` intentionally omitted — focus is role + norm,
        not eligibility.

    Each deontic value is a **list of pre-formatted fact sentences** (one per
    rule). `execution_constraint` is appended to its fact as `(Constraint: ...)`
    when present. Duplicate facts within a deontic group are collapsed.

    The KG `summary` field is trimmed to its first sentence (≤ 400 chars).

    Final rendering into the M1/C1/X1-numbered block for the prompt is handled
    by `_render_actor_context_block()`.
    """
    sentences: dict[str, list[str]] = {d: [] for d in _VALID_DEONTICS}
    for e in kg_context.get("related_edges") or []:
        if (e.get("edge_category") or "").lower() != "regulative":
            continue
        d = (e.get("deontic") or "").lower()
        if d not in sentences:
            continue
        if e.get("direction") != "outgoing":
            continue
        fact = (e.get("fact") or "").strip()
        if not fact:
            continue
        cons = (e.get("execution_constraint") or "").strip()
        if cons:
            if not fact.endswith((".", "?", "!")):
                fact += "."
            fact = f"{fact} (Constraint: {cons})"
        if fact not in sentences[d]:
            sentences[d].append(fact)

    summary = (kg_context.get("summary") or "").strip()
    if summary:
        first_stop = summary.find(". ")
        summary = (summary[: first_stop + 1] if first_stop > 0 else summary)[:400]

    return {
        "name": kg_context.get("name"),
        "entity_type": kg_context.get("entity_type"),
        "summary": summary,
        "role_family": kg_context.get("role_family"),
        "eligibility_criteria": (kg_context.get("eligibility_criteria") or "").strip() or None,
        "role_scope":           (kg_context.get("role_scope") or "").strip() or None,
        "must":   sentences["must"],
        "can":    sentences["can"],
        "cannot": sentences["cannot"],
    }


_FREEFORM_ATTR_SKIP = {
    # Skip framework-only keys promoted into attributes by the IAD/IG reader,
    # so the free-form persona prompt is not polluted by leftover IAD labels.
    "iad_category", "node_family", "iad_rule_type", "agent_modality",
    "role_family", "is_actor", "persona_eligible", "rationale", "source_spans",
}


def _freeform_actor_context(kg_context: dict[str, Any]) -> dict[str, Any]:
    """Free-form actor context for the IAD/IG ablation (Stage 1 KG without
    institutional typed slots).

    Renders ALL grounded attributes as flat key/value lines and ALL outgoing
    edges as ``this actor [TYPE] [target]: [fact]`` lines, bypassing the
    must/can/cannot regulative filter. The IAD/IG-only slots (role_family,
    must, can, cannot, eligibility_criteria, role_scope) are kept as
    None/empty so the prompt-side switch can distinguish modes.
    """
    attrs = kg_context.get("attributes") or {}
    flat: list[str] = []
    for k, v in sorted(attrs.items()):
        if k in _FREEFORM_ATTR_SKIP or v in (None, "", [], {}):
            continue
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
        flat.append(f"  {k}: {v}")

    edge_lines: list[str] = []
    seen_edges: set[str] = set()
    for e in kg_context.get("related_edges") or []:
        if e.get("direction") != "outgoing":
            continue
        # persona/scope_definer.build_kg_context stores the edge label as
        # ``edge_name`` (not ``type``). Honor that first; fall back to ``type``
        # for direct callers that pass raw graph edges.
        et = (e.get("edge_name") or e.get("type") or "RELATES_TO").strip()
        target = (e.get("target_name") or e.get("target") or "?").strip()
        fact = (e.get("fact") or "").strip()
        line = f"  this actor {et} {target}"
        if fact:
            line += f": {fact}"
        if line in seen_edges:
            continue
        seen_edges.add(line)
        edge_lines.append(line)

    summary = (kg_context.get("summary") or "").strip()
    if summary:
        first_stop = summary.find(". ")
        summary = (summary[: first_stop + 1] if first_stop > 0 else summary)[:400]

    return {
        "name": kg_context.get("name"),
        "entity_type": kg_context.get("entity_type"),
        "summary": summary,
        "attributes_flat": flat,
        "edges_flat": edge_lines,
        # IAD/IG-only slots — kept so downstream code distinguishes modes:
        "role_family": None,
        "eligibility_criteria": None,
        "role_scope": None,
        "must": [], "can": [], "cannot": [],
        "_freeform": True,
    }


def _render_freeform_actor_context_block(ctx: dict[str, Any]) -> str:
    """Render the free-form actor_context: flat attributes + flat outgoing edges.

    Plugs into the same ``{actor_context}`` prompt slot as
    ``_render_actor_context_block`` but uses a non-numbered Grounded
    attributes / Policy relations layout instead of the IAD/IG M1/C1/X1 block.
    """
    parts: list[str] = []
    parts.append(f"Actor: {ctx.get('name') or '?'} (entity type: {ctx.get('entity_type') or '?'})")
    parts.append(f"Summary: {ctx.get('summary') or '(none)'}")
    parts.append("")
    parts.append("Grounded attributes:")
    attrs_flat = ctx.get("attributes_flat") or []
    if attrs_flat:
        parts.extend(attrs_flat)
    else:
        parts.append("  (none)")
    parts.append("")
    parts.append("Policy relations (this actor is the subject):")
    edges_flat = ctx.get("edges_flat") or []
    if edges_flat:
        parts.extend(edges_flat)
    else:
        parts.append("  (none)")
    return "\n".join(parts)


def _actor_context_for(
    kg_context: dict[str, Any], mode: str = "auto"
) -> tuple[dict[str, Any], "callable[[dict[str, Any]], str]"]:
    """Dispatch (context_dict, renderer) by mode.

    - ``mode == "freeform"``: always free-form.
    - ``mode == "iadig"``: always IAD/IG (legacy).
    - ``mode == "auto"`` (default): sniff ``kg_context["_builder"]`` —
      ``"local_freeform"`` → free-form, anything else → IAD/IG.
    """
    if mode == "freeform" or (
        mode == "auto" and kg_context.get("_builder") == "local_freeform"
    ):
        return _freeform_actor_context(kg_context), _render_freeform_actor_context_block
    return _slim_actor_context(kg_context), _render_actor_context_block


def _render_actor_context_block(ctx: dict[str, Any]) -> str:
    """Render the slim actor_context into a numbered-list text block.

    Output shape (multi-line):

        Actor: <name> (entity type: <type>)
        Role family: <role_family>
        Summary: <summary>

        Norms (this actor is the subject; cite by ID in kg_evidence):

        [must]
          M1: <fact>. (Constraint: ...)
          M2: <fact>.

        [can]
          C1: <fact>.

        [cannot]
          (none)
    """
    parts: list[str] = []
    parts.append(f"Actor: {ctx.get('name') or '?'} (entity type: {ctx.get('entity_type') or '?'})")
    parts.append(f"Role family: {ctx.get('role_family') or '?'}")
    if ctx.get("eligibility_criteria"):
        parts.append(f"Eligibility: {ctx['eligibility_criteria']}")
    if ctx.get("role_scope"):
        parts.append(f"Role scope: {ctx['role_scope']}")
    parts.append(f"Summary: {ctx.get('summary') or '(none)'}")
    parts.append("")
    parts.append("Norms (this actor is the subject; cite by ID in kg_evidence):")
    for deontic in _VALID_DEONTICS:
        parts.append("")
        parts.append(f"[{deontic}]")
        items = ctx.get(deontic) or []
        if not items:
            parts.append("  (none)")
            continue
        prefix = _DEONTIC_PREFIX[deontic]
        for i, sentence in enumerate(items, start=1):
            parts.append(f"  {prefix}{i}: {sentence}")
    return "\n".join(parts)


def extract_policy_globals(policy_graph: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Backwards-compat stub. Returns empty dict; the new refiner does not consume
    policy-global resource/outcome lists. Kept so callers that already pass
    `policy_globals=` keep working without modification.
    """
    return {"resources": [], "outcomes": []}
