"""Compute `policy_role_actions` (one shared list per persona-eligible entity).

Two outputs per entity:
  - `policy_role_actions_raw`  — deterministic edge facts with deontic suffixes
  - `policy_role_actions`      — LLM-synthesized 3-7 polished action statements

Both are attached to every persona of that entity so simulation participants can
ground their behaviour in the policy-specific actions captured in the KG.
"""

from __future__ import annotations

import json
import re
from typing import Any

from llm.client import chat_completion
from models.prompts import (
    ROLE_ACTIONS_SYNTHESIZER_SYSTEM,
    ROLE_ACTIONS_SYNTHESIZER_USER,
)


_DEONTIC_SUFFIX = {
    "must": " (required)",
    "required": " (required)",
    "shall": " (required)",
    "cannot": " (prohibited)",
    "prohibited": " (prohibited)",
    "can": " (allowed)",
    "may": " (allowed)",
}


def compute_raw_actions(entity: Any, kg_context: dict[str, Any] | None = None) -> list[str]:
    """Return verbatim action statements from the entity's related edges.

    Each statement is `fact` (or `source_text` if `fact` is missing), with a
    deontic suffix appended for normative edges. Near-duplicates are deduped
    on a normalized (lowercase, whitespace-collapsed) key.
    """
    edges = list(getattr(entity, "related_edges", None) or [])
    if not edges and kg_context:
        edges = list(kg_context.get("related_edges") or [])

    seen: set[str] = set()
    out: list[str] = []
    for e in edges:
        text = (e.get("fact") or e.get("source_text") or "").strip()
        if not text:
            continue
        sentence = text if text.endswith((".", "?", "!")) else text + "."
        deontic = (e.get("deontic") or "").strip().lower()
        suffix = _DEONTIC_SUFFIX.get(deontic, "")
        if suffix:
            sentence = sentence.rstrip(".") + "." + suffix if not sentence.endswith(suffix) else sentence
            # Make sure suffix sits cleanly after the trailing period:
            if not sentence.endswith(suffix):
                sentence = sentence.rstrip(".") + "." + suffix
        key = re.sub(r"\s+", " ", sentence.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(sentence)
    return out


def synthesize_actions(
    entity: Any,
    kg_context: dict[str, Any],
    raw_actions: list[str],
    config: dict[str, Any],
) -> list[str]:
    """Call an LLM to produce 3-7 polished action statements from the KG context.

    Falls back to `raw_actions[:7]` if the LLM call fails or returns malformed JSON.
    """
    if not raw_actions:
        return []

    actor_context = {
        "name": kg_context.get("name"),
        "entity_type": kg_context.get("entity_type"),
        "summary": kg_context.get("summary"),
        "role_family": kg_context.get("role_family"),
        "agent_modality": kg_context.get("agent_modality"),
        "role_scope": kg_context.get("role_scope"),
        "eligibility_criteria": kg_context.get("eligibility_criteria"),
        "exclusion_criteria": kg_context.get("exclusion_criteria"),
        "related_edges": kg_context.get("related_edges") or [],
        "related_nodes": kg_context.get("related_nodes") or [],
    }

    raw_block = "\n".join(f"- {line}" for line in raw_actions)
    messages = [
        {"role": "system", "content": ROLE_ACTIONS_SYNTHESIZER_SYSTEM},
        {
            "role": "user",
            "content": ROLE_ACTIONS_SYNTHESIZER_USER.format(
                actor_context=json.dumps(actor_context, ensure_ascii=False),
                raw_actions=raw_block,
            ),
        },
    ]
    try:
        raw = chat_completion(messages, config)
        payload = json.loads(raw)
        actions = payload.get("actions") or []
        cleaned = [str(a).strip() for a in actions if str(a).strip()]
        return cleaned[:7] if cleaned else raw_actions[:7]
    except Exception as exc:
        print(f"  [role_actions] synthesis failed for {kg_context.get('name')}: {exc}")
        return raw_actions[:7]


def compute_for_entity(
    entity: Any,
    kg_context: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return both raw and synthesized actions for one entity."""
    raw = compute_raw_actions(entity, kg_context)
    synthesized = synthesize_actions(entity, kg_context, raw, config)
    return {
        "policy_role_actions_raw": raw,
        "policy_role_actions": synthesized,
    }
