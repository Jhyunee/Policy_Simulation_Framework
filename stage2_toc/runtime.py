"""Shared runtime context for stage2_toc round prompts.

The Stage 2 ToC pipeline injects `{org_kg_contexts}` into six persona/phase-facing
prompts (SIM_PERSONA_CONSTRUCTOR_USER, SEQUENTIAL_FORWARD_POST_USER,
SEQUENTIAL_BACKWARD_POST_USER, REFINEMENT_JUDGE_USER, PHASE_SUMMARY_USER,
PHASE_STAKEHOLDER_SYNTHESIZER_USER). Rather than threading the serialized payload
through every function signature in the ported pipeline, we stash it here at the
top of `run_stage2_toc` and read it from each `.format()` call site.

The payload is the same across all scenarios (it's shared scenario context), so
a plain module-level string is sufficient — no per-thread state needed.
"""

from __future__ import annotations

import json
from typing import Any


ORG_KG_CONTEXTS_JSON: str = "(no organizations registered for this run)"


def set_org_kg_contexts(payload: list[dict[str, Any]] | None) -> None:
    """Serialize and stash the org_kg_contexts payload for prompt injection."""
    global ORG_KG_CONTEXTS_JSON
    if not payload:
        ORG_KG_CONTEXTS_JSON = "(no organizations registered for this run)"
        return
    try:
        ORG_KG_CONTEXTS_JSON = json.dumps(payload, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        ORG_KG_CONTEXTS_JSON = str(payload)


def get_org_kg_contexts() -> str:
    return ORG_KG_CONTEXTS_JSON
