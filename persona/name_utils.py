from __future__ import annotations

import re
from typing import Any, Mapping


_ENGLISH_LEADING_NAME = re.compile(
    r"^(?P<name>(?:(?:Dr|Mr|Mrs|Ms)\.?\s+)?"
    r"[A-Z][A-Za-z'’-]+"
    r"(?:\s+(?:[A-Z]\.\s+)?[A-Z][A-Za-z'’-]+){1,3})"
    r"(?=,|\s+is\b)"
)


def infer_persona_name(record: Mapping[str, Any]) -> str | None:
    """Return an explicit name or conservatively infer one from narrative text."""
    for key in ("name", "full_name", "person_name", "display_name"):
        value = str(record.get(key) or "").strip()
        if value:
            return value

    for key in ("professional_persona", "persona", "persona_en"):
        text = str(record.get(key) or "").strip()
        if not text:
            continue
        match = _ENGLISH_LEADING_NAME.match(text)
        if match:
            return match.group("name").strip()
    return None
