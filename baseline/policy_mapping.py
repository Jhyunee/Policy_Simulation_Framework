"""Resolve comparison inputs from config/policies.yaml, without historical runs."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PDF_CACHE_DIR = REPO_ROOT / "dataset/processed/pdf_cache/opendataloader"
OUTPUTS_DIR = REPO_ROOT / "outputs"
POLICY_REGISTRY_PATH = REPO_ROOT / "config/policies.yaml"


def _load_registry() -> dict:
    with POLICY_REGISTRY_PATH.open(encoding="utf-8") as stream:
        registry = yaml.safe_load(stream) or {}
    if not isinstance(registry, dict):
        raise ValueError(f"Expected a policy mapping in {POLICY_REGISTRY_PATH}")
    for key, spec in registry.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Policy {key!r} must map to input paths")
    return registry


POLICY_MAPPING = _load_registry()


def resolve_md_path(policy_key: str) -> Path:
    spec = POLICY_MAPPING[policy_key]
    if "md_path" in spec:
        path = REPO_ROOT / spec["md_path"]
        if path.is_file():
            return path
        raise FileNotFoundError(f"Policy Markdown not found: {path}; edit config/policies.yaml")
    # Compatibility for users who explicitly register a structured PDF cache.
    if "md_glob" in spec and "md_filename" in spec:
        candidates = sorted(PDF_CACHE_DIR.glob(spec["md_glob"]), key=lambda p: p.stat().st_mtime, reverse=True)
        for candidate in candidates:
            path = candidate / spec["md_filename"]
            if path.is_file():
                return path
    raise FileNotFoundError(f"Set md_path for {policy_key!r} in config/policies.yaml")


def resolve_personas_path(policy_key: str) -> Path:
    value = POLICY_MAPPING[policy_key].get("personas_path")
    if not value:
        raise ValueError(f"Set personas_path for {policy_key!r} in config/policies.yaml")
    path = REPO_ROOT / value
    if not path.is_file():
        raise FileNotFoundError(f"Stage 1 personas not found: {path}")
    return path


def all_policy_keys() -> list[str]:
    return list(POLICY_MAPPING)
