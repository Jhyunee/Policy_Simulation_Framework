"""Generate personas from raw policy MD (PersonaHub KG-ablation baseline).

KG-ablation variant of scripts/generate_kg_personas.py: replaces all KG context
injection (ECOSYSTEM + FOCAL ACTOR blocks) with the raw policy markdown
document. Used to measure the contribution of KG construction to persona
quality in downstream stage2-toc.

Two-stage LLM:
  1. One call → identify K stakeholder types from MD.
  2. Batched calls per stakeholder → generate M personas each.

Output schema matches simulation_personas_baseline.json so stage2-toc can
consume the file unchanged. Scenarios are assembled fresh (template's
stakeholder slots are NOT preserved since LLM-derived stakeholders differ).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import string
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.client import chat_completion


# ---------------------------------------------------------------- helpers

_LANG_TO_LABEL = {
    "kor": "Korean", "jp": "Japanese", "singapore": "English",
    "fr": "French", "usa": "English",
}

_COUNTRY_LABEL = {
    "kor": "Republic of Korea", "jp": "Japan", "singapore": "Singapore",
    "fr": "France", "usa": "United States",
}


def load_md(md_path: str | Path) -> str:
    p = Path(md_path)
    if not p.exists():
        raise FileNotFoundError(md_path)
    if p.suffix.lower() != ".md":
        raise ValueError(f"expected .md, got {p.suffix}")
    return p.read_text(encoding="utf-8")


def _short_uid(k: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=k))


# ---------------------------------------------------------------- Step 1: stakeholders

def _build_stakeholder_extraction_prompt(md_text: str, max_stakeholders: int) -> list[dict[str, str]]:
    system = (
        "You read a policy plan document and identify the distinct stakeholder "
        "types who participate in or are affected by this policy. Output valid "
        "JSON only — no markdown, no code fences."
    )
    user = f"""--- POLICY DOCUMENT ---
{md_text}
---

Identify up to {max_stakeholders} distinct stakeholder types most relevant to this policy's outcomes (people who participate, fund, evaluate, or are directly affected). Use concise PascalCase identifiers as `stakeholder_type` names (e.g. "GraduateStudent", "PrincipalInvestigator", "FundingAgency").

Output a JSON object:
{{
  "stakeholders": [
    {{"stakeholder_type": "<PascalCase>", "rationale": "<1-2 sentences why this type is central>"}}
  ]
}}
No prose, no markdown.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _extract_stakeholders(md_text: str, max_stakeholders: int, config: dict[str, Any]) -> list[dict[str, str]]:
    messages = _build_stakeholder_extraction_prompt(md_text, max_stakeholders)
    raw = chat_completion(messages, config)
    payload = json.loads(raw)
    stakeholders = payload.get("stakeholders") or payload.get("results") or []
    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    for s in stakeholders:
        if not isinstance(s, dict):
            continue
        stype = (s.get("stakeholder_type") or "").strip()
        if not stype or stype in seen:
            continue
        seen.add(stype)
        cleaned.append({
            "stakeholder_type": stype,
            "rationale": (s.get("rationale") or "").strip(),
        })
        if len(cleaned) >= max_stakeholders:
            break
    if not cleaned:
        raise RuntimeError("LLM did not return any stakeholder types")
    return cleaned


# ---------------------------------------------------------------- Step 2: persona generation

def _build_persona_prompt(
    md_text: str,
    stakeholder_type: str,
    rationale: str,
    n_personas: int,
    language: str,
    country_label: str,
) -> list[dict[str, str]]:
    system = (
        "You generate personas grounded in a policy plan document. Each persona "
        "is a distinct individual with self-consistent demographics, professional "
        f"narrative, skills, and goals. Output language for narrative fields must "
        f"be {language}. Demographic field values must be plausible for "
        f"{country_label} and consistent with the stakeholder role inferred from "
        "the policy document. Output valid JSON only — no markdown, no code fences."
    )
    user = f"""Generate {n_personas} personas who would plausibly participate as "{stakeholder_type}" in the policy described below. Each persona must reflect a distinct individual with varied background, motivation, and circumstances.

--- POLICY DOCUMENT ---
{md_text}

--- STAKEHOLDER ROLE ---
{stakeholder_type}: {rationale}

--- PERSONA SCHEMA ---
Each persona is a JSON object with these fields:
- persona_id: "P_{stakeholder_type}__<8-char hex>" (unique per persona)
- stakeholder_type: "{stakeholder_type}"
- representative_type: "{stakeholder_type}"
- professional_persona: 2-3 sentences in {language}
- skills_and_expertise: 1-2 sentences in {language}
- career_goals_and_ambitions: 1-2 sentences in {language}
- sex
- age (integer)
- marital_status
- military_status
- education_level
- occupation
- district
- province
- bachelors_field

--- VALUE CONSTRAINTS (no vocabulary provided — infer from context) ---
- All demographic values must be plausible {country_label} values.
- All values must satisfy the stakeholder role inferred from the policy document above.
- Use natural {language}-language values for categorical fields (sex, marital_status, education_level, occupation, province, district, bachelors_field).
- `age` must be a plain integer.

--- DIVERSITY ---
Across the {n_personas} personas, vary background, motivation, life circumstances, region, and bachelor's field naturally. Do NOT homogenize.

--- OUTPUT ---
Return a JSON object with key "personas" containing an array of {n_personas} personas. No prose, no markdown.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _call_llm_batch(messages: list[dict[str, str]], config: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        raw = chat_completion(messages, config)
        payload = json.loads(raw)
        personas = payload.get("personas") or payload.get("results") or []
        if not isinstance(personas, list):
            return []
        return personas
    except Exception as exc:
        print(f"  [gen] LLM call failed: {exc}")
        return []


# ---------------------------------------------------------------- Step 3: scenario assembly

def _assemble_scenarios_stratified(
    n_scenarios: int,
    participants_per_scenario: int,
    persona_pool_by_stype: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build N scenarios of `participants_per_scenario` personas each.

    Strategy: for each scenario, fill participant slots by round-robin across
    stakeholder types, cycling through each stype's pool. This balances
    representation across stakeholders.
    """
    stypes = list(persona_pool_by_stype.keys())
    if not stypes:
        return []
    cursors: dict[str, int] = {st: 0 for st in stypes}
    scenarios = []
    for i in range(n_scenarios):
        new_participants = []
        used_in_scenario: set[str] = set()
        for slot in range(participants_per_scenario):
            stype = stypes[slot % len(stypes)]
            pool = persona_pool_by_stype.get(stype, [])
            if not pool:
                continue
            # pick next persona for this stype, prefer unseen-in-scenario
            picked = None
            for _ in range(len(pool)):
                idx = cursors[stype] % len(pool)
                cursors[stype] += 1
                cand = pool[idx]
                pid = cand.get("persona_id")
                if pid not in used_in_scenario:
                    picked = cand
                    used_in_scenario.add(pid)
                    break
            if picked is None:
                picked = pool[cursors[stype] % len(pool)]
                cursors[stype] += 1
            new_participants.append(picked)
        scenarios.append({
            "scenario_id": f"S{i+1}",
            "participants": new_participants,
        })
    return scenarios


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--md", type=Path, required=True,
                    help="Policy plan markdown file")
    ap.add_argument("--template-personas", type=Path, default=None,
                    help="Optional simulation_personas.json: read scenario_count + "
                         "participants/scenario to match KG variant scale. If absent, "
                         "use --scenarios and --participants-per-scenario.")
    ap.add_argument("--config-path", type=str, default="config/base.yaml")
    ap.add_argument("--country", choices=["kor", "jp", "singapore", "fr", "usa"], default="kor",
                    help="Used for language label + country label in prompt")
    ap.add_argument("--language", default=None,
                    help="Override language label (default: derived from --country)")
    ap.add_argument("--max-stakeholders", type=int, default=6,
                    help="Cap on K stakeholder types LLM may extract")
    ap.add_argument("--scenarios", type=int, default=50,
                    help="Total scenarios to assemble (used if --template-personas not given)")
    ap.add_argument("--participants-per-scenario", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--margin-pct", type=float, default=10.0,
                    help="Over-generation margin (e.g. 10.0 = 10%% extra)")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--out-filename", type=str, default="simulation_personas_md_only.json")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(open(args.config_path))
    config.setdefault("llm", {})["temperature"] = args.temperature

    print(f"  Loading MD: {args.md}")
    md_text = load_md(args.md)
    print(f"    {len(md_text):,} chars loaded")

    # Resolve scenario count + participants
    if args.template_personas and args.template_personas.exists():
        tpl = json.load(open(args.template_personas))
        scenarios_tpl = tpl.get("scenarios") or []
        n_scenarios = len(scenarios_tpl) or args.scenarios
        participants_per_sc = (
            len(scenarios_tpl[0].get("participants", []))
            if scenarios_tpl else args.participants_per_scenario
        )
        print(f"  Template: {n_scenarios} scenarios × {participants_per_sc} participants")
    else:
        n_scenarios = args.scenarios
        participants_per_sc = args.participants_per_scenario
        print(f"  No template — using --scenarios={n_scenarios}, --participants-per-scenario={participants_per_sc}")

    language = args.language or _LANG_TO_LABEL.get(args.country, "Korean")
    country_label = _COUNTRY_LABEL.get(args.country, "Republic of Korea")

    # Step 1: extract stakeholders
    print(f"\n  [Step 1] Extracting stakeholder types (max={args.max_stakeholders})...")
    stakeholders = _extract_stakeholders(md_text, args.max_stakeholders, config)
    print(f"    K = {len(stakeholders)} stakeholder types:")
    for s in stakeholders:
        print(f"      - {s['stakeholder_type']}: {s['rationale'][:80]}")

    # Plan target persona counts: ceil(N × P / K), with margin
    total_slots = n_scenarios * participants_per_sc
    target_per_stype = math.ceil(total_slots / len(stakeholders))
    target_with_margin = math.ceil(target_per_stype * (1 + args.margin_pct / 100.0))
    print(f"\n  [Step 2] Target per stakeholder: {target_per_stype} "
          f"(+{int(args.margin_pct)}% margin = {target_with_margin})")

    # Build batches
    plan: list[tuple[str, str, int]] = []  # (stype, rationale, n_in_batch)
    for s in stakeholders:
        remaining = target_with_margin
        while remaining > 0:
            this_n = min(args.batch_size, remaining)
            plan.append((s["stakeholder_type"], s["rationale"], this_n))
            remaining -= this_n
    print(f"  Total batches: {len(plan)} (batch_size={args.batch_size}, temp={config['llm']['temperature']})")

    # Run batches in parallel
    persona_pool: dict[str, list[dict[str, Any]]] = {s["stakeholder_type"]: [] for s in stakeholders}

    def _task(stype: str, rationale: str, n: int) -> tuple[str, list[dict[str, Any]]]:
        messages = _build_persona_prompt(md_text, stype, rationale, n, language, country_label)
        personas = _call_llm_batch(messages, config)
        cleaned = []
        for p in personas:
            if not isinstance(p, dict):
                continue
            uid = _short_uid()
            p["persona_id"] = f"P_{stype}__{uid}"
            p["stakeholder_type"] = stype
            p["representative_type"] = f"{stype} | MD-Gen-{uid}"
            cleaned.append(p)
        return stype, cleaned

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(_task, st, rat, n) for st, rat, n in plan]
        done = 0
        for fut in as_completed(futures):
            stype, personas = fut.result()
            persona_pool[stype].extend(personas)
            done += 1
            if done % 5 == 0 or done == len(plan):
                pool_str = ", ".join(f"{k}={len(v)}" for k, v in persona_pool.items())
                print(f"    {done}/{len(plan)} batches | pool: {pool_str}")

    # Trim to exact target per stype (extras dropped)
    for st in list(persona_pool.keys()):
        if len(persona_pool[st]) > target_per_stype:
            persona_pool[st] = persona_pool[st][:target_per_stype]
        elif len(persona_pool[st]) < target_per_stype:
            print(f"  WARNING: {st} short by {target_per_stype - len(persona_pool[st])} (cycle in assembly)")

    # Step 3: assemble scenarios
    print(f"\n  [Step 3] Assembling {n_scenarios} scenarios × {participants_per_sc} participants...")
    new_scenarios = _assemble_scenarios_stratified(n_scenarios, participants_per_sc, persona_pool)

    # Build output bundle (shape compatible with simulation_personas_baseline.json)
    bundle = {
        "scenario_count": len(new_scenarios),
        "scenarios": new_scenarios,
        "shared_context": {},  # KG-derived org_kg_contexts intentionally empty for ablation
        "sampling_metadata": {
            "persona_source": "md_text_to_persona",
            "generation_llm": config.get("llm", {}).get("model", "?"),
            "md_path": str(args.md.resolve()),
            "stakeholder_types": [s["stakeholder_type"] for s in stakeholders],
            "stakeholder_rationale": {s["stakeholder_type"]: s["rationale"] for s in stakeholders},
            "max_stakeholders": args.max_stakeholders,
            "target_per_stakeholder": target_per_stype,
            "margin_pct": args.margin_pct,
            "temperature": args.temperature,
        },
    }

    out_path = args.out_dir / args.out_filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    print(f"\n  Wrote {out_path}")
    print(f"  Persona pool sizes: " +
          ", ".join(f"{k}={len(v)}" for k, v in persona_pool.items()))


if __name__ == "__main__":
    main()
