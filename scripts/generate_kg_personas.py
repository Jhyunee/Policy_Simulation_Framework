"""Generate personas from policy KG context (Persona Hub baseline).

For each policy stakeholder, prompt LLM to generate personas grounded in:
  - shared_context.org_kg_contexts (policy ecosystem)
  - focal actor's KG context (role + must/can/cannot norms)

Personas are output in the same schema as Nemotron+ECO_ROLE
(PP/SE/CG + demographics) so downstream stage2-toc consumes them unchanged.

The new personas are stamped into the SAME scenario structure as an existing
refined simulation_personas.json (stakeholder slots preserved, persona content
swapped).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import string
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.client import chat_completion
from persona.refiner import (
    _slim_actor_context,
    _render_actor_context_block,
    _actor_context_for,
    _render_policy_ecosystem_block,
)
from graph.local_graph_reader import LocalGraphReader
from persona.scope_definer import build_kg_context


# ---------------------------------------------------------------- KG utilities

def _build_actor_kg_lookup(policy_graph: str) -> dict[str, dict[str, Any]]:
    """Map stakeholder entity_type → full KG context (for _slim_actor_context).

    Stamps each ctx with ``_builder`` (e.g. ``"local_iad_ig"`` or
    ``"local_freeform"``) read from the graph metadata so downstream code can
    auto-dispatch the IAD/IG vs free-form actor-context renderer.
    """
    reader = LocalGraphReader(policy_graph)
    builder = (reader.graph.get("metadata") or {}).get("builder", "local_iad_ig")
    out: dict[str, dict[str, Any]] = {}
    for e in reader.get_filtered_entities(persona_only=False):
        et = e.get_entity_type() or ""
        if not et:
            continue
        attrs = e.attributes or {}
        is_elig = bool(attrs.get("persona_eligible"))
        # Prefer the persona-eligible variant if multiple entities share entity_type
        if et not in out or (is_elig and not out[et].get("_persona_eligible")):
            ctx = build_kg_context(e, et)
            ctx["_persona_eligible"] = is_elig
            ctx["_builder"] = builder
            out[et] = ctx
    return out


# ---------------------------------------------------------------- prompt build

_LANG_TO_LABEL = {
    "kor": "Korean",
    "jp": "Japanese",
    "singapore": "English",
    "fr": "French",
    "usa": "English",
}

_COUNTRY_LABEL = {
    "kor": "Republic of Korea",
    "jp": "Japan",
    "singapore": "Singapore",
    "fr": "France",
    "usa": "United States",
}


def _persona_id_seed(prefix: str) -> str:
    return prefix + "__" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


def _build_prompt(
    focal_actor_ctx: dict[str, Any],
    shared_context: dict[str, Any],
    n_personas: int,
    language: str,
    country_label: str,
    persona_context_mode: str = "auto",
) -> list[dict[str, str]]:
    """Build messages for one batch of N persona generations.

    ``persona_context_mode`` selects how the focal actor's KG slice is rendered:
      - ``"auto"`` (default): sniff ``focal_actor_ctx['_builder']`` — free-form
        for ``"local_freeform"`` graphs, IAD/IG for everything else.
      - ``"iadig"``: force IAD/IG must/can/cannot block.
      - ``"freeform"``: force free-form attributes + edges block.
    """
    actor_name = focal_actor_ctx.get("name") or "?"
    actor_etype = focal_actor_ctx.get("entity_type") or actor_name

    slim_ctx, renderer = _actor_context_for(focal_actor_ctx, mode=persona_context_mode)
    actor_block = renderer(slim_ctx)
    ecosystem_block = _render_policy_ecosystem_block(
        shared_context, focal_name=actor_name, focal_etype=actor_etype
    ) or "(no ecosystem context)"

    system = (
        f"You generate personas grounded in policy knowledge graph context. "
        f"Each persona is a distinct individual with self-consistent demographics, "
        f"professional narrative, skills, and goals. Output language for narrative "
        f"fields must be {language}. Demographic field values must be plausible for "
        f"{country_label} and consistent with the focal actor's eligibility, inferred "
        f"from the provided policy context. Output valid JSON only — no markdown, no code fences."
    )

    user = f"""Generate {n_personas} personas who would plausibly participate in this policy as the focal actor ({actor_name}). Each persona must reflect a distinct individual with varied background, motivation, and circumstances.

--- POLICY ECOSYSTEM (background) ---
{ecosystem_block}

--- FOCAL ACTOR (this persona's role) ---
{actor_block}

--- PERSONA SCHEMA ---
Each persona is a JSON object with these fields:
- persona_id: "P_{actor_etype}__<8-char hex>" (unique per persona)
- stakeholder_type: "{actor_etype}"
- representative_type: "{actor_etype}"
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
- All values must satisfy the focal actor's eligibility (age range, education, occupation type, etc.) inferred from the policy ecosystem and focal actor sections above.
- Use natural {language}-language values for categorical fields (sex, marital_status, education_level, occupation, province, district, bachelors_field).
- `age` must be a plain integer.

--- DIVERSITY ---
Across the {n_personas} personas, vary background, motivation, life circumstances, region, and bachelor's field naturally. Do NOT homogenize.

--- OUTPUT ---
Return a JSON object with key "personas" containing an array of {n_personas} personas. No prose, no markdown.
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------- demo hints

# Korean Nemotron demographic vocabulary (extracted from existing pools)
_KOR_DEMO_HINTS = {
    "sex": ["남자", "여자"],
    "marital_status": ["미혼", "배우자있음", "이혼", "사별"],
    "military_status": ["현역", "비현역"],
    "education_level": ["고등학교", "대학", "대학원"],
    "province": [
        "서울", "경기", "인천", "부산", "대구", "광주", "대전", "울산", "세종",
        "강원", "충청북", "충청남", "전라북", "전라남", "경상북", "경상남", "제주",
    ],
    "bachelors_field": [
        "예술·인문", "사회과학·언론", "경영·행정·법", "교육", "공학·제조·건설",
        "자연과학·수학", "정보통신기술", "보건·복지", "서비스", "농림어업·수의학",
    ],
}

_JP_DEMO_HINTS = {
    "sex": ["男", "女"],
    "marital_status": ["未婚", "既婚 (子供なし)", "既婚 (子供あり)", "離別", "死別"],
    "military_status": ["未経験"],
    "education_level": ["中学卒", "高校卒", "高専卒", "短大卒", "大学卒 文系", "大学卒 理系", "大学院卒 文系", "大学院卒 理系"],
    "province": [
        "東日本", "西日本", "東京都", "大阪府", "京都府", "北海道", "神奈川県", "愛知県",
    ],
    "bachelors_field": [
        "人文・芸術", "社会科学", "経営・行政", "教育", "工学・製造・建設",
        "自然科学・数学", "情報通信技術", "保健・福祉", "サービス", "農林水産",
    ],
}


def _demo_hints_for_country(country: str) -> dict[str, Any]:
    if country == "kor":
        return _KOR_DEMO_HINTS
    if country == "jp":
        return _JP_DEMO_HINTS
    return _KOR_DEMO_HINTS  # default


# ---------------------------------------------------------------- LLM call

def _call_llm_batch(messages: list[dict[str, str]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Returns parsed personas list, or empty list on failure."""
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


# ---------------------------------------------------------------- assembly

def _assemble_scenarios(
    template_scenarios: list[dict[str, Any]],
    persona_pool_by_stype: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Replace participant content in each scenario while preserving stakeholder slots.

    Allocation strategy: for each (scenario, slot of stakeholder_type X), use a
    cycling pool — each scenario gets distinct personas as much as the pool size
    allows, then cycles back if pool < participants needed.
    """
    cursors: dict[str, int] = {st: 0 for st in persona_pool_by_stype}
    out = []
    for s in template_scenarios:
        new_s = dict(s)
        new_participants = []
        slot_usage_in_scenario: dict[str, set] = {}
        for p in s.get("participants", []):
            st = p.get("stakeholder_type") or "?"
            pool = persona_pool_by_stype.get(st, [])
            if not pool:
                new_participants.append(p)  # fallback: keep original (should be rare)
                continue
            # Try to pick a persona not yet used in this scenario
            used = slot_usage_in_scenario.setdefault(st, set())
            picked = None
            for _attempt in range(len(pool)):
                idx = cursors[st] % len(pool)
                cursors[st] += 1
                cand = pool[idx]
                cand_pid = cand.get("persona_id")
                if cand_pid not in used:
                    picked = cand
                    used.add(cand_pid)
                    break
            if picked is None:
                picked = pool[cursors[st] % len(pool)]
                cursors[st] += 1
            new_participants.append(picked)
        new_s["participants"] = new_participants
        out.append(new_s)
    return out


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simulation-personas", type=Path, required=True,
                    help="Existing simulation_personas.json (provides scenario structure + shared_context)")
    ap.add_argument("--policy-graph", type=Path, required=True,
                    help="policy_graph.json from KG extraction")
    ap.add_argument("--config-path", type=str, default="config/base.yaml")
    ap.add_argument("--country", choices=["kor", "jp", "singapore", "fr", "usa"], default="kor",
                    help="Used for language + demographic vocabulary")
    ap.add_argument("--language", default=None,
                    help="Override language label (default: derived from --country)")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--margin-pct", type=float, default=10.0,
                    help="Over-generation margin (e.g. 10.0 = 10%% extra)")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9,
                    help="Override LLM temperature for diversity (applied to config)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--out-filename", type=str, default="simulation_personas_baseline.json")
    ap.add_argument("--persona-context-mode", choices=["auto", "iadig", "freeform"],
                    default="auto",
                    help="How to render the focal actor's KG slice into the LLM prompt. "
                         "'auto' (default) sniffs policy_graph.metadata.builder. "
                         "'freeform' is used by the IAD/IG ablation experiment.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(open(args.config_path))
    if args.temperature is not None:
        config.setdefault("llm", {})["temperature"] = args.temperature

    sim_data = json.load(open(args.simulation_personas))
    shared_context = sim_data.get("shared_context") or {}

    kg_lookup = _build_actor_kg_lookup(str(args.policy_graph))
    if kg_lookup:
        sample_builder = next(iter(kg_lookup.values())).get("_builder")
        effective_mode = args.persona_context_mode
        if effective_mode == "auto":
            effective_mode = "freeform" if sample_builder == "local_freeform" else "iadig"
        print(f"  graph metadata.builder = {sample_builder!r} -> persona-context-mode = {effective_mode}")

    # Count target personas per stakeholder type from the scenario template
    scenarios = sim_data.get("scenarios") or []
    target_by_stype: dict[str, int] = {}
    seen = set()
    for s in scenarios:
        for p in s.get("participants", []):
            pid = p.get("persona_id")
            if pid in seen:
                continue
            seen.add(pid)
            st = p.get("stakeholder_type") or "?"
            target_by_stype[st] = target_by_stype.get(st, 0) + 1

    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Target unique personas by stakeholder_type:")
    for st, n in target_by_stype.items():
        margin_n = int(n * (1 + args.margin_pct / 100.0) + 0.999)  # ceil
        print(f"    {st}: {n} (gen target +{int(args.margin_pct)}% = {margin_n})")

    language = args.language or _LANG_TO_LABEL.get(args.country, "Korean")
    country_label = _COUNTRY_LABEL.get(args.country, "Republic of Korea")

    # Plan batches: per stakeholder type, ceil(target_with_margin / batch_size) calls
    plan: list[tuple[str, int]] = []  # (stype, n_in_this_batch)
    for st, n in target_by_stype.items():
        if st not in kg_lookup:
            print(f"  WARNING: no KG context for stakeholder_type '{st}' — skipping")
            continue
        margin_n = int(n * (1 + args.margin_pct / 100.0) + 0.999)
        remaining = margin_n
        while remaining > 0:
            this_n = min(args.batch_size, remaining)
            plan.append((st, this_n))
            remaining -= this_n
    print(f"  Total batches: {len(plan)} (batch_size={args.batch_size}, temp={config['llm']['temperature']})")

    # Run batches in parallel
    persona_pool: dict[str, list[dict[str, Any]]] = {st: [] for st in target_by_stype}

    def _task(stype: str, n: int) -> tuple[str, list[dict[str, Any]]]:
        focal_ctx = kg_lookup[stype]
        focal_name = (focal_ctx.get("name") or stype).strip()
        messages = _build_prompt(
            focal_ctx, shared_context, n, language, country_label,
            persona_context_mode=args.persona_context_mode,
        )
        personas = _call_llm_batch(messages, config)
        # ALWAYS override persona_id + representative_type so they are unique per persona,
        # regardless of what the LLM returned (LLMs often echo the prompt placeholder).
        cleaned = []
        for p in personas:
            if not isinstance(p, dict):
                continue
            uid = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
            p["persona_id"] = f"P_{stype}__{uid}"
            p["stakeholder_type"] = stype
            p["representative_type"] = f"{focal_name} | KG-Gen-{uid}"
            cleaned.append(p)
        return stype, cleaned

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(_task, st, n) for st, n in plan]
        done = 0
        for fut in as_completed(futures):
            stype, personas = fut.result()
            persona_pool[stype].extend(personas)
            done += 1
            if done % 5 == 0 or done == len(plan):
                pool_str = ", ".join(f"{k}={len(v)}" for k, v in persona_pool.items())
                print(f"    {done}/{len(plan)} batches | pool: {pool_str}")

    # Trim to exact target counts (drop extras from end)
    for st, n in target_by_stype.items():
        if len(persona_pool[st]) > n:
            persona_pool[st] = persona_pool[st][:n]
        elif len(persona_pool[st]) < n:
            print(f"  WARNING: {st} short by {n - len(persona_pool[st])} (will cycle in scenario assembly)")

    # Assemble scenarios with new persona content
    new_scenarios = _assemble_scenarios(scenarios, persona_pool)

    # Build output bundle (mirror structure of input)
    bundle = dict(sim_data)
    bundle["scenarios"] = new_scenarios
    bundle["scenario_count"] = len(new_scenarios)
    bundle.setdefault("sampling_metadata", {})
    bundle["sampling_metadata"]["persona_source"] = "kg_text_to_persona"
    bundle["sampling_metadata"]["generation_llm"] = config.get("llm", {}).get("model", "?")

    out_path = args.out_dir / args.out_filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    print(f"\n  Wrote {out_path}")
    print(f"  Persona pool sizes by stakeholder: " +
          ", ".join(f"{k}={len(v)}" for k, v in persona_pool.items()))


if __name__ == "__main__":
    main()
