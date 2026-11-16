"""Sequential 5-persona discussion runner — shared by Conditions (1) and (3).

Mirrors the ToC `_phase_round` mechanic from stage2_toc/modules.py but stripped
of phase / forward-backward machinery. Per scenario:

1. random.shuffle(personas) with seed = scenario_index (reproducible)
2. For each persona in order: LLM call with prior speakers' utterances inlined
3. Mean across 5 personas' prediction_values → scenario prediction

Concurrency: WITHIN a scenario speakers are sequential (each sees prior). ACROSS
scenarios we use a thread pool.
"""
from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from llm.client import chat_completion_with_usage


def _format_sequential_context(prior_posts: list[dict]) -> str:
    """Mirror of stage2_toc/modules.py:_format_sequential_context, but operates
    on plain dicts (no PersonaPhaseEstimate dependency).

    prior_posts entries: {"persona_label": str, "prediction_values": dict, "narrative": str}
    """
    if not prior_posts:
        return "--- Other Participants' Posts So Far ---\n(you are the first to speak)"
    lines = []
    for i, post in enumerate(prior_posts, 1):
        lines.append(
            f"  [{i}] {post.get('persona_label') or 'persona'}: "
            f"values={json.dumps(post.get('prediction_values') or {}, ensure_ascii=False)}; "
            f"narrative={post.get('narrative') or ''}"
        )
    return "--- Other Participants' Posts So Far ---\n" + "\n".join(lines)


def _persona_label(persona: dict) -> str:
    """Pick a short label for a persona — first word of professional_persona,
    occupation, or 'persona'. Used only in sequential_context (LLM-visible peer
    identifier; not used for routing or aggregation).
    """
    prof = (persona.get("professional_persona") or "").strip()
    if prof:
        return prof.split()[0]
    occ = (persona.get("occupation") or "").strip()
    if occ:
        return occ
    return "persona"


def _coerce_target_value(prediction_values: dict, target_key: str) -> float | None:
    """Pull simulation_target's value from prediction_values; coerce to float."""
    v = prediction_values.get(target_key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _build_prediction_keys_hint(target_keys: list[str]) -> str:
    """JSON object literal showing every expected key. Used as `{prediction_keys_hint}` slot."""
    inner = ", ".join(f'"{k}": <number>' for k in target_keys)
    return "{" + inner + "}"


def _run_one_persona(
    persona: dict,
    sequential_context_text: str,
    policy_context_text: str,
    org_kg_contexts_text: str | None,
    simulation_target: str,
    target_keys: list[str],
    system_prompt: str,
    user_template: str,
    config: dict,
) -> dict:
    """One LLM call for one persona. Returns {prediction_values, narrative, evidence, usage}.

    `user_template` is either SEQ_PREDICT_MD_USER or SEQ_PREDICT_KG_USER. The MD
    variant does not have {org_kg_contexts} placeholder; we pass it only when
    the template expects it.
    """
    fmt_kwargs = dict(
        simulation_target=simulation_target,
        prediction_keys_hint=_build_prediction_keys_hint(target_keys),
        sequential_context=sequential_context_text,
        policy_context=policy_context_text,
        persona=json.dumps(persona, ensure_ascii=False, indent=2),
    )
    if "{org_kg_contexts}" in user_template:
        fmt_kwargs["org_kg_contexts"] = org_kg_contexts_text or "(none)"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(**fmt_kwargs)},
    ]
    try:
        content, usage = chat_completion_with_usage(messages, config)
        data = json.loads(content)
        pred_values = data.get("prediction_values") or {}
        return {
            "prediction_values": pred_values,
            "narrative": data.get("narrative") or "",
            "evidence": data.get("evidence") or [],
            "usage": usage,
        }
    except Exception as exc:  # noqa: BLE001 — driver-level: log + continue
        return {
            "prediction_values": {},
            "narrative": "",
            "evidence": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_scenario_sequential(
    scenario_id: str,
    participants: list[dict],
    policy_context_text: str,
    org_kg_contexts_text: str | None,
    simulation_target: str,
    target_keys: list[str],
    system_prompt: str,
    user_template: str,
    config: dict,
    shuffle_seed: int,
) -> dict:
    """Run one scenario sequentially: shuffle, then 5 ordered LLM calls.

    Returns: {scenario_id, posts: [...5...], prediction_values_per_persona: [...]}
    """
    rng = random.Random(shuffle_seed)
    order = list(range(len(participants)))
    rng.shuffle(order)

    posts: list[dict] = []
    persona_records: list[dict] = []
    for idx, p_idx in enumerate(order):
        persona = participants[p_idx]
        seq_text = _format_sequential_context(posts)
        result = _run_one_persona(
            persona=persona,
            sequential_context_text=seq_text,
            policy_context_text=policy_context_text,
            org_kg_contexts_text=org_kg_contexts_text,
            simulation_target=simulation_target,
            target_keys=target_keys,
            system_prompt=system_prompt,
            user_template=user_template,
            config=config,
        )
        label = _persona_label(persona)
        posts.append({
            "persona_label": label,
            "prediction_values": result["prediction_values"],
            "narrative": result["narrative"],
        })
        persona_records.append({
            "speak_order": idx,
            "original_idx": p_idx,
            "persona_label": label,
            "prediction_values": result["prediction_values"],
            "narrative": result["narrative"],
            "evidence": result["evidence"],
            "usage": result["usage"],
            **({"error": result["error"]} if "error" in result else {}),
        })
    return {"scenario_id": scenario_id, "persona_records": persona_records}


def run_all_scenarios(
    scenarios: list[dict],
    policy_context_text: str,
    org_kg_contexts_text: str | None,
    simulation_target: str,
    target_keys: list[str],
    system_prompt: str,
    user_template: str,
    config: dict,
    max_concurrent_scenarios: int = 5,
    checkpoint_cb: Callable[[list[dict]], None] | None = None,
) -> list[dict]:
    """Drive sequential runner across N scenarios with cross-scenario concurrency.

    `scenarios` items must have `scenario_id` and `participants` keys (list of persona dicts).
    `checkpoint_cb` is invoked with the partial results list after each completion
    (caller can persist a results.json checkpoint).
    """
    out: list[dict | None] = [None] * len(scenarios)

    with ThreadPoolExecutor(max_workers=max_concurrent_scenarios) as ex:
        fut_to_idx = {}
        for s_idx, scen in enumerate(scenarios):
            fut = ex.submit(
                run_scenario_sequential,
                scenario_id=scen["scenario_id"],
                participants=scen["participants"],
                policy_context_text=policy_context_text,
                org_kg_contexts_text=org_kg_contexts_text,
                simulation_target=simulation_target,
                target_keys=target_keys,
                system_prompt=system_prompt,
                user_template=user_template,
                config=config,
                shuffle_seed=s_idx + 1,
            )
            fut_to_idx[fut] = s_idx
        done = 0
        for fut in as_completed(fut_to_idx):
            s_idx = fut_to_idx[fut]
            out[s_idx] = fut.result()
            done += 1
            if checkpoint_cb:
                checkpoint_cb([r for r in out if r is not None])
    return [r for r in out if r is not None]


def extract_persona_predictions(
    scenario_result: dict,
    target_keys: list[str],
) -> list[dict[str, float | None]]:
    """Convert one scenario's persona_records into the list-of-dicts shape expected
    by `aggregate_scenario` (each entry: {target_key: float|None}).

    Behavior: the model is asked to put a SINGLE key in prediction_values matching
    the simulation_target. We map that single value to ALL target_keys (when gt has
    multiple keys per policy, we take the same value for each). If multiple keys
    are present in the model output, we match by key name first.
    """
    out: list[dict[str, float | None]] = []
    for rec in scenario_result.get("persona_records", []):
        pv = rec.get("prediction_values") or {}
        # Try direct key match; otherwise fall back to the first numeric value.
        per_persona: dict[str, float | None] = {}
        for k in target_keys:
            v = pv.get(k)
            if v is None:
                # fall back: first numeric value in dict
                for vv in pv.values():
                    try:
                        v = float(vv)
                        break
                    except (TypeError, ValueError):
                        continue
            try:
                per_persona[k] = float(v) if v is not None else None
            except (TypeError, ValueError):
                per_persona[k] = None
        out.append(per_persona)
    return out
