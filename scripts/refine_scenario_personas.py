"""Apply role+norm persona refinement to the unique participants of a simulation_personas.json.

Produces `simulation_personas_refined.json` alongside the input. The refiner is
defined in `persona/refiner.py` and edits three narrative fields
(professional_persona, skills_and_expertise, career_goals_and_ambitions) grounded
in the policy KG actor's role_family + must/can/cannot regulative norms.

Features
--------
- **Scenario slicing**: `--scenario-start / --scenario-end` (1-indexed inclusive)
- **Batching**: `--batch-size N` groups personas of the same stakeholder_type into
  one LLM call sharing actor_context (saves tokens + round-trips). Set to 1 for
  per-persona mode.
- **Checkpoint / resume**: each completed persona is written atomically to
  `<checkpoint-dir>/<persona_id>.json`. Rerunning with `--resume` (default) skips
  already-completed persona_ids. Use `--no-resume` to force a full re-run.
- **Parallelism**: `--max-workers` controls concurrent LLM calls (one per batch).

Unique participants are deduplicated by `persona_id` across the selected scenarios.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from persona.refiner import refine_persona, refine_persona_batch
from persona.scope_definer import build_kg_context
from graph.local_graph_reader import LocalGraphReader


@dataclass
class _StubPersona:
    """Lightweight stand-in for a PersonaProfile that the refiner can mutate."""
    representative_type: str
    demographics: dict = field(default_factory=dict)


def _build_kg_lookups(
    policy_graph_path: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build KG context lookups by stakeholder type and stable entity id.

    Mechanism-panel personas use normalized stakeholder types such as ``payer``
    and retain the source KG entity in ``role_anchor_entity_id``.  The entity-id
    lookup preserves that exact grounding while the type lookup remains a
    backwards-compatible fallback for legacy persona bundles.
    """
    reader = LocalGraphReader(policy_graph_path)
    entities = reader.get_filtered_entities(persona_only=False)
    by_type: dict[str, dict[str, Any]] = {}
    by_entity_id: dict[str, dict[str, Any]] = {}
    for entity in entities:
        etype = entity.get_entity_type() or ""
        if not etype:
            continue
        attrs = entity.attributes or {}
        is_eligible = bool(attrs.get("persona_eligible"))
        ctx = build_kg_context(entity, etype)
        ctx["_persona_eligible"] = is_eligible

        entity_id = str(getattr(entity, "uuid", "") or attrs.get("uuid") or "")
        if entity_id:
            by_entity_id[entity_id] = ctx

        existing = by_type.get(etype)
        if existing is None or (is_eligible and not existing.get("_persona_eligible")):
            by_type[etype] = ctx
    return by_type, by_entity_id


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON to `path` atomically (write to temp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _scan_checkpoint(ck_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all completed personas from the checkpoint directory."""
    out: dict[str, dict[str, Any]] = {}
    if not ck_dir.is_dir():
        return out
    for f in ck_dir.glob("*.json"):
        try:
            d = json.load(open(f))
            pid = d.get("persona_id") or f.stem
            out[pid] = d
        except Exception as exc:
            print(f"  [checkpoint] skip unreadable {f.name}: {exc}")
    return out


def _restore_pre_refine_narratives(
    sim_data: dict[str, Any], source_data: Any
) -> tuple[dict[str, Any], int]:
    """Restore selected personas' original narratives from a Stage 1 artifact.

    ``personas_initial.json`` retains ``*_pre_refine`` fields that the compact
    simulation bundle intentionally omits.  This hook makes prompt revisions
    auditable: a corrected refiner can start from the same pre-refinement text
    instead of treating an earlier refiner's output as source material.
    """
    source_by_type: dict[str, dict[str, Any]] = {}

    def _visit(value: Any) -> None:
        if isinstance(value, dict):
            representative_type = value.get("representative_type")
            demographics = value.get("demographics")
            if representative_type and isinstance(demographics, dict):
                source_by_type[str(representative_type)] = demographics
            for child in value.values():
                _visit(child)
        elif isinstance(value, list):
            for child in value:
                _visit(child)

    _visit(source_data)
    out = copy.deepcopy(sim_data)
    restored = 0
    fields = (
        "professional_persona",
        "skills_and_expertise",
        "career_goals_and_ambitions",
    )
    for scenario in out.get("scenarios") or []:
        for participant in scenario.get("participants") or []:
            source = source_by_type.get(str(participant.get("representative_type") or ""))
            if not source:
                continue
            changed = False
            for field_name in fields:
                original = source.get(f"{field_name}_pre_refine")
                if original:
                    participant[field_name] = original
                    changed = True
            if changed:
                restored += 1
    return out, restored


def _refine_one_persona(
    participant: dict[str, Any],
    kg_context: dict[str, Any],
    config: dict[str, Any],
    shared_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a per-persona refine. Returns the mutated demographics dict."""
    persona = _StubPersona(
        representative_type=str(participant.get("representative_type") or participant.get("persona_id") or ""),
        demographics=copy.deepcopy(participant),
    )
    refine_persona(persona, kg_context, config, shared_context=shared_context)
    return persona.demographics


def _refine_one_batch(
    batch: list[dict[str, Any]],
    kg_context: dict[str, Any],
    config: dict[str, Any],
    shared_context: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run a batched refine. Returns persona_id → refined demographics.

    On batch failure, falls back to per-persona refinement for the affected entries.
    """
    personas = [
        _StubPersona(
            representative_type=str(p.get("representative_type") or p.get("persona_id") or ""),
            demographics=copy.deepcopy(p),
        )
        for p in batch
    ]
    result = refine_persona_batch(personas, kg_context, config, shared_context=shared_context)

    refined: dict[str, dict[str, Any]] = {}
    if result is not None:
        for pers in personas:
            pid = pers.demographics.get("persona_id")
            if pid and pers.demographics.get("professional_persona_pre_refine") is not None:
                refined[str(pid)] = pers.demographics

    # Per-persona fallback for any that didn't make it through the batch.
    missing = [p for p in batch if str(p.get("persona_id")) not in refined]
    if missing:
        print(f"  [batch] fallback to per-persona for {len(missing)} entries")
        for p in missing:
            try:
                demos = _refine_one_persona(p, kg_context, config, shared_context=shared_context)
                pid = demos.get("persona_id") or p.get("persona_id")
                if pid:
                    refined[str(pid)] = demos
            except Exception as exc:
                print(f"  [fallback] failed for {p.get('persona_id')}: {exc}")

    return refined


def _stamp_scenarios(sim_data: dict[str, Any], refined_by_pid: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return a deep copy of sim_data with each participant replaced by its refined version."""
    out = copy.deepcopy(sim_data)
    for scenario in out.get("scenarios") or []:
        new_participants = []
        for p in scenario.get("participants") or []:
            pid = p.get("persona_id")
            new_participants.append(refined_by_pid.get(pid) or p)
        scenario["participants"] = new_participants
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-personas", type=Path, required=True)
    parser.add_argument(
        "--pre-refine-source",
        type=Path,
        default=None,
        help="Optional Stage 1 personas_initial.json used to restore *_pre_refine narratives.",
    )
    parser.add_argument(
        "--restore-only",
        action="store_true",
        help="Restore pre-refine narratives and write the bundle without calling an LLM.",
    )
    parser.add_argument("--policy-graph",        type=Path, required=True)
    parser.add_argument("--config-path",         type=str,  default="config/base.yaml")
    parser.add_argument("--out-dir",             type=Path, default=None)
    # scenario slicing (1-indexed inclusive)
    parser.add_argument("--scenario-start", type=int, default=1, help="First scenario to include (1-indexed)")
    parser.add_argument("--scenario-end",   type=int, default=None, help="Last scenario to include (1-indexed, inclusive). Default: all.")
    # batching + parallelism
    parser.add_argument("--batch-size",  type=int, default=5, help="Personas per LLM call (sharing actor_context). 1 = per-persona mode.")
    parser.add_argument("--max-workers", type=int, default=8, help="Concurrent LLM calls (one per batch)")
    # checkpoint
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Dir for per-persona checkpoint files. Default: <out-dir>/refine_checkpoint/")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoint and re-refine everything")
    # output naming
    parser.add_argument("--out-filename", type=str, default="simulation_personas_refined.json")
    args = parser.parse_args()

    out_dir = args.out_dir or args.simulation_personas.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_dir = args.checkpoint_dir or (out_dir / "refine_checkpoint")
    ck_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(open(args.config_path))
    sim_data = json.load(open(args.simulation_personas))
    if args.pre_refine_source:
        source_data = json.load(open(args.pre_refine_source))
        sim_data, restored_count = _restore_pre_refine_narratives(sim_data, source_data)
        print(f"  Restored pre-refine narratives: {restored_count} personas")
    if args.restore_only:
        if not args.pre_refine_source:
            parser.error("--restore-only requires --pre-refine-source")
        out_dir = args.out_dir or args.simulation_personas.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / args.out_filename
        _atomic_write_json(out_path, sim_data)
        print(f"  Wrote {out_path} (no refiner call)")
        return
    kg_lookup, kg_lookup_by_entity_id = _build_kg_lookups(str(args.policy_graph))
    print(f"  KG context lookup keys: {sorted(kg_lookup)}")
    print(f"  KG entity anchors: {len(kg_lookup_by_entity_id)}")

    shared_context = sim_data.get("shared_context") if isinstance(sim_data, dict) else None
    if isinstance(shared_context, dict) and shared_context.get("org_kg_contexts"):
        n_ctx = len(shared_context["org_kg_contexts"])
        print(f"  Policy ecosystem: {n_ctx} actor contexts loaded from shared_context")
    else:
        print("  Policy ecosystem: shared_context missing or empty (refiner falls back to focal-only)")

    # ---- scenario slice ----
    all_scenarios = sim_data.get("scenarios") or []
    start_idx = max(0, args.scenario_start - 1)
    end_idx = len(all_scenarios) if args.scenario_end is None else min(args.scenario_end, len(all_scenarios))
    scenarios = all_scenarios[start_idx:end_idx]
    print(f"  Scenarios: {len(scenarios)} (indices {start_idx + 1}..{end_idx} of {len(all_scenarios)})")

    # ---- collect unique personas ----
    unique: dict[str, dict[str, Any]] = {}
    for s in scenarios:
        for p in s.get("participants") or []:
            pid = p.get("persona_id")
            if pid and pid not in unique:
                unique[pid] = p
    print(f"  Unique personas: {len(unique)}")

    # ---- load checkpoint ----
    completed = {} if args.no_resume else _scan_checkpoint(ck_dir)
    if completed:
        print(f"  Checkpoint: {len(completed)} personas already done in {ck_dir}")
    todo_pids = [pid for pid in unique if pid not in completed]
    print(f"  Remaining: {len(todo_pids)} personas to refine")

    # ---- group by stakeholder_type and chunk into batches ----
    by_context: dict[str, list[dict[str, Any]]] = {}
    context_lookup: dict[str, dict[str, Any]] = {}
    skipped_no_kg: list[str] = []
    for pid in todo_pids:
        p = unique[pid]
        stype = p.get("stakeholder_type") or ""
        anchor_id = str(p.get("role_anchor_entity_id") or "")
        kg_context = kg_lookup_by_entity_id.get(anchor_id) or kg_lookup.get(stype)
        if not kg_context:
            skipped_no_kg.append(pid)
            continue
        context_key = f"anchor:{anchor_id}" if anchor_id in kg_lookup_by_entity_id else f"type:{stype}"
        context_lookup[context_key] = kg_context
        by_context.setdefault(context_key, []).append(p)
    if skipped_no_kg:
        print(f"  Skipped {len(skipped_no_kg)} personas (no KG context for their stakeholder_type)")

    batch_size = max(1, args.batch_size)
    batches: list[tuple[str, list[dict[str, Any]]]] = []
    for context_key, plist in by_context.items():
        for i in range(0, len(plist), batch_size):
            batches.append((context_key, plist[i : i + batch_size]))
    print(f"  Batches: {len(batches)} (size={batch_size})")
    if not batches:
        print("  Nothing to do. Will assemble output from checkpoint only.")

    # ---- run batches in parallel ----
    refined_total = dict(completed)

    def _task(context_key: str, batch: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        kg_ctx = context_lookup[context_key]
        if batch_size == 1 or len(batch) == 1:
            try:
                demos = _refine_one_persona(batch[0], kg_ctx, config, shared_context=shared_context)
                return {str(demos.get("persona_id")): demos}
            except Exception as exc:
                print(f"  [single] failed for {batch[0].get('persona_id')}: {exc}")
                return {}
        return _refine_one_batch(batch, kg_ctx, config, shared_context=shared_context)

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(_task, context_key, b) for context_key, b in batches]
        done = 0
        for fut in as_completed(futures):
            batch_result = fut.result()
            for pid, demos in batch_result.items():
                refined_total[pid] = demos
                _atomic_write_json(ck_dir / f"{pid}.json", demos)
            done += 1
            if done % 5 == 0 or done == len(batches):
                print(f"    {done}/{len(batches)} batches · {len(refined_total)}/{len(unique)} personas done")

    # ---- assemble final bundle (only over the SELECTED scenario slice) ----
    sliced_bundle = dict(sim_data)
    sliced_bundle["scenarios"] = scenarios
    bundle = _stamp_scenarios(sliced_bundle, refined_total)
    out_path = out_dir / args.out_filename
    _atomic_write_json(out_path, bundle)
    print(f"\n  Wrote {out_path}")
    print(f"  Total refined personas in output: {sum(1 for p in unique if p in refined_total)}/{len(unique)}")


if __name__ == "__main__":
    main()
