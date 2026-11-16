"""Build the quantitative aggregation file (aggregated_final + implementation_gap_report)
from a stage2-toc run dir whose scenarios completed but whose aggregator step
failed (e.g. when the per-(phase, stakeholder) LLM synthesizer crashed on a
too-large prompt).

Reads:
  - <run-dir>/inner_schema.json
  - <run-dir>/scenario_*.json

Writes:
  - <run-dir>/aggregated_quantitative.json

The math mirrors `stage2_toc.modules.aggregate_across_scenarios` exactly:
  * aggregated_final = confidence-weighted mean of scenario_impact per Impact field
                       (falls back to unweighted mean when no positive weights)
  * implementation_gap_report = per-(phase, variable) mean across scenarios

This skips the two LLM aggregators (`run_final_toc_reporter` and
`run_phase_stakeholder_synthesizer`) — those produce qualitative narrative
content that is unnecessary for the quantitative analysis tables.

Usage:
    python scripts/build_quantitative_report.py \\
        --run-dir outputs/kor/ssf/stage2_toc/<dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Optional


def _scenario_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("scenario_*.json"), key=lambda p: int(p.stem.split("_")[1]))


def _impact_fields(schema: dict[str, Any]) -> list[dict[str, Any]]:
    for phase in schema.get("phases") or []:
        if phase.get("phase") == "Impact":
            return [
                f for f in (phase.get("prediction_fields") or [])
                if f.get("type") in ("int", "float")
            ]
    return []


def _aggregated_final(impact_fields: list[dict[str, Any]], scenarios: list[dict[str, Any]]) -> dict[str, Optional[float]]:
    """Confidence-weighted mean per Impact field."""
    out: dict[str, Optional[float]] = {}
    for field in impact_fields:
        name = field["name"]
        num = denom = 0.0
        unweighted: list[float] = []
        for s in scenarios:
            v = (s.get("scenario_impact") or {}).get(name)
            w = (s.get("scenario_confidence") or {}).get(name)
            if not isinstance(v, (int, float)):
                continue
            unweighted.append(float(v))
            if isinstance(w, (int, float)) and w > 0:
                num += float(v) * float(w)
                denom += float(w)
        if denom > 0:
            out[name] = num / denom
        elif unweighted:
            out[name] = mean(unweighted)
        else:
            out[name] = None
    return out


def _impl_gap_report(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-(phase, variable) mean across scenarios."""
    acc: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for s in scenarios:
        for c in (s.get("cross_checks") or []):
            key = (c.get("phase"), c.get("variable"))
            acc.setdefault(key, []).append(c)

    out: list[dict[str, Any]] = []
    for (phase, variable), checks in acc.items():
        def m(attr: str) -> Optional[float]:
            vals = [c.get(attr) for c in checks if isinstance(c.get(attr), (int, float))]
            return float(sum(vals)) / len(vals) if vals else None
        out.append({
            "phase": phase,
            "variable": variable,
            "forward_agg": m("forward_agg"),
            "backward_agg": m("backward_agg"),
            "gap": m("gap"),
            "relative_gap": m("relative_gap"),
            "consistency": m("consistency"),
            "aggregated_value": m("aggregated_value"),
        })
    # Stable sort by inner_schema phase order then variable name.
    phase_order = ["Inputs", "Activities", "Outputs", "Outcomes", "Impact"]
    out.sort(key=lambda r: (phase_order.index(r["phase"]) if r["phase"] in phase_order else 99, r["variable"]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="Stage 2 ToC run dir (contains scenario_*.json + inner_schema.json)")
    p.add_argument("--out", type=Path, default=None, help="Output JSON path (default: <run-dir>/aggregated_quantitative.json)")
    args = p.parse_args()

    if not args.run_dir.is_dir():
        raise SystemExit(f"--run-dir not a directory: {args.run_dir}")

    schema_path = args.run_dir / "inner_schema.json"
    if not schema_path.exists():
        raise SystemExit(f"inner_schema.json not found in {args.run_dir}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    scenario_paths = _scenario_files(args.run_dir)
    if not scenario_paths:
        raise SystemExit(f"No scenario_*.json files in {args.run_dir}")
    scenarios = [json.loads(p.read_text(encoding="utf-8")) for p in scenario_paths]

    impact_fields = _impact_fields(schema)
    aggregated_final = _aggregated_final(impact_fields, scenarios)
    impl_gap = _impl_gap_report(scenarios)

    out_path = args.out or (args.run_dir / "aggregated_quantitative.json")
    payload = {
        "run_dir": str(args.run_dir),
        "scenario_count": len(scenarios),
        "scenarios_used": [p.stem for p in scenario_paths],
        "aggregated_final": aggregated_final,
        "implementation_gap_report": impl_gap,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"  scenarios: {len(scenarios)}")
    print(f"  aggregated_final fields: {len(aggregated_final)}")
    print(f"  implementation_gap_report rows: {len(impl_gap)}")
    print(f"\n  aggregated_final = {json.dumps(aggregated_final, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
