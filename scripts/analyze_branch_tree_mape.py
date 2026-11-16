#!/usr/bin/env python3
"""Compute outcome-level MAPE summaries for a pathway branch tree."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--ground-truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-node",
        default="root/baseline/baseline/baseline",
    )
    return parser.parse_args()


def mape(prediction: float, ground_truth: float) -> float:
    return abs(prediction - ground_truth) / abs(ground_truth) * 100.0


def load_ground_truth(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def node_means(tree: dict, target_fields: set[str]) -> list[dict[str, object]]:
    nodes = tree["nodes"].values() if isinstance(tree["nodes"], dict) else tree["nodes"]
    rows: list[dict[str, object]] = []
    for node in nodes:
        phase_result = node["phase_result"]
        if phase_result["phase"] != "Outcomes":
            continue
        posts = phase_result.get("revised_posts") or phase_result.get("initial_posts") or []
        for field in target_fields:
            values = [
                float(post["prediction_values"][field])
                for post in posts
                if field in post.get("prediction_values", {})
                and isinstance(post["prediction_values"][field], (int, float))
            ]
            if values:
                rows.append({
                    "node_id": node["node_id"],
                    "phase": "Outcomes",
                    "local_transition": node.get("transition_mode"),
                    "field": field,
                    "node_mean": mean(values),
                    "persona_min": min(values),
                    "persona_max": max(values),
                    "persona_count": len(values),
                })
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    tree = json.loads(args.tree.read_text(encoding="utf-8"))
    ground_truth_rows = load_ground_truth(args.ground_truth_csv)
    ground_truth = {
        row["field"]: {
            "label": row["label"],
            "value": float(row["ground_truth"]),
        }
        for row in ground_truth_rows
    }
    rows = node_means(tree, set(ground_truth))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    quantitative_path = args.output_dir / "tree_node_quantitative_summary.csv"
    write_csv(
        quantitative_path,
        [
            "node_id", "phase", "local_transition", "field", "node_mean",
            "persona_min", "persona_max", "persona_count",
        ],
        rows,
    )

    grouped: dict[str, list[dict]] = {field: [] for field in ground_truth}
    for row in rows:
        grouped[row["field"]].append(row)

    mean_rows = []
    range_rows = []
    for field, truth_meta in ground_truth.items():
        field_rows = grouped[field]
        if not field_rows:
            continue
        truth = truth_meta["value"]
        tree_mean = mean(float(row["node_mean"]) for row in field_rows)
        mean_rows.append({
            "field": field,
            "label": truth_meta["label"],
            "ground_truth": truth,
            "tree_mean_prediction": round(tree_mean, 6),
            "mape_pct": round(mape(tree_mean, truth), 6),
            "node_count": len(field_rows),
        })

        scored = [
            (mape(float(row["node_mean"]), truth), row)
            for row in field_rows
        ]
        min_error, min_row = min(scored, key=lambda pair: pair[0])
        max_error, max_row = max(scored, key=lambda pair: pair[0])
        baseline_row = next(
            (row for row in field_rows if row["node_id"] == args.baseline_node),
            None,
        )
        range_rows.append({
            "field": field,
            "label": truth_meta["label"],
            "ground_truth": truth,
            "max_mape_pct": round(max_error, 6),
            "max_mape_prediction": round(float(max_row["node_mean"]), 6),
            "max_mape_path": max_row["node_id"],
            "baseline_mape_pct": (
                round(mape(float(baseline_row["node_mean"]), truth), 6)
                if baseline_row else None
            ),
            "baseline_prediction": (
                round(float(baseline_row["node_mean"]), 6)
                if baseline_row else None
            ),
            "baseline_path": args.baseline_node if baseline_row else None,
            "min_mape_pct": round(min_error, 6),
            "min_mape_prediction": round(float(min_row["node_mean"]), 6),
            "min_mape_path": min_row["node_id"],
        })

    mean_path = args.output_dir / "tree_outcome_mean_prediction_mape.csv"
    write_csv(
        mean_path,
        [
            "field", "label", "ground_truth", "tree_mean_prediction",
            "mape_pct", "node_count",
        ],
        mean_rows,
    )
    range_path = args.output_dir / "tree_outcome_mape_max_baseline_min.csv"
    write_csv(
        range_path,
        [
            "field", "label", "ground_truth", "max_mape_pct",
            "max_mape_prediction", "max_mape_path", "baseline_mape_pct",
            "baseline_prediction", "baseline_path", "min_mape_pct",
            "min_mape_prediction", "min_mape_path",
        ],
        range_rows,
    )
    print(f"Saved {len(rows)} node-field estimates from {len(rows) // len(ground_truth)} outcome nodes")
    print(mean_path)
    print(range_path)


if __name__ == "__main__":
    main()
