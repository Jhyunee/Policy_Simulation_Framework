"""Zero-shot baseline: raw policy .md + ToC concept explanation -> single-prompt
prediction -> MAPE. No KG, no personas, no multi-phase deliberation.

Reuses the same policy .md resolution as `baseline/predict.py` (`baseline.policy_mapping`)
and the same target/gt table (`config/policy_targets.json`).

Since a single LLM call is inherently noisy, `--repeats` independent calls are
dispatched concurrently per policy and averaged (mirrors the 5-persona mean used
elsewhere) — set `--repeats 1` for a literal one-shot run.

Run:
    python -m zero_shot.predict --policy kor/masked_bk21_v3 \\
        --config-path config/base.yaml --repeats 20 --run-tag v1
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from ablation.common.io import (
    aggregate_scenario,
    load_policy_targets,
    normalize_gt,
    save_json,
)
from baseline.policy_mapping import POLICY_MAPPING, all_policy_keys, resolve_md_path
from llm.client import chat_completion_with_usage
from zero_shot.prompts import ZERO_SHOT_SYSTEM, ZERO_SHOT_USER

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
ZERO_SHOT_OUTPUTS_ROOT = REPO_ROOT / "outputs" / "zero_shot"
MAX_TEXT_LENGTH = 50_000
MIN_EVIDENCE_QUOTES = 3
MIN_QUOTE_WORDS = 8


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_grounding_text(text: str) -> str:
    """Compare source excerpts while ignoring Markdown punctuation and bullets."""
    return " ".join(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _has_grounded_evidence(evidence: Any, md_text: str) -> bool:
    """Require direct, document-verbatim evidence to reject closed-book leakage."""
    if not isinstance(evidence, list) or len(evidence) != MIN_EVIDENCE_QUOTES:
        return False

    normalized_document = _normalize_grounding_text(md_text)
    for item in evidence:
        if not isinstance(item, str):
            return False
        matches = re.findall(r'["“]([^"”]+)["”]', item)
        if len(matches) > 1:
            return False
        quote = _normalize_text(matches[0] if matches else item)
        normalized_quote = _normalize_grounding_text(quote)
        if len(normalized_quote.split()) < MIN_QUOTE_WORDS or normalized_quote not in normalized_document:
            return False
        if matches and _normalize_text(item) != f'"{quote}"':
            return False
    return True


def _add_usage(total: dict[str, int], usage: dict | None) -> None:
    for key in total:
        total[key] += int((usage or {}).get(key, 0) or 0)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_run_dir(policy_key: str, model: str, tag: str | None) -> Path:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    suffix = f"_{tag}" if tag else ""
    safe_policy = policy_key.replace("/", "_")
    safe_model = model.replace("/", "_")
    run_dir = ZERO_SHOT_OUTPUTS_ROOT / safe_policy / f"{ts}_{safe_model}_result{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def predict_one(
    md_text: str,
    simulation_target: str,
    target_keys: list[str],
    config: dict,
) -> dict:
    """Single zero-shot call. Returns {prediction, evidence, usage} or {error, usage}."""
    prediction_keys_hint = ", ".join(f'"{k}": <number>' for k in target_keys)
    messages = [
        {"role": "system", "content": ZERO_SHOT_SYSTEM},
        {
            "role": "user",
            "content": ZERO_SHOT_USER.format(
                simulation_target=simulation_target,
                prediction_keys_hint="{" + prediction_keys_hint + "}",
                policy_md_text=md_text,
            ),
        },
    ]
    max_attempts = 1 + int(config.get("llm", {}).get("grounding_retries", 2))
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    last_error = "No response received."
    rejected_responses = []

    for attempt in range(1, max_attempts + 1):
        try:
            content, usage = chat_completion_with_usage(messages, config)
            _add_usage(total_usage, usage)
            import json

            parsed = json.loads(content)
            evidence = parsed.get("evidence") or []
            if not _has_grounded_evidence(evidence, md_text):
                last_error = "Ungrounded evidence: expected three verbatim policy-document quotes."
                rejected_responses.append({
                    "attempt": attempt,
                    "raw_response": content,
                    "rejection_reason": last_error,
                })
                continue

            prediction = parsed.get("prediction") or {}
            cleaned_pred: dict[str, float | None] = {}
            for k in target_keys:
                v = prediction.get(k)
                if v is None:
                    cleaned_pred[k] = None
                    continue
                try:
                    cleaned_pred[k] = float(v)
                except (TypeError, ValueError):
                    cleaned_pred[k] = None
            return {
                "prediction": cleaned_pred,
                "evidence": evidence,
                "usage": total_usage,
                "grounding_attempts": attempt,
            }
        except Exception as exc:  # noqa: BLE001 — driver-level, log + continue
            last_error = f"{type(exc).__name__}: {exc}"

    return {
        "prediction": {k: None for k in target_keys},
        "evidence": [],
        "usage": total_usage,
        "error": last_error,
        "rejected_responses": rejected_responses,
    }


def run_policy(
    policy_key: str,
    config: dict,
    config_path: str,
    repeats: int,
    run_dir: Path,
) -> dict:
    print(f"\n=== {policy_key} ===")
    md_path = resolve_md_path(policy_key)
    print(f"  md: {md_path}")
    md_text = md_path.read_text(encoding="utf-8")
    original_length = len(md_text)
    if original_length > MAX_TEXT_LENGTH:
        md_text = md_text[:MAX_TEXT_LENGTH] + (
            f"\n\n...(original text was {original_length} characters; "
            f"truncated to first {MAX_TEXT_LENGTH} characters)..."
        )

    targets_all = load_policy_targets()
    target_info = targets_all[policy_key]
    simulation_target = target_info["simulation_target"]
    gt = normalize_gt(target_info["gt"])
    target_keys = list(gt.keys())
    print(f"  simulation_target: {simulation_target}")
    print(f"  gt: {gt}")
    print(f"  repeats: {repeats}")

    t0 = time.time()
    results: list[dict] = [None] * repeats  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=repeats) as ex:
        futures = {
            ex.submit(predict_one, md_text, simulation_target, target_keys, config): i
            for i in range(repeats)
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")

    pred_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    call_records = []
    for i, r in enumerate(results):
        u = r["usage"]
        for k in pred_tokens:
            pred_tokens[k] += u.get(k, 0)
        call_records.append({
            "call_idx": i,
            "prediction": r["prediction"],
            "evidence": r["evidence"],
            "usage": u,
            **({"error": r["error"]} if "error" in r else {}),
            **({"rejected_responses": r["rejected_responses"]} if "rejected_responses" in r else {}),
        })

    agg = aggregate_scenario([r["prediction"] for r in results], gt, min_valid_personas=max(1, repeats // 2))
    print(f"  mean: {agg['scenario_mean_prediction']}   mape: {agg['scenario_mape']}")

    out = {
        "policy_key": policy_key,
        "condition": "zero_shot",
        "config_model": config.get("llm", {}).get("model"),
        "config_path": config_path,
        "md_path": str(md_path),
        "simulation_target": simulation_target,
        "gt": gt,
        "repeats": repeats,
        "elapsed_seconds": elapsed,
        "calls": call_records,
        "mean_prediction": agg["scenario_mean_prediction"],
        "mape": agg["scenario_mape"],
        "n_valid_calls_per_key": agg["n_valid_personas_per_key"],
        "skipped": agg["skipped"],
        "total_usage": pred_tokens,
    }
    save_json(run_dir / "results.json", out)
    save_json(run_dir / "run_log.json", {
        "policy_key": policy_key,
        "condition": "zero_shot",
        "model": config.get("llm", {}).get("model"),
        "md_path": str(md_path),
        "repeats": repeats,
        "elapsed_seconds": elapsed,
        "mape": agg["scenario_mape"],
        "total_usage": pred_tokens,
    })

    return {
        "policy_key": policy_key,
        "gt": gt,
        "mape": agg["scenario_mape"],
        "repeats": repeats,
        "pred_tokens": pred_tokens,
    }


def write_summary_csv(run_dirs_root: Path, policy_results: list[dict]) -> None:
    csv_path = run_dirs_root / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "policy_key", "target_name", "gt", "mape", "repeats",
            "pred_prompt_tokens", "pred_completion_tokens", "pred_total_tokens",
        ])
        for pr in policy_results:
            for k, gt_v in pr["gt"].items():
                w.writerow([
                    pr["policy_key"], k, gt_v, pr["mape"].get(k), pr["repeats"],
                    pr["pred_tokens"].get("prompt_tokens", 0),
                    pr["pred_tokens"].get("completion_tokens", 0),
                    pr["pred_tokens"].get("total_tokens", 0),
                ])
    print(f"\nSummary CSV: {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Zero-shot baseline: ToC-explained single prompt -> MAPE")
    parser.add_argument("--config-path", default="config/base.yaml")
    parser.add_argument("--policies", default=",".join(all_policy_keys()),
                         help="Comma-separated policy keys (default: all)")
    parser.add_argument("--repeats", type=int, default=20,
                         help="Independent single-shot calls per policy, averaged (default 20)")
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()

    config = load_config(args.config_path)
    model = config.get("llm", {}).get("model", "unknown")

    policy_keys = [p.strip() for p in args.policies.split(",") if p.strip()]
    unknown = [p for p in policy_keys if p not in POLICY_MAPPING]
    if unknown:
        print(f"ERROR: unknown policy keys: {unknown}", file=sys.stderr)
        print(f"Available: {list(POLICY_MAPPING.keys())}", file=sys.stderr)
        return 2

    policy_results = []
    failed_policies = []
    for policy_key in policy_keys:
        run_dir = make_run_dir(policy_key, model, args.run_tag)
        print(f"Run dir: {run_dir}")
        try:
            pr = run_policy(policy_key, config, args.config_path, args.repeats, run_dir)
            policy_results.append(pr)
        except Exception as exc:
            print(f"  !! policy {policy_key} FAILED: {type(exc).__name__}: {exc}")
            failed_policies.append(policy_key)
            continue

    if failed_policies:
        print(f"\nFailed policies (skipped): {failed_policies}")
    if policy_results:
        write_summary_csv(ZERO_SHOT_OUTPUTS_ROOT, policy_results)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
