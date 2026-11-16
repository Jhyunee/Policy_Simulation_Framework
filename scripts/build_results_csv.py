"""Build per-scenario results CSV merging a full sequential run with a backward-only run.

For one or more targets, produces two CSV files per target:
  - results_<tag>_with_outliers.csv      raw 10-scenario means + summary rows
  - results_<tag>_outliers_removed.csv   same shape; cells outside IQR×1.5 emptied

Saved into BOTH the seq run dir and the bwd-only run dir.

Columns (fixed order, per user request):
  scenario | Fwd only | Bwd only | Fwd->Bwd | Fwd->Bwd->Fwd(Final) | Fwd->Bwd->Fwd_seq | Fwd->Bwd->Fwd->Bwd_seq

Mapping:
  Fwd only               ← seq run, forward_pass[Outcomes].revised_posts  mean
  Bwd only               ← bwd-only run, backward_pass[Outcomes].revised_posts mean
  Fwd->Bwd               ← seq run, backward_pass[Outcomes].revised_posts mean
  Fwd->Bwd->Fwd(Final)   ← seq run, forward_pass[Outcomes].refined_posts  mean
  Fwd->Bwd->Fwd_seq      ← seq run, fwd_bwd_fwd_pass[Outcomes].revised_posts mean (5th chain)
  Fwd->Bwd->Fwd->Bwd_seq ← seq run, fwd_bwd_fwd_bwd_pass[Outcomes].revised_posts mean (6th chain)

Usage:
    python scripts/build_results_csv.py \\
        --seq-run outputs/stage2_toc/<seq_dir> \\
        --bwd-only-run outputs/stage2_toc/<bwd_dir> \\
        --target approved:115 \\
        --target deployed:73
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Optional

OUTCOMES_IDX = 3  # phase index for Outcomes in the 5-phase ToC


# ----------------------------------------------------------------------------- helpers

def _slot_mean(phase: dict, slot: str, key: str) -> Optional[float]:
    posts = phase.get(slot) or []
    vals = [p.get("prediction_values", {}).get(key) for p in posts]
    vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return mean(vals) if vals else None


def _find_target_key(run_dir: Path, hint: str) -> Optional[str]:
    """Find the Outcomes field name in this run whose name contains `hint`."""
    fr_path = run_dir / "final_report.json"
    if not fr_path.exists():
        return None
    with fr_path.open("r", encoding="utf-8") as f:
        fr = json.load(f)
    outcomes = [r["variable"] for r in (fr.get("implementation_gap_report") or []) if r.get("phase") == "Outcomes"]
    for v in outcomes:
        if hint.lower() in v.lower():
            return v
    return outcomes[0] if outcomes else None


def _scenario_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("scenario_*.json"), key=lambda p: int(p.stem.split("_")[1]))


def _read_scenarios(run_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in _scenario_files(run_dir)]


def _iqr_outlier_mask(values: list[Optional[float]]) -> list[bool]:
    """Per Tukey: outlier if value < Q1-1.5*IQR or > Q3+1.5*IQR."""
    pure = sorted(v for v in values if v is not None)
    if len(pure) < 4:
        return [False] * len(values)
    q1 = pure[len(pure) // 4]
    q3 = pure[3 * len(pure) // 4]
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return [v is not None and (v < lo or v > hi) for v in values]


def _err_pct(mu: Optional[float], gt: float) -> Optional[float]:
    if mu is None:
        return None
    return abs(mu - gt) / gt * 100.0


# ----------------------------------------------------------------------------- builders

COLUMNS = ["Fwd only", "Bwd only", "Fwd->Bwd", "Fwd->Bwd->Fwd(Final)",
           "Fwd->Bwd->Fwd_seq", "Fwd->Bwd->Fwd->Bwd_seq"]


def build_rows_for_target(
    seq_run: Path,
    bwd_run: Path,
    target_seq: str,
    target_bwd: str,
) -> list[list[Optional[float]]]:
    """Return one row per scenario index (1..N) with the 6 column values."""
    seq_scs = _read_scenarios(seq_run)
    bwd_scs = _read_scenarios(bwd_run)
    n = max(len(seq_scs), len(bwd_scs))
    rows: list[list[Optional[float]]] = []
    for i in range(n):
        # Fwd only / Fwd->Bwd / Final / Fwd->Bwd->Fwd_seq / Fwd->Bwd->Fwd->Bwd_seq from seq run
        fwd_only = bwd_seq = fin = fbf_seq = fbfb_seq = None
        if i < len(seq_scs):
            sc = seq_scs[i]
            try:
                fp = sc["forward_pass"][OUTCOMES_IDX]
                fwd_only = _slot_mean(fp, "revised_posts", target_seq) or _slot_mean(fp, "initial_posts", target_seq)
                fin = _slot_mean(fp, "refined_posts", target_seq) or fwd_only
            except (KeyError, IndexError, TypeError):
                pass
            try:
                bp = sc["backward_pass"][OUTCOMES_IDX]
                bwd_seq = (
                    _slot_mean(bp, "refined_posts", target_seq)
                    or _slot_mean(bp, "revised_posts", target_seq)
                    or _slot_mean(bp, "initial_posts", target_seq)
                )
            except (KeyError, IndexError, TypeError):
                pass
            try:
                fbf = (sc.get("fwd_bwd_fwd_pass") or [])[OUTCOMES_IDX]
                fbf_seq = _slot_mean(fbf, "revised_posts", target_seq) or _slot_mean(fbf, "initial_posts", target_seq)
            except (KeyError, IndexError, TypeError):
                pass
            try:
                fbfb = (sc.get("fwd_bwd_fwd_bwd_pass") or [])[OUTCOMES_IDX]
                fbfb_seq = _slot_mean(fbfb, "revised_posts", target_seq) or _slot_mean(fbfb, "initial_posts", target_seq)
            except (KeyError, IndexError, TypeError):
                pass
        # Bwd only from bwd-only run
        bwd_only = None
        if i < len(bwd_scs):
            sc = bwd_scs[i]
            try:
                bp = sc["backward_pass"][OUTCOMES_IDX]
                bwd_only = (
                    _slot_mean(bp, "refined_posts", target_bwd)
                    or _slot_mean(bp, "revised_posts", target_bwd)
                    or _slot_mean(bp, "initial_posts", target_bwd)
                )
            except (KeyError, IndexError, TypeError):
                pass
        rows.append([fwd_only, bwd_only, bwd_seq, fin, fbf_seq, fbfb_seq])
    return rows


def _format_cell(v: Optional[float]) -> str:
    if v is None:
        return ""
    return f"{v:.1f}"


def write_csv(
    path: Path,
    rows: list[list[Optional[float]]],
    gt: float,
    *,
    blank_outliers: bool,
) -> None:
    """Write CSV with header, data rows (S1..SN), mean row, err% row."""
    n_cols = len(COLUMNS)
    # Compute outlier mask per column
    col_values = [[r[c] for r in rows] for c in range(n_cols)]
    masks = [_iqr_outlier_mask(col_values[c]) for c in range(n_cols)]

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario"] + COLUMNS)
        for i, row in enumerate(rows):
            display = []
            for c, v in enumerate(row):
                if blank_outliers and masks[c][i]:
                    display.append("")
                else:
                    display.append(_format_cell(v))
            w.writerow([f"S{i+1}"] + display)

        # Summary rows — mean and err%
        mean_cells: list[str] = []
        err_cells: list[str] = []
        for c in range(n_cols):
            if blank_outliers:
                kept = [v for v, out in zip(col_values[c], masks[c]) if v is not None and not out]
            else:
                kept = [v for v in col_values[c] if v is not None]
            if kept:
                mu = mean(kept)
                mean_cells.append(f"{mu:.1f}")
                err_cells.append(f"{_err_pct(mu, gt):.1f}%")
            else:
                mean_cells.append("")
                err_cells.append("")
        w.writerow(["mean"] + mean_cells)
        w.writerow(["err%"] + err_cells)


# ----------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-run", type=Path, required=True, help="Full sequential stage2 run dir (with forward+backward)")
    p.add_argument("--bwd-only-run", type=Path, required=True, help="Backward-only stage2 run dir (--no-forward)")
    p.add_argument(
        "--target",
        action="append",
        required=True,
        metavar="HINT:GT",
        help="Target hint (substring match against final_report Outcomes field names) and ground truth, "
             "e.g. 'approved:115'. Repeatable.",
    )
    args = p.parse_args()

    if not args.seq_run.is_dir():
        raise SystemExit(f"seq-run not a directory: {args.seq_run}")
    if not args.bwd_only_run.is_dir():
        raise SystemExit(f"bwd-only-run not a directory: {args.bwd_only_run}")

    for spec in args.target:
        if ":" not in spec:
            raise SystemExit(f"--target must be HINT:GT, got {spec!r}")
        hint, gt_str = spec.split(":", 1)
        hint = hint.strip()
        try:
            gt = float(gt_str)
        except ValueError:
            raise SystemExit(f"GT must be numeric, got {gt_str!r}")

        target_seq = _find_target_key(args.seq_run, hint)
        target_bwd = _find_target_key(args.bwd_only_run, hint)
        if not target_seq:
            print(f"[skip] {hint}: no Outcomes field in seq run matches")
            continue
        if not target_bwd:
            print(f"[warn] {hint}: no Outcomes field in bwd-only run matches — Bwd only column will be empty")
            target_bwd = ""

        rows = build_rows_for_target(args.seq_run, args.bwd_only_run, target_seq, target_bwd)

        # Save into BOTH dirs
        for out_dir in (args.seq_run, args.bwd_only_run):
            with_out_path = out_dir / f"results_{hint}_with_outliers.csv"
            no_out_path = out_dir / f"results_{hint}_outliers_removed.csv"
            write_csv(with_out_path, rows, gt, blank_outliers=False)
            write_csv(no_out_path, rows, gt, blank_outliers=True)
            print(f"  wrote {with_out_path}")
            print(f"  wrote {no_out_path}")

        # Console summary
        n_kept_all = sum(1 for r in rows if all(v is not None for v in r))
        print(f"[{hint}] GT={gt}  target_seq={target_seq}  target_bwd={target_bwd}  scenarios={len(rows)} (complete={n_kept_all})")


if __name__ == "__main__":
    main()
