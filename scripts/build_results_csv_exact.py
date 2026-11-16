"""Build per-scenario results CSV with EXACT field-name matching.

Sibling of ``scripts/build_results_csv.py``. That tool uses ``hint``-based
substring matching against ``final_report.json`` Outcomes fields — convenient
when one short hint uniquely names a variable, but it picks the wrong field
when a short variable name (e.g. ``employment_rate``) is a substring of a
longer one (e.g. ``graduate_employment_rate_of_bk21_participants_in_2022``).

This script avoids that hazard: it takes the **exact** field name to pull
from each scenario's prediction_values, and writes the same 5-column CSV
layout the original tool emits (``Fwd only | Bwd only | Fwd->Bwd |
Fwd->Bwd->Fwd(Final) | Fwd->Bwd->Fwd_seq``).

Use this when:
  - The target field name is a substring of another variable (e.g. BK21
    short auxiliary fields).
  - The seq run's inner_schema named the variable differently from the
    bwd-only run's inner_schema (each stage2-toc run rebuilds the schema
    via LLM and minor name drift is expected). In that case provide the
    ``BWD_FIELD`` slot in ``--target``.

Output filenames mirror the original tool: ``results_<LABEL>_with_outliers.csv``
and ``results_<LABEL>_outliers_removed.csv``, written into **both** the seq
run dir and the bwd-only run dir.

Usage:
    python scripts/build_results_csv_exact.py \\
        --seq-run outputs/kor/bk21/stage2_toc/<seq_dir> \\
        --bwd-only-run outputs/kor/bk21/stage2_toc/<bwd_dir> \\
        --target employment_rate:82.2 \\
        --target international_collab_rate:36.0:international_collaboration_rate:intl_research_collab_rate

--target format: ``FIELD:GT[:LABEL[:BWD_FIELD]]``
    FIELD     — exact field name in the SEQ run (used for Fwd / Fwd->Bwd /
                Final / 5th chain columns).
    GT        — ground truth value (float).
    LABEL     — filename slug for the CSV (default = FIELD).
    BWD_FIELD — bwd-only run's field name when it differs from FIELD
                (default = FIELD; used only for the Bwd-only column).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Optional


OUTCOMES_IDX = 3
COLUMNS = ["Fwd only", "Bwd only", "Fwd->Bwd", "Fwd->Bwd->Fwd(Final)",
           "Fwd->Bwd->Fwd_seq", "Fwd->Bwd->Fwd->Bwd_seq"]


def _slot_mean(phase: Optional[dict], slot: str, key: str) -> Optional[float]:
    if not phase:
        return None
    posts = phase.get(slot) or []
    vals = [p.get('prediction_values', {}).get(key) for p in posts]
    vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return mean(vals) if vals else None


def _scenario_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob('scenario_*.json'), key=lambda p: int(p.stem.split('_')[1]))


def _load_scenarios(run_dir: Path) -> list[tuple[int, dict]]:
    return [(int(p.stem.split('_')[1]), json.loads(p.read_text(encoding='utf-8')))
            for p in _scenario_files(run_dir)]


def _iqr_outlier_mask(values: list[Optional[float]]) -> list[bool]:
    """Tukey 1.5×IQR mask — matches scripts/build_results_csv.py exactly."""
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
    return None if mu is None else abs(mu - gt) / gt * 100


def _fmt(v: Optional[float]) -> str:
    return '' if v is None else f'{v:.1f}'


def _parse_scenarios_arg(spec: Optional[str], available: list[int]) -> Optional[set[int]]:
    if not spec:
        return None
    out: set[int] = set()
    for tok in spec.split(','):
        tok = tok.strip()
        if '-' in tok:
            a, b = tok.split('-', 1)
            for i in range(int(a), int(b) + 1):
                out.add(i)
        else:
            out.add(int(tok))
    return out & set(available)


def build_rows(
    seq_dir: Path,
    bwd_dir: Path,
    seq_field: str,
    bwd_field: str,
    scenarios_keep: Optional[set[int]] = None,
) -> list[list[Optional[float]]]:
    seq = _load_scenarios(seq_dir)
    bwd = _load_scenarios(bwd_dir)
    if scenarios_keep is not None:
        seq = [(i, sc) for i, sc in seq if i in scenarios_keep]
        bwd = [(i, sc) for i, sc in bwd if i in scenarios_keep]

    rows: list[list[Optional[float]]] = []
    for (i, sc), (_, sb) in zip(seq, bwd):
        fp = (sc.get('forward_pass') or [None] * 5)[OUTCOMES_IDX]
        bp_seq = (sc.get('backward_pass') or [None] * 5)[OUTCOMES_IDX]
        fbf = (sc.get('fwd_bwd_fwd_pass') or [None] * 5)[OUTCOMES_IDX]
        fbfb = (sc.get('fwd_bwd_fwd_bwd_pass') or [None] * 5)[OUTCOMES_IDX]
        bp_only = (sb.get('backward_pass') or [None] * 5)[OUTCOMES_IDX]

        fwd_only = _slot_mean(fp, 'revised_posts', seq_field) or _slot_mean(fp, 'initial_posts', seq_field)
        fin = _slot_mean(fp, 'refined_posts', seq_field) or fwd_only
        bwd_seq = (_slot_mean(bp_seq, 'refined_posts', seq_field)
                   or _slot_mean(bp_seq, 'revised_posts', seq_field)
                   or _slot_mean(bp_seq, 'initial_posts', seq_field))
        bwd_only = (_slot_mean(bp_only, 'refined_posts', bwd_field)
                    or _slot_mean(bp_only, 'revised_posts', bwd_field)
                    or _slot_mean(bp_only, 'initial_posts', bwd_field))
        fbf_seq = _slot_mean(fbf, 'revised_posts', seq_field) if fbf else None
        fbfb_seq = _slot_mean(fbfb, 'revised_posts', seq_field) if fbfb else None

        rows.append([i, fwd_only, bwd_only, bwd_seq, fin, fbf_seq, fbfb_seq])
    return rows


def write_csv(path: Path, rows: list[list[Optional[float]]], gt: float, *, blank_outliers: bool) -> None:
    cols = list(zip(*[r[1:] for r in rows]))
    masks = [_iqr_outlier_mask(list(c)) for c in cols]
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['scenario'] + COLUMNS)
        for i, row in enumerate(rows):
            sid, *cells = row
            display = []
            for c, v in enumerate(cells):
                if blank_outliers and masks[c][i]:
                    display.append('')
                else:
                    display.append(_fmt(v))
            w.writerow([f'S{sid}'] + display)
        # mean + err% summary rows
        mean_cells, err_cells = [], []
        for c in range(len(COLUMNS)):
            if blank_outliers:
                kept = [v for v, o in zip(cols[c], masks[c]) if v is not None and not o]
            else:
                kept = [v for v in cols[c] if v is not None]
            if kept:
                mu = mean(kept)
                mean_cells.append(_fmt(mu))
                err_cells.append(f'{_err_pct(mu, gt):.1f}%')
            else:
                mean_cells.append('')
                err_cells.append('')
        w.writerow(['mean'] + mean_cells)
        w.writerow(['err%'] + err_cells)


def _parse_target(spec: str) -> tuple[str, float, str, str]:
    """Parse 'FIELD:GT[:LABEL[:BWD_FIELD]]'. Returns (seq_field, gt, label, bwd_field)."""
    parts = spec.split(':')
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(f"--target must be FIELD:GT[:LABEL[:BWD_FIELD]], got {spec!r}")
    seq_field = parts[0]
    try:
        gt = float(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"GT must be numeric, got {parts[1]!r}") from e
    label = parts[2] if len(parts) >= 3 and parts[2] else seq_field
    bwd_field = parts[3] if len(parts) >= 4 and parts[3] else seq_field
    return seq_field, gt, label, bwd_field


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--seq-run', type=Path, required=True, help='Full sequential stage2-toc run dir')
    p.add_argument('--bwd-only-run', type=Path, required=True, help='Backward-only stage2-toc run dir (--no-forward)')
    p.add_argument(
        '--target',
        action='append',
        required=True,
        metavar='FIELD:GT[:LABEL[:BWD_FIELD]]',
        help='Exact target field, ground truth, optional CSV label, optional bwd-run field name. Repeatable.',
    )
    p.add_argument('--scenarios', type=str, default=None,
                   help="Subset to keep, e.g. '1-50' or '1,3,5'. Default: all scenarios.")
    args = p.parse_args()

    if not args.seq_run.is_dir():
        raise SystemExit(f'--seq-run not a directory: {args.seq_run}')
    if not args.bwd_only_run.is_dir():
        raise SystemExit(f'--bwd-only-run not a directory: {args.bwd_only_run}')

    available = [int(p.stem.split('_')[1]) for p in _scenario_files(args.seq_run)]
    scenarios_keep = _parse_scenarios_arg(args.scenarios, available)

    for spec in args.target:
        seq_field, gt, label, bwd_field = _parse_target(spec)
        rows = build_rows(args.seq_run, args.bwd_only_run, seq_field, bwd_field, scenarios_keep)
        for out_dir in (args.seq_run, args.bwd_only_run):
            with_p = out_dir / f'results_{label}_with_outliers.csv'
            no_p = out_dir / f'results_{label}_outliers_removed.csv'
            write_csv(with_p, rows, gt, blank_outliers=False)
            write_csv(no_p, rows, gt, blank_outliers=True)
            print(f'  wrote {with_p}')
            print(f'  wrote {no_p}')

        # Console summary
        n_complete = sum(1 for r in rows if all(v is not None for v in r[1:]))
        print(f'[{label}] GT={gt}  seq_field={seq_field}  bwd_field={bwd_field}  '
              f'scenarios={len(rows)} (complete={n_complete})')


if __name__ == '__main__':
    main()
