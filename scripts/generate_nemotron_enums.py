"""Generate {age_range.py, education_level_values.py, occupation_values.py} for a
Nemotron-Personas-* dataset by scanning its parquet shards via DuckDB.

Usage:
    python scripts/generate_nemotron_enums.py \
        --parquet-glob 'dataset/processed/nemotron_sg/Nemotron-Personas-Singapore/data/*.parquet' \
        --out-dir       'dataset/processed/nemotron_sg' \
        --label         'Nemotron-Personas-Singapore'

Produces:
    <out-dir>/age_range.py            → AGE_MIN, AGE_MAX
    <out-dir>/education_level_values.py → EDUCATION_LEVELS (frequency-sorted)
    <out-dir>/occupation_values.py    → OCCUPATIONS (frequency-sorted)
    <out-dir>/__init__.py             → empty marker
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def _quote(value: str) -> str:
    """Render a Python-quoted string literal that handles embedded quotes safely."""
    if "'" not in value:
        return f"'{value}'"
    if '"' not in value:
        return f'"{value}"'
    # Both quote types present — escape doubles
    escaped = value.replace('"', '\\"')
    return f'"{escaped}"'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet-glob", required=True, help="Glob pattern for parquet shards")
    p.add_argument("--out-dir", required=True, help="Destination dir to write the three enum files")
    p.add_argument("--label", default="Nemotron-Personas", help="Human label for header comments")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Ensure __init__.py exists (Python package requirement)
    init_path = out / "__init__.py"
    if not init_path.exists():
        init_path.write_text("", encoding="utf-8")

    # 1. age range
    age_min, age_max = duckdb.sql(
        f"SELECT MIN(age), MAX(age) FROM read_parquet('{args.parquet_glob}') WHERE age IS NOT NULL"
    ).fetchone()
    (out / "age_range.py").write_text(
        f"# Auto-generated from {args.label}.\n"
        f"# Verified observed ages run from {age_min} to {age_max} inclusive.\n"
        f"AGE_MIN = {int(age_min)}\n"
        f"AGE_MAX = {int(age_max)}\n",
        encoding="utf-8",
    )
    print(f"  wrote age_range.py    AGE_MIN={age_min} AGE_MAX={age_max}")

    # 2. education_level values (frequency-sorted)
    edu_rows = duckdb.sql(
        f"SELECT education_level, COUNT(*) AS n "
        f"FROM read_parquet('{args.parquet_glob}') "
        f"WHERE education_level IS NOT NULL AND education_level != '' "
        f"GROUP BY education_level ORDER BY n DESC"
    ).fetchall()
    edu_lines = "\n".join(f"    {_quote(v)},  # n={n:,}" for v, n in edu_rows)
    (out / "education_level_values.py").write_text(
        f"# Unique education_level values from {args.label}, frequency-sorted descending.\n"
        f"# Use as the LLM-facing enum so scope_definer's picks always match a parquet row.\n\n"
        f"EDUCATION_LEVELS = [\n{edu_lines}\n]\n",
        encoding="utf-8",
    )
    print(f"  wrote education_level_values.py  ({len(edu_rows)} distinct values)")

    # 3. occupation values (frequency-sorted, no cap — caller can slice if needed)
    occ_rows = duckdb.sql(
        f"SELECT occupation, COUNT(*) AS n "
        f"FROM read_parquet('{args.parquet_glob}') "
        f"WHERE occupation IS NOT NULL AND occupation != '' "
        f"GROUP BY occupation ORDER BY n DESC"
    ).fetchall()
    occ_lines = "\n".join(f"    {_quote(v)},  # n={n:,}" for v, n in occ_rows)
    (out / "occupation_values.py").write_text(
        f"# Unique occupation values from {args.label}, sorted by frequency (descending).\n\n"
        f"OCCUPATIONS = [\n{occ_lines}\n]\n",
        encoding="utf-8",
    )
    print(f"  wrote occupation_values.py       ({len(occ_rows)} distinct values)")

    print(f"\nDone. Files placed in {out}")


if __name__ == "__main__":
    main()
