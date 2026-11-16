"""Print unique values + counts for occupation and education_level in Nemotron-Personas-Korea.

Backend priority:
  1. DuckDB over local parquet (fastest, ~0.5s for 1M rows)
  2. pyarrow row iteration over local parquet (slower)
  3. HuggingFace streaming (slowest; requires network)

Usage:
    conda run -n mirofish-policy python3 scripts/nemotron_unique_values.py
    conda run -n mirofish-policy python3 scripts/nemotron_unique_values.py \
        --columns occupation,education_level,bachelors_field --top 50
    conda run -n mirofish-policy python3 scripts/nemotron_unique_values.py --limit 50000  # caps streaming
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

DATASET_NAME = "nvidia/Nemotron-Personas-Korea"
LOCAL_DATA_DIR = Path("dataset/processed/nemotron/Nemotron-Personas-Korea/data")
CACHE_DIR = Path("dataset/processed/nemotron")


def parquet_glob() -> str | None:
    if not LOCAL_DATA_DIR.exists():
        return None
    if not any(LOCAL_DATA_DIR.glob("*.parquet")):
        return None
    return str(LOCAL_DATA_DIR / "*.parquet")


def scan_with_duckdb(glob_pattern: str, columns: list[str], top: int) -> None:
    import duckdb
    print(f"[duckdb] scanning {glob_pattern}")
    total = duckdb.sql(f"SELECT COUNT(*) FROM read_parquet('{glob_pattern}')").fetchone()[0]
    print(f"\nScanned {total:,} rows.\n")
    for col in columns:
        result = duckdb.sql(f"""
            SELECT {col} AS value, COUNT(*) AS n
            FROM read_parquet('{glob_pattern}')
            WHERE {col} IS NOT NULL AND {col} != ''
            GROUP BY 1
            ORDER BY n DESC
        """).fetchall()
        print(f"=== {col} === ({len(result):,} unique non-empty values)")
        for value, n in result[:top]:
            print(f"  {n:>7,}  {value}")
        if len(result) > top:
            print(f"  ... and {len(result) - top:,} more")
        print()


def scan_with_pyarrow_or_stream(columns: list[str], limit: int | None, top: int) -> None:
    counters = {col: Counter() for col in columns}

    glob_pattern = parquet_glob()
    if glob_pattern:
        import pyarrow.parquet as pq
        print(f"[pyarrow] reading local parquet from {LOCAL_DATA_DIR}")
        files = sorted(LOCAL_DATA_DIR.glob("*.parquet"))
        total = 0
        for path in files:
            for batch in pq.ParquetFile(str(path)).iter_batches(batch_size=8192, columns=columns):
                for row in batch.to_pylist():
                    total += 1
                    for col in columns:
                        v = row.get(col)
                        if v not in (None, ""):
                            counters[col][str(v)] += 1
                    if total % 100000 == 0:
                        print(f"  scanned {total:,} rows...")
    else:
        from datasets import load_dataset
        print(f"[hf-stream] no local parquet; streaming {DATASET_NAME}")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(DATASET_NAME, split="train", streaming=True, cache_dir=str(CACHE_DIR))
        total = 0
        for row in ds:
            if limit is not None and total >= limit:
                break
            total += 1
            for col in columns:
                v = row.get(col)
                if v not in (None, ""):
                    counters[col][str(v)] += 1
            if total % 50000 == 0:
                print(f"  scanned {total:,} rows...")

    print(f"\nScanned {total:,} rows.\n")
    for col, counter in counters.items():
        print(f"=== {col} === ({len(counter):,} unique non-empty values)")
        for value, n in counter.most_common(top):
            print(f"  {n:>7,}  {value}")
        if len(counter) > top:
            print(f"  ... and {len(counter) - top:,} more")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--columns", default="occupation,education_level",
                        help="Comma-separated column names to summarize.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap rows scanned (only honored when streaming from HF).")
    parser.add_argument("--top", type=int, default=30,
                        help="Show only the top N most frequent values per column.")
    parser.add_argument("--no-duckdb", action="store_true",
                        help="Force the pyarrow / streaming path even if DuckDB is available.")
    args = parser.parse_args()

    columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    glob_pattern = parquet_glob()

    if glob_pattern and not args.no_duckdb:
        try:
            scan_with_duckdb(glob_pattern, columns, args.top)
            return
        except ImportError:
            print("[duckdb] not installed; falling back to pyarrow")

    scan_with_pyarrow_or_stream(columns, args.limit, args.top)


if __name__ == "__main__":
    main()
