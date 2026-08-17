#!/usr/bin/env python3
"""Benchmark icebug-format's vertex/edge Parquet backends (RSS + runtime).

Runs each backend in a fresh subprocess (see run_one.py) for clean RSS
measurement, repeats a few times, and prints a Markdown table.

Usage:
    uv run bench/benchmark_ldbc.py --source-dir ldbc/cit-Patents \
        --out-dir /tmp/bench --repeat 3 [--memory-limit 32GB] [--no-cleanup]
"""

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

BACKENDS = ["pyarrow", "duckdb", "datafusion"]
REPO_ROOT = Path(__file__).resolve().parent.parent


def run_one(
    source_dir: Path, out_dir: Path, backend: str, memory_limit: str | None
) -> dict:
    """Run a single conversion; returns {elapsed_s, max_rss_mib} or raises."""
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "bench" / "run_one.py"),
            "--source-dir",
            str(source_dir),
            "--output-db",
            str(out_dir / f"{backend}.duckdb"),
            "--backend",
            backend,
            *(["--memory-limit", memory_limit] if memory_limit else []),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{backend} failed:\n{proc.stderr[-2000:]}")
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(
            f"{backend} produced unparsable output:\n{proc.stdout}"
        ) from exc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source-dir", required=True, help="LDBC dataset dir with *-v/-e pairs"
    )
    ap.add_argument("--out-dir", default="/tmp/icebug-bench", help="scratch/output dir")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--memory-limit", default=None, help="SQL engine memory limit (e.g. 32GB)"
    )
    ap.add_argument("--keep", action="store_true", help="keep the output parquet dirs")
    args = ap.parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for backend in BACKENDS:
        samples = []
        for rep in range(1, args.repeats + 1):
            run_dir = Path(
                tempfile.mkdtemp(dir=str(out_dir), prefix=f"{backend}_{rep}_")
            )
            result = run_one(source_dir, run_dir, backend, args.memory_limit)
            elapsed = result["elapsed_s"]
            rss = result["max_rss_mib"]
            print(
                f"  {backend:11s} rep {rep}: {elapsed:8.2f}s  {rss:9.1f} MiB",
                file=sys.stderr,
            )
            samples.append((elapsed, rss))
            if not args.keep:
                shutil.rmtree(run_dir, ignore_errors=True)
        rows.append((backend, samples))

    def fmt(values: list[float]) -> str:
        median = statistics.median(values)
        lo, hi = min(values), max(values)
        return f"{median:7.2f} [{lo:.2f}, {hi:.2f}]"

    print(f"\nDataset: {source_dir}")
    print("Backend     | Runtime (s, median [min,max])")
    print("            | Peak RSS (MiB, median [min,max])")
    print("------------|----------------------------------")
    for backend, samples in rows:
        etimes = [s[0] for s in samples]
        rsses = [s[1] for s in samples]
        print(f"{backend:11s} | {fmt(etimes)} | {fmt(rsses)}")


if __name__ == "__main__":
    main()
