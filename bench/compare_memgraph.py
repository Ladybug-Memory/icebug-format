#!/usr/bin/env python3
"""Compare the pyarrow Parquet-pair backend vs IcebugMemGraph.from_arrow_tables().

Both implementations produce the same CSR structure; the differences are the
engine (pure PyArrow vs DuckDB SQL) and the I/O scope (the pyarrow backend
streams from Parquet and writes icebug-disk files; from_arrow_tables requires
both input Arrow tables to be resident and keeps the graph in memory).

Each path runs in a fresh subprocess for clean RSS measurement.

Usage:
    uv run bench/compare_memgraph.py --source-dir ldbc/cit-Patents \
        --out-dir /tmp/bench --repeats 3 [--memory-limit 32GB]
"""

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], backend: str) -> dict:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
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
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--out-dir", default="/tmp/icebug-bench")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--memory-limit", default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[dict]] = {}

    # pyarrow Parquet-pair backend (streams from parquet, writes icebug-disk)
    for rep in range(1, args.repeats + 1):
        run_dir = Path(tempfile.mkdtemp(dir=str(out_dir), prefix=f"pyarrow_{rep}_"))
        r = _run(
            [
                sys.executable,
                str(REPO_ROOT / "bench" / "run_one.py"),
                "--source-dir",
                args.source_dir,
                "--output-db",
                str(run_dir / "out.duckdb"),
                "--backend",
                "pyarrow",
                *(["--memory-limit", args.memory_limit] if args.memory_limit else []),
            ],
            "pyarrow",
        )
        results.setdefault("pyarrow backend (full pipeline)", []).append(r)
        print(
            f"  pyarrow   rep {rep}: {r['elapsed_s']:7.2f}s  {r['max_rss_mib']:8.1f} MiB",
            file=sys.stderr,
        )
        if not args.keep:
            shutil.rmtree(run_dir, ignore_errors=True)

    # IcebugMemGraph.from_arrow_tables (tables already in memory)
    for rep in range(1, args.repeats + 1):
        r = _run(
            [
                sys.executable,
                str(REPO_ROOT / "bench" / "run_one_memgraph.py"),
                "--source-dir",
                args.source_dir,
            ],
            "from_arrow_tables",
        )
        results.setdefault("IcebugMemGraph.from_arrow_tables", []).append(r)
        print(
            f"  memgraph rep {rep}: {r['total_s']:7.2f}s "
            f"(read {r['read_s']:.2f}s + conv {r['convert_s']:.2f}s)  "
            f"{r['max_rss_mib']:8.1f} MiB",
            file=sys.stderr,
        )

    def fmt(vals: list[float]) -> str:
        return f"{statistics.median(vals):7.2f} [{min(vals):.2f}, {max(vals):.2f}]"

    print(f"\nDataset: {args.source_dir}")
    print(
        "Implementation                | Runtime (s, median [min,max]) | Peak RSS (MiB, median [min,max])"
    )
    print(
        "------------------------------|------------------------------|---------------------------------"
    )
    for label, runs in results.items():
        rss = [r["max_rss_mib"] for r in runs]
        if label.startswith("pyarrow"):
            times = [r["elapsed_s"] for r in runs]
        else:
            times = [r["total_s"] for r in runs]
        print(f"{label:29s} | {fmt(times):28s} | {fmt(rss)}")


if __name__ == "__main__":
    main()
