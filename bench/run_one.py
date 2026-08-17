#!/usr/bin/env python3
"""Run one icebug-format conversion in a fresh process; report time + peak RSS.

Used by benchmark_ldbc.py as the subprocess target so each backend gets a clean
interpreter and its peak RSS is measured from process start (python startup +
imports included, which is a small constant for all backends).
"""

import argparse
import json
import resource
import time

from icebug_format.convert_parquet import convert_parquet_dir_to_csr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--output-db", required=True)
    ap.add_argument(
        "--backend", required=True, choices=["pyarrow", "duckdb", "datafusion"]
    )
    ap.add_argument("--memory-limit", default=None)
    args = ap.parse_args()

    start = time.perf_counter()
    results = convert_parquet_dir_to_csr(
        source_dir=args.source_dir,
        output_db=args.output_db,
        backend=args.backend,
        memory_limit=args.memory_limit,
    )
    elapsed = time.perf_counter() - start
    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "backend": args.backend,
                "elapsed_s": round(elapsed, 3),
                "max_rss_mib": round(rss_kib / 1024, 1),
                "graphs": results,
            }
        )
    )


if __name__ == "__main__":
    main()
