#!/usr/bin/env python3
"""Run one icebug-format conversion in a fresh process; report time + peak RSS.

Used by benchmark_ldbc.py as the subprocess target so each backend gets a clean
interpreter and its peak RSS is measured from process start (python startup +
imports included, which is a small constant for all backends).

Note: ``ru_maxrss`` is reported in bytes on macOS but in KiB on Linux/BSD,
so it is normalized to MiB accordingly.
"""

import argparse
import json
import resource
import sys
import time

from icebug_format.convert_parquet import convert_parquet_dir_to_csr


def _max_rss_mib() -> float:
    """Return peak RSS in MiB, normalizing per-platform ``ru_maxrss`` units."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)  # bytes on macOS
    return rss / 1024  # KiB on Linux/BSD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--backend", required=True, choices=["pyarrow", "duckdb", "datafusion"]
    )
    ap.add_argument("--memory-limit", default=None)
    args = ap.parse_args()

    start = time.perf_counter()
    results = convert_parquet_dir_to_csr(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        backend=args.backend,
        memory_limit=args.memory_limit,
    )
    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "backend": args.backend,
                "elapsed_s": round(elapsed, 3),
                "max_rss_mib": round(_max_rss_mib(), 1),
                "graphs": results,
            }
        )
    )


if __name__ == "__main__":
    main()
