#!/usr/bin/env python3
"""Run IcebugMemGraph.from_arrow_tables() in a fresh process; report metrics.

Used by compare_memgraph.py as the subprocess target.  Loads the vertex/edge
Parquet pair into Arrow tables (as any caller of the in-memory API must) and
then converts via ``from_arrow_tables``.  Reports total wall time (parquet
read + conversion) and conversion-only time, plus peak RSS.
"""

import argparse
import json
import resource
import time

import pyarrow.parquet as pq

from icebug_format.convert_parquet import discover_graphs
from icebug_format.memory import IcebugMemGraph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True)
    args = ap.parse_args()

    graphs = discover_graphs(args.source_dir)
    if not graphs:
        raise SystemExit(f"No vertex/edge Parquet pairs found in {args.source_dir}")
    vpath, epath = str(graphs[0]["vertex"]), str(graphs[0]["edge"])

    start = time.perf_counter()
    v = pq.read_table(vpath)
    e = pq.read_table(epath)
    read_elapsed = time.perf_counter() - start

    conv_start = time.perf_counter()
    g = IcebugMemGraph.from_arrow_tables(v, e)
    conv_elapsed = time.perf_counter() - conv_start

    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "backend": "from_arrow_tables",
                "read_s": round(read_elapsed, 3),
                "convert_s": round(conv_elapsed, 3),
                "total_s": round(read_elapsed + conv_elapsed, 3),
                "max_rss_mib": round(rss_kib / 1024, 1),
                "indices_len": len(g.indices),
                "indptr_len": len(g.indptr),
            }
        )
    )


if __name__ == "__main__":
    main()
