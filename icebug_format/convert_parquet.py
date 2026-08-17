"""Convert vertex/edge Parquet pairs (LDBC-style) to icebug-disk CSR format.

Benchmark graph datasets (e.g. the LDBC ones under ``ldbc/``) ship as a vertex
table (``{name}-v.parquet``, first column is the node primary key) and an edge
table (``{name}-e.parquet`` with ``source``/``target`` columns), rather than as
a DuckDB database of ``nodes_*`` / ``edges_*`` tables.  This module adapts the
icebug-format converter to that layout, producing the same icebug-disk output
as :mod:`icebug_format.cli` for DuckDB sources:

- ``nodes_<name>.parquet``   vertex table sorted by primary key
- ``indices_<name>.parquet`` dense target id per edge, sorted by (source, target)
- ``indptr_<name>.parquet``  CSR row pointers (``N + 1`` entries)
- ``schema.cypher``          LadybugDB mount schema

Three interchangeable backends are provided:

================  ========================================================
backend           characteristics
================  ========================================================
``pyarrow``       pure in-memory PyArrow pipeline; no extra dependencies.
                  Radix sort + dense-id lookup tables => fastest, highest RSS.
``duckdb``        DuckDB SQL engine; external sort bounded by ``memory_limit``
                  => lower RSS, moderate runtime.  Requires ``convert`` extra.
``datafusion``    Apache DataFusion SQL engine; streaming execution with
                  ``execute_stream()`` => lowest RSS for large graphs.
                  Requires ``convert-datafusion`` extra.
================  ========================================================
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ICEBUG_DISK_VERSION = "v1"

_SOURCE_ALIASES = ("source", "src", "from")
_TARGET_ALIASES = ("target", "destination", "dest", "to")

BACKENDS = ("auto", "pyarrow", "duckdb", "datafusion")


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------


def sanitize_graph_name(name: str) -> str:
    """Turn a file stem into a SQL/Cypher-safe graph identifier."""
    sanitized = re.sub(r"\W+", "_", name).strip("_")
    if not sanitized:
        sanitized = "graph"
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def resolve_rel_column_names(names: list[str]) -> tuple[str, str]:
    """
    Return the (source_col, target_col) names from a relationship table.

    Checks known aliases in priority order, then falls back to the 0th and 1st
    columns respectively (matching :mod:`icebug_format.memory`).
    """
    name_set = set(names)
    src_col = next((a for a in _SOURCE_ALIASES if a in name_set), names[0])
    dst_col = next((a for a in _TARGET_ALIASES if a in name_set), names[1])
    return src_col, dst_col


def discover_graphs(source_dir: str | Path) -> list[dict]:
    """
    Discover vertex/edge Parquet pairs under *source_dir*.

    Supported layouts:

    - LDBC-style pairs: ``<name>-v.parquet`` + ``<name>-e.parquet``
    - generic pair:     ``vertex.parquet`` + ``edge.parquet``

    Returns a list of dicts::

        {"name": <sanitized graph name>, "vertex": Path, "edge": Path}
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise ValueError(f"Source directory does not exist: {source_dir}")

    graphs: list[dict] = []
    for v_file in sorted(source_dir.glob("*-v.parquet")):
        e_file = source_dir / f"{v_file.stem[:-2]}-e.parquet"
        if e_file.exists():
            graphs.append(
                {
                    "name": sanitize_graph_name(v_file.stem[:-2]),
                    "vertex": v_file,
                    "edge": e_file,
                }
            )

    v_file = source_dir / "vertex.parquet"
    e_file = source_dir / "edge.parquet"
    if v_file.exists() and e_file.exists():
        graphs.append(
            {
                "name": sanitize_graph_name(source_dir.name),
                "vertex": v_file,
                "edge": e_file,
            }
        )

    # De-duplicate by (vertex, edge) paths.
    seen = set()
    unique: list[dict] = []
    for g in graphs:
        key = (g["vertex"], g["edge"])
        if key not in seen:
            seen.add(key)
            unique.append(g)
    return unique


# ---------------------------------------------------------------------------
# Output helpers shared by all backends
# ---------------------------------------------------------------------------


def _write_icebug_parquet(
    table: pa.Table, path: Path, compression: str = "zstd"
) -> None:
    """Write an Arrow table as Parquet with icebug_disk_version metadata."""
    table = table.replace_schema_metadata({"icebug_disk_version": ICEBUG_DISK_VERSION})
    pq.write_table(table, path, compression=compression)


def _arrow_type_to_cypher(t: pa.DataType) -> str:
    """Map an Arrow type to the Cypher/Ladybug type used in schema.cypher."""
    if pa.types.is_int8(t):
        return "INT8"
    if pa.types.is_int16(t):
        return "INT16"
    if pa.types.is_int32(t):
        return "INT32"
    if pa.types.is_int64(t):
        return "INT64"
    if pa.types.is_uint8(t):
        return "UINT8"
    if pa.types.is_uint16(t):
        return "UINT16"
    if pa.types.is_uint32(t):
        return "UINT32"
    if pa.types.is_uint64(t):
        return "UINT64"
    if pa.types.is_float32(t):
        return "FLOAT"
    if pa.types.is_float64(t):
        return "DOUBLE"
    if pa.types.is_boolean(t):
        return "BOOL"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "STRING"
    if pa.types.is_timestamp(t):
        return "TIMESTAMP"
    if pa.types.is_date(t):
        return "DATE"
    if pa.types.is_time(t):
        return "TIME"
    if (
        pa.types.is_binary(t)
        or pa.types.is_large_binary(t)
        or pa.types.is_fixed_size_binary(t)
    ):
        return "BLOB"
    return "STRING"


def _generate_schema_cypher(graph_name: str, output_dir: Path, storage: str) -> str:
    """
    Build schema.cypher for one graph from the parquet files just written.

    The graph is homogeneous: both edge endpoints use the single vertex table.
    """
    node_pq = output_dir / f"nodes_{graph_name}.parquet"
    indices_pq = output_dir / f"indices_{graph_name}.parquet"
    node_schema = pq.ParquetFile(node_pq).schema_arrow
    indices_schema = pq.ParquetFile(indices_pq).schema_arrow

    node_cols = [f"{f.name} {_arrow_type_to_cypher(f.type)}" for f in node_schema]
    pk = node_schema.names[0]
    node_stmt = (
        f"CREATE NODE TABLE {graph_name}({', '.join(node_cols)}, PRIMARY KEY({pk})) "
        f"WITH (storage = '{storage}', format = 'icebug-disk');"
    )
    lines = [node_stmt]

    props = [
        f"{f.name} {_arrow_type_to_cypher(f.type)}"
        for f in indices_schema
        if f.name != "target"
    ]
    props_str = (", " + ", ".join(props)) if props else ""
    lines.append(
        f"CREATE REL TABLE {graph_name}(FROM {graph_name} TO {graph_name}{props_str}) "
        f"WITH (storage = '{storage}', format = 'icebug-disk');"
    )
    return "\n".join(lines) + "\n"


def _build_indptr(csr_source: pa.Array, n_nodes: int) -> pa.Array:
    """
    Build the ``N + 1`` uint64 CSR row-pointer array from dense source ids.

    ``csr_source`` values must be dense (0-based) and in ``[0, n_nodes)``.
    """
    if n_nodes == 0:
        return pa.array([0], type=pa.uint64())
    counts = pc.value_counts(csr_source)  # struct<values, counts>
    deg = pc.fill_null(
        pc.scatter(
            counts.field("counts"), counts.field("values"), max_index=n_nodes - 1
        ),
        0,
    )
    ptr = pc.cumulative_sum(deg)
    return pa.concat_arrays([pa.array([0], type=pa.uint64()), ptr.cast(pa.uint64())])


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------


def _select_backend(backend: str) -> Callable:
    """Return the converter callable for the requested backend."""
    if backend == "pyarrow":
        from icebug_format import _convert_pyarrow

        return _convert_pyarrow.convert_graph
    if backend == "duckdb":
        from icebug_format import _convert_duckdb

        return _convert_duckdb.convert_graph
    if backend == "datafusion":
        from icebug_format import _convert_datafusion

        return _convert_datafusion.convert_graph
    if backend == "auto":
        try:
            from icebug_format import _convert_duckdb

            return _select_backend("duckdb")
        except ImportError:
            pass
        try:
            from icebug_format import _convert_datafusion

            return _select_backend("datafusion")
        except ImportError:
            pass
        return _select_backend("pyarrow")
    raise ValueError(
        f"Unknown backend {backend!r}; expected one of {', '.join(BACKENDS)}"
    )


def convert_parquet_dir_to_csr(
    source_dir: str | Path,
    output_db: str | Path | None = None,
    graph_name: str | None = None,
    backend: str = "auto",
    add_reverse_edges: bool = False,
    memory_limit: str | None = None,
    storage: str | None = None,
) -> list[dict]:
    """
    Convert vertex/edge Parquet pairs in *source_dir* to icebug-disk output.

    Args:
        source_dir: Directory containing ``<name>-v.parquet``/``<name>-e.parquet``
            pairs (and/or ``vertex.parquet``/``edge.parquet``).
        output_db: Output database path used only as a naming convention; the
            Parquet files are written to a sibling directory named after its
            stem.  Defaults to ``<parent>/<source_dir.name>_csr.duckdb``.
        graph_name: Optional filter; when omitted, every discovered graph is
            converted.
        backend: One of ``auto``, ``pyarrow``, ``duckdb``, ``datafusion``.
        add_reverse_edges: Emit symmetric adjacency by adding reverse edges.
        memory_limit: DuckDB/DataFusion memory limit (size string or GB number).
        storage: Storage path recorded in ``schema.cypher``.

    Returns:
        List of dicts with ``name``, ``output_dir`` and ``elapsed`` (seconds)
        for each converted graph.
    """
    graphs = discover_graphs(source_dir)
    if not graphs:
        raise ValueError(
            f"No vertex/edge Parquet pairs found in {source_dir}. "
            "Expected '<name>-v.parquet' + '<name>-e.parquet' or "
            "'vertex.parquet' + 'edge.parquet'."
        )

    if graph_name:
        wanted = sanitize_graph_name(graph_name)
        graphs = [g for g in graphs if g["name"] == wanted]
        if not graphs:
            raise ValueError(
                f"No graph named {graph_name!r} in {source_dir}; found "
                f"{', '.join(g['name'] for g in graphs)}"
            )

    output_db = (
        Path(output_db)
        if output_db
        else Path(source_dir).parent / f"{Path(source_dir).name}_csr.duckdb"
    )
    output_base = Path(output_db).parent / Path(output_db).stem
    if len(graphs) == 1:
        out_dirs = [output_base]
    else:
        out_dirs = [output_base / g["name"] for g in graphs]

    storage_path = storage if storage is not None else f"./{Path(output_db).stem}"

    convert = _select_backend(backend)
    results: list[dict] = []
    for g, out_dir in zip(graphs, out_dirs):
        out_dir.mkdir(parents=True, exist_ok=True)
        ctx = {**g, "output_dir": out_dir, "storage": storage_path}
        start = time.perf_counter()
        convert(ctx, add_reverse_edges=add_reverse_edges, memory_limit=memory_limit)
        elapsed = time.perf_counter() - start
        (out_dir / "schema.cypher").write_text(
            _generate_schema_cypher(g["name"], out_dir, storage_path)
        )
        results.append(
            {"name": g["name"], "output_dir": str(out_dir), "elapsed": elapsed}
        )
    return results
