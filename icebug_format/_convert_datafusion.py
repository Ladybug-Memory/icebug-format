"""DataFusion backend for vertex/edge Parquet -> icebug-disk CSR conversion.

Runs the same relational plan as the DuckDB backend on Apache DataFusion
(``convert-datafusion`` extra).  Unlike the DuckDB backend, results are pulled
out of the engine with ``execute_stream()`` and streamed into Parquet with
``pyarrow.parquet.ParquetWriter``, so only a bounded working set (the sort
window) is ever resident.  This gives the lowest peak RSS of the three
backends for large graphs.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from icebug_format.convert_parquet import (
    ICEBUG_DISK_VERSION,
    _write_icebug_parquet,
    resolve_rel_column_names,
)

_COMPRESSION = "zstd"


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _stream_to_parquet(df, path: Path, target_to_uint64: bool = False) -> None:
    """
    Stream a DataFusion DataFrame into a Parquet file with icebug metadata.

    ``execute_stream()`` keeps memory bounded to the engine's working set
    rather than materialising the whole result, which is the point of the
    DataFusion backend.  Falls back to ``to_arrow_table()`` only for empty
    results (where there is nothing to stream).
    """
    writer = None
    for batch in df.execute_stream():
        rb = batch.to_pyarrow()
        if writer is None:
            schema = rb.schema
            if target_to_uint64 and "target" in schema.names:
                i = schema.get_field_index("target")
                schema = schema.set(i, pa.field(schema.field(i).name, pa.uint64()))
            schema = schema.with_metadata({"icebug_disk_version": ICEBUG_DISK_VERSION})
            writer = pq.ParquetWriter(path, schema, compression=_COMPRESSION)
        if target_to_uint64 and "target" in rb.schema.names:
            i = rb.schema.get_field_index("target")
            rb = rb.set_column(
                i, rb.schema.field(i).name, rb.column(i).cast(pa.uint64())
            )
        writer.write_batch(rb)
    if writer is None:
        table = df.to_arrow_table()
        if target_to_uint64 and "target" in table.schema.names:
            i = table.schema.get_field_index("target")
            table = table.set_column(
                i,
                table.schema.field(i).name,
                table.column(i).cast(pa.uint64()),
            )
        _write_icebug_parquet(table, path, compression=_COMPRESSION)
        return
    writer.close()


def convert_graph(
    graph: dict,
    add_reverse_edges: bool = False,
    memory_limit: str | None = None,
) -> None:
    """Convert one vertex/edge Parquet pair with the DataFusion SQL engine.

    ``memory_limit`` is accepted for interface compatibility with the DuckDB
    backend; datafusion 54's Python bindings do not expose operator memory
    limits (``SessionConfig``), so it is not applied.
    """
    from icebug_format._datafusion import require_datafusion

    name = graph["name"]
    out_dir = Path(graph["output_dir"])
    vertex_path = str(graph["vertex"])
    edge_path = str(graph["edge"])

    vf = pq.ParquetFile(vertex_path)
    pk = vf.schema_arrow.names[0]
    n_nodes = vf.metadata.num_rows

    ef = pq.ParquetFile(edge_path)
    src_col, dst_col = resolve_rel_column_names(ef.schema_arrow.names)
    prop_cols = [c for c in ef.schema_arrow.names if c not in (src_col, dst_col)]

    datafusion = require_datafusion("the 'convert-datafusion' feature")
    ctx = datafusion.SessionContext()
    ctx.register_parquet("vertices", vertex_path)
    ctx.register_parquet("edges", edge_path)

    # Dense id -> CSR index mapping, ordered by primary key.
    df_map = ctx.sql(f"""
        SELECT {_q(pk)} AS original_node_id,
               CAST(ROW_NUMBER() OVER (ORDER BY {_q(pk)}) AS BIGINT) - 1 AS csr_index
        FROM vertices
        """)
    ctx.register_view("src_map", df_map)
    ctx.register_view("dst_map", df_map)

    props_fwd = (", " + ", ".join(f"e.{_q(c)}" for c in prop_cols)) if prop_cols else ""
    props_rev = props_fwd
    # Columns of the `relations` view (no table alias in scope there).
    props_plain = (", " + ", ".join(_q(c) for c in prop_cols)) if prop_cols else ""
    join_clause = f"""
        FROM edges e
        JOIN src_map m1 ON e.{_q(src_col)} = m1.original_node_id
        JOIN dst_map m2 ON e.{_q(dst_col)} = m2.original_node_id
    """
    if not add_reverse_edges:
        rel_sql = (
            f"SELECT m1.csr_index AS csr_source, m2.csr_index AS csr_target{props_fwd} "
            f"{join_clause}"
        )
    else:
        # Self-loops appear once (forward only); non-self edges get both directions.
        rel_sql = f"""
            SELECT m1.csr_index AS csr_source, m2.csr_index AS csr_target{props_fwd}
            {join_clause}
            UNION ALL
            SELECT m2.csr_index AS csr_source, m1.csr_index AS csr_target{props_rev}
            {join_clause}
            WHERE e.{_q(src_col)} != e.{_q(dst_col)}
        """
    ctx.register_view("relations", ctx.sql(rel_sql))

    # indices: neighbour list sorted by (source, target), streamed to Parquet.
    idx_sql = (
        f"SELECT CAST(csr_target AS BIGINT) AS target{props_plain} "
        f"FROM relations ORDER BY csr_source, csr_target"
    )
    _stream_to_parquet(
        ctx.sql(idx_sql), out_dir / f"indices_{name}.parquet", target_to_uint64=True
    )

    # indptr: degrees from a streaming GROUP BY, then histogram + prefix sum.
    deg_df = ctx.sql(
        "SELECT csr_source AS src, COUNT(*) AS deg FROM relations GROUP BY csr_source"
    )
    deg = deg_df.to_arrow_table()
    src = deg["src"].cast(pa.int32()).combine_chunks()
    counts = deg["deg"].combine_chunks()
    indptr = pa.table(
        {
            "ptr": pa.concat_arrays(
                [
                    pa.array([0], type=pa.uint64()),
                    pc.cumulative_sum(
                        pc.fill_null(pc.scatter(counts, src, max_index=n_nodes - 1), 0)
                    ).cast(pa.uint64()),
                ]
            )
        }
    )
    if n_nodes == 0:
        indptr = pa.table({"ptr": pa.array([0], type=pa.uint64())})

    # vertices: copy through (already sorted by primary key) and write outputs.
    _stream_to_parquet(
        ctx.sql(f"SELECT * FROM vertices ORDER BY {_q(pk)}"),
        out_dir / f"nodes_{name}.parquet",
    )
    _write_icebug_parquet(
        indptr, out_dir / f"indptr_{name}.parquet", compression=_COMPRESSION
    )
