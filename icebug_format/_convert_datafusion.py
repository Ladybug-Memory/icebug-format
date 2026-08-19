"""DataFusion backend for vertex/edge Parquet -> icebug-disk CSR conversion.

Runs the same relational plan as the DuckDB backend on Apache DataFusion
(``convert-datafusion`` extra).  Unlike the DuckDB backend, results are pulled
out of the engine with ``execute_stream()`` and streamed into Parquet with
``pyarrow.parquet.ParquetWriter``, so only a bounded working set is ever
resident.

When ``memory_limit`` is given it is enforced as a spill-aware memory pool on
the session's ``RuntimeEnv`` (the Python binding for DataFusion's runtime
memory limit) plus the ``datafusion.execution.sort_*`` session options that
tune SortExec's spilling.  A *fair* spill pool is used (not greedy): greedy
lets the sort buffer until the pool is completely full, leaving no headroom
for ``SortPreservingMerge`` to allocate its in-memory merge buffers, so tight
limits fail with ``ResourcesExhausted``.  The fair pool caps each spillable
operator's share and guarantees the merge headroom, spilling sort runs and
window output to the OS temp dir instead of growing resident memory.

The same SQL plan is available from in-memory Arrow tables via
:func:`build_csr_from_arrow_tables`, which backs
:meth:`icebug_format.memory.IcebugMemGraph.from_arrow_tables` when a memory
limit is requested.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

from icebug_format.convert_parquet import (
    ICEBUG_DISK_VERSION,
    _write_icebug_parquet,
    csr_rel_name,
    parse_size_to_bytes,
    resolve_rel_column_names,
)

_COMPRESSION = "zstd"

# Row groups are flushed on every ``ParquetWriter.write_table``/``write_batch``
# call (the default row group size is ``min(rows, 1 MiB)`` *per call*, not
# accumulated across calls), so DataFusion's 8192-row stream batches would each
# become their own tiny row group, fragmenting zstd's compression window and
# row-group metadata.  Buffer batches to this many rows and write each chunk as
# a single row group.  122880 matches the row-group size of the LDBC input
# parquet files (DuckDB's writer).
_ROWS_PER_ROW_GROUP = 122880

# Arrow tables are registered with the engine in chunks of at most this many
# rows.  A single-chunk table would otherwise arrive as one huge record batch
# and force operators (e.g. SortExec feeding the ROW_NUMBER window) to reserve
# the whole table's worth of memory at once, which fails under a tight memory
# pool.  Small batches keep per-operator reservations bounded so the pool can
# spill instead.
_REGISTER_CHUNK_ROWS = 65_536

# Keep SortExec's spill-reservation headroom small: a large value (e.g. a
# fraction of the pool) makes the final in-memory merge try to reserve far more
# than the pool can hand out and fails with ResourcesExhausted.  A modest ~64
# MiB cap guarantees the merge buffers while leaving the rest of the pool to
# the sort data.
_SPILL_RESERVATION_FLOOR = 1 << 20  # 1 MiB
_SPILL_RESERVATION_CAP = 1 << 26  # 64 MiB
_IN_PLACE_FLOOR = 1 << 20  # 1 MiB
_IN_PLACE_CAP = 1 << 26  # 64 MiB


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _datafusion_temp_size(gb: float) -> str:
    """Format a GB number as a DataFusion ``max_temp_directory_size`` value.

    The ``SET datafusion.runtime.max_temp_directory_size`` path only accepts
    integer sizes with ``K``/``M``/``G`` units, so the byte count is expressed
    in the largest integral binary unit (e.g. 10 -> ``'10G'``, 1.5 ->
    ``'1536M'``).
    """
    size = max(1, int(round(gb * (1 << 30))))
    if size % (1 << 30) == 0:
        return f"{size >> 30}G"
    if size % (1 << 20) == 0:
        return f"{size >> 20}M"
    return f"{size >> 10}K"


def _build_context(
    datafusion, memory_limit: str | None, max_tmp_dir_gb: float | None = None
):
    """Return a ``SessionContext`` that honors *memory_limit* as a spill pool
    and *max_tmp_dir_gb* as the temp directory size cap.

    Without a limit (or with an unparseable/non-positive one) a default
    ``SessionContext`` is returned.  With a limit, the session's runtime uses
    a fair spill pool of that size (the Python equivalent of DataFusion's
    ``datafusion.runtime.memory_limit``; greedy cannot be used because it does
    not leave the headroom ``SortPreservingMerge`` needs to allocate its
    in-memory merge buffers) and SortExec is tuned to spill through
    ``datafusion.execution.sort_spill_reservation_bytes`` and
    ``datafusion.execution.sort_in_place_threshold_bytes``.  With a temp dir
    size, ``datafusion.runtime.max_temp_directory_size`` caps how much spill
    data may accumulate on disk (the DataFusion analogue of DuckDB's
    ``max_temp_directory_size`` PRAGMA).
    """
    if not memory_limit and not max_tmp_dir_gb:
        return datafusion.SessionContext()

    config = None
    runtime = None
    limit_bytes = parse_size_to_bytes(memory_limit) if memory_limit else None
    if limit_bytes:
        spill_reservation = min(
            max(limit_bytes // 8, _SPILL_RESERVATION_FLOOR), _SPILL_RESERVATION_CAP
        )
        in_place = min(max(limit_bytes // 64, _IN_PLACE_FLOOR), _IN_PLACE_CAP)
        config = datafusion.SessionConfig(
            {
                "datafusion.execution.sort_spill_reservation_bytes": str(
                    spill_reservation
                ),
                "datafusion.execution.sort_in_place_threshold_bytes": str(in_place),
            }
        )
        runtime = datafusion.RuntimeEnvBuilder().with_fair_spill_pool(limit_bytes)

    ctx = datafusion.SessionContext(config=config, runtime=runtime)
    if max_tmp_dir_gb:
        ctx.sql(
            f"SET datafusion.runtime.max_temp_directory_size = "
            f"'{_datafusion_temp_size(max_tmp_dir_gb)}'"
        )
    return ctx


def _stream_to_parquet(
    df,
    path: Path,
    target_to_uint64: bool = False,
    plain_columns: list[str] | None = None,
    total_rows: int | None = None,
) -> None:
    """
    Stream a DataFusion DataFrame into a Parquet file with icebug metadata.

    ``execute_stream()`` keeps memory bounded to the engine's working set
    rather than materialising the whole result, which is the point of the
    DataFusion backend.  Falls back to ``to_arrow_table()`` only for empty
    results (where there is nothing to stream).

    ``plain_columns`` names columns whose values are effectively unique or
    random per row group (CSR ``target`` neighbour ids, node primary keys).
    pyarrow's default dictionary encoding writes a dictionary page per row
    group that can never pay off for such columns (a dictionary of N distinct
    values costs roughly 2x the plain data), inflating the file; those columns
    are written with PLAIN encoding while any other (low-cardinality property)
    columns keep dictionary encoding.

    ``total_rows`` enables a tqdm progress bar over the streamed rows (used
    when the caller knows the result size up front, e.g. from parquet
    metadata); when ``None`` no bar is shown.
    """
    writer = None
    plain = set(plain_columns or ())
    buf: list[pa.RecordBatch] = []
    buf_rows = 0
    bar = (
        tqdm(total=total_rows, unit="rows", desc=f"Writing {path.name}")
        if total_rows is not None
        else None
    )

    def _flush() -> None:
        """Write the buffered batches as one row group."""
        nonlocal buf, buf_rows
        if not buf:
            return
        # row_group_size=None -> min(rows, 1 MiB) per call, i.e. exactly one
        # row group of _ROWS_PER_ROW_GROUP rows (the tail chunk may be smaller).
        writer.write_table(pa.Table.from_batches(buf))
        buf = []
        buf_rows = 0

    for batch in df.execute_stream():
        rb = batch.to_pyarrow()
        if writer is None:
            schema = rb.schema
            if target_to_uint64 and "target" in schema.names:
                i = schema.get_field_index("target")
                schema = schema.set(i, pa.field(schema.field(i).name, pa.uint64()))
            schema = schema.with_metadata({"icebug_disk_version": ICEBUG_DISK_VERSION})
            # Enable dictionary encoding only where it can pay off; a listed
            # use_dictionary of column names disables it for all others.
            dict_cols = [c for c in schema.names if c not in plain]
            writer = pq.ParquetWriter(
                path, schema, compression=_COMPRESSION, use_dictionary=dict_cols
            )
        if target_to_uint64 and "target" in rb.schema.names:
            i = rb.schema.get_field_index("target")
            rb = rb.set_column(
                i, rb.schema.field(i).name, rb.column(i).cast(pa.uint64())
            )
        buf.append(rb)
        buf_rows += rb.num_rows
        if bar is not None:
            bar.update(rb.num_rows)
        if buf_rows >= _ROWS_PER_ROW_GROUP:
            _flush()
    if bar is not None:
        bar.close()
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
    _flush()
    writer.close()


def _indptr_from_degrees(deg: pa.Table, n_nodes: int) -> pa.Table:
    """Build the ``N + 1`` CSR row-pointer table from a ``(src, deg)`` table."""
    if n_nodes == 0:
        return pa.table({"ptr": pa.array([0], type=pa.uint64())})
    src = deg["src"].cast(pa.int32()).combine_chunks()
    counts = deg["deg"].combine_chunks()
    return pa.table(
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


def _register_table(ctx, name: str, table: pa.Table) -> None:
    """Register an Arrow table with the DataFusion session in bounded chunks."""
    batches = table.to_batches(max_chunksize=_REGISTER_CHUNK_ROWS)
    if batches:
        ctx.register_record_batches(name, [batches])
    else:  # empty table: no rows to chunk, register directly
        ctx.register_table(name, ctx.from_arrow(table))


def build_csr_from_arrow_tables(
    node_table: pa.Table,
    rel_table: pa.Table,
    *,
    to_node_table: pa.Table | None = None,
    add_reverse_edges: bool = False,
    memory_limit: str | None = None,
    max_tmp_dir_gb: float | None = None,
    input_sorted: bool = False,
) -> tuple[pa.Table, pa.Table, pa.Table, pa.Table]:
    """
    Build ``(src_out, dst_out, indices, indptr)`` with the DataFusion engine.

    Same contract as :func:`icebug_format._convert_pyarrow.build_csr_from_tables`
    (the pure-PyArrow path), but the node sort, endpoint mapping and
    ``(source, target)`` sort run inside DataFusion so *memory_limit* bounds
    the engine's memory pool and spillable operators write to the OS temp dir
    instead of growing RSS, and *max_tmp_dir_gb* caps the temp directory size.
    Mirrors :func:`convert_graph` for Parquet pairs, with the vertex/edge
    tables registered from Arrow rather than Parquet files.  Unless
    *input_sorted* is set, node tables are ordered by primary key; with
    *input_sorted* the caller guarantees they are already ordered, so CSR ids
    are assigned by row position and the sort is skipped.
    """
    from icebug_format._datafusion import require_datafusion

    if add_reverse_edges and to_node_table is not None:
        raise ValueError(
            "to_node_arrow_table must not be provided when adding reverse edges; "
            "from and to node tables must be the same."
        )
    if rel_table.num_columns < 2:
        raise ValueError(
            f"rel_table must have at least 2 columns (source and target), "
            f"got {rel_table.num_columns}"
        )

    if to_node_table is None:
        to_node_table = node_table

    src_pk = node_table.schema.names[0]
    dst_pk = to_node_table.schema.names[0]
    src_col, dst_col = resolve_rel_column_names(rel_table.schema.names)
    prop_cols = [c for c in rel_table.schema.names if c not in (src_col, dst_col)]
    n_src = len(node_table)

    datafusion = require_datafusion("the 'convert-datafusion' feature")
    ctx = _build_context(datafusion, memory_limit, max_tmp_dir_gb)
    _register_table(ctx, "vertices", node_table)
    _register_table(ctx, "vertices_dst", to_node_table)
    _register_table(ctx, "edges", rel_table)

    def _map_view(table: str, pk: str) -> str:
        # input_sorted: CSR ids follow input row order, so skip the sort.
        row_order = "" if input_sorted else f"ORDER BY {_q(pk)}"
        return (
            f"SELECT {_q(pk)} AS original_node_id, "
            f"CAST(ROW_NUMBER() OVER ({row_order}) AS BIGINT) - 1 "
            f"AS csr_index FROM {table}"
        )

    ctx.register_view("src_map", ctx.sql(_map_view("vertices", src_pk)))
    ctx.register_view("dst_map", ctx.sql(_map_view("vertices_dst", dst_pk)))

    props_fwd = ", " + ", ".join(f"e.{_q(c)}" for c in prop_cols) if prop_cols else ""
    # Columns of the `relations` view (no table alias in scope there).
    props_plain = (", " + ", ".join(_q(c) for c in prop_cols)) if prop_cols else ""
    join_clause = f"""
        FROM edges e
        JOIN src_map m1 ON e.{_q(src_col)} = m1.original_node_id
        JOIN dst_map m2 ON e.{_q(dst_col)} = m2.original_node_id
    """
    if not add_reverse_edges:
        rel_sql = (
            f"SELECT m1.csr_index AS csr_source, m2.csr_index AS csr_target"
            f"{props_fwd} {join_clause}"
        )
    else:
        # Self-loops appear once (forward only); non-self edges get both directions.
        rel_sql = f"""
            SELECT m1.csr_index AS csr_source, m2.csr_index AS csr_target{props_fwd}
            {join_clause}
            UNION ALL
            SELECT m2.csr_index AS csr_source, m1.csr_index AS csr_target{props_fwd}
            {join_clause}
            WHERE e.{_q(src_col)} != e.{_q(dst_col)}
        """
    ctx.register_view("relations", ctx.sql(rel_sql))

    # indices: neighbour list sorted by (source, target).
    indices = ctx.sql(
        f"SELECT CAST(csr_target AS BIGINT) AS target{props_plain} "
        f"FROM relations ORDER BY csr_source, csr_target"
    ).to_arrow_table()
    i = indices.schema.get_field_index("target")
    indices = indices.set_column(
        i, indices.schema.field(i).name, indices.column(i).cast(pa.uint64())
    )

    # indptr: degrees from a GROUP BY, then histogram + prefix sum.
    deg = ctx.sql(
        "SELECT csr_source AS src, COUNT(*) AS deg FROM relations GROUP BY csr_source"
    ).to_arrow_table()
    indptr = _indptr_from_degrees(deg, n_src)

    # vertices: ordered by primary key (or input row order when input_sorted).
    src_out = ctx.sql(
        f"SELECT * FROM vertices{' ORDER BY ' + _q(src_pk) if not input_sorted else ''}"
    ).to_arrow_table()
    dst_out = ctx.sql(
        f"SELECT * FROM vertices_dst{' ORDER BY ' + _q(dst_pk) if not input_sorted else ''}"
    ).to_arrow_table()

    return src_out, dst_out, indices, indptr


def convert_graph(
    graph: dict,
    add_reverse_edges: bool = False,
    memory_limit: str | None = None,
    max_tmp_dir_gb: float | None = None,
    input_sorted: bool = False,
) -> None:
    """Convert one vertex/edge Parquet pair with the DataFusion SQL engine.

    ``memory_limit`` bounds the session's memory pool (size string, percent of
    RAM, or GB number); sorts and window operators spill to disk when the pool
    is exhausted instead of growing RSS.  ``max_tmp_dir_gb`` caps the temp
    directory size (``datafusion.runtime.max_temp_directory_size``).
    ``input_sorted`` skips the node-table sort: CSR ids are then assigned by
    row position, which is only valid when the vertex parquet is already
    ordered by primary key.
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
    ctx = _build_context(datafusion, memory_limit, max_tmp_dir_gb)
    ctx.register_parquet("vertices", vertex_path)
    ctx.register_parquet("edges", edge_path)

    # Dense id -> CSR index mapping, ordered by primary key (or input row
    # order when input_sorted).
    row_order = "" if input_sorted else f"ORDER BY {_q(pk)}"
    df_map = ctx.sql(f"""
        SELECT {_q(pk)} AS original_node_id,
               CAST(ROW_NUMBER() OVER ({row_order}) AS BIGINT) - 1 AS csr_index
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
        ctx.sql(idx_sql),
        out_dir / f"indices_{csr_rel_name(name)}.parquet",
        target_to_uint64=True,
        plain_columns=["target"],
        total_rows=ef.metadata.num_rows * (2 if add_reverse_edges else 1),
    )

    # indptr: degrees from a streaming GROUP BY, then histogram + prefix sum.
    deg = ctx.sql(
        "SELECT csr_source AS src, COUNT(*) AS deg FROM relations GROUP BY csr_source"
    ).to_arrow_table()
    indptr = _indptr_from_degrees(deg, n_nodes)

    # vertices: copy through (already sorted by primary key) and write outputs.
    _stream_to_parquet(
        ctx.sql(
            f"SELECT * FROM vertices{' ORDER BY ' + _q(pk) if not input_sorted else ''}"
        ),
        out_dir / f"nodes_{name}.parquet",
        plain_columns=[pk],
        total_rows=n_nodes,
    )
    _write_icebug_parquet(
        indptr,
        out_dir / f"indptr_{csr_rel_name(name)}.parquet",
        compression=_COMPRESSION,
    )
