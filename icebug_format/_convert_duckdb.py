"""DuckDB backend for vertex/edge Parquet -> icebug-disk CSR conversion.

Runs the same relational plan the original DuckDB-source converter uses, but
reads the vertex/edge tables straight from the Parquet pair into an in-memory
DuckDB instance.  DuckDB performs the endpoint joins and the ``(source, target)``
sort, with its external sort spilling to disk when ``memory_limit`` is
exceeded, so peak RSS stays bounded for large graphs.

Output Parquet files are streamed from DuckDB via ``COPY ... TO ...``.

The same plan is available from in-memory Arrow tables via
:func:`build_csr_from_arrow_tables`, which backs
:meth:`icebug_format.memory.IcebugMemGraph.from_arrow_tables` when the
``"duckdb"`` backend is selected.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from icebug_format.convert_parquet import (
    ICEBUG_DISK_VERSION,
    csr_rel_name,
    resolve_rel_column_names,
)


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def convert_graph(
    graph: dict,
    add_reverse_edges: bool = False,
    memory_limit: str | None = None,
    max_tmp_dir_gb: float | None = None,
    input_sorted: bool = False,
    threads: int | None = None,
) -> None:
    """Convert one vertex/edge Parquet pair with the DuckDB SQL engine."""
    from icebug_format._duckdb import require_duckdb
    from icebug_format.cli import (
        _write_parquet_with_icebug_metadata,
        set_max_temp_dir_size,
        set_memory_limit,
        set_threads,
        step_progress,
    )

    name = graph["name"]
    out_dir = Path(graph["output_dir"])
    vertex_path = str(graph["vertex"])
    edge_path = str(graph["edge"])

    pk = pq.ParquetFile(vertex_path).schema_arrow.names[0]
    ef = pq.ParquetFile(edge_path)
    edge_cols = ef.schema_arrow.names
    src_col, dst_col = resolve_rel_column_names(edge_cols)
    prop_cols = [c for c in edge_cols if c not in (src_col, dst_col)]

    duckdb = require_duckdb("the 'convert' feature (vertex/edge Parquet conversion)")
    con = duckdb.connect()
    if memory_limit:
        set_memory_limit(con, memory_limit)
    set_max_temp_dir_size(con, max_tmp_dir_gb)
    set_threads(con, threads)
    # Note: we deliberately do NOT set ``preserve_insertion_order=false``
    # here.  The indexed/CSR outputs (and, with ``input_sorted``, the
    # skip-the-sort path that relies on the parquet scan order) must keep row
    # order; disabling insertion-order preservation would let DuckDB reorder
    # table rows and corrupt the CSR output.
    try:
        # input_sorted: the caller guarantees the vertex parquet is already
        # ordered by primary key, so skip the sort; CSR ids = row position.
        order_by = "" if input_sorted else f" ORDER BY {_q(pk)}"
        con.execute(
            f"CREATE TABLE vertices AS SELECT * FROM read_parquet(?){order_by}",
            [vertex_path],
        )
        n_nodes = con.execute("SELECT COUNT(*) FROM vertices").fetchone()[0]

        # Dense id -> CSR index mapping, ordered by primary key (or input row
        # order when input_sorted).
        row_order = "" if input_sorted else f"ORDER BY {_q(pk)}"
        con.execute(f"""
            CREATE TABLE src_map AS
            SELECT row_number() OVER ({row_order}) - 1 AS csr_index,
                   {_q(pk)} AS original_node_id
            FROM vertices
            """)

        select_fwd = "m1.csr_index AS csr_source, m2.csr_index AS csr_target"
        select_rev = "m2.csr_index AS csr_source, m1.csr_index AS csr_target"
        if prop_cols:
            props = ", " + ", ".join(f"e.{_q(c)}" for c in prop_cols)
            select_fwd += props
            select_rev += props
        # The edge file is scanned straight from Parquet and never materialized
        # as a DuckDB table, and there is no materialized `relations` copy
        # either: the indices sort and the indptr GROUP BY each run their own
        # join over the streaming parquet scan.  Peak temp usage is therefore
        # just the external sort spill, not a full copy of the (potentially
        # multi-billion-row) edge set.
        join_clause = f"""
            FROM read_parquet(?) AS e
            JOIN src_map m1 ON e.{_q(src_col)} = m1.original_node_id
            JOIN src_map m2 ON e.{_q(dst_col)} = m2.original_node_id
        """
        # Inner relation: dense csr ids per edge (plus properties).  One
        # read_parquet(?) placeholder per branch, hence the parameter lists
        # below.
        if not add_reverse_edges:
            inner = f"SELECT {select_fwd} {join_clause}"
            inner_params = [edge_path]
        else:
            # Self-loops appear once (forward only); non-self edges get both directions.
            inner = f"""
                SELECT {select_fwd} {join_clause}
                UNION ALL
                SELECT {select_rev} {join_clause}
                WHERE e.{_q(src_col)} != e.{_q(dst_col)}
            """
            inner_params = [edge_path, edge_path]

        # indices: neighbour list sorted by (source, target), streamed directly
        # to Parquet via COPY.  The external sort spills to temp only when
        # memory_limit is exceeded, then the sorted stream is written out.
        # Sort keys are packed into one 8-byte UBIGINT ((source << 32) | target)
        # so each row through the external sort is just key + properties: for
        # multi-billion-row edge sets this halves the spill footprint versus
        # sorting two 8-byte keys, which otherwise blows past the temp cap.
        select_props = (
            ", " + ", ".join(_q(c) for c in prop_cols) if prop_cols else ""
        )
        if input_sorted and not add_reverse_edges:
            # With ``input_sorted`` the caller guarantees the vertex file is
            # ordered by primary key and the edge file by (source, target).
            # DuckDB's hash join preserves the left input's probe order, and
            # assigning CSR ids by vertex row position is a monotonic map, so
            # the join output is already ordered by (csr_source, csr_target).
            # Skip the external sort entirely -- it would only re-sort (and
            # spill) data that is already in the required order.
            idx_query = (
                f"SELECT m2.csr_index::UBIGINT AS target{select_props}"
                f" {join_clause}"
            )
            step = "Mapping edges (duckdb)"
        else:
            # Otherwise sort by (source, target), packing both into one 8-byte
            # UBIGINT ((source << 32) | target) so each row through the
            # external sort is key + projected properties rather than two
            # 8-byte keys -- roughly halves the spill footprint for
            # multi-billion-row edge sets.
            idx_query = (
                f"SELECT CAST(sort_key & 4294967295 AS UBIGINT) AS target"
                f"{select_props} "
                f"FROM (SELECT ((csr_source::UBIGINT << 32) | "
                f"csr_target::UBIGINT) AS sort_key{select_props} FROM ({inner})) "
                f"ORDER BY sort_key"
            )
            step = "Sorting edges (duckdb)"
        with step_progress(step):
            con.execute(
                f"""
                COPY ({idx_query}) TO ?
                (FORMAT PARQUET, COMPRESSION ZSTD,
                 KV_METADATA {{ icebug_disk_version: '{ICEBUG_DISK_VERSION}' }})
                """,
                [str(out_dir / f"indices_{csr_rel_name(name)}.parquet")]
                + inner_params,
            )

        # indptr: cumulative degree per source node, zero-filled, N+1 entries.
        with step_progress("Building indptr (duckdb)"):
            con.execute(
                f"""
                CREATE TABLE indptr AS
                WITH node_range AS (
                    SELECT unnest(range(0, {n_nodes})) AS node_id
                ),
                degrees AS (
                    SELECT csr_source AS src, COUNT(*) AS deg
                    FROM ({inner})
                    GROUP BY csr_source
                ),
                cumulative AS (
                    SELECT
                        node_range.node_id,
                        COALESCE(
                            SUM(degrees.deg) OVER (
                                ORDER BY node_range.node_id
                                ROWS UNBOUNDED PRECEDING
                            ), 0
                        ) AS ptr
                    FROM node_range
                    LEFT JOIN degrees ON node_range.node_id = degrees.src
                )
                SELECT ptr FROM cumulative
                ORDER BY node_id
                """,
                inner_params,
            )
        con.execute("""
            CREATE OR REPLACE TABLE indptr AS
            SELECT 0::UBIGINT AS ptr
            UNION ALL
            SELECT ptr::UBIGINT FROM indptr
            ORDER BY ptr
            """)

        _write_parquet_with_icebug_metadata(
            con, "vertices", out_dir / f"nodes_{name}.parquet"
        )
        _write_parquet_with_icebug_metadata(
            con, "indptr", out_dir / f"indptr_{csr_rel_name(name)}.parquet"
        )
    finally:
        con.close()


def build_csr_from_arrow_tables(
    node_table: pa.Table,
    rel_table: pa.Table,
    *,
    to_node_table: pa.Table | None = None,
    add_reverse_edges: bool = False,
    memory_limit: str | None = None,
    max_tmp_dir_gb: float | None = None,
    input_sorted: bool = False,
    threads: int | None = None,
) -> tuple[pa.Table, pa.Table, pa.Table, pa.Table]:
    """
    Build ``(src_out, dst_out, indices, indptr)`` with the DuckDB engine.

    Same contract as :func:`icebug_format._convert_pyarrow.build_csr_from_tables`
    (the pure-PyArrow path), but the node sort, endpoint mapping and
    ``(source, target)`` sort run inside DuckDB with the vertex/edge tables
    registered from Arrow, so *memory_limit* bounds DuckDB's buffer pool and
    the external sort spills to disk instead of growing RSS, and
    *max_tmp_dir_gb* caps the spill directory size.  *threads* sets DuckDB's
    worker thread count.  Mirrors
    :func:`convert_graph` for Parquet pairs, with the tables registered from
    Arrow rather than Parquet files.  Unless *input_sorted* is set, node
    tables are ordered by primary key; with *input_sorted* the caller
    guarantees they are already ordered, so CSR ids are assigned by row
    position and the sort is skipped.
    """
    from icebug_format._duckdb import require_duckdb
    from icebug_format.cli import (
        set_max_temp_dir_size,
        set_memory_limit,
        set_threads,
    )

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

    duckdb = require_duckdb("the 'convert' feature (Arrow table conversion)")
    con = duckdb.connect()
    if memory_limit:
        set_memory_limit(con, memory_limit)
    set_max_temp_dir_size(con, max_tmp_dir_gb)
    set_threads(con, threads)
    # Disable insertion-order preservation: saves memory/disk during the
    # external sort of large edge sets (relations/indices are read back
    # fully sorted by ORDER BY anyway).
    con.execute("SET preserve_insertion_order = false")
    try:
        con.register("vertices", node_table)
        con.register("vertices_dst", to_node_table)
        con.register("edges", rel_table)

        # Dense id -> CSR index mapping, ordered by primary key (or input row
        # order when input_sorted).
        src_row_order = "" if input_sorted else f"ORDER BY {_q(src_pk)}"
        con.execute(f"""
            CREATE TABLE src_map AS
            SELECT row_number() OVER ({src_row_order}) - 1 AS csr_index,
                   {_q(src_pk)} AS original_node_id
            FROM vertices
            """)
        if to_node_table is node_table:
            dst_map = "src_map"
        else:
            dst_map = "dst_map"
            dst_row_order = "" if input_sorted else f"ORDER BY {_q(dst_pk)}"
            con.execute(f"""
                CREATE TABLE dst_map AS
                SELECT row_number() OVER ({dst_row_order}) - 1 AS csr_index,
                       {_q(dst_pk)} AS original_node_id
                FROM vertices_dst
                """)

        select_fwd = "m1.csr_index AS csr_source, m2.csr_index AS csr_target"
        select_rev = "m2.csr_index AS csr_source, m1.csr_index AS csr_target"
        if prop_cols:
            props = ", " + ", ".join(f"e.{_q(c)}" for c in prop_cols)
            select_fwd += props
            select_rev += props
        join_clause = f"""
            FROM edges e
            JOIN src_map m1 ON e.{_q(src_col)} = m1.original_node_id
            JOIN {dst_map} m2 ON e.{_q(dst_col)} = m2.original_node_id
        """

        if not add_reverse_edges:
            rel_query = f"SELECT {select_fwd} {join_clause}"
        else:
            # Self-loops appear once (forward only); non-self edges get both directions.
            rel_query = f"""
                SELECT {select_fwd} {join_clause}
                UNION ALL
                SELECT {select_rev} {join_clause}
                WHERE e.{_q(src_col)} != e.{_q(dst_col)}
            """
        con.execute(f"CREATE TABLE relations AS {rel_query}")

        # indices: neighbour list sorted by (source, target)
        select_props = ", " + ", ".join(_q(c) for c in prop_cols) if prop_cols else ""
        con.execute(f"""
            CREATE TABLE indices AS
            SELECT csr_target::UBIGINT AS target{select_props}
            FROM relations
            ORDER BY csr_source, csr_target
            """)

        # indptr: cumulative degree per source node, zero-filled, N+1 entries.
        con.execute(f"""
            CREATE TABLE indptr AS
            WITH node_range AS (
                SELECT unnest(range(0, {n_src})) AS node_id
            ),
            degrees AS (
                SELECT csr_source AS src, COUNT(*) AS deg
                FROM relations
                GROUP BY csr_source
            ),
            cumulative AS (
                SELECT
                    node_range.node_id,
                    COALESCE(
                        SUM(degrees.deg) OVER (
                            ORDER BY node_range.node_id
                            ROWS UNBOUNDED PRECEDING
                        ), 0
                    ) AS ptr
                FROM node_range
                LEFT JOIN degrees ON node_range.node_id = degrees.src
            )
            SELECT ptr FROM cumulative
            ORDER BY node_id
            """)
        con.execute("""
            CREATE OR REPLACE TABLE indptr AS
            SELECT 0::UBIGINT AS ptr
            UNION ALL
            SELECT ptr::UBIGINT FROM indptr
            ORDER BY ptr
            """)

        src_out = con.execute(
            f"SELECT * FROM vertices{' ORDER BY ' + _q(src_pk) if not input_sorted else ''}"
        ).to_arrow_table()
        dst_out = con.execute(
            f"SELECT * FROM vertices_dst{' ORDER BY ' + _q(dst_pk) if not input_sorted else ''}"
        ).to_arrow_table()
        indices = con.execute("SELECT * FROM indices").to_arrow_table()
        indptr = con.execute("SELECT * FROM indptr").to_arrow_table()
        return src_out, dst_out, indices, indptr
    finally:
        con.close()
