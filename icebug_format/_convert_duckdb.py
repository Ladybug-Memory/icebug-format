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

from icebug_format.convert_parquet import resolve_rel_column_names


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def convert_graph(
    graph: dict,
    add_reverse_edges: bool = False,
    memory_limit: str | None = None,
) -> None:
    """Convert one vertex/edge Parquet pair with the DuckDB SQL engine."""
    from icebug_format._duckdb import require_duckdb
    from icebug_format.cli import _write_parquet_with_icebug_metadata, set_memory_limit

    name = graph["name"]
    out_dir = Path(graph["output_dir"])
    vertex_path = str(graph["vertex"])
    edge_path = str(graph["edge"])

    pk = pq.ParquetFile(vertex_path).schema_arrow.names[0]

    duckdb = require_duckdb("the 'convert' feature (vertex/edge Parquet conversion)")
    con = duckdb.connect()
    if memory_limit:
        set_memory_limit(con, memory_limit)
    try:
        con.execute(
            f"CREATE TABLE vertices AS SELECT * FROM read_parquet(?) ORDER BY {_q(pk)}",
            [vertex_path],
        )
        con.execute("CREATE TABLE edges AS SELECT * FROM read_parquet(?)", [edge_path])
        n_nodes = con.execute("SELECT COUNT(*) FROM vertices").fetchone()[0]

        edge_cols = [r[0] for r in con.execute("DESCRIBE edges").fetchall()]
        src_col, dst_col = resolve_rel_column_names(edge_cols)
        prop_cols = [c for c in edge_cols if c not in (src_col, dst_col)]

        # Dense id -> CSR index mapping, ordered by primary key.
        con.execute(f"""
            CREATE TABLE src_map AS
            SELECT row_number() OVER (ORDER BY {_q(pk)}) - 1 AS csr_index,
                   {_q(pk)} AS original_node_id
            FROM vertices
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
            JOIN src_map m2 ON e.{_q(dst_col)} = m2.original_node_id
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
        select_props = (", " + ", ".join(_q(c) for c in prop_cols)) if prop_cols else ""
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
                SELECT unnest(range(0, {n_nodes})) AS node_id
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

        _write_parquet_with_icebug_metadata(
            con, "vertices", out_dir / f"nodes_{name}.parquet"
        )
        _write_parquet_with_icebug_metadata(
            con, "indices", out_dir / f"indices_{name}.parquet"
        )
        _write_parquet_with_icebug_metadata(
            con, "indptr", out_dir / f"indptr_{name}.parquet"
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
) -> tuple[pa.Table, pa.Table, pa.Table, pa.Table]:
    """
    Build ``(src_out, dst_out, indices, indptr)`` with the DuckDB engine.

    Same contract as :func:`icebug_format._convert_pyarrow.build_csr_from_tables`
    (the pure-PyArrow path), but the node sort, endpoint mapping and
    ``(source, target)`` sort run inside DuckDB with the vertex/edge tables
    registered from Arrow, so *memory_limit* bounds DuckDB's buffer pool and
    the external sort spills to disk instead of growing RSS.  Mirrors
    :func:`convert_graph` for Parquet pairs, with the tables registered from
    Arrow rather than Parquet files.  Node tables are always ordered by primary
    key (there is no ``input_sorted`` shortcut).
    """
    from icebug_format._duckdb import require_duckdb
    from icebug_format.cli import set_memory_limit

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
    try:
        con.register("vertices", node_table)
        con.register("vertices_dst", to_node_table)
        con.register("edges", rel_table)

        # Dense id -> CSR index mapping, ordered by primary key.
        con.execute(f"""
            CREATE TABLE src_map AS
            SELECT row_number() OVER (ORDER BY {_q(src_pk)}) - 1 AS csr_index,
                   {_q(src_pk)} AS original_node_id
            FROM vertices
            """)
        if to_node_table is node_table:
            dst_map = "src_map"
        else:
            dst_map = "dst_map"
            con.execute(f"""
                CREATE TABLE dst_map AS
                SELECT row_number() OVER (ORDER BY {_q(dst_pk)}) - 1 AS csr_index,
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
            f"SELECT * FROM vertices ORDER BY {_q(src_pk)}"
        ).to_arrow_table()
        dst_out = con.execute(
            f"SELECT * FROM vertices_dst ORDER BY {_q(dst_pk)}"
        ).to_arrow_table()
        indices = con.execute("SELECT * FROM indices").to_arrow_table()
        indptr = con.execute("SELECT * FROM indptr").to_arrow_table()
        return src_out, dst_out, indices, indptr
    finally:
        con.close()
