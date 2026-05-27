"""
Convert Arrow tables to icebug-memory format (IcebugMemGraph).

icebug-memory is the in-memory counterpart of icebug-disk: instead of
parquet files, graph data is stored as PyArrow tables in a CSR
(Compressed Sparse Row) structure.
"""

from dataclasses import dataclass

import duckdb
import pyarrow as pa

_SOURCE_ALIASES = ("source", "src", "from")
_TARGET_ALIASES = ("target", "destination", "dest", "to")


def _resolve_rel_columns(schema: pa.Schema) -> tuple[str, str]:
    """
    Return the (source_col, target_col) column names from a relationship schema.

    Checks for known aliases in priority order, then falls back to the 0th and
    1st columns respectively.

    Source aliases (in order): source, src, from
    Target aliases (in order): target, destination, dest, to
    """
    names = schema.names
    name_set = set(names)

    src_col = next((a for a in _SOURCE_ALIASES if a in name_set), names[0])
    dst_col = next((a for a in _TARGET_ALIASES if a in name_set), names[1])
    return src_col, dst_col


@dataclass
class IcebugMemGraph:
    """
    CSR graph representation using Arrow tables.

    Attributes:
        src: Source node table (passed as-is from input).
        dest: Destination node table (passed as-is from input).
        indices: Arrow table with a 'target' column (and optional edge
                 property columns) sorted by source node, then target node.
        indptr: Arrow table with a 'ptr' column of length
                len(src) + 1 giving the start offset of each source
                node's adjacency list in *indices*.
    """

    src: pa.Table
    dest: pa.Table
    indices: pa.Table
    indptr: pa.Table

    @classmethod
    def from_arrow_tables(
        cls,
        from_node_arrow_table: pa.Table,
        rel_arrow_table: pa.Table,
        *,
        to_node_arrow_table: pa.Table | None = None,
        add_reverse_edges: bool = False,
    ) -> "IcebugMemGraph":
        """
        Convert node and relationship Arrow tables to an IcebugMemGraph.

        The first column of each node table is treated as the primary key used
        to map node IDs to dense 0-based CSR indices.

        The relationship table's source and target columns are resolved by name
        in the following priority order, falling back to positional columns:

        - Source: ``source`` → ``src`` → ``from`` → 0th column
        - Target: ``target`` → ``destination`` → ``dest`` → ``to`` → 1st column

        Any remaining columns in *rel_arrow_table* are preserved as edge
        properties in the *indices* output table.

        When ``add_reverse_edges=True``, ``to_node_arrow_table`` must not be
        provided: the from-node table is used for both sides of every edge.

        Args:
            from_node_arrow_table: Source node table.
            rel_arrow_table:       Relationship table.
            to_node_arrow_table:   Destination node table (directed graphs only).
                                   Defaults to *from_node_arrow_table* when
                                   ``None`` (i.e., homogeneous edges).
            add_reverse_edges:     If ``False`` (default), only input edges are
                                   stored.  If ``True``, reverse edges are added
                                   for symmetric adjacency.

        Returns:
            IcebugMemGraph where *src* and *dest* are the original node tables
            and *indices*/*indptr* encode the CSR adjacency structure.

        Raises:
            ValueError: If *rel_arrow_table* has fewer than 2 columns.
            ValueError: If ``add_reverse_edges=True`` and *to_node_arrow_table*
                        is provided.
        """
        if add_reverse_edges and to_node_arrow_table is not None:
            raise ValueError(
                "to_node_arrow_table must not be provided when adding reverse edges; "
                "from and to node tables must be the same."
            )

        if to_node_arrow_table is None:
            to_node_arrow_table = from_node_arrow_table

        if rel_arrow_table.num_columns < 2:
            raise ValueError(
                f"rel_arrow_table must have at least 2 columns (source and target), "
                f"got {rel_arrow_table.num_columns}"
            )

        src_pk = from_node_arrow_table.schema.names[0]
        dst_pk = to_node_arrow_table.schema.names[0]
        num_src_nodes = len(from_node_arrow_table)

        src_col, dst_col = _resolve_rel_columns(rel_arrow_table.schema)
        edge_cols = [
            c for c in rel_arrow_table.schema.names if c not in (src_col, dst_col)
        ]

        select_fwd = "m1.csr_index AS csr_source, m2.csr_index AS csr_target"
        select_rev = "m2.csr_index AS csr_source, m1.csr_index AS csr_target"

        def q(name: str) -> str:
            return '"' + name.replace('"', '""') + '"'

        if edge_cols:
            props = ", ".join(f"e.{q(c)}" for c in edge_cols)
            select_fwd += f", {props}"
            select_rev += f", {props}"

        map_cte = f"""
            src_map AS (
                SELECT row_number() OVER () - 1 AS csr_index,
                       {q(src_pk)} AS original_node_id
                FROM from_nodes
            ),
            dst_map AS (
                SELECT row_number() OVER () - 1 AS csr_index,
                       {q(dst_pk)} AS original_node_id
                FROM to_nodes
            )
        """

        join_clause = f"""
            FROM edges e
            JOIN src_map m1 ON e.{q(src_col)} = m1.original_node_id
            JOIN dst_map m2 ON e.{q(dst_col)} = m2.original_node_id
        """

        if not add_reverse_edges:
            rel_query = f"WITH {map_cte} SELECT {select_fwd} {join_clause}"
        else:
            # Self-loops appear once (forward only); non-self edges get both directions.
            rel_query = f"""
                WITH {map_cte}
                SELECT {select_fwd} {join_clause}
                UNION ALL
                SELECT {select_rev} {join_clause}
                WHERE e.{q(src_col)} != e.{q(dst_col)}
            """

        edge_props_select = (
            (", " + ", ".join(q(c) for c in edge_cols)) if edge_cols else ""
        )

        con = duckdb.connect()
        try:
            con.register("from_nodes", from_node_arrow_table)
            con.register("to_nodes", to_node_arrow_table)
            con.register("edges", rel_arrow_table)

            con.execute(f"CREATE TABLE relations AS {rel_query}")

            # Build indptr: cumulative degree per source node
            con.execute(f"""
                CREATE TABLE indptr_table AS
                WITH node_range AS (
                    SELECT unnest(range(0, {num_src_nodes})) AS node_id
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

            # Prepend leading zero so indptr[i] = start of node i's adjacency list
            con.execute("""
                CREATE OR REPLACE TABLE indptr_table AS
                SELECT 0::UINT64 AS ptr
                UNION ALL
                SELECT ptr::UINT64 FROM indptr_table
                ORDER BY ptr
            """)

            # Build indices: neighbour list sorted by (source, target)
            con.execute(f"""
                CREATE TABLE indices_table AS
                SELECT csr_target::UINT64 AS target{edge_props_select}
                FROM relations
                ORDER BY csr_source, csr_target
            """)

            indices = con.execute("SELECT * FROM indices_table").arrow().read_all()
            indptr = con.execute("SELECT * FROM indptr_table").arrow().read_all()
        finally:
            con.close()

        return cls(
            src=from_node_arrow_table,
            dest=to_node_arrow_table,
            indices=indices,
            indptr=indptr,
        )
