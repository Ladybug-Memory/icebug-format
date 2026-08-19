"""
Convert Arrow tables to icebug-memory format (IcebugMemGraph).

icebug-memory is the in-memory counterpart of icebug-disk: instead of
parquet files, graph data is stored as PyArrow tables in a CSR
(Compressed Sparse Row) structure.

The conversion is performed by the pure-PyArrow pipeline in
:mod:`icebug_format._convert_pyarrow` (the same engine behind the
``pyarrow`` Parquet-pair backend); DuckDB is not required.  The ``"duckdb"``
and ``"datafusion"`` backends run the same plan inside a SQL engine so a
``memory_limit`` can spill sorts to disk; requesting a memory limit with the
default backend switches to datafusion.
"""

import warnings
from dataclasses import dataclass

import pyarrow as pa

from icebug_format._convert_pyarrow import build_csr_from_tables


@dataclass
class IcebugMemGraph:
    """
    CSR graph representation using Arrow tables.

    Attributes:
        src: Source node table in CSR index order (sorted by primary key).
        dest: Destination node table in CSR index order (sorted by primary key).
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
        input_sorted: bool = False,
        backend: str = "pyarrow",
        memory_limit: str | None = None,
        max_tmp_dir_gb: float | None = None,
        threads: int | None = None,
    ) -> "IcebugMemGraph":
        """
        Convert node and relationship Arrow tables to an IcebugMemGraph.

        The first column of each node table is treated as the primary key used
        to map node IDs to dense 0-based CSR indices.  Unless
        ``input_sorted=True``, node tables are sorted by primary key, so the
        returned ``src``/``dest`` tables are in CSR index order (row ``i`` of
        ``src`` is CSR node ``i``).

        The relationship table's source and target columns are resolved by name
        in the following priority order, falling back to positional columns:

        - Source: ``source`` → ``src`` → ``from`` → 0th column
        - Target: ``target`` → ``destination`` → ``dest`` → ``to`` → 1st column

        Any remaining columns in *rel_arrow_table* are preserved as edge
        properties in the *indices* output table.  Edges whose endpoints do
        not appear in the node tables are dropped.

        When ``add_reverse_edges=True``, ``to_node_arrow_table`` must not be
        provided: the from-node table is used for both sides of every edge.
        Self-loops appear once; non-self edges get both directions.

        Args:
            from_node_arrow_table: Source node table.
            rel_arrow_table:       Relationship table.
            to_node_arrow_table:   Destination node table (directed graphs only).
                                   Defaults to *from_node_arrow_table* when
                                   ``None`` (i.e., homogeneous edges).
            add_reverse_edges:     If ``False`` (default), only input edges are
                                   stored.  If ``True``, reverse edges are added
                                   for symmetric adjacency.
            input_sorted:          If ``True``, the caller guarantees the node
                                   tables are already ordered by primary key,
                                   so the sort step is skipped and CSR ids are
                                   assigned by input row position (faster, but
                                   the mapping is only valid for pre-sorted
                                   input).  Honored by all backends; defaults
                                   to ``False``.
            backend:               Conversion engine: ``"pyarrow"`` (default;
                                   pure in-memory pipeline), ``"duckdb"``
                                   (requires the ``convert`` extra) or
                                   ``"datafusion"`` (requires the
                                   ``convert-datafusion`` extra).  When
                                   *memory_limit* is set and *backend* is left
                                   at the default, the backend automatically
                                   switches to ``"datafusion"`` because the
                                   pyarrow pipeline cannot bound its memory.
            memory_limit:          Cap on the memory the SQL engine may use
                                   for sorting (size string like ``"4GB"``,
                                   percent of RAM like ``"50%"``, or a GB
                                   number); sorts spill to disk.  Only honored
                                   by the ``"duckdb"`` and ``"datafusion"``
                                   backends.
            max_tmp_dir_gb:        Cap on the temp/spill directory size in GiB
                                   (DuckDB's ``max_temp_directory_size``
                                   PRAGMA, DataFusion's
                                   ``datafusion.runtime.max_temp_directory_size``).
                                   Only honored by the ``"duckdb"`` and
                                   ``"datafusion"`` backends.
            threads:               Worker thread count for the SQL engines
                                   (DuckDB's ``threads`` setting, DataFusion's
                                   ``datafusion.execution.target_partitions``).
                                   Only honored by the ``"duckdb"`` and
                                   ``"datafusion"`` backends.

        Returns:
            IcebugMemGraph where *src* and *dest* are the node tables in CSR
            index order and *indices*/*indptr* encode the CSR adjacency
            structure.

        Raises:
            ValueError: If *rel_arrow_table* has fewer than 2 columns.
            ValueError: If ``add_reverse_edges=True`` and *to_node_arrow_table*
                        is provided.
            ValueError: If *backend* is not ``"pyarrow"``, ``"duckdb"`` or
                        ``"datafusion"``.
        """
        if backend not in ("pyarrow", "duckdb", "datafusion"):
            raise ValueError(
                f"Unknown backend {backend!r}; expected 'pyarrow', 'duckdb' or "
                f"'datafusion'"
            )
        if backend == "pyarrow" and memory_limit:
            warnings.warn(
                "memory_limit is not supported by the pyarrow backend; "
                "switching to the datafusion backend",
                stacklevel=2,
            )
            backend = "datafusion"

        if backend == "datafusion":
            from icebug_format._convert_datafusion import (
                build_csr_from_arrow_tables as _df_build_csr,
            )

            src, dest, indices, indptr = _df_build_csr(
                from_node_arrow_table,
                rel_arrow_table,
                to_node_table=to_node_arrow_table,
                add_reverse_edges=add_reverse_edges,
                memory_limit=memory_limit,
                max_tmp_dir_gb=max_tmp_dir_gb,
                input_sorted=input_sorted,
                threads=threads,
            )
        elif backend == "duckdb":
            from icebug_format._convert_duckdb import (
                build_csr_from_arrow_tables as _dd_build_csr,
            )

            src, dest, indices, indptr = _dd_build_csr(
                from_node_arrow_table,
                rel_arrow_table,
                to_node_table=to_node_arrow_table,
                add_reverse_edges=add_reverse_edges,
                memory_limit=memory_limit,
                max_tmp_dir_gb=max_tmp_dir_gb,
                input_sorted=input_sorted,
                threads=threads,
            )
        else:
            src, dest, indices, indptr = build_csr_from_tables(
                from_node_arrow_table,
                rel_arrow_table,
                to_node_table=to_node_arrow_table,
                add_reverse_edges=add_reverse_edges,
                input_sorted=input_sorted,
            )
        return cls(src=src, dest=dest, indices=indices, indptr=indptr)
