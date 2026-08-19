"""Pure-PyArrow CSR construction for vertex/edge graph data.

In-memory pipeline (no SQL engine, no extra dependencies beyond pyarrow):

1. Sort node tables by primary key (unless the caller guarantees ``input_sorted``);
   CSR index = row position.
2. Map edge ``source``/``target`` ids to dense CSR indices using the cheapest
   strategy for each endpoint's id layout:
   - contiguous ``0..N-1`` ids  -> identity mapping (no work);
   - bounded non-negative ids   -> O(max_id) lookup table + vectorised gather;
   - anything else              -> hash lookup (``index_in``).
3. Sort mapped edges by ``(csr_source, csr_target)`` with pyarrow's radix sort.
4. Derive ``indptr`` from a histogram + prefix sum (``value_counts`` +
   ``scatter`` + ``cumulative_sum``), so no second pass over the edges.

``build_csr_from_tables`` is the in-memory core shared by
:meth:`icebug_format.memory.IcebugMemGraph.from_arrow_tables` and by the
Parquet-pair converter (:func:`convert_graph`), which streams the edge table
in batches to keep peak memory bounded.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from icebug_format.convert_parquet import (
    _build_indptr,
    _write_icebug_parquet,
    csr_rel_name,
    resolve_rel_column_names,
)

_BATCH_SIZE = 4_000_000  # mapped edge rows buffered at a time


def _csr_dtype(n_nodes: int) -> pa.DataType:
    """Smallest signed int type that can hold a CSR index < n_nodes."""
    if n_nodes > 2**31 - 1:
        return pa.int64()
    return pa.int32()


def _mapping_plan(ids_sorted: pa.Array) -> tuple[str, object]:
    """
    Decide how to map original ids to dense CSR indices.

    Returns ``(strategy, lookup_or_none)`` where ``strategy`` is one of
    ``"dense"`` (id == csr index), ``"lookup"`` (gather from a lookup table) or
    ``"hash"`` (``index_in``).
    """
    n = len(ids_sorted)
    if n == 0:
        return "hash", None
    first = ids_sorted[0].as_py()
    last = ids_sorted[-1].as_py()
    if first == 0 and last == n - 1:
        # Sorted ascending with min 0 and max n-1 must be exactly 0..n-1.
        return "dense", None
    if first >= 0 and last < n * 32:  # bounded id range keeps lookup cheap
        lookup = pc.fill_null(
            pc.scatter(
                pa.array(range(n), type=pa.int32()),
                ids_sorted,
                max_index=last,
            ),
            -1,
        )
        return "lookup", lookup
    return "hash", None


def _side(plan: tuple[str, object], ids_sorted: pa.Array, n: int, v64: pa.Array):
    """
    Compute an endpoint column's validity and (optionally) pre-mapped values.

    Returns ``(pre_mapped_or_none, valid_mask)``.  ``pre_mapped`` is only set
    for the hash strategy, where mapping is needed to know validity; dense and
    lookup strategies defer the (cheap) gather until the combined mask is known.
    """
    strategy, lookup = plan
    if strategy == "dense":
        valid = pc.and_(pc.greater_equal(v64, 0), pc.less(v64, n))
        return None, valid
    if strategy == "lookup":
        max_id = len(lookup) - 1
        valid = pc.and_(pc.greater_equal(v64, 0), pc.less_equal(v64, max_id))
        return None, valid
    mapped = pc.index_in(v64, ids_sorted)
    return mapped, pc.is_valid(mapped)


def _map_side(
    v64: pa.Array,
    pre: pa.Array | None,
    plan: tuple[str, object],
    valid: pa.Array,
    dtype: pa.DataType,
) -> pa.Array:
    """Map one endpoint column's valid rows to CSR ids."""
    if pre is not None:
        return pre.filter(valid).cast(dtype)
    strategy, lookup = plan
    if strategy == "dense":
        return v64.filter(valid).cast(dtype)
    return pc.take(lookup, v64.filter(valid)).cast(dtype)


def _finalize_csr(
    rel: pa.Table,
    prop_cols: list[str],
    add_reverse_edges: bool,
    n_nodes: int,
) -> tuple[pa.Table, pa.Table]:
    """Sort mapped edges and return ``(indices, indptr)`` CSR tables."""
    if add_reverse_edges and len(rel):
        # Self-loops appear once (forward only); non-self edges get both directions.
        rev_mask = pc.not_equal(rel["csr_source"], rel["csr_target"])
        rev_cols = {
            "csr_source": rel["csr_target"].filter(rev_mask),
            "csr_target": rel["csr_source"].filter(rev_mask),
        }
        for c in prop_cols:
            rev_cols[c] = rel[c].filter(rev_mask)
        rel = pa.concat_tables([rel, pa.table(rev_cols)])

    rel = rel.take(
        pc.sort_indices(
            rel, sort_keys=[("csr_source", "ascending"), ("csr_target", "ascending")]
        )
    )

    indptr = pa.table({"ptr": _build_indptr(rel["csr_source"], n_nodes)})

    idx_cols = {"target": rel["csr_target"].cast(pa.uint64())}
    for c in prop_cols:
        idx_cols[c] = rel[c]
    return pa.table(idx_cols), indptr


def _map_batch(
    rb: pa.RecordBatch,
    src_plan: tuple[str, object],
    dst_plan: tuple[str, object],
    src_ids: pa.Array,
    dst_ids: pa.Array,
    n_src: int,
    n_dst: int,
    src_col: str,
    dst_col: str,
) -> tuple:
    """Map one record batch's endpoints to CSR ids; drop unknown endpoints."""
    s64 = rb[src_col].cast(pa.int64())
    t64 = rb[dst_col].cast(pa.int64())
    s_pre, s_valid = _side(src_plan, src_ids, n_src, s64)
    t_pre, t_valid = _side(dst_plan, dst_ids, n_dst, t64)
    valid = pc.and_(s_valid, t_valid)
    dtype = _csr_dtype(max(n_src, n_dst))
    return (
        _map_side(s64, s_pre, src_plan, valid, dtype),
        _map_side(t64, t_pre, dst_plan, valid, dtype),
        valid,
    )


def _empty_mapped(
    schema: pa.Schema,
    src_col: str,
    dst_col: str,
    prop_cols: list[str],
    n_src: int,
    n_dst: int,
) -> pa.Table:
    dtype = _csr_dtype(max(n_src, n_dst))
    return pa.table(
        {
            "csr_source": pa.array([], type=dtype),
            "csr_target": pa.array([], type=dtype),
            **{
                c: pa.array([], type=schema.field(c).type)
                for c in prop_cols
                if c not in (src_col, dst_col)
            },
        }
    )


def _read_mapped_edges(
    epf: pq.ParquetFile,
    src_col: str,
    dst_col: str,
    prop_cols: list[str],
    ids_sorted: pa.Array,
    n_nodes: int,
    plan: tuple[str, object],
) -> pa.Table:
    """Stream the edge parquet, mapping ids to CSR indices batch by batch."""
    if n_nodes == 0:
        return _empty_mapped(
            epf.schema_arrow, src_col, dst_col, prop_cols, n_nodes, n_nodes
        )

    batches = []
    for rb in epf.iter_batches(
        batch_size=_BATCH_SIZE,
        columns=[src_col, dst_col, *prop_cols],
    ):
        s, t, valid = _map_batch(
            rb, plan, plan, ids_sorted, ids_sorted, n_nodes, n_nodes, src_col, dst_col
        )
        cols = {"csr_source": s, "csr_target": t}
        for c in prop_cols:
            cols[c] = rb[c].filter(valid)
        batches.append(pa.table(cols))

    if not batches:
        return _empty_mapped(
            epf.schema_arrow, src_col, dst_col, prop_cols, n_nodes, n_nodes
        )
    return pa.concat_tables(batches)


def build_csr_from_tables(
    node_table: pa.Table,
    rel_table: pa.Table,
    *,
    to_node_table: pa.Table | None = None,
    add_reverse_edges: bool = False,
    input_sorted: bool = False,
) -> tuple[pa.Table, pa.Table, pa.Table]:
    """
    Build ``(src_out, dst_out, indices, indptr)`` from in-memory Arrow tables.

    The first column of each node table is the primary key used to map node ids
    to dense 0-based CSR indices.  Source and destination endpoints may come
    from different node tables (``to_node_table``); when omitted, the from-node
    table is used for both sides.

    Args:
        node_table: Source node table.
        rel_table: Relationship table with source/target columns (resolved by
            alias name, then positionally) plus optional edge property columns.
        to_node_table: Destination node table (directed heterogeneous graphs
            only); defaults to *node_table*.
        add_reverse_edges: If ``True``, emit reverse edges for symmetric
            adjacency (only valid when *to_node_table* is omitted).
        input_sorted: If ``True``, the caller guarantees the node tables are
            already ordered by primary key, so the sort step is skipped and CSR
            ids are assigned by row position.

    Returns:
        ``(src_out, dst_out, indices, indptr)`` where ``src_out``/``dst_out``
        are the node tables in CSR index order, ``indices`` holds the
        ``target`` column (plus edge properties) sorted by
        ``(csr_source, csr_target)``, and ``indptr`` is the ``N + 1`` row
        pointer array.
    """
    if add_reverse_edges and to_node_table is not None:
        raise ValueError(
            "to_node_arrow_table must not be provided when adding reverse edges; "
            "from and to node tables must be the same."
        )
    if rel_table.num_columns < 2:
        raise ValueError(
            f"rel_arrow_table must have at least 2 columns (source and target), "
            f"got {rel_table.num_columns}"
        )

    if to_node_table is None:
        to_node_table = node_table

    src_pk = node_table.schema.names[0]
    dst_pk = to_node_table.schema.names[0]
    src_col, dst_col = resolve_rel_column_names(rel_table.schema.names)
    prop_cols = [c for c in rel_table.schema.names if c not in (src_col, dst_col)]

    # --- nodes: sort by primary key unless the caller guarantees sorted -----
    if input_sorted:
        src_out, dst_out = node_table, to_node_table
    else:
        src_out = node_table.take(pc.sort_indices(node_table.column(src_pk)))
        dst_out = to_node_table.take(pc.sort_indices(to_node_table.column(dst_pk)))
    src_ids = src_out.column(src_pk)
    dst_ids = dst_out.column(dst_pk)
    n_src, n_dst = len(src_out), len(dst_out)

    # --- edges: map endpoints to dense CSR indices -------------------------
    if len(rel_table) == 0 or n_src == 0 or n_dst == 0:
        rel = _empty_mapped(rel_table.schema, src_col, dst_col, prop_cols, n_src, n_dst)
    else:
        s64 = rel_table[src_col].cast(pa.int64())
        t64 = rel_table[dst_col].cast(pa.int64())
        src_plan = _mapping_plan(src_ids)
        dst_plan = _mapping_plan(dst_ids)
        s_pre, s_valid = _side(src_plan, src_ids, n_src, s64)
        t_pre, t_valid = _side(dst_plan, dst_ids, n_dst, t64)
        valid = pc.and_(s_valid, t_valid)
        dtype = _csr_dtype(max(n_src, n_dst))
        cols = {
            "csr_source": _map_side(s64, s_pre, src_plan, valid, dtype),
            "csr_target": _map_side(t64, t_pre, dst_plan, valid, dtype),
        }
        for c in prop_cols:
            cols[c] = rel_table[c].filter(valid)
        rel = pa.table(cols)

    indices, indptr = _finalize_csr(rel, prop_cols, add_reverse_edges, n_src)
    return src_out, dst_out, indices, indptr


def convert_graph(
    graph: dict,
    add_reverse_edges: bool = False,
    memory_limit: str | None = None,
    max_tmp_dir_gb: float | None = None,
    input_sorted: bool = False,
) -> None:
    """
    Convert one vertex/edge Parquet pair with the pure-PyArrow pipeline.

    ``memory_limit`` and ``max_tmp_dir_gb`` are accepted for interface
    compatibility with the SQL backends but are not used (pyarrow manages its
    own memory).  ``input_sorted`` skips the node-table sort: CSR ids are then
    assigned by row position, which is only valid when the vertex parquet is
    already ordered by primary key.
    """
    name = graph["name"]
    out_dir = Path(graph["output_dir"])

    # --- vertices: CSR index = row position; sort by primary key unless -----
    # --- the caller guarantees the input is already sorted -------------------
    vtable = pq.read_table(graph["vertex"])
    pk = vtable.schema.names[0]
    if not input_sorted:
        vtable = vtable.take(pc.sort_indices(vtable.column(pk)))
    ids_sorted = vtable.column(pk)
    n_nodes = len(vtable)

    # --- edges: map endpoints to dense CSR indices -------------------------
    epf = pq.ParquetFile(graph["edge"])
    src_col, dst_col = resolve_rel_column_names(epf.schema_arrow.names)
    prop_cols = [c for c in epf.schema_arrow.names if c not in (src_col, dst_col)]

    plan = _mapping_plan(ids_sorted)
    rel = _read_mapped_edges(
        epf, src_col, dst_col, prop_cols, ids_sorted, n_nodes, plan
    )

    indices, indptr = _finalize_csr(rel, prop_cols, add_reverse_edges, n_nodes)

    _write_icebug_parquet(vtable, out_dir / f"nodes_{name}.parquet")
    _write_icebug_parquet(indices, out_dir / f"indices_{csr_rel_name(name)}.parquet")
    _write_icebug_parquet(indptr, out_dir / f"indptr_{csr_rel_name(name)}.parquet")
