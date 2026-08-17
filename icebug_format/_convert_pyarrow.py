"""Pure-PyArrow backend for vertex/edge Parquet -> icebug-disk CSR conversion.

In-memory pipeline (no SQL engine, no extra dependencies beyond pyarrow):

1. Read the vertex table, sort by primary key; CSR index = row position.
2. Map edge ``source``/``target`` ids to dense CSR indices using the cheapest
   strategy for the id layout:
   - contiguous ``0..N-1`` ids  -> identity mapping (no work);
   - bounded non-negative ids   -> O(max_id) lookup table + vectorised gather;
   - anything else              -> hash lookup (``index_in``).
3. Sort mapped edges by ``(csr_source, csr_target)`` with pyarrow's radix sort.
4. Derive ``indptr`` from a histogram + prefix sum (``value_counts`` +
   ``scatter`` + ``cumulative_sum``), so no second pass over the edges.

This is the fastest backend but keeps the whole (mapped, sorted) edge table in
memory, so its peak RSS scales with the edge count.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from icebug_format.convert_parquet import (
    _build_indptr,
    _write_icebug_parquet,
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


def _map_batch(
    rb: pa.RecordBatch,
    plan: tuple[str, object],
    ids_sorted: pa.Array,
    n_nodes: int,
    src_col: str,
    dst_col: str,
) -> tuple:
    """Map one record batch's endpoints to dense CSR indices; drop unknowns."""
    strategy, lookup = plan
    dtype = _csr_dtype(n_nodes)
    s = rb[src_col].cast(pa.int64())
    t = rb[dst_col].cast(pa.int64())

    if strategy == "dense":
        valid = pc.and_(
            pc.and_(pc.greater_equal(s, 0), pc.less(s, n_nodes)),
            pc.and_(pc.greater_equal(t, 0), pc.less(t, n_nodes)),
        )
        return s.filter(valid).cast(dtype), t.filter(valid).cast(dtype), valid

    if strategy == "lookup":
        max_id = len(lookup) - 1
        valid = pc.and_(
            pc.and_(pc.greater_equal(s, 0), pc.less_equal(s, max_id)),
            pc.and_(pc.greater_equal(t, 0), pc.less_equal(t, max_id)),
        )
        return (
            pc.take(lookup, s.filter(valid)).cast(dtype),
            pc.take(lookup, t.filter(valid)).cast(dtype),
            valid,
        )

    # hash fallback (handles negative/arbitrary ids and unknown endpoints)
    s = pc.index_in(s, ids_sorted)
    t = pc.index_in(t, ids_sorted)
    valid = pc.and_(pc.is_valid(s), pc.is_valid(t))
    return s.filter(valid).cast(dtype), t.filter(valid).cast(dtype), valid


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

    def empty_table():
        return pa.table(
            {
                "csr_source": pa.array([], type=_csr_dtype(n_nodes)),
                "csr_target": pa.array([], type=_csr_dtype(n_nodes)),
                **{
                    c: pa.array([], type=epf.schema_arrow.field(c).type)
                    for c in prop_cols
                },
            }
        )

    if n_nodes == 0:
        return empty_table()

    batches = []
    for rb in epf.iter_batches(
        batch_size=_BATCH_SIZE,
        columns=[src_col, dst_col, *prop_cols],
    ):
        s, t, valid = _map_batch(rb, plan, ids_sorted, n_nodes, src_col, dst_col)
        cols = {"csr_source": s, "csr_target": t}
        for c in prop_cols:
            cols[c] = rb[c].filter(valid)
        batches.append(pa.table(cols))

    if not batches:
        return empty_table()
    return pa.concat_tables(batches)


def convert_graph(
    graph: dict,
    add_reverse_edges: bool = False,
    memory_limit: str | None = None,
) -> None:
    """
    Convert one vertex/edge Parquet pair with the pure-PyArrow pipeline.

    ``memory_limit`` is accepted for interface compatibility with the SQL
    backends but is not used (pyarrow manages its own memory).
    """
    name = graph["name"]
    out_dir = Path(graph["output_dir"])

    # --- vertices: sort by primary key; CSR index = row position -----------
    vtable = pq.read_table(graph["vertex"])
    pk = vtable.schema.names[0]
    order = pc.sort_indices(vtable.column(pk))
    vtable = vtable.take(order)
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

    # --- reverse-edge expansion (self-loops stay forward-only) -------------
    if add_reverse_edges and len(rel):
        rev_mask = pc.not_equal(rel["csr_source"], rel["csr_target"])
        rev_cols = {
            "csr_source": rel["csr_target"].filter(rev_mask),
            "csr_target": rel["csr_source"].filter(rev_mask),
        }
        for c in prop_cols:
            rev_cols[c] = rel[c].filter(rev_mask)
        rel = pa.concat_tables([rel, pa.table(rev_cols)])

    # --- sort by (csr_source, csr_target) ---------------------------------
    rel = rel.take(
        pc.sort_indices(
            rel, sort_keys=[("csr_source", "ascending"), ("csr_target", "ascending")]
        )
    )

    # --- indptr: histogram + prefix sum (no second edge pass) --------------
    indptr = pa.table({"ptr": _build_indptr(rel["csr_source"], n_nodes)})

    # --- indices: target column (uint64) + preserved edge properties -------
    idx_cols = {"target": rel["csr_target"].cast(pa.uint64())}
    for c in prop_cols:
        idx_cols[c] = rel[c]
    indices = pa.table(idx_cols)

    _write_icebug_parquet(vtable, out_dir / f"nodes_{name}.parquet")
    _write_icebug_parquet(indices, out_dir / f"indices_{name}.parquet")
    _write_icebug_parquet(indptr, out_dir / f"indptr_{name}.parquet")
