"""Tests for the icebug-memory converter (IcebugMemGraph.from_arrow_tables)."""

import pyarrow as pa
import pytest

from icebug_format.memory import IcebugMemGraph


def _nodes(*ids, pk="id"):
    return pa.table({pk: pa.array(ids, type=pa.int64())})


def _rels(sources, targets, src_col="source", dst_col="destination", **props):
    data = {
        src_col: pa.array(sources, type=pa.int64()),
        dst_col: pa.array(targets, type=pa.int64()),
    }
    data.update(props)
    return pa.table(data)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_icebug_mem_graph():
    nodes = _nodes(0, 1)
    rels = _rels([0], [1])
    result = IcebugMemGraph.from_arrow_tables(nodes, rels)
    assert isinstance(result, IcebugMemGraph)


# ---------------------------------------------------------------------------
# Column types
# ---------------------------------------------------------------------------


def test_indices_target_column_is_uint64():
    nodes = _nodes(0, 1, 2)
    rels = _rels([0, 1], [1, 2])
    g = IcebugMemGraph.from_arrow_tables(nodes, rels)
    assert g.indices.schema.field("target").type == pa.uint64()


def test_indptr_ptr_column_is_uint64():
    nodes = _nodes(0, 1, 2)
    rels = _rels([0, 1], [1, 2])
    g = IcebugMemGraph.from_arrow_tables(nodes, rels)
    assert g.indptr.schema.field("ptr").type == pa.uint64()


# ---------------------------------------------------------------------------
# Directed graph: CSR structure
# ---------------------------------------------------------------------------


def test_directed_linear_chain():
    # Graph: 0 -> 1 -> 2
    nodes = _nodes(0, 1, 2)
    rels = _rels([0, 1], [1, 2])
    g = IcebugMemGraph.from_arrow_tables(nodes, rels)

    # indices: neighbour list in source order
    assert g.indices["target"].to_pylist() == [1, 2]
    # indptr: [0, 1, 2, 2]  (node 2 has no outgoing edges)
    assert g.indptr["ptr"].to_pylist() == [0, 1, 2, 2]


def test_directed_fan_out():
    # Graph: 0 -> 1, 0 -> 2, 0 -> 3
    nodes = _nodes(0, 1, 2, 3)
    rels = _rels([0, 0, 0], [1, 2, 3])
    g = IcebugMemGraph.from_arrow_tables(nodes, rels)

    assert sorted(g.indices["target"].to_pylist()) == [1, 2, 3]
    assert g.indptr["ptr"].to_pylist() == [0, 3, 3, 3, 3]


def test_indptr_length_equals_num_src_nodes_plus_one():
    nodes = _nodes(0, 1, 2, 3, 4)
    rels = _rels([0, 2], [1, 3])
    g = IcebugMemGraph.from_arrow_tables(nodes, rels)
    assert len(g.indptr) == len(nodes) + 1


def test_indptr_starts_with_zero():
    nodes = _nodes(0, 1, 2)
    rels = _rels([0], [1])
    g = IcebugMemGraph.from_arrow_tables(nodes, rels)
    assert g.indptr["ptr"][0].as_py() == 0


def test_indptr_ends_with_edge_count():
    nodes = _nodes(0, 1, 2)
    rels = _rels([0, 1], [1, 2])
    g = IcebugMemGraph.from_arrow_tables(nodes, rels)
    assert g.indptr["ptr"][-1].as_py() == 2


def test_indices_length_equals_edge_count():
    nodes = _nodes(0, 1, 2)
    rels = _rels([0, 1], [1, 2])
    g = IcebugMemGraph.from_arrow_tables(nodes, rels)
    assert len(g.indices) == 2


def test_directed_preserves_self_loops():
    # Self-loops must not be filtered in directed graphs.
    nodes = _nodes(0, 1)
    rels = _rels([0, 0], [0, 1])  # 0->0 self-loop + 0->1
    g = IcebugMemGraph.from_arrow_tables(nodes, rels)
    assert len(g.indices) == 2
    assert sorted(g.indices["target"].to_pylist()) == [0, 1]


# ---------------------------------------------------------------------------
# Undirected graph
# ---------------------------------------------------------------------------


def test_undirected_adds_reverse_edges():
    # Graph: 0 -- 1
    nodes = _nodes(0, 1)
    rels = _rels([0], [1])
    g = IcebugMemGraph.from_arrow_tables(nodes, rels, directed=False)

    assert len(g.indices) == 2
    targets = sorted(g.indices["target"].to_pylist())
    assert targets == [0, 1]


def test_undirected_indptr_reflects_bidirectional_degree():
    # 0 -- 1 -- 2
    nodes = _nodes(0, 1, 2)
    rels = _rels([0, 1], [1, 2])
    g = IcebugMemGraph.from_arrow_tables(nodes, rels, directed=False)

    ptr = g.indptr["ptr"].to_pylist()
    # node 0: 1 neighbour, node 1: 2 neighbours, node 2: 1 neighbour
    degrees = [ptr[i + 1] - ptr[i] for i in range(len(nodes))]
    assert degrees == [1, 2, 1]


# ---------------------------------------------------------------------------
# Self-loop handling
# ---------------------------------------------------------------------------

def test_self_loops_appear_once_in_undirected_graph():
    nodes = _nodes(0, 1)
    rels = _rels([0, 0], [0, 1])  # 0->0 self-loop + 0->1
    g = IcebugMemGraph.from_arrow_tables(nodes, rels, directed=False)

    # 0->0 (once), 0->1, 1->0  → 3 entries total
    assert len(g.indices) == 3
    targets_by_src = {
        0: g.indices["target"].to_pylist()[: g.indptr["ptr"][1].as_py()],
        1: g.indices["target"].to_pylist()[g.indptr["ptr"][1].as_py() :],
    }
    assert sorted(targets_by_src[0]) == [0, 1]
    assert targets_by_src[1] == [0]


# ---------------------------------------------------------------------------
# Edge properties
# ---------------------------------------------------------------------------


def test_edge_properties_preserved_in_indices():
    nodes = _nodes(0, 1, 2)
    weight = pa.array([0.5, 1.5], type=pa.float32())
    rels = pa.table(
        {
            "source": pa.array([0, 1], type=pa.int64()),
            "destination": pa.array([1, 2], type=pa.int64()),
            "weight": weight,
        }
    )
    g = IcebugMemGraph.from_arrow_tables(nodes, rels)

    assert "weight" in g.indices.schema.names
    assert g.indices["weight"].to_pylist() == pytest.approx([0.5, 1.5])


def test_edge_properties_not_in_indptr():
    nodes = _nodes(0, 1)
    rels = pa.table(
        {
            "source": pa.array([0], type=pa.int64()),
            "destination": pa.array([1], type=pa.int64()),
            "weight": pa.array([1.0], type=pa.float32()),
        }
    )
    g = IcebugMemGraph.from_arrow_tables(nodes, rels)
    assert g.indptr.schema.names == ["ptr"]


# ---------------------------------------------------------------------------
# Column name aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("src_col", ["source", "src", "from"])
def test_source_column_aliases(src_col):
    nodes = _nodes(0, 1)
    rels = _rels([0], [1], src_col=src_col)
    g = IcebugMemGraph.from_arrow_tables(nodes, rels)
    assert len(g.indices) == 1


@pytest.mark.parametrize("dst_col", ["target", "destination", "dest", "to"])
def test_target_column_aliases(dst_col):
    nodes = _nodes(0, 1)
    rels = _rels([0], [1], dst_col=dst_col)
    g = IcebugMemGraph.from_arrow_tables(nodes, rels)
    assert len(g.indices) == 1


# ---------------------------------------------------------------------------
# Node tables passed through unchanged
# ---------------------------------------------------------------------------


def test_src_and_dest_tables_are_passed_through():
    src_nodes = pa.table({"id": pa.array([10, 20], type=pa.int64()), "label": pa.array(["a", "b"])})
    dst_nodes = pa.table({"id": pa.array([10, 20], type=pa.int64()), "label": pa.array(["c", "d"])})
    rels = _rels([10], [20])
    g = IcebugMemGraph.from_arrow_tables(src_nodes, rels, to_node_arrow_table=dst_nodes)
    assert g.src.equals(src_nodes)
    assert g.dest.equals(dst_nodes)


def test_omitting_to_node_table_uses_from_node_for_both():
    nodes = _nodes(0, 1)
    rels = _rels([0], [1])
    g = IcebugMemGraph.from_arrow_tables(nodes, rels)
    assert g.src.equals(nodes)
    assert g.dest.equals(nodes)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rel_table_with_fewer_than_two_columns_raises():
    nodes = _nodes(0, 1)
    bad_rels = pa.table({"source": pa.array([0], type=pa.int64())})
    with pytest.raises(ValueError, match="at least 2 columns"):
        IcebugMemGraph.from_arrow_tables(nodes, bad_rels)


def test_undirected_with_to_node_table_raises():
    nodes = _nodes(0, 1)
    rels = _rels([0], [1])
    with pytest.raises(ValueError, match="to_node_arrow_table must not be provided"):
        IcebugMemGraph.from_arrow_tables(nodes, rels, to_node_arrow_table=nodes, directed=False)


def test_column_names_with_spaces_are_handled():
    nodes = pa.table({"node id": pa.array([0, 1], type=pa.int64())})
    rels = pa.table({
        "source": pa.array([0], type=pa.int64()),
        "destination": pa.array([1], type=pa.int64()),
    })
    g = IcebugMemGraph.from_arrow_tables(nodes, rels)
    assert len(g.indices) == 1


def test_empty_edges_produces_zero_indptr():
    nodes = _nodes(0, 1, 2)
    rels = _rels([], [])
    g = IcebugMemGraph.from_arrow_tables(nodes, rels)

    assert len(g.indices) == 0
    assert g.indptr["ptr"].to_pylist() == [0, 0, 0, 0]
