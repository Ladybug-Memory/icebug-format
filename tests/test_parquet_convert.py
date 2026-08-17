"""Tests for vertex/edge Parquet pair -> icebug-disk CSR conversion."""

import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from icebug_format.convert_parquet import (
    convert_parquet_dir_to_csr,
    discover_graphs,
    parse_size_to_bytes,
    resolve_rel_column_names,
    sanitize_graph_name,
)

BACKENDS = ["pyarrow", "duckdb", "datafusion"]


def _write_graph(dir_: Path, name: str, vertex_ids, edges, prop=None):
    """Write an LDBC-style <name>-v.parquet / <name>-e.parquet pair."""
    v = pa.table({"id": pa.array(vertex_ids, type=pa.int64())})
    e_data = {
        "source": pa.array([e[0] for e in edges], type=pa.int64()),
        "target": pa.array([e[1] for e in edges], type=pa.int64()),
    }
    if prop is not None:
        e_data[prop[0]] = pa.array(prop[1])
    e = pa.table(e_data)
    pq.write_table(v, dir_ / f"{name}-v.parquet")
    pq.write_table(e, dir_ / f"{name}-e.parquet")


def _read_csr(out_dir: Path, name: str):
    indices = pq.read_table(out_dir / f"indices_{name}_rel.parquet")
    indptr = pq.read_table(out_dir / f"indptr_{name}_rel.parquet")
    nodes = pq.read_table(out_dir / f"nodes_{name}.parquet")
    return nodes, indices, indptr


@pytest.mark.parametrize("backend", BACKENDS)
def test_directed_sparse_ids(backend):
    """Sparse, unsorted vertex ids map to dense CSR indices by id order."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        # vertex ids sorted: 10, 20, 30, 40 -> csr 10=0, 20=1, 30=2, 40=3
        # edges: 20->30 (1->2), 10->40 (0->3), 10->20 (0->1), 99->30 (dropped)
        _write_graph(
            src, "g", [30, 10, 20, 40], [(20, 30), (10, 40), (10, 20), (99, 30)]
        )

        res = convert_parquet_dir_to_csr(
            src, output_dir=Path(tmp) / "out", backend=backend
        )
        assert [r["name"] for r in res] == ["g"]
        out_dir = Path(res[0]["output_dir"])

        nodes, indices, indptr = _read_csr(out_dir, "g")
        assert nodes["id"].to_pylist() == [10, 20, 30, 40]
        # (0,1), (0,3), (1,2) sorted by source then target
        assert indices["target"].to_pylist() == [1, 3, 2]
        assert indptr["ptr"].to_pylist() == [0, 2, 3, 3, 3]
        assert indices.schema.field("target").type == pa.uint64()
        assert indptr.schema.field("ptr").type == pa.uint64()


@pytest.mark.parametrize("backend", BACKENDS)
def test_dense_contiguous_ids(backend):
    """wiki-Talk-style dense ids use the identity mapping path."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        _write_graph(src, "g", [2, 0, 1], [(0, 1), (2, 0), (1, 1)])

        res = convert_parquet_dir_to_csr(
            src, output_dir=Path(tmp) / "out", backend=backend
        )
        out_dir = Path(res[0]["output_dir"])
        _, indices, indptr = _read_csr(out_dir, "g")

        assert indices["target"].to_pylist() == [1, 1, 0]  # 0->[1], 1->[1], 2->[0]
        assert indptr["ptr"].to_pylist() == [0, 1, 2, 3]


@pytest.mark.parametrize("backend", BACKENDS)
def test_reverse_edges(backend):
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        _write_graph(src, "g", [0, 1, 2], [(0, 1), (1, 2)])

        res = convert_parquet_dir_to_csr(
            src,
            output_dir=Path(tmp) / "out",
            backend=backend,
            add_reverse_edges=True,
        )
        out_dir = Path(res[0]["output_dir"])
        _, indices, indptr = _read_csr(out_dir, "g")

        # degrees: node0=1, node1=2, node2=1
        assert indptr["ptr"].to_pylist() == [0, 1, 3, 4]
        assert sorted(indices["target"].to_pylist()) == [0, 1, 1, 2]


@pytest.mark.parametrize("backend", BACKENDS)
def test_self_loops_appear_once_with_reverse_edges(backend):
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        _write_graph(src, "g", [0, 1], [(0, 0), (0, 1)])

        res = convert_parquet_dir_to_csr(
            src,
            output_dir=Path(tmp) / "out",
            backend=backend,
            add_reverse_edges=True,
        )
        out_dir = Path(res[0]["output_dir"])
        _, indices, _ = _read_csr(out_dir, "g")

        assert len(indices) == 3  # 0->0, 0->1, 1->0


@pytest.mark.parametrize("backend", BACKENDS)
def test_self_loops_preserved_directed(backend):
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        _write_graph(src, "g", [0, 1], [(0, 0), (0, 1)])

        res = convert_parquet_dir_to_csr(
            src, output_dir=Path(tmp) / "out", backend=backend
        )
        out_dir = Path(res[0]["output_dir"])
        _, indices, _ = _read_csr(out_dir, "g")

        assert sorted(indices["target"].to_pylist()) == [0, 1]


@pytest.mark.parametrize("backend", BACKENDS)
def test_edge_properties_preserved(backend):
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        _write_graph(src, "g", [0, 1], [(0, 1)], prop=("weight", [2.5]))

        res = convert_parquet_dir_to_csr(
            src, output_dir=Path(tmp) / "out", backend=backend
        )
        out_dir = Path(res[0]["output_dir"])
        _, indices, _ = _read_csr(out_dir, "g")

        assert "weight" in indices.schema.names
        assert indices["weight"].to_pylist() == [2.5]


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_edges(backend):
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        _write_graph(src, "g", [0, 1, 2], [])

        res = convert_parquet_dir_to_csr(
            src, output_dir=Path(tmp) / "out", backend=backend
        )
        out_dir = Path(res[0]["output_dir"])
        _, indices, indptr = _read_csr(out_dir, "g")

        assert len(indices) == 0
        assert indptr["ptr"].to_pylist() == [0, 0, 0, 0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_icebug_disk_metadata_written(backend):
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        _write_graph(src, "g", [0, 1], [(0, 1)])

        res = convert_parquet_dir_to_csr(
            src, output_dir=Path(tmp) / "out", backend=backend
        )
        out_dir = Path(res[0]["output_dir"])

        for f in (
            "nodes_g.parquet",
            "indices_g_rel.parquet",
            "indptr_g_rel.parquet",
        ):
            meta = pq.ParquetFile(out_dir / f).metadata.metadata or {}
            assert meta.get(b"icebug_disk_version") == b"v1"


@pytest.mark.parametrize("backend", BACKENDS)
def test_schema_cypher_generated(backend):
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        _write_graph(src, "g", [0, 1], [(0, 1)])

        res = convert_parquet_dir_to_csr(
            src, output_dir=Path(tmp) / "out", backend=backend
        )
        out_dir = Path(res[0]["output_dir"])
        schema = (out_dir / "schema.cypher").read_text()

        assert "CREATE NODE TABLE g(id INT64, PRIMARY KEY(id))" in schema
        assert "CREATE REL TABLE g_rel(FROM g TO g)" in schema
        assert "icebug-disk" in schema


def test_default_output_dir_csr_suffix_and_schema_names():
    """Default output dir is <source_dir>-csr; rel table name differs from node."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "graph500-24"
        src.mkdir()
        _write_graph(src, "graph500-24", [0, 1], [(0, 1)])

        res = convert_parquet_dir_to_csr(src, backend="pyarrow")
        assert Path(res[0]["output_dir"]) == Path(tmp) / "graph500-24-csr"

        schema = (Path(res[0]["output_dir"]) / "schema.cypher").read_text()
        assert "CREATE NODE TABLE graph500_24(id INT64, PRIMARY KEY(id))" in schema
        assert (
            "CREATE REL TABLE graph500_24_rel(FROM graph500_24 TO graph500_24)"
            in schema
        )
        # NODE and REL table names must not clash
        assert "CREATE REL TABLE graph500_24(" not in schema


@pytest.mark.parametrize("backend", ["duckdb", "datafusion"])
def test_memory_limit_accepted(backend):
    """SQL backends honor memory_limit; output is unchanged."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        _write_graph(src, "g", [0, 1, 2], [(0, 1), (2, 0), (1, 1)])

        res = convert_parquet_dir_to_csr(
            src,
            output_dir=Path(tmp) / "out",
            backend=backend,
            memory_limit="128MB",
        )
        out_dir = Path(res[0]["output_dir"])
        _, indices, indptr = _read_csr(out_dir, "g")

        assert indices["target"].to_pylist() == [1, 1, 0]
        assert indptr["ptr"].to_pylist() == [0, 1, 2, 3]


def test_parse_size_to_bytes():
    assert parse_size_to_bytes(None) is None
    assert parse_size_to_bytes("") is None
    assert parse_size_to_bytes("0") is None
    assert parse_size_to_bytes("0GB") is None
    assert parse_size_to_bytes("32") == 32 << 30  # bare number => GB
    assert parse_size_to_bytes("0.5") == int(0.5 * (1 << 30))
    assert parse_size_to_bytes("32GB") == 32 << 30
    assert parse_size_to_bytes("2GiB") == 2 << 30
    assert parse_size_to_bytes("1500MB") == 1500 << 20
    assert parse_size_to_bytes("1.5GB") == int(1.5 * (1 << 30))
    assert parse_size_to_bytes("10KB") == 10 << 10
    # percent of physical RAM
    pct = parse_size_to_bytes("50%")
    assert pct is not None and pct > 0
    with pytest.raises(ValueError):
        parse_size_to_bytes("lots")
    with pytest.raises(ValueError):
        parse_size_to_bytes("%")


def test_discover_graphs_ldbc_and_generic_layouts():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        _write_graph(src, "cit-Patents", [0, 1], [(0, 1)])
        generic = src / "generic"
        generic.mkdir()
        pq.write_table(
            pa.table({"id": pa.array([0, 1], type=pa.int64())}),
            generic / "vertex.parquet",
        )
        pq.write_table(
            pa.table(
                {
                    "source": pa.array([0], type=pa.int64()),
                    "target": pa.array([1], type=pa.int64()),
                }
            ),
            generic / "edge.parquet",
        )

        graphs = discover_graphs(src)
        assert {g["name"] for g in graphs} == {"cit_Patents"}
        # generic vertex.parquet / edge.parquet pair in its own directory
        assert {g["name"] for g in discover_graphs(generic)} == {"generic"}


def test_discover_graphs_missing_pair_ignored():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        pq.write_table(
            pa.table({"id": pa.array([0], type=pa.int64())}), src / "orphan-v.parquet"
        )
        assert discover_graphs(src) == []


def test_discover_graphs_no_parquet_raises():
    with (
        tempfile.TemporaryDirectory() as tmp,
        pytest.raises(ValueError, match="No vertex/edge Parquet pairs"),
    ):
        convert_parquet_dir_to_csr(tmp)


def test_resolve_rel_column_names():
    assert resolve_rel_column_names(["source", "target", "w"]) == ("source", "target")
    assert resolve_rel_column_names(["a", "b", "w"]) == ("a", "b")
    assert resolve_rel_column_names(["src", "dst"]) == ("src", "dst")


def test_sanitize_graph_name():
    assert sanitize_graph_name("cit-Patents") == "cit_Patents"
    assert sanitize_graph_name("2025.graph") == "_2025_graph"
    assert sanitize_graph_name("!!!") == "graph"


def test_graph_name_filter():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        _write_graph(src, "one", [0, 1], [(0, 1)])
        _write_graph(src, "two", [0, 1], [(1, 0)])

        res = convert_parquet_dir_to_csr(
            src, output_dir=Path(tmp) / "out", graph_name="two", backend="pyarrow"
        )
        assert [r["name"] for r in res] == ["two"]


def test_multi_graph_output_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        _write_graph(src, "one", [0, 1], [(0, 1)])
        _write_graph(src, "two", [0, 1], [(1, 0)])

        res = convert_parquet_dir_to_csr(
            src, output_dir=Path(tmp) / "out", backend="pyarrow"
        )
        assert {r["name"] for r in res} == {"one", "two"}
        dirs = {Path(r["output_dir"]).name for r in res}
        assert dirs == {"one", "two"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_backends_agree(backend):
    """Randomized sparse-id graph: all backends produce identical CSR output."""
    import random

    rng = random.Random(42)
    n_nodes = 200
    ids = rng.sample(range(10_000), n_nodes)  # sparse, unsorted
    edges = [(rng.choice(ids), rng.choice(ids)) for _ in range(1_000)]

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        _write_graph(src, "g", ids, edges)

        convert_parquet_dir_to_csr(
            src, output_dir=Path(tmp) / "base", backend="pyarrow"
        )
        convert_parquet_dir_to_csr(src, output_dir=Path(tmp) / "other", backend=backend)

        base = pq.read_table(Path(tmp) / "base" / "indices_g_rel.parquet")
        base_ptr = pq.read_table(Path(tmp) / "base" / "indptr_g_rel.parquet")
        other = pq.read_table(Path(tmp) / "other" / "indices_g_rel.parquet")
        other_ptr = pq.read_table(Path(tmp) / "other" / "indptr_g_rel.parquet")

        assert other.equals(base)
        assert other_ptr.equals(base_ptr)
