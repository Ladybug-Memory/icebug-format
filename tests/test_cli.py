"""Tests for the icebug-disk converter (create_csr_graph_to_duckdb)."""

import sys
import tempfile
from pathlib import Path

import duckdb
import pytest

from icebug_format.cli import create_csr_graph_to_duckdb, default_csr_table_name, main

_MEM = "1GB"


def _make_source_db(path: str, edges: list[tuple], self_loop: bool = False) -> None:
    """Create a minimal source DuckDB with nodes and edges tables."""
    con = duckdb.connect(path)
    con.execute("CREATE TABLE nodes (id BIGINT)")
    # Collect unique node IDs
    node_ids = sorted({n for e in edges for n in e})
    for nid in node_ids:
        con.execute(f"INSERT INTO nodes VALUES ({nid})")

    con.execute("CREATE TABLE edges (source BIGINT, target BIGINT)")
    for src, dst in edges:
        con.execute(f"INSERT INTO edges VALUES ({src}, {dst})")
    con.close()


def _make_hetero_source_db(path: str) -> None:
    """Create a source DuckDB with two node types and a heterogeneous edge table."""
    con = duckdb.connect(path)
    con.execute("CREATE TABLE nodes_user (id BIGINT)")
    con.execute("INSERT INTO nodes_user VALUES (0), (1)")
    con.execute("CREATE TABLE nodes_city (id BIGINT)")
    con.execute("INSERT INTO nodes_city VALUES (10), (11)")
    con.execute("CREATE TABLE edges_livesin (source BIGINT, target BIGINT)")
    con.execute("INSERT INTO edges_livesin VALUES (0, 10), (1, 11)")
    con.close()


def _make_multi_edge_source_db(path: str) -> None:
    """Create a source DB with two edge tables: edges_follows and edges_likes."""
    con = duckdb.connect(path)
    con.execute("CREATE TABLE nodes (id BIGINT)")
    for i in range(4):
        con.execute(f"INSERT INTO nodes VALUES ({i})")
    con.execute("CREATE TABLE edges_follows (source BIGINT, target BIGINT)")
    con.execute("INSERT INTO edges_follows VALUES (0,1),(1,2)")
    con.execute("CREATE TABLE edges_likes (source BIGINT, target BIGINT)")
    con.execute("INSERT INTO edges_likes VALUES (0,2),(1,3)")
    con.close()


def _make_multi_node_source_db(path: str) -> None:
    """Create a source DB with two node tables and one edge table."""
    con = duckdb.connect(path)
    con.execute("CREATE TABLE nodes_user (id BIGINT)")
    for i in range(3):
        con.execute(f"INSERT INTO nodes_user VALUES ({i})")
    con.execute("CREATE TABLE nodes_admin (id BIGINT)")
    for i in range(10, 12):
        con.execute(f"INSERT INTO nodes_admin VALUES ({i})")
    con.execute("CREATE TABLE edges (source BIGINT, target BIGINT)")
    con.execute("INSERT INTO edges VALUES (0,1),(1,2)")
    con.close()


def _parquet_dir(out_path: str) -> Path:
    """Return the parquet output directory for a given output_db_path."""
    p = Path(out_path)
    return p.parent / p.stem


# ---------------------------------------------------------------------------
# Directed graph
# ---------------------------------------------------------------------------


def test_directed_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        src = str(Path(tmpdir) / "src.duckdb")
        out = str(Path(tmpdir) / "out.duckdb")
        # 0 -> 1 -> 2
        _make_source_db(src, [(0, 1), (1, 2)])
        create_csr_graph_to_duckdb(src, out, add_reverse_edges=False, memory_limit=_MEM)

        con = duckdb.connect(out)
        indices = con.execute(
            "SELECT target FROM csr_graph_indices_edges ORDER BY rowid"
        ).fetchall()
        indptr = con.execute(
            "SELECT ptr FROM csr_graph_indptr_edges ORDER BY rowid"
        ).fetchall()
        metadata = con.execute(
            "SELECT CAST(value AS VARCHAR) FROM parquet_kv_metadata(?) "
            "WHERE key = 'icebug_disk_version'",
            [str(_parquet_dir(out) / "indices_edges.parquet")],
        ).fetchone()
        con.close()

        assert [r[0] for r in indices] == [1, 2]
        assert [r[0] for r in indptr] == [0, 1, 2, 2]
        assert metadata == ("v1",)


def test_csr_columns_are_uint64():
    with tempfile.TemporaryDirectory() as tmpdir:
        src = str(Path(tmpdir) / "src.duckdb")
        out = str(Path(tmpdir) / "out.duckdb")
        _make_source_db(src, [(0, 1), (1, 2)])
        create_csr_graph_to_duckdb(src, out, add_reverse_edges=False, memory_limit=_MEM)

        con = duckdb.connect(out)
        indices_desc = con.execute("DESCRIBE csr_graph_indices_edges").fetchall()
        indptr_desc = con.execute("DESCRIBE csr_graph_indptr_edges").fetchall()
        con.close()

        indices_target_type = next(col[1] for col in indices_desc if col[0] == "target")
        indptr_ptr_type = next(col[1] for col in indptr_desc if col[0] == "ptr")
        assert indices_target_type == "UBIGINT"
        assert indptr_ptr_type == "UBIGINT"


def test_directed_preserves_self_loops():
    """Self-loops must not be filtered from directed graphs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = str(Path(tmpdir) / "src.duckdb")
        out = str(Path(tmpdir) / "out.duckdb")
        # 0->0 (self-loop) + 0->1
        _make_source_db(src, [(0, 0), (0, 1)])
        create_csr_graph_to_duckdb(src, out, add_reverse_edges=False, memory_limit=_MEM)

        con = duckdb.connect(out)
        indices = con.execute(
            "SELECT target FROM csr_graph_indices_edges ORDER BY target"
        ).fetchall()
        con.close()

        assert sorted(r[0] for r in indices) == [0, 1]


# ---------------------------------------------------------------------------
# Reverse-edge expansion
# ---------------------------------------------------------------------------


def test_add_reverse_edges_adds_reverse_edges():
    with tempfile.TemporaryDirectory() as tmpdir:
        src = str(Path(tmpdir) / "src.duckdb")
        out = str(Path(tmpdir) / "out.duckdb")
        # 0 -- 1
        _make_source_db(src, [(0, 1)])
        create_csr_graph_to_duckdb(src, out, add_reverse_edges=True, memory_limit=_MEM)

        con = duckdb.connect(out)
        count = con.execute("SELECT COUNT(*) FROM csr_graph_indices_edges").fetchone()[
            0
        ]
        con.close()

        assert count == 2  # forward + reverse


def test_add_reverse_edges_self_loop_appears_once():
    """Self-loops must appear exactly once when reverse edges are added."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = str(Path(tmpdir) / "src.duckdb")
        out = str(Path(tmpdir) / "out.duckdb")
        # 0->0 self-loop + 0->1
        _make_source_db(src, [(0, 0), (0, 1)])
        create_csr_graph_to_duckdb(src, out, add_reverse_edges=True, memory_limit=_MEM)

        con = duckdb.connect(out)
        count = con.execute("SELECT COUNT(*) FROM csr_graph_indices_edges").fetchone()[
            0
        ]
        con.close()

        # Edges: 0--0 (once) + 0--1 (forward) + 1--0 (reverse) = 3
        assert count == 3


# ---------------------------------------------------------------------------
# Reverse-edge validation
# ---------------------------------------------------------------------------


def test_add_reverse_edges_heterogeneous_edges_raise():
    """Reverse-edge expansion requires homogeneous edge tables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = str(Path(tmpdir) / "src.duckdb")
        out = str(Path(tmpdir) / "out.duckdb")
        _make_hetero_source_db(src)

        schema_path = Path(tmpdir) / "schema.cypher"
        schema_path.write_text(
            "CREATE REL TABLE livesin(FROM user TO city) WITH (storage='x', format='icebug-disk');\n"
        )

        with pytest.raises(ValueError, match="same node table"):
            create_csr_graph_to_duckdb(
                src,
                out,
                add_reverse_edges=True,
                schema_path=str(schema_path),
                memory_limit=_MEM,
            )


# ---------------------------------------------------------------------------
# limit_rels
# ---------------------------------------------------------------------------


def test_limit_rels_caps_edge_count():
    """limit_rels restricts how many edges are stored in the CSR indices table."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = str(Path(tmpdir) / "src.duckdb")
        out = str(Path(tmpdir) / "out.duckdb")
        # 10 edges: 0->1, 1->2, ..., 9->10
        _make_source_db(src, [(i, i + 1) for i in range(10)])
        create_csr_graph_to_duckdb(
            src, out, add_reverse_edges=False, limit_rels=3, memory_limit=_MEM
        )

        con = duckdb.connect(out)
        count = con.execute("SELECT COUNT(*) FROM csr_graph_indices_edges").fetchone()[
            0
        ]
        con.close()

        assert count <= 3


def test_limit_rels_add_reverse_edges_adds_reverse_within_limit():
    """limit_rels applies before reverse edges are added."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = str(Path(tmpdir) / "src.duckdb")
        out = str(Path(tmpdir) / "out.duckdb")
        # 6 distinct edges: 0-1, 1-2, 2-3, 3-4, 4-5, 5-0
        _make_source_db(src, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
        create_csr_graph_to_duckdb(
            src, out, add_reverse_edges=True, limit_rels=2, memory_limit=_MEM
        )

        con = duckdb.connect(out)
        count = con.execute("SELECT COUNT(*) FROM csr_graph_indices_edges").fetchone()[
            0
        ]
        con.close()

        # 2 forward edges → 4 total (each gets a reverse)
        assert count == 4


# ---------------------------------------------------------------------------
# csr_table_name
# ---------------------------------------------------------------------------


def test_csr_table_name_prefix():
    """All output tables should be prefixed with the custom csr_table_name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = str(Path(tmpdir) / "src.duckdb")
        out = str(Path(tmpdir) / "out.duckdb")
        _make_source_db(src, [(0, 1), (1, 2)])
        create_csr_graph_to_duckdb(
            src,
            out,
            add_reverse_edges=False,
            csr_table_name="mygraph",
            memory_limit=_MEM,
        )

        con = duckdb.connect(out)
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        con.close()

        assert "mygraph_nodes" in tables
        assert "mygraph_indices_edges" in tables
        assert "mygraph_indptr_edges" in tables
        # Default prefix must NOT appear
        assert "csr_graph_indices_edges" not in tables


# ---------------------------------------------------------------------------
# node_table / edge_table
# ---------------------------------------------------------------------------


def test_node_table_selects_single_table():
    """node_table restricts processing to exactly one node table."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = str(Path(tmpdir) / "src.duckdb")
        out = str(Path(tmpdir) / "out.duckdb")
        _make_multi_node_source_db(src)
        create_csr_graph_to_duckdb(
            src,
            out,
            add_reverse_edges=False,
            node_table="nodes_user",
            memory_limit=_MEM,
        )

        con = duckdb.connect(out)
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        con.close()

        assert "csr_graph_nodes_user" in tables
        assert "csr_graph_nodes_admin" not in tables


def test_edge_table_selects_single_table():
    """edge_table restricts processing to exactly one edge table."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = str(Path(tmpdir) / "src.duckdb")
        out = str(Path(tmpdir) / "out.duckdb")
        _make_multi_edge_source_db(src)
        create_csr_graph_to_duckdb(
            src,
            out,
            add_reverse_edges=False,
            edge_table="edges_follows",
            memory_limit=_MEM,
        )

        con = duckdb.connect(out)
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        con.close()

        assert "csr_graph_indices_follows" in tables
        assert "csr_graph_indices_likes" not in tables


def test_edge_table_not_found_raises():
    """Specifying a non-existent edge_table should raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = str(Path(tmpdir) / "src.duckdb")
        out = str(Path(tmpdir) / "out.duckdb")
        _make_source_db(src, [(0, 1)])
        with pytest.raises(ValueError, match="No edge tables found"):
            create_csr_graph_to_duckdb(
                src,
                out,
                add_reverse_edges=False,
                edge_table="edges_nonexistent",
                memory_limit=_MEM,
            )


# ---------------------------------------------------------------------------
# schema_path
# ---------------------------------------------------------------------------


def test_schema_path_maps_from_to_node_types():
    """schema_path controls which node types appear in FROM/TO of the output schema.cypher."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = str(Path(tmpdir) / "src.duckdb")
        out = str(Path(tmpdir) / "out.duckdb")
        _make_hetero_source_db(src)

        schema_path = Path(tmpdir) / "in_schema.cypher"
        schema_path.write_text(
            "CREATE REL TABLE livesin(FROM user TO city) WITH (storage='x', format='icebug-disk');\n"
        )

        create_csr_graph_to_duckdb(
            src,
            out,
            add_reverse_edges=False,
            schema_path=str(schema_path),
            memory_limit=_MEM,
        )

        out_schema = (_parquet_dir(out) / "schema.cypher").read_text()
        assert "FROM user TO city" in out_schema


# ---------------------------------------------------------------------------
# storage_path
# ---------------------------------------------------------------------------


def test_storage_path_appears_in_schema_cypher():
    """Custom storage_path should appear in the WITH clause of the output schema.cypher."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = str(Path(tmpdir) / "src.duckdb")
        out = str(Path(tmpdir) / "out.duckdb")
        _make_source_db(src, [(0, 1), (1, 2)])

        create_csr_graph_to_duckdb(
            src,
            out,
            add_reverse_edges=False,
            storage_path="./my_custom_store",
            memory_limit=_MEM,
        )

        out_schema = (_parquet_dir(out) / "schema.cypher").read_text()
        assert "./my_custom_store" in out_schema


def test_storage_path_default_uses_output_stem():
    """When storage_path is omitted the output DB stem is used as the default."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = str(Path(tmpdir) / "src.duckdb")
        out = str(Path(tmpdir) / "out.duckdb")
        _make_source_db(src, [(0, 1)])

        create_csr_graph_to_duckdb(src, out, add_reverse_edges=False, memory_limit=_MEM)

        out_schema = (_parquet_dir(out) / "schema.cypher").read_text()
        # Default storage_path is "./out" (stem of out.duckdb)
        assert "./out" in out_schema


def test_default_csr_table_name_is_sql_safe():
    assert default_csr_table_name("wikidata-20250625") == "wikidata_20250625"
    assert default_csr_table_name("2025.graph") == "_2025_graph"
    assert default_csr_table_name("---") == "csr_graph"


def test_cli_default_csr_table_allows_dashed_source_db(tmp_path, monkeypatch):
    source_db = tmp_path / "wikidata-20250625.db"
    output_db = tmp_path / "wikidata-20250625_csr.duckdb"

    con = duckdb.connect(str(source_db))
    try:
        con.execute("CREATE TABLE nodes (id BIGINT);")
        con.execute("INSERT INTO nodes VALUES (1), (2), (3);")
        con.execute("CREATE TABLE edges (source BIGINT, target BIGINT);")
        con.execute("INSERT INTO edges VALUES (1, 2), (2, 3);")
    finally:
        con.close()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "icebug-format",
            "--source-db",
            str(source_db),
            "--memory-limit",
            "1GB",
        ],
    )

    main()

    con = duckdb.connect(str(output_db), read_only=True)
    try:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        assert "wikidata_20250625_nodes" in tables
        assert "wikidata_20250625_indices_edges" in tables
        assert "wikidata_20250625_indptr_edges" in tables
        assert con.execute(
            "SELECT ptr FROM wikidata_20250625_indptr_edges"
        ).fetchall() == [(0,), (1,), (2,), (2,)]
    finally:
        con.close()
