# Icebug Format

Icebug is a standardized graph format designed for efficient graph data interchange. It comes in two flavours:

| Format | Storage | Use case |
|---|---|---|
| **icebug-disk** | Parquet files | Object storage, persistence |
| **icebug-memory** | Apache Arrow tables | In-process, zero-copy access |

Both represent *directed* graphs in [CSR (Compressed Sparse Row)](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) format, which enables fast adjacency-list traversal.

> **icebug-disk is Parquet + CSR — not a proprietary format.**
>
> An icebug-disk graph is stored as ordinary [Apache Parquet](https://parquet.apache.org/) files whose columns form a standard CSR array: a dense `target` column (the per-edge CSR `indices` array) plus a row-pointer `ptr` column (the CSR `indptr` array). Parquet is an open, cross-language industry standard for columnar storage, and CSR is an equally standard sparse-matrix layout used by scipy, NetworkX, and the broader graph/ML ecosystem.
>
> Nothing about icebug-disk locks you into a specific implementation. `ladybugdb` is the only known implementation *today*, and it consumes the format by reading these standard parquet files; but the on-disk representation itself is plain parquet structured as CSR. Any tool that can read parquet can consume an icebug-disk graph directly, and any workflow that can produce a CSR sparse adjacency can write one. If and when other graph databases adopt the same parquet layout, you can move between implementations without re-converting.

---

## icebug-disk v1

Two command paths convert graph data into icebug-disk. They share the same output layout (a directory of Parquet files plus a `schema.cypher`) and differ only in the source format:

```
┌──────────────┐   ┌─────────────────┐   ┌───────────────────────────┐
│  duckdb      │──▶│   duckdb-csr    │──▶│    icebug-disk parquet    │
│  nodes_*/    │   │  (intermediate  │   │  nodes / indices / indptr │
│  edges_*     │   │     .duckdb)    │   │  + schema.cypher          │
└──────────────┘   └─────────────────┘   └───────────────────────────┘
        --source-db

┌──────────────┐   ┌───────────────────────────┐
│ parquet      │──▶│    icebug-disk parquet    │
│ <name>-v /   │   │  nodes / indices / indptr │
│ <name>-e     │   │  + schema.cypher          │
└──────────────┘   └───────────────────────────┘
        --source-dir
```

### Path 1: DuckDB → icebug-disk (`--source-db`)

Reads a DuckDB database containing `nodes_*` / `edges_*` tables. The converter first builds an intermediate CSR DuckDB database (`<stem>_csr.duckdb`, the "duckdb-csr" stage), then exports it to the icebug-disk Parquet directory. A `schema.cypher` is generated that a graph database can mount directly:

```bash
uv run icebug-format \
  --source-db examples/karate/duckdb/karate_random.duckdb \
  --schema examples/karate/duckdb/schema.cypher      # input schema for rel tables
```

When `--output-db` is omitted it is derived from the source stem (e.g. `karate_random_csr.duckdb`), and the Parquet directory is created next to it.

### Path 2: Parquet → icebug-disk (`--source-dir`)

Benchmark datasets such as the ones under `ldbc/` ship as a vertex table
(`<name>-v.parquet`, first column is the node id) and an edge table
(`<name>-e.parquet` with `source`/`target` columns), or as plain
`vertex.parquet` / `edge.parquet`. Pass the directory to `--source-dir`:

```bash
uv run icebug-format --source-dir ldbc/cit-Patents --backend pyarrow
```

There is no intermediate DuckDB file — the output is the icebug-disk Parquet directory + `schema.cypher` directly. With multiple graphs in one source directory each graph gets its own subdirectory (single graph writes to the `--output-dir` root). The output directory is named after the source dir stem (e.g. `ldbc/cit-Patents-csr`) unless overridden with `--output-dir`.

Three interchangeable conversion backends are available (select with
`--backend`, default `auto`):

| Backend | Extra | Characteristics |
|---|---|---|
| `pyarrow` | (none) | Pure in-memory PyArrow pipeline: dense-id lookup tables + radix sort. Fastest; RSS scales with edge count. |
| `duckdb` | `convert` | DuckDB SQL engine with external sort (bounded by `--memory-limit`). |
| `datafusion` | `convert-datafusion` | Apache DataFusion SQL engine; streams results with `execute_stream()`. Lowest RSS for large graphs. |

### Output structure

Both paths produce the same ice-disk layout. For each node table `nodes_<name>` and edge table `edges_<name>` (DuckDB) or each vertex/edge pair `<name>` (Parquet):

| Name | Description |
|---|---|
| `nodes_<name>.parquet` | Original node table with attributes |
| `indices_<name>.parquet` | Target node for each edge, sorted by source (the CSR `indices` array; size E) |
| `indptr_<name>.parquet` | Row-pointer array of size N+1 (the CSR `indptr` array) |
| `schema.cypher` | Cypher schema for mounting in a graph database |

NOTE: Each parquet file stores `icebug_disk_version` in its metadata.

### Example (DuckDB path)

Starting from a `demo-db.duckdb` with `nodes_user`, `nodes_city`, `edges_follows`, and `edges_livesin` tables:

```bash
uv run icebug-format \
  --source-db demo-db.duckdb \
  --schema demo-db/schema.cypher
```

Verify the result with `test_csr_duckdb.py`:

```bash
uv run ./icebug-format/test_csr_duckdb.py --input demo-db_csr
```

```
Metadata: 7 nodes, 8 edges, directed=True

Node Tables:
Table: demo_nodes_user
(100, 'Adam', 30) ...

Edge Tables (reconstructed from CSR):
Table: follows (FROM user TO user)
(100, 250, 2020) ...
```

### Example (Parquet path)

Starting from an LDBC-style pair (`graph-v.parquet` + `graph-e.parquet`):

```bash
uv run icebug-format --source-dir ldbc/cit-Patents --backend pyarrow
```

produces `nodes_cit_Patents.parquet`, `indices_cit_Patents_rel.parquet`,
`indptr_cit_Patents_rel.parquet`, and `schema.cypher` in the output directory
(here REL table name is `<graph>_rel` so it stays distinct from the NODE table,
both saved from the CSR filenames).

---

## icebug-memory v1

### Python API

Convert Arrow tables directly into an in-memory CSR graph

```python
from icebug_format import IcebugMemGraph

# Directed heterogeneous graph (different node types on each end)
graph: IcebugMemGraph = IcebugMemGraph.from_arrow_tables(
    from_node_arrow_table=users,   # pa.Table, first column is the primary key
    rel_arrow_table=livesin,       # pa.Table with 'source' and 'target' columns
    to_node_arrow_table=cities,    # pa.Table, first column is the primary key
)

# Directed graph, or homogeneous graph with reverse edges added
graph: IcebugMemGraph = IcebugMemGraph.from_arrow_tables(
    from_node_arrow_table=users,   # pa.Table, first column is the primary key
    rel_arrow_table=follows,       # pa.Table with 'source' and 'target' columns
    add_reverse_edges=True,        # to_node_arrow_table must be omitted
)

# Node tables are passed through unchanged
graph.src    # pa.Table — source nodes
graph.dest  # pa.Table — destination nodes

# CSR adjacency structure
graph.indices  # pa.Table — 'target' column (+ any edge properties), sorted by source
graph.indptr   # pa.Table — 'ptr' column of length len(src) + 1
```

The `rel_arrow_table` source and target columns are resolved by name in priority order, with a positional fallback:

| Role | Accepted names (in order) | Fallback |
|---|---|---|
| Source | `source`, `src`, `from` | 0th column |
| Target | `target`, `destination`, `dest`, `to` | 1st column |

Any remaining columns are preserved as edge properties in `graph.indices`.

Use `--add-reverse-edges` in the CLI, or `add_reverse_edges=True` in the Python API, to emit a symmetric adjacency by adding reverse edges. For reverse-edge expansion, `to_node_arrow_table` must be omitted; the same node table is used for both sides of every edge.

## Caveats

- icebug-format will always output a directed graph
- If an algorithm needs symmetric adjacency, pass `--add-reverse-edges` to the CLI or `add_reverse_edges=True` to the Python API. Reverse edges will be added automatically. Reverse-edge expansion is supported only for rel tables with the same node type on both ends.
- Reverse-edge expansion is all or nothing for a conversion. If your graph mixes edge types that should be symmetric, such as `friends`, with edge types that should stay directed, such as `follows`, run separate conversions or add reverse edges before calling icebug-format; `--add-reverse-edges` cannot be applied selectively per edge type.

---

## Further reading

[Blog post: Graph Archiving with Apache GraphAR](https://adsharma.github.io/graph-archiving/)
