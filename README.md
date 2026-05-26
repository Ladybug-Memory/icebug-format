# Icebug Format

Icebug is a standardized graph format designed for efficient graph data interchange. It comes in two flavours:

| Format | Storage | Use case |
|---|---|---|
| **icebug-disk** | Parquet files | Object storage, persistence |
| **icebug-memory** | Apache Arrow tables | In-process, zero-copy access |

Both represent *directed* graphs in [CSR (Compressed Sparse Row)](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) format, which enables fast adjacency-list traversal.

---

## icebug-disk v1

### CLI

Convert a DuckDB source database containing `nodes_*` / `edges_*` tables into Parquet files and a `schema.cypher` that a graph database can mount directly:

```bash
uv run icebug-format \
  --source-db examples/karate/duckdb/karate_random.duckdb \
  --schema examples/karate/duckdb/schema.cypher      // input schema for rel tables
```

### Output structure

For each node table `nodes_<name>` and edge table `edges_<name>`, the following files/tables are produced:

| Name | Description |
|---|---|
| `nodes_<name>.parquet` | Original node table with attributes |
| `indices_<name>.parquet` | Target node for each edge, sorted by source (size E) |
| `indptr_<name>.parquet` | Row-pointer array of size N+1 |
| `schema.cypher` | Cypher schema for mounting in a graph database |

NOTE: Each parquet file stores `icebug_disk_version` in its metadata

### Example

Starting from a `demo-db.duckdb` with `nodes_user`, `nodes_city`, `edges_follows`, and `edges_livesin` tables:

```bash
uv run icebug-format \
  --directed \
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

# Directed or undirected homogeneous graph (same node type on both ends)
graph: IcebugMemGraph = IcebugMemGraph.from_arrow_tables(
    from_node_arrow_table=users,   # pa.Table, first column is the primary key
    rel_arrow_table=follows,       # pa.Table with 'source' and 'target' columns
    undirected=True,                 # undirected=True for undirected (to_node_arrow_table must be omitted)
)

# Node tables are passed through unchanged
graph.src    # pa.Table — source nodes
graph.dest   # pa.Table — destination nodes

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

Set `undirected=True` to automatically add reverse edges (undirected graph).  For undirected graphs, `to_node_arrow_table` must be omitted; the same node table is used for both sides of every edge.

## Caveats

- icebug-format will always output a directed graph
- If you want an undirected graph to be converted, pass undirected=True to the CLI or Python API, and the reverse edges will be added automatically. But do note that undirected graphs are supported for rel tables with same node type on both ends only

---

## Further reading

[Blog post: Graph Archiving with Apache GraphAR](https://adsharma.github.io/graph-archiving/)

