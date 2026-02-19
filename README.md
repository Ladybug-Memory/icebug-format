# Icebug Format

> **Note**: This project was formerly called **graph-std**.

Icebug is a standardized graph format designed for efficient graph data interchange. It comes in two formats:

- **icebug-disk**: Parquet-based format for object storage
- **icebug-memory**: Apache Arrow-based format for in-memory processing

This project provides tools to convert graph data from simple DuckDB databases or Parquet files containing `nodes_*` and `edges_*` tables, along with a `schema.cypher` file, into standardized graph formats for efficient processing.

## Sample Usage

```bash
uv run icebug-format.py \
--source-db karate/karate_random.duckdb \
--output-db karate/karate_csr.duckdb \
--csr-table karate \
--schema karate/karate_csr/schema.cypher
```

This will create a CSR representation with multiple tables depending on the number of node and edge types:

- `{table_name}_indptr_{edge_name}`: Array of size N+1 for row pointers (one per edge table)
- `{table_name}_indices_{edge_name}`: Array of size E containing column indices (one per edge table)
- `{table_name}_nodes_{node_name}`: Original nodes table with node attributes (one per node table)
- `{table_name}_mapping_{node_name}`: Maps original node IDs to contiguous indices (one per node table)
- `{table_name}_metadata`: Global graph metadata (node count, edge count, directed flag)
- `schema.cypher`: A cypher schema that a graph database can mount without ingesting

## More information about Icebug and Apache GraphAR

[Blog Post](https://adsharma.github.io/graph-archiving/)

## Recreating demo-db/icebug-disk

Start from a simple demo-db.duckdb that looks like this

```
Querying database: demo-db.duckdb
================================

--- Table: edges_follows ---
┌────────┬────────┬───────┐
│ source │ target │ since │
│ int32  │ int32  │ int32 │
├────────┼────────┼───────┤
│    100 │    250 │  2020 │
│    300 │     75 │  2022 │
│    250 │    300 │  2021 │
│    100 │    300 │  2020 │
└────────┴────────┴───────┘
================================

--- Table: edges_livesin ---
┌────────┬────────┐
│ source │ target │
│ int32  │ int32  │
├────────┼────────┤
│    100 │    700 │
│    250 │    700 │
│    300 │    600 │
│     75 │    500 │
└────────┴────────┘
================================

--- Table: nodes_city ---
┌───────┬───────────┬────────────┐
│  id   │   name    │ population │
│ int32 │  varchar  │   int64    │
├───────┼───────────┼────────────┤
│   500 │ Guelph    │      75000 │
│   600 │ Kitchener │     200000 │
│   700 │ Waterloo  │     150000 │
└───────┴───────────┴────────────┘
================================

--- Table: nodes_user ---
┌───────┬─────────┬───────┐
│  id   │  name   │  age  │
│ int32 │ varchar │ int64 │
├───────┼─────────┼───────┤
│   100 │ Adam    │    30 │
│   250 │ Karissa │    40 │
│    75 │ Noura   │    25 │
│   300 │ Zhang   │    50 │
└───────┴─────────┴───────┘
================================

--- Schema: schema.cypher --
CREATE NODE TABLE User(id INT64, name STRING, age INT64, PRIMARY KEY (id));
CREATE NODE TABLE City(id INT64, name STRING, population INT64, PRIMARY KEY (id));
CREATE REL TABLE Follows(FROM User TO User, since INT64);
CREATE REL TABLE LivesIn(FROM User TO City);
```

and run:

```
uv run icebug-format.py \
--directed \
--source-db demo-db.duckdb \
--output-db demo-db_csr.duckdb \
--csr-table demo \
--schema demo-db/schema.cypher
```

You'll get a demo-db_csr.duckdb AND the object storage ready representation aka icebug-disk.

## Verification

You can verify that the conversion went ok by running `scan.py`. It's also a good way to understand the icebug-disk format.

```
uv run scan.py --input demo-db_csr --prefix demo
Metadata: 7 nodes, 8 edges, directed=True

Node Tables:

Table: demo_nodes_user
(100, 'Adam', 30)
(250, 'Karissa', 40)
(75, 'Noura', 25)
(300, 'Zhang', 50)

Table: demo_nodes_city
(500, 'Guelph', 75000)
(600, 'Kitchener', 200000)
(700, 'Waterloo', 150000)

Edge Tables (reconstructed from CSR):

Table: follows (FROM user TO user)
(100, 250, 2020)
(100, 300, 2020)
(250, 300, 2021)
(300, 75, 2022)

Table: livesin (FROM user TO city)
(75, 500)
(100, 700)
(250, 700)
(300, 600)
```
