# Icebug Format CLI Specification

## Purpose

`icebug-format` converts a property graph stored as relational node and edge
tables into an Icebug disk layout backed by CSR (Compressed Sparse Row) adjacency
tables.

The conversion is intended to make graph data cheap to scan from object storage
or local files without ingesting it into a graph database first. It preserves the
original node records and edge properties, creates dense integer node mappings
for efficient adjacency traversal, emits CSR tables per edge type, and writes a
`schema.cypher` file that describes how a graph engine can mount the generated
Parquet files.

Although the current implementation uses DuckDB for SQL execution, sorting,
joining, table introspection, and Parquet export, the required behavior is not
DuckDB-specific. A different backend can reimplement the functionality by
providing equivalent table discovery, relational joins, deterministic ordering,
table creation, and Parquet writing.

## Command

```bash
icebug-format \
  --source-db SOURCE_DB \
  [--output-db OUTPUT_DB] \
  [--csr-table CSR_PREFIX] \
  [--node-table NODE_TABLE] \
  [--edge-table EDGE_TABLE] \
  [--schema SCHEMA_CYPHER] \
  [--storage STORAGE_PATH] \
  [--directed] \
  [--test] \
  [--limit N] \
  [--memory-limit LIMIT]
```

The source is a database containing node and edge tables. The output is both:

- a database at `OUTPUT_DB` containing generated CSR and metadata tables
- a sibling directory named after `OUTPUT_DB` without its extension, containing
  Parquet exports of every generated table plus `schema.cypher`

`--source-db` is required. `--output-db` is optional; when omitted, it defaults
to a DuckDB file next to `SOURCE_DB` using the source stem plus `_csr.duckdb`.
For example, `--source-db path/to/karate.duckdb` defaults `--output-db` to:

```text
path/to/karate_csr.duckdb
```

`--csr-table` is optional; when omitted, it defaults to the stem of
`SOURCE_DB`. For example, `--source-db path/to/karate.duckdb` defaults
`--csr-table` to:

```text
karate
```

## Expected Input Format

### Table Discovery

Input tables are discovered by name:

- node tables have names beginning with `nodes`
- edge tables have names beginning with `edges`

Examples:

- `nodes`
- `nodes_user`
- `nodes_city`
- `edges`
- `edges_follows`
- `edges_livesin`

If `--node-table` is supplied, only that exact node table is used. If the named
table is not present among discovered node tables, no node table is selected.

If `--edge-table` is supplied, only that exact edge table is used. If no edge
tables remain after discovery/filtering, conversion fails.

### Node Tables

Each node table is a relational table whose first column is treated as the node
primary key. The first column is also used as the stable ordering key for dense
CSR ID assignment.

Required properties:

- The first column must uniquely identify nodes in that table.
- Edge `source` and `target` values must refer to first-column values in the
  appropriate node tables.
- All node columns are preserved in the output node table.

For a node table named `nodes_user`, the node type name is `user`. For a table
named `nodes`, the node type name is `nodes`.

### Edge Tables

Each edge table must contain:

- `source`: original source node ID
- `target`: original target node ID

Any other columns are treated as edge properties and are preserved in the CSR
indices table.

Self-loops are excluded. Any row where `source = target` is not emitted.

For an edge table named `edges_follows`, the edge type name is `follows`. For a
table named `edges`, the edge type name is `edges`.

### Optional Schema Input

`--schema` points to a Cypher schema file used only to determine edge endpoints.
The converter parses relationship definitions shaped like:

```cypher
CREATE REL TABLE Follows(FROM User TO User, since INT64);
CREATE REL TABLE LivesIn(FROM User TO City);
```

Identifier matching is case-insensitive after lowercasing. Backtick-quoted
identifiers are accepted for the relationship name and endpoint node names.

The parsed relationship map is:

```text
edge type -> (source node type, destination node type)
```

For example:

```text
follows -> (user, user)
livesin -> (user, city)
```

If no schema is supplied, or an edge type is missing from the schema, that edge
type falls back to the first selected node table for both source and target
mapping.

## Functionality

### Output Reset

Before conversion, the output database is opened and all existing tables in it
are dropped. The generated output is therefore a full replacement, not an
incremental update.

### Node Copy and Dense Mapping

For each selected node table `NT`:

1. Copy all rows and columns to:

   ```text
   {CSR_PREFIX}_{NT}
   ```

2. Sort copied rows by the first column of `NT`.

3. Create a dense mapping table:

   ```text
   {CSR_PREFIX}_mapping_{node_type}
   ```

   with columns:

   ```text
   csr_index BIGINT-like integer
   original_node_id same logical type as node primary key
   ```

4. Assign `csr_index` values using zero-based row numbers ordered by the node
   primary key:

   ```text
   0, 1, 2, ...
   ```

This mapping is per node type. Node IDs do not need to be globally dense or
globally unique across node tables.

### Edge Endpoint Mapping

For each selected edge table `ET`:

1. Derive `edge_type` from the table name.
2. Determine source and destination node types from `--schema`, if available.
3. Select the source and destination mapping tables for those node types.
4. Join edge rows to mapping tables:

   ```text
   edge.source -> source_mapping.original_node_id
   edge.target -> destination_mapping.original_node_id
   ```

5. Convert original IDs into:

   ```text
   csr_source
   csr_target
   ```

Only edges whose endpoints both exist in the selected node mappings are emitted,
because the conversion uses inner-join semantics.

### Directed and Undirected Modes

With `--directed`, each non-self-loop input edge produces one output edge:

```text
source -> target
```

Without `--directed`, the graph is treated as undirected. Each non-self-loop
input edge produces two output edges:

```text
source -> target
target -> source
```

Edge properties are copied onto both directions in undirected mode.

### Test Limit

`--test` enables limiting. The effective limit is `--limit`; its default is
`50000`.

When multiple edge tables are selected, the per-table limit is:

```text
floor(limit / number_of_edge_tables)
```

The limit is applied before undirected edge duplication. In undirected mode, a
per-table limit of `L` can therefore emit up to `2L` rows for that edge type.

The implementation does not define a deterministic ordering before applying the
limit, so callers should treat test-mode sampling as backend-dependent unless
their backend explicitly imposes an order.

## Output Format

### Output Database Tables

The output database contains generated tables with names based on `CSR_PREFIX`.

For every node table:

```text
{CSR_PREFIX}_{node_table}
{CSR_PREFIX}_mapping_{node_type}
```

For every edge table:

```text
{CSR_PREFIX}_indptr_{edge_type}
{CSR_PREFIX}_indices_{edge_type}
```

Once per conversion:

```text
{CSR_PREFIX}_metadata
```

Temporary relation tables may be used internally, but must not remain in the
final output.

### Node Tables

Generated node tables preserve the source node table columns and values:

```text
{CSR_PREFIX}_{node_table}
```

Rows are ordered by the source node table's first column.

### Mapping Tables

Mapping tables are named:

```text
{CSR_PREFIX}_mapping_{node_type}
```

Columns:

```text
csr_index
original_node_id
```

Rows are ordered by `csr_index`.

### CSR Indptr Tables

For each edge type, the indptr table is named:

```text
{CSR_PREFIX}_indptr_{edge_type}
```

Columns:

```text
ptr INT64-like integer
```

The table contains `N + 1` rows, where `N` is the number of source nodes for
that edge type.

`ptr[i]` is the starting row offset in the matching indices table for source
node `i`. `ptr[i + 1] - ptr[i]` is the out-degree of source node `i`.

The first row is always `0`. The final row is the number of emitted edges for
that edge type.

### CSR Indices Tables

For each edge type, the indices table is named:

```text
{CSR_PREFIX}_indices_{edge_type}
```

Columns:

```text
target
{edge property columns...}
```

`target` is the dense CSR index of the destination node. Edge property columns
are all columns from the source edge table except `source` and `target`,
preserved with their original values and logical types.

Rows are ordered by:

```text
csr_source, csr_target
```

The `csr_source` column is not stored in the final indices table. It is encoded
by the offsets in the matching indptr table.

### Metadata Table

The metadata table is named:

```text
{CSR_PREFIX}_metadata
```

Columns:

```text
n_nodes
n_edges
directed
```

`n_nodes` is the sum of selected node table row counts.

`n_edges` is the sum of emitted rows across all generated indices tables. In
undirected mode this includes duplicated reverse edges.

`directed` records whether `--directed` was supplied.

### Parquet Directory

For `--output-db path/to/name.duckdb`, the Parquet directory is:

```text
path/to/name/
```

Every final output database table is exported to one Parquet file in that
directory:

```text
{lowercase_table_name}.parquet
```

The directory also contains:

```text
schema.cypher
```

Old `schema.sql` and `load.sql` files in the directory are removed if present.

### Generated schema.cypher

The generated schema contains one `CREATE NODE TABLE` statement per selected
node table and one `CREATE REL TABLE` statement per selected edge table.

Node table display names:

```text
nodes       -> nodes
nodes_user  -> user
nodes_city  -> city
```

Edge table display names:

```text
edges          -> edges
edges_follows  -> follows
edges_livesin  -> livesin
```

Node columns are inferred from the generated node tables. The first column is
declared as the primary key.

Relationship endpoints are taken from the parsed input schema when available.
If endpoint information is unavailable, both endpoints use the first selected
node table's display name.

Relationship properties are inferred from the generated indices table, excluding
the `target` column.

Every generated table statement includes a LadybugDB-specific Cypher extension:

```cypher
WITH (storage = 'STORAGE_PATH')
```

This extension tells the columnar database to query the referenced
Parquet files in-place instead of ingesting the data into an internal
row store. It is not standard Cypher, and it is not relevant to
row-oriented graph databases such as Neo4j, which typically load
data into their own storage engine before querying it.

`STORAGE_PATH` is `--storage` if supplied. Otherwise it defaults to:

```text
./{output_db_stem}/{CSR_PREFIX}
```

Future enhancement: a `schema.gql` file may be added if GQL becomes more widely
used. For now, `schema.cypher` is the required schema artifact.

### Type Mapping for schema.cypher

When generating Cypher types, backend column types should be mapped as follows:

```text
BIGINT      -> INT64
INTEGER     -> INT32
SMALLINT    -> INT16
TINYINT     -> INT8
HUGEINT     -> INT128
UBIGINT     -> UINT64
UINTEGER    -> UINT32
USMALLINT   -> UINT16
UTINYINT    -> UINT8
DOUBLE      -> DOUBLE
FLOAT       -> FLOAT
REAL        -> FLOAT
BOOLEAN     -> BOOL
VARCHAR     -> STRING
TEXT        -> STRING
CHAR        -> STRING
DATE        -> DATE
TIMESTAMP   -> TIMESTAMP
TIME        -> TIME
BLOB        -> BLOB
```

For parameterized types such as `DECIMAL(10,2)`, use the base type before the
first `(`. Unknown types are emitted as `STRING`.

## Backend Requirements for Reimplementation

A non-DuckDB backend must provide equivalent behavior for:

- listing source tables by name
- reading table schemas and column types
- copying node tables while ordering by the first column
- assigning zero-based dense IDs with deterministic ordering
- joining edge tables to source and destination mapping tables
- filtering self-loops
- duplicating edges for undirected mode
- preserving edge property columns and values
- grouping emitted edges by `csr_source` to compute degrees
- creating cumulative CSR offsets for all source node IDs from `0` to `N - 1`,
  including nodes with degree zero
- sorting final indices rows by `csr_source, csr_target`
- exporting all generated tables as Parquet
- generating the Cypher schema with the naming and type rules above

The most important compatibility requirements are deterministic node mapping,
correct CSR offsets, matching table/file names, and preservation of node and edge
properties.
