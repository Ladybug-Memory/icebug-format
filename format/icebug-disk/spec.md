# Icebug-Disk graph specification

The Icebug-Disk graph format is designed to store graph data on disk efficiently, enabling high read performance for large graphs.

## Versioning

### Version 1

Initial version of the format. Uses compressed sparse row-based (CSR) adjacency list/join indices for efficient graph traversals.

## Architecture Diagram

![Icebug-Disk Architecture Overview](architecture_overview.svg)

## Components

### Initialization files

Initialization files could be format version, schema definitions, table creation statements etc. These files are used to initialize the graph

### Node tables

Node tables store the actual node data in columnar formats like Parquet, Lance, Vortex, etc. Ideally, one file per table should be sufficient for storing the node data. However, for very large graphs, multiple files can be used to store partitions of the node data.

Example node table:

| id | name  | age |
| -- | ----- | --- |
| 1  | Alice | 30  |
| 2  | Bob   | 25  |

### Indices files

Each indices file[^1] is a table that stores relationship data (i.e., `from_node_id`, `to_node_id`, and edge properties) in columnar format. Similar to node tables, there can be multiple files to store partitions of the relationship data for very large graphs.

Example indices file:

| from | to | weight |
| ---- | -- | ------ |
| 1    | 2  | 0.5    |
| 2    | 1  | 0.8    |

### Indptr files

Each indptr file[^1] (per rel table) stores the offset range (in the indices file) for each node. These files are used to efficiently retrieve the edges connected to a specific node during graph traversals.

Example indptr file:

| edge_id |
| ------- |
| 0       |
| 1       |
| 2       |

[^1]: https://www.usenix.org/system/files/login/articles/login_winter20_16_kelly.pdf

