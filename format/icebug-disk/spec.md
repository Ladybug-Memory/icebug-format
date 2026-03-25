# Icebug-Disk graph specification

The Icebug-Disk graph format is designed to store graph data on disk efficiently, enabling high read performance for large graphs.

## Versioning

### Version 1
Initial version of the format. Uses compressed sparse row-based (CSR) adjacency list/join indices for efficient graph traversals.

## Architecture Diagram

![Icebug-Disk Architecture Overview](architecture_overview.svg)


## Components

### Metadata
A metadata file containing version information and the initialization script path.

Example metadata file:
```json
{
    "version": "1",
    "init_script": "path/to/schema.cypher"
}
```

### Initialization script
A script (typically containing node and relationship table creation rules) that can be executed by the query engine to create the graph in the database. This script is responsible for creating the graph structure, including node tables and relationship tables, and performing necessary validations.

Example initialization script:
```cypher
CREATE NODE TABLE node_table_1 (
    id STRING PRIMARY KEY,
    name STRING,
    age INT
) WITH (
    format = 'parquet',
    file_path = 'path/to/node_table_1.parquet'
);

CREATE REL TABLE rel_table_1 (
    from_node_id STRING,
    to_node_id STRING,
    weight FLOAT
) WITH (
    format = 'parquet',
    file_path = 'path/to/rel_table_1.parquet'
);
```

### Node tables
Node tables store the actual node data in columnar formats like Parquet, Lance, Vortex, etc. Ideally, one file per table should be sufficient for storing the node data. However, for very large graphs, multiple files can be used to store partitions of the node data.

Example node table:

| id   | name  | age |
|------|-------|-----|
| 1    | Alice | 30  |
| 2    | Bob   | 25  |

### Relationship tables
Relationship tables store the relationship data (i.e., `from_node_id`, `to_node_id`, and edge properties) in columnar format. Similar to node tables, there can be multiple files to store partitions of the relationship data for very large graphs.

Example relationship table:

| from | to  | weight |
|------|-----|--------|
| 1    | 2   | 0.5    |
| 2    | 1   | 0.8    |

### Internal ID mapping files
Each internal ID mapping file(per node table) stores the mapping between the original node IDs (`primary_key`) and the internal node offsets (`row_idx`) used in the graph. This is necessary for efficient storage and retrieval of nodes and relationships, as well as for performing graph traversals.

Example internal ID mapping file:

| node_id                                    | node_offset |
|--------------------------------------------|-------------|
| 0020a216-0626-11ea-9f44-8c16456798f1       | 0           |
| 0020a216-0626-11ea-9f44-8c16456798f2       | 1           |

### Edge range files
Each edge range file(per rel table) stores the edge ranges (in the relationship tables). These files are used to efficiently retrieve the edges connected to a specific node during graph traversals.

Example edge range file:

| edge_id                                                       |
|---------------------------------------------------------------|
| 0 -> start edge offset for node 1                             | 
| 1 -> end edge offset for node 1, start edge offset for node 2 | 
...
| 2 -> end edge offset for node n                               |
