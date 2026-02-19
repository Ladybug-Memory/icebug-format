# Generating Graphs

This section describes how to generate various graphs using the provided tools.

## Supported Graphs

- Karate Club graph: A social network of friendships between 34 members of a karate club
- Complete graph: A graph where every pair of distinct vertices is connected by a unique edge
- Cycle graph: A graph that consists of a single cycle
- Path graph: A graph whose vertices can be listed in an order such that the edges connect consecutive vertices
- Kronecker graph: A scale-free graph with properties similar to Kronecker graphs

## Usage

To generate graph data in various formats, run:

```bash
uv run python generate/gen.py [--type TYPE] [--size SIZE] [--randomize-ids]
```

Options:
- `--type`: Type of graph to generate (karate, complete, cycle, path, kronecker) - default: karate
- `--size`: Size of the graph (number of nodes for applicable graph types)
- `--randomize-ids`: Randomize node IDs in a space 10x the number of nodes

This will generate five outputs:
- `{prefix}_nodes.csv` and `{prefix}_edges.csv` - CSV format
- `{prefix}.duckdb` - DuckDB database with nodes and edges tables
- `{prefix}.snap` - SNAP format (edge list)
- `{prefix}.snap.bin` - SNAP binary format (efficient binary format)
- `{prefix}.lbdb` - LadybugDB database with nodes and edges tables

Example for Karate Club graph:
```bash
python gen.py --type karate
```

Example for a randomized Karate Club graph:
```bash
python gen.py --type karate --randomize-ids
```

Example for a complete graph with 20 nodes:
```bash
python gen.py --type complete --size 20
```

## Format Details

### CSV
Two CSV files are generated:
- `{prefix}_nodes.csv`: Contains node IDs and their attributes (if any)
- `{prefix}_edges.csv`: Contains source and target node IDs for each edge

### DuckDB
A DuckDB database with two tables:
- `nodes`: Contains node_id and attributes (if any)
- `edges`: Contains source and target columns

### SNAP
A text file with the SNAP format:
- Comment lines starting with #
- Each line represents an edge with source and target node IDs

### SNAP Binary
A binary file with the SNAP binary format:
- Efficient binary representation of graph data
- Contains header with graph metadata
- Stores nodes and edges in a compact binary format
- Supports optional node and edge attributes

### LadybugDB
A LadybugDB database with two tables:
- `nodes`: Contains node_id (INT64) and attributes (if any) with node_id as primary key
- `edges`: A relationship table connecting nodes to nodes

### CSR (Compressed Sparse Row)
A representation optimized for fast graph algorithms:
- Node mapping tables: `{table_name}_mapping_{node_name}` - Maps original sparse node IDs to 0-based contiguous indices (one per node table)
- Row pointers (indptr): `{table_name}_indptr_{edge_name}` - For each node, points to the start of its edges in the indices array (one per edge table)
- Column indices (indices): `{table_name}_indices_{edge_name}` - Contains the target node for each edge (one per edge table)
- Metadata: `{table_name}_metadata` - Stores global graph properties (node count, edge count, directed flag)
- Nodes: `{table_name}_nodes_{node_name}` - Original nodes table with node attributes (one per node table)
