#!/usr/bin/env python3
"""
Script to scan graph data in icebug-disk format from parquet files and print metadata, node tables, and reconstructed edge tables.

Usage:
    uv run scan.py --input demo-db_csr
"""

import argparse
import re
from pathlib import Path

import duckdb


def parse_schema_cypher(schema_path: Path) -> dict:
    """
    Parse schema.cypher to extract edge relationships (FROM/TO node types).

    Returns:
        Dictionary mapping edge names to (from_node_type, to_node_type) tuples
    """
    edge_relationships = {}

    if not schema_path.exists():
        return edge_relationships

    content = schema_path.read_text()

    # Parse REL TABLE definitions: CREATE REL TABLE Follows(FROM User TO User, ...);
    # Also handles backtick-quoted identifiers: CREATE REL TABLE `edges` (FROM `nodes` TO `nodes`, ...)
    rel_pattern = (
        r"CREATE\s+REL\s+TABLE\s+`?(\w+)`?\s*\(\s*FROM\s+`?(\w+)`?\s+TO\s+`?(\w+)`?"
    )
    for match in re.finditer(rel_pattern, content, re.IGNORECASE):
        edge_name = match.group(1).lower()
        from_node = match.group(2).lower()
        to_node = match.group(3).lower()
        edge_relationships[edge_name] = (from_node, to_node)

    return edge_relationships


def scan_icebug_disk(input_dir: Path, schema_path: Path | None = None):
    """
    Scan the graph data in icebug-disk format from parquet files and print nodes and edges.
    """
    con = duckdb.connect()  # In-memory connection

    try:
        # Node tables: nodes_*.parquet
        node_parquets = sorted(input_dir.glob("nodes_*.parquet"))
        print("Node Tables:")
        for np in node_parquets:
            nt = np.stem  # e.g. "nodes_city"
            print(f"\nTable: {nt}")

            # verify metadata
            metadataQuery = f"""
                SELECT CAST(value AS VARCHAR) AS metadata_value
                FROM parquet_kv_metadata('{np}')
                WHERE key = 'icebug_disk_version'
            """
            metadata = con.execute(metadataQuery).fetchone()

            if not metadata or metadata[0].lower() != "v1":
                print(f"Warning: {np} has missing or incompatible icebug_disk_version metadata")

            rows = con.execute(f"SELECT * FROM '{np}'").fetchall()
            for row in rows:
                print(row)

        # Edge tables - reconstruct from CSR using node table row order as ID map
        print("\nEdge Tables (reconstructed from CSR):")

        # Build node-type -> pk-values-by-row-order map from nodes_*.parquet
        node_id_map: dict[str, list] = {}
        for np in node_parquets:
            node_type = np.stem[len("nodes_"):]  # e.g. "city"
            rows = con.execute(f"SELECT * FROM '{np}'").fetchall()
            node_id_map[node_type] = [row[0] for row in rows]

        # Parse schema for edge FROM/TO relationships
        edge_relationships = {}
        if schema_path:
            edge_relationships = parse_schema_cypher(schema_path)

        for indptr_p in sorted(input_dir.glob("indptr_*.parquet")):
            edge_name = indptr_p.stem[len("indptr_"):]
            indices_p = input_dir / f"indices_{edge_name}.parquet"

            if not indices_p.exists():
                print(f"\nSkipping {edge_name}: indices parquet not found")
                continue

            from_node, to_node = edge_relationships.get(edge_name, (None, None))
            if not from_node or not to_node:
                print(f"\nSkipping {edge_name}: no relationship info in schema.cypher")
                continue

            source_ids = node_id_map.get(from_node)
            target_ids = node_id_map.get(to_node)

            if source_ids is None:
                print(f"\nSkipping {edge_name}: no node table for '{from_node}'")
                continue
            if target_ids is None:
                print(f"\nSkipping {edge_name}: no node table for '{to_node}'")
                continue

            print(f"\nTable: {edge_name} (FROM {from_node} TO {to_node})")

            indptr = [row[0] for row in con.execute(f"SELECT ptr FROM '{indptr_p}'").fetchall()]
            indices_result = con.execute(f"SELECT * FROM '{indices_p}'").fetchall()

            for i in range(len(indptr) - 1):
                start = indptr[i]
                end = indptr[i + 1]
                source_orig = source_ids[i]
                for j in range(start, end):
                    row = indices_result[j]
                    target_orig = target_ids[row[0]]
                    edge_data = [source_orig, target_orig]
                    if len(row) > 1:
                        edge_data.extend(row[1:])
                    print(tuple(edge_data))

    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(
        description="Scan CSR graph data from parquet files"
    )
    parser.add_argument(
        "--input", required=True, help="Input directory containing parquet files"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Directory {input_dir} not found")
        return

    schema_path = input_dir / "schema.cypher"
    if not schema_path.exists():
        schema_path = None

    scan_icebug_disk(input_dir, schema_path)


if __name__ == "__main__":
    main()
