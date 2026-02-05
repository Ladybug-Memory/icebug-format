#!/usr/bin/env python3
"""
Script to scan graph data in icebug-disk format from parquet files and print metadata, node tables, and reconstructed edge tables.

Usage:
    uv run scan.py --input demo-db_csr --prefix demo
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


def scan_graph_std(input_dir: Path, prefix: str, schema_path: Path | None = None):
    """
    Scan the graph data in icebug-disk format from parquet files and print metadata, nodes, and edges.
    """
    con = duckdb.connect()  # In-memory connection

    try:
        # Use provided prefix
        metadata_parquet = input_dir / f"{prefix}_metadata.parquet"

        if not metadata_parquet.exists():
            print(f"Metadata parquet {metadata_parquet} not found")
            return

        # Get metadata
        metadata = con.execute(f"SELECT * FROM '{metadata_parquet}'").fetchone()
        if metadata:
            n_nodes, n_edges, directed = metadata
            print(f"Metadata: {n_nodes} nodes, {n_edges} edges, directed={directed}")
        else:
            print("No metadata found")
            return

        # Node tables
        node_parquets = list(input_dir.glob(f"{prefix}_nodes*.parquet"))
        print("\nNode Tables:")
        for np in node_parquets:
            nt = np.stem  # remove .parquet
            print(f"\nTable: {nt}")
            rows = con.execute(f"SELECT * FROM '{np}'").fetchall()
            for row in rows:
                print(row)

        # Verify CSR index == node offset invariant
        print("\nVerifying CSR index == node offset invariant:")
        for np in node_parquets:
            nt = np.stem
            # Extract node type: prefix_nodes[_type]
            if nt.startswith(f"{prefix}_nodes"):
                suffix = nt[len(f"{prefix}_nodes") :]
                if suffix == "":
                    node_type = "nodes"
                else:
                    node_type = suffix[1:]  # remove leading _
                mapping_p = input_dir / f"{prefix}_mapping_{node_type}.parquet"
                if mapping_p.exists():
                    # Load node rows
                    node_rows = con.execute(f"SELECT * FROM '{np}'").fetchall()
                    # Load mapping ordered by csr_index
                    mapping = con.execute(
                        f"SELECT original_node_id FROM '{mapping_p}' ORDER BY csr_index"
                    ).fetchall()
                    mapping_ids = [row[0] for row in mapping]
                    # Assume pk is first column
                    pk_values = [row[0] for row in node_rows]
                    if len(pk_values) != len(mapping_ids):
                        print(
                            f"  {nt}: Length mismatch: {len(pk_values)} vs {len(mapping_ids)}"
                        )
                        continue
                    violations = [
                        i
                        for i in range(len(pk_values))
                        if pk_values[i] != mapping_ids[i]
                    ]
                    if violations:
                        print(
                            f"  {nt}: Invariant violated at rows {violations[:5]}{'...' if len(violations) > 5 else ''}"
                        )
                    else:
                        print(f"  {nt}: Invariant holds ✓")
                else:
                    print(f"  {nt}: No mapping file found")

        # Edge tables - reconstruct from CSR
        print("\nEdge Tables (reconstructed from CSR):")

        # Parse schema for edge relationships
        edge_relationships = {}
        if schema_path:
            edge_relationships = parse_schema_cypher(schema_path)

        indptr_parquets = list(input_dir.glob(f"{prefix}_indptr_*.parquet"))
        for indptr_p in indptr_parquets:
            indptr_t = indptr_p.stem
            edge_name = indptr_t[len(f"{prefix}_indptr_") :]
            indices_p = input_dir / f"{prefix}_indices_{edge_name}.parquet"

            if not indices_p.exists():
                print(f"\nSkipping {edge_name}: indices parquet not found")
                continue

            # Get source and target node types
            from_node, to_node = edge_relationships.get(edge_name, (None, None))
            if not from_node or not to_node:
                print(f"\nSkipping {edge_name}: no relationship info")
                continue

            source_mapping_p = input_dir / f"{prefix}_mapping_{from_node}.parquet"
            target_mapping_p = input_dir / f"{prefix}_mapping_{to_node}.parquet"

            if not source_mapping_p.exists():
                print(f"\nSkipping {edge_name}: source mapping parquet not found")
                continue
            if not target_mapping_p.exists():
                print(f"\nSkipping {edge_name}: target mapping parquet not found")
                continue

            print(f"\nTable: {edge_name} (FROM {from_node} TO {to_node})")

            # Fetch data
            indptr = [
                row[0]
                for row in con.execute(f"SELECT ptr FROM '{indptr_p}'").fetchall()
            ]
            indices_result = con.execute(f"SELECT * FROM '{indices_p}'").fetchall()
            source_map = [
                row[0]
                for row in con.execute(
                    f"SELECT original_node_id FROM '{source_mapping_p}' ORDER BY csr_index"
                ).fetchall()
            ]
            target_map = [
                row[0]
                for row in con.execute(
                    f"SELECT original_node_id FROM '{target_mapping_p}' ORDER BY csr_index"
                ).fetchall()
            ]

            # Reconstruct edges
            for i in range(len(indptr) - 1):
                start = indptr[i]
                end = indptr[i + 1]
                source_orig = source_map[i]
                for j in range(start, end):
                    row = indices_result[j]
                    target_csr = row[0]  # target is first column
                    target_orig = target_map[target_csr]
                    # Print source, target, and any additional properties
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
    parser.add_argument("--prefix", help="Table prefix (default: input)")

    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Directory {input_dir} not found")
        return

    prefix = args.prefix if args.prefix else args.input

    schema_path = input_dir / "schema.cypher"
    if not schema_path.exists():
        schema_path = None

    scan_graph_std(input_dir, prefix, schema_path)


if __name__ == "__main__":
    main()
