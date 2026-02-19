#!/usr/bin/env python3
"""
Script to convert graph data from DuckDB to CSR (Compressed Sparse Row) format.

This script reads graph data from a DuckDB database containing an edges table
with source and target columns representing edges, and converts it to CSR format for
efficient processing with NetworkKit.

The conversion process:
1. Reads graph data from DuckDB (edges table with source, target columns)
2. Handles sparse node IDs by creating a dense mapping (original_id -> csr_index)
3. Converts edges to CSR (Compressed Sparse Row) format
4. Pre-sorts edges by source using DuckDB for memory efficiency
5. Saves CSR data and node mapping to DuckDB for reuse
6. Exports to parquet format and generates schema.cypher for ladybugdb

Key Features:
- Memory efficient: Uses database-level sorting and PyArrow for large graph processing
- Handles sparse node IDs: Works with any node ID range (e.g., 1000, 5000, 9999)
- Scalable: Optimized for large graphs using DuckDB's efficient sorting
- Multi-table support: Processes multiple node/edge tables (prefix: nodes*, edges*)

Usage Examples:
    # Convert edges in karate_random.duckdb to CSR format and save to csr_graph.db
    python convert_csr.py --source-db karate_random.duckdb --output-db csr_graph.db

    # Convert with limited data for testing
    python convert_csr.py --source-db karate_random.duckdb --test --limit 50000 --output-db test.db
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


def get_node_and_edge_tables(
    con, db_alias: str = "orig"
) -> tuple[list[str], list[str]]:
    """
    Discover node and edge tables in the source database.

    Tables starting with 'nodes' are considered node tables.
    Tables starting with 'edges' are considered edge tables.

    Returns:
        Tuple of (node_table_names, edge_table_names)
    """
    result = con.execute(
        f"SELECT table_name FROM information_schema.tables WHERE table_catalog = '{db_alias}'"
    ).fetchall()
    all_tables = [row[0] for row in result]

    node_tables = [t for t in all_tables if t.startswith("nodes")]
    edge_tables = [t for t in all_tables if t.startswith("edges")]

    return node_tables, edge_tables


def duckdb_type_to_cypher_type(duckdb_type: str) -> str:
    """Convert DuckDB column type to Cypher/Ladybug type."""
    duckdb_type = duckdb_type.upper()
    type_map = {
        "BIGINT": "INT64",
        "INTEGER": "INT32",
        "SMALLINT": "INT16",
        "TINYINT": "INT8",
        "HUGEINT": "INT128",
        "UBIGINT": "UINT64",
        "UINTEGER": "UINT32",
        "USMALLINT": "UINT16",
        "UTINYINT": "UINT8",
        "DOUBLE": "DOUBLE",
        "FLOAT": "FLOAT",
        "REAL": "FLOAT",
        "BOOLEAN": "BOOL",
        "VARCHAR": "STRING",
        "TEXT": "STRING",
        "CHAR": "STRING",
        "DATE": "DATE",
        "TIMESTAMP": "TIMESTAMP",
        "TIME": "TIME",
        "BLOB": "BLOB",
    }
    # Handle parameterized types like DECIMAL(10,2)
    base_type = duckdb_type.split("(")[0].strip()
    return type_map.get(base_type, "STRING")


def generate_schema_cypher(
    con,
    csr_table_name: str,
    node_tables: list[str],
    edge_tables: list[str],
    parquet_dir: Path,
    edge_relationships: dict,
    node_type_to_table: dict,
    storage_path: str,
) -> str:
    """
    Generate schema.cypher content for ladybugdb.

    Args:
        con: DuckDB connection
        csr_table_name: Prefix for CSR tables
        node_tables: List of original node table names
        edge_tables: List of original edge table names
        parquet_dir: Path to the parquet output directory (for storage path)
        edge_relationships: Dict of edge relationships from schema
        node_type_to_table: Mapping of node types to table names
        storage_path: Storage path string for schema.cypher

    Returns:
        String containing the schema.cypher content
    """
    lines = []

    # Helper to derive display name from table name (lowercase)
    # nodes => nodes, nodes_person => person, nodes_foo => foo
    def get_node_display_name(table_name: str) -> str:
        if table_name == "nodes":
            return "nodes"
        elif table_name.startswith("nodes_"):
            return table_name[6:].lower()  # Remove "nodes_" prefix and lowercase
        return table_name.lower()

    def get_edge_display_name(table_name: str) -> str:
        if table_name == "edges":
            return "edges"
        elif table_name.startswith("edges_"):
            return table_name[6:].lower()  # Remove "edges_" prefix and lowercase
        return table_name.lower()

    # Build mapping of original table names to display names
    node_display_names = {nt: get_node_display_name(nt) for nt in node_tables}

    # Generate NODE TABLE definitions for each node table
    for node_table in node_tables:
        table_name = f"{csr_table_name}_{node_table}"
        try:
            cols = con.execute(f"DESCRIBE {table_name}").fetchall()
            col_defs = []
            pk_col = None
            for col in cols:
                col_name, col_type = col[0], col[1]
                cypher_type = duckdb_type_to_cypher_type(col_type)
                col_defs.append(f"{col_name} {cypher_type}")
                # First column is typically the primary key
                if pk_col is None:
                    pk_col = col_name

            cols_str = ", ".join(col_defs)
            display_name = node_display_names[node_table]
            lines.append(
                f"CREATE NODE TABLE {display_name}({cols_str}, PRIMARY KEY({pk_col})) "
                f"WITH (storage = '{storage_path}');"
            )
        except Exception as e:
            print(
                f"Warning: Could not generate schema for node table {table_name}: {e}"
            )

    # Generate REL TABLE definitions for each edge table
    for edge_table in edge_tables:
        rel_name = get_edge_display_name(edge_table)
        edge_name = (
            edge_table[6:].lower()
            if edge_table.startswith("edges_")
            else edge_table.lower()
        )
        src_node_type, dst_node_type = edge_relationships.get(edge_name, (None, None))
        if (
            src_node_type
            and dst_node_type
            and src_node_type in node_type_to_table
            and dst_node_type in node_type_to_table
        ):
            src_nt = node_type_to_table[src_node_type]
            dst_nt = node_type_to_table[dst_node_type]
            src_table = node_display_names[src_nt]
            dst_table = node_display_names[dst_nt]
        else:
            src_table = node_display_names[node_tables[0]] if node_tables else "nodes"
            dst_table = src_table

        # Get columns from indices table
        indices_table = f"{csr_table_name}_indices_{edge_name}"
        try:
            cols = con.execute(f"DESCRIBE {indices_table}").fetchall()
            col_defs = []
            for col in cols:
                col_name, col_type = col[0], col[1]
                if col_name == "target":
                    continue
                cypher_type = duckdb_type_to_cypher_type(col_type)
                col_defs.append(f"{col_name} {cypher_type}")
            props_str = ", ".join(col_defs)
            lines.append(
                f"CREATE REL TABLE {rel_name}(FROM {src_table} TO {dst_table}"
                f"{', ' + props_str if props_str else ''}) WITH (storage = '{storage_path}');"
            )
        except Exception as e:
            print(f"Warning: Could not generate schema for rel table {rel_name}: {e}")

    return "\n".join(lines) + "\n"


def export_to_parquet_and_cypher(
    con,
    output_db_path: str,
    csr_table_name: str,
    node_tables: list[str],
    edge_tables: list[str],
    edge_relationships: dict,
    node_type_to_table: dict,
    storage_path: str | None = None,
) -> None:
    """
    Export all tables to parquet format and generate schema.cypher.

    Args:
        con: DuckDB connection
        output_db_path: Path to output DuckDB database
        csr_table_name: Prefix for CSR tables
        node_tables: List of original node table names
        edge_tables: List of original edge table names
        storage_path: Storage path for schema.cypher (default: output_db without .duckdb + csr_table_name)
    """
    print("\n=== Exporting to Parquet and Generating schema.cypher ===")

    # Create output directory next to the database
    output_path = Path(output_db_path)
    parquet_dir = output_path.parent / output_path.stem
    parquet_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parquet output directory: {parquet_dir}")

    # Compute storage path if not provided
    if storage_path is None:
        storage_path = f"./{output_path.stem}/{csr_table_name}"

    # Get all tables to export
    result = con.execute("SHOW TABLES").fetchall()
    all_tables = [row[0] for row in result]

    # Export each table to parquet (lowercase filenames)
    for table_name in all_tables:
        parquet_file = parquet_dir / f"{table_name.lower()}.parquet"
        con.execute(f"COPY {table_name} TO '{parquet_file}' (FORMAT 'parquet')")
        print(f"  Exported: {table_name} -> {parquet_file.name}")

    # Generate schema.cypher
    schema_cypher = generate_schema_cypher(
        con,
        csr_table_name,
        node_tables,
        edge_tables,
        parquet_dir,
        edge_relationships,
        node_type_to_table,
        storage_path,
    )
    schema_file = parquet_dir / "schema.cypher"
    schema_file.write_text(schema_cypher)
    print(f"  Generated: {schema_file.name}")

    # Remove old SQL files if they exist
    for old_file in ["schema.sql", "load.sql"]:
        old_path = parquet_dir / old_file
        if old_path.exists():
            old_path.unlink()
            print(f"  Removed: {old_file}")

    print(f"✓ Export complete. Files saved to: {parquet_dir}")


def create_csr_graph_to_duckdb(
    source_db_path: str,
    output_db_path: str,
    limit_rels: int | None = None,
    directed: bool = False,
    csr_table_name: str = "csr_graph",
    node_table: str | None = None,
    edge_table: str | None = None,
    schema_path: str | None = None,
    storage_path: str | None = None,
) -> None:
    """
    Create CSR graph data and save to DuckDB using optimized SQL approach.

    Args:
        source_db_path: Path to source DuckDB with edges table
        output_db_path: Path to output DuckDB for CSR data
        limit_rels: Limit number of relationships for testing
        directed: Whether graph is directed
        csr_table_name: Name of table to store CSR data
        node_table: Specific node table to use (default: auto-discover)
        edge_table: Specific edge table to use (default: auto-discover)
        schema_path: Path to schema.cypher for edge relationship info
        storage_path: Storage path for schema.cypher (default: output_db without .duckdb + csr_table_name)
    """
    print("\n=== Creating CSR Graph Data (Optimized SQL Approach) ===")

    # Connect to a fresh DuckDB database for output
    con = duckdb.connect(output_db_path)

    # Drop all existing tables to recreate from scratch
    result = con.execute("SHOW TABLES").fetchall()
    existing_tables = [row[0] for row in result]
    for table in existing_tables:
        con.execute(f"DROP TABLE IF EXISTS {table}")
    if existing_tables:
        print(f"Dropped {len(existing_tables)} existing tables")

    try:
        print("Step 0: Loading edges and nodes from original DB into new DB...")

        # Import the edges table from the original database
        con.execute(f"ATTACH '{source_db_path}' AS orig;")

        # Discover node and edge tables
        node_tables, edge_tables = get_node_and_edge_tables(con, "orig")

        # Use specified tables or discovered ones
        if node_table:
            node_tables = [node_table] if node_table in node_tables else []
        if edge_table:
            edge_tables = [edge_table] if edge_table in edge_tables else []

        if not edge_tables:
            raise ValueError(
                "No edge tables found in source database (tables must start with 'edges')"
            )

        print(f"Discovered node tables: {node_tables}")
        print(f"Discovered edge tables: {edge_tables}")

        # Parse schema.cypher for edge relationships
        edge_relationships = {}
        if schema_path:
            schema_file = Path(schema_path)
            edge_relationships = parse_schema_cypher(schema_file)
            print(f"Parsed edge relationships from schema: {edge_relationships}")

        # Build mapping from node type names to table names
        # e.g., "user" -> "nodes_user", "city" -> "nodes_city"
        node_type_to_table = {}
        for nt in node_tables:
            if nt == "nodes":
                node_type_to_table["nodes"] = nt
            elif nt.startswith("nodes_"):
                node_type_name = nt[6:].lower()  # Remove "nodes_" prefix and lowercase
                node_type_to_table[node_type_name] = nt

        print(f"Node type to table mapping: {node_type_to_table}")

        # Copy all node tables with proper prefixing and create per-table mappings
        node_counts = {}  # Track node counts per table
        for nt in node_tables:
            try:
                # Get the primary key column (first column of original node table)
                cols = con.execute(f"DESCRIBE orig.{nt}").fetchall()
                pk_col = cols[0][0] if cols else "id"

                con.execute(
                    f"CREATE TABLE {csr_table_name}_{nt} AS SELECT * FROM orig.{nt} ORDER BY {pk_col};"
                )
                print(f"  Copied node table: {nt} -> {csr_table_name}_{nt}")

                # Create per-table node mapping
                node_type = nt[6:].lower() if nt.startswith("nodes_") else nt.lower()
                mapping_table = f"{csr_table_name}_mapping_{node_type}"
                con.execute(
                    f"""
                    CREATE TABLE {mapping_table} AS
                    SELECT
                        row_number() OVER (ORDER BY {pk_col}) - 1 AS csr_index,
                        {pk_col} AS original_node_id
                    FROM {csr_table_name}_{nt}
                    ORDER BY csr_index;
                """
                )
                print(f"  Created node mapping: {mapping_table}")

                # Track node count
                result = con.execute(
                    f"SELECT COUNT(*) FROM {csr_table_name}_{nt}"
                ).fetchone()
                node_counts[nt] = result[0] if result else 0
            except Exception as e:
                print(f"Warning: Could not copy node table {nt}: {e}")

        # Process each edge table separately to create per-edge CSR structures
        print("\nStep 1: Building per-edge-table CSR structures...")

        for et in edge_tables:
            # Determine source and target node types from schema
            edge_name = (
                et[6:].lower() if et.startswith("edges_") else et.lower()
            )  # Remove "edges_" prefix and lowercase
            src_node_type, dst_node_type = edge_relationships.get(
                edge_name, (None, None)
            )

            # Find the corresponding node tables
            src_table = node_type_to_table.get(src_node_type)
            dst_table = node_type_to_table.get(dst_node_type)

            fallback_node_type = None
            if src_table and dst_table:
                src_mapping = f"{csr_table_name}_mapping_{src_node_type}"
                dst_mapping = f"{csr_table_name}_mapping_{dst_node_type}"
                num_src_nodes = node_counts.get(src_table, 0)
                print(
                    f"\n  Processing {et}: {src_node_type} ({num_src_nodes} nodes) -> {dst_node_type}"
                )
            else:
                # Fallback: use first node table for both
                fallback_table = node_tables[0] if node_tables else "nodes"
                fallback_node_type = (
                    fallback_table[6:].lower()
                    if fallback_table.startswith("nodes_")
                    else fallback_table.lower()
                )
                src_mapping = f"{csr_table_name}_mapping_{fallback_node_type}"
                dst_mapping = src_mapping
                num_src_nodes = node_counts.get(fallback_table, 0)
                print(f"\n  Processing {et}: using fallback mapping {src_mapping}")

            # Get edge columns excluding source and target
            edge_cols_result = con.execute(f"DESCRIBE orig.{et}").fetchall()
            edge_col_names = [col[0] for col in edge_cols_result]
            edge_cols = [c for c in edge_col_names if c not in ["source", "target"]]

            # Prepare select column strings
            select_cols = "m1.csr_index AS csr_source, m2.csr_index AS csr_target"
            if edge_cols:
                select_cols += ", " + ", ".join([f"e.{c}" for c in edge_cols])
            reverse_select_cols = (
                "m2.csr_index AS csr_source, m1.csr_index AS csr_target"
            )
            if edge_cols:
                reverse_select_cols += ", " + ", ".join([f"e.{c}" for c in edge_cols])
            reverse_cols = "csr_target AS csr_source, csr_source AS csr_target"
            if edge_cols:
                reverse_cols += ", " + ", ".join(edge_cols)

            # Create relations table for this edge type
            if limit_rels:
                limit_per_table = limit_rels // len(edge_tables)
                if directed:
                    rel_query = f"""
                        SELECT {select_cols}
                        FROM orig.{et} e
                        JOIN {src_mapping} m1 ON e.source = m1.original_node_id
                        JOIN {dst_mapping} m2 ON e.target = m2.original_node_id
                        WHERE e.source != e.target
                        LIMIT {limit_per_table}
                    """
                else:
                    rel_query = f"""
                        WITH limited AS (
                            SELECT {select_cols}
                            FROM orig.{et} e
                            JOIN {src_mapping} m1 ON e.source = m1.original_node_id
                            JOIN {dst_mapping} m2 ON e.target = m2.original_node_id
                            WHERE e.source != e.target
                            LIMIT {limit_per_table}
                        )
                        SELECT * FROM limited
                        UNION ALL
                        SELECT {reverse_cols} FROM limited
                    """
            else:
                if directed:
                    rel_query = f"""
                        SELECT {select_cols}
                        FROM orig.{et} e
                        JOIN {src_mapping} m1 ON e.source = m1.original_node_id
                        JOIN {dst_mapping} m2 ON e.target = m2.original_node_id
                        WHERE e.source != e.target
                    """
                else:
                    rel_query = f"""
                        SELECT {select_cols}
                        FROM orig.{et} e
                        JOIN {src_mapping} m1 ON e.source = m1.original_node_id
                        JOIN {dst_mapping} m2 ON e.target = m2.original_node_id
                        WHERE e.source != e.target
                        UNION ALL
                        SELECT {reverse_select_cols}
                        FROM orig.{et} e
                        JOIN {src_mapping} m1 ON e.source = m1.original_node_id
                        JOIN {dst_mapping} m2 ON e.target = m2.original_node_id
                        WHERE e.source != e.target
                    """

            con.execute(f"CREATE TABLE relations_{edge_name} AS {rel_query};")

            result = con.execute(
                f"SELECT COUNT(*) FROM relations_{edge_name}"
            ).fetchone()
            edge_count = result[0] if result else 0
            print(f"    Edges: {edge_count:,}")

            # Build CSR indptr for this edge type
            indptr_table = f"{csr_table_name}_indptr_{edge_name}"
            con.execute(
                f"""
                CREATE TABLE {indptr_table} AS
                WITH node_range AS (
                    SELECT unnest(range(0, {num_src_nodes})) AS node_id
                ),
                degrees AS (
                    SELECT csr_source AS src, COUNT(*) AS deg
                    FROM relations_{edge_name}
                    GROUP BY csr_source
                ),
                cumulative AS (
                    SELECT
                        node_range.node_id,
                        COALESCE(SUM(degrees.deg) OVER (ORDER BY node_range.node_id ROWS UNBOUNDED PRECEDING), 0) AS ptr
                    FROM node_range
                    LEFT JOIN degrees ON node_range.node_id = degrees.src
                )
                SELECT ptr FROM cumulative
                ORDER BY node_id;
            """
            )

            # Recreate with leading zero
            con.execute(
                f"""
                CREATE OR REPLACE TABLE {indptr_table} AS
                SELECT 0::BIGINT AS ptr
                UNION ALL
                SELECT ptr::int64 FROM {indptr_table}
                ORDER BY ptr;
            """
            )

            result = con.execute(f"SELECT COUNT(*) FROM {indptr_table}").fetchone()
            indptr_size = result[0] if result else 0
            print(f"    indptr: {indptr_size} entries")

            # Build CSR indices for this edge type
            indices_table = f"{csr_table_name}_indices_{edge_name}"
            con.execute(
                f"""
                CREATE TABLE {indices_table} AS
                SELECT csr_target AS target{', ' + ', '.join(edge_cols) if edge_cols else ''}
                FROM relations_{edge_name}
                ORDER BY csr_source, csr_target;
            """
            )

            result = con.execute(f"SELECT COUNT(*) FROM {indices_table}").fetchone()
            indices_size = result[0] if result else 0
            print(f"    indices: {indices_size} entries")

            # Drop temporary relations table
            con.execute(f"DROP TABLE IF EXISTS relations_{edge_name.lower()};")

        # Count total nodes and edges for summary
        total_nodes = sum(node_counts.values())
        total_edges = 0
        for et in edge_tables:
            edge_name = et[6:].lower() if et.startswith("edges_") else et.lower()
            result = con.execute(
                f"SELECT COUNT(*) FROM {csr_table_name}_indices_{edge_name}"
            ).fetchone()
            total_edges += result[0] if result else 0

        # Create global metadata
        con.execute(
            f"""
        CREATE TABLE {csr_table_name}_metadata AS
        SELECT {total_nodes} AS n_nodes, {total_edges} AS n_edges, {directed} AS directed
        """
        )

        # List per-table node mappings for output
        node_mapping_tables = [
            f"{csr_table_name}_mapping_{nt[6:].lower() if nt.startswith('nodes_') else nt.lower()}"
            for nt in node_tables
        ]

        print("\n✅ CSR format built and cleaned up. Final tables:")
        for mapping_table in node_mapping_tables:
            print(f"  - {mapping_table} (orig_id → mapped_id)")
        for i, et in enumerate(edge_tables):
            edge_name = et[6:].lower() if et.startswith("edges_") else et.lower()
            print(f"  - {csr_table_name}_indptr_{edge_name}")
            print(f"  - {csr_table_name}_indices_{edge_name}")
        print(f"  - {csr_table_name}_metadata (global)")

        print(
            f"\n✓ Built CSR format: {total_nodes} nodes, {total_edges} edges across {len(edge_tables)} edge types"
        )
        print(f"✓ Saved CSR graph data to {output_db_path}")

        # Export to parquet and generate schema.cypher
        export_to_parquet_and_cypher(
            con,
            output_db_path,
            csr_table_name,
            node_tables,
            edge_tables,
            edge_relationships,
            node_type_to_table,
            storage_path,
        )

    except Exception as e:
        print(f"Error building CSR format: {e}")
        raise
    finally:
        con.close()

    print(f"\nAll data saved to: {output_db_path}")


def main():
    """Main function to convert DuckDB edges to CSR format."""
    parser = argparse.ArgumentParser(
        description="Convert graph data from DuckDB to CSR format"
    )
    parser.add_argument(
        "--source-db",
        type=str,
        default="karate_random.duckdb",
        help="Source DuckDB database path (default: karate_random.duckdb)",
    )
    parser.add_argument(
        "--output-db",
        type=str,
        default="csr_graph.db",
        help="Output DuckDB database path (default: csr_graph.db)",
    )
    parser.add_argument(
        "--csr-table",
        type=str,
        default="csr_graph",
        help="Table name prefix for CSR data (default: csr_graph)",
    )
    parser.add_argument(
        "--node-table",
        type=str,
        default=None,
        help="Specific node table to use (default: auto-discover tables starting with 'nodes')",
    )
    parser.add_argument(
        "--edge-table",
        type=str,
        default=None,
        help="Specific edge table to use (default: auto-discover tables starting with 'edges')",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with limited data"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50000,
        help="Number of edges to use in test mode (default: 50000)",
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        help="Treat graph as directed (default: undirected)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Storage path for schema.cypher (default: output_db path without .duckdb extension)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="Path to schema.cypher for edge relationship info (FROM/TO node types)",
    )

    args = parser.parse_args()

    print("=== DuckDB to CSR Format Converter ===\n")

    # Configuration
    source_db_path = args.source_db  # DuckDB source

    # Create CSR graph
    test_limit = args.limit if args.test else None

    if test_limit:
        print(f"Creating CSR graph in TEST MODE with limit: {test_limit} edges")
    else:
        print("Creating CSR graph on FULL DATASET")

    print(f"Source database: {source_db_path}")
    print(f"CSR output database: {args.output_db}")
    print(f"CSR table prefix: {args.csr_table}")
    print(f"Directed: {args.directed}")

    # Compute default storage path from output_db if not specified
    storage_path = args.storage
    if storage_path is None:
        # Use output_db path without .duckdb extension + csr_table_name
        storage_path = f"./{Path(args.output_db).stem}/{args.csr_table}"
    print(f"Storage path: {storage_path}")

    if args.node_table:
        print(f"Node table filter: {args.node_table}")
    if args.edge_table:
        print(f"Edge table filter: {args.edge_table}")
    if args.schema:
        print(f"Schema file: {args.schema}")

    create_csr_graph_to_duckdb(
        source_db_path=source_db_path,
        output_db_path=args.output_db,
        limit_rels=test_limit,
        directed=args.directed,
        csr_table_name=args.csr_table,
        node_table=args.node_table,
        edge_table=args.edge_table,
        schema_path=args.schema,
        storage_path=storage_path,
    )

    print("\n=== Conversion Completed Successfully! ===")
    print(f"CSR graph data saved to: {args.output_db}")


if __name__ == "__main__":
    main()
