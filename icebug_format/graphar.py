#!/usr/bin/env python3
"""
Script to convert GraphAr to icebug-format.

This script reads graph data in GraphAr format from a directory
and converts it to icebug-format (CSR representation) for use with
icebug-format compatible databases.

The conversion process:
1. Reads vertex data from GraphAr parquet files
2. Reads edge adjacency lists and properties from GraphAr parquet files
3. Creates a mapping from original vertex IDs to contiguous indices
4. Converts edges to CSR (Compressed Sparse Row) format
5. Saves to DuckDB and exports to parquet format with schema.cypher

Usage Examples:
    # Convert graphar graph to icebug-format
    uv run graphar.py --graphar-dir path/to/graphar --output-db output.duckdb
"""

import argparse
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


def duckdb_type_to_cypher_type(duckdb_type: str) -> str:
    """Convert DuckDB column type to Cypher/Kuzu type."""
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
    base_type = duckdb_type.split("(")[0].strip()
    return type_map.get(base_type, "STRING")


def read_graphar_vertices(vertex_info, graph_path: Path) -> dict:
    """
    Read vertices from GraphAr parquet files.

    Returns:
        Dictionary mapping vertex type to list of vertex data
    """
    vertices = {}
    vertex_type = vertex_info.get_type()
    
    # Find the vertex directory (has property groups)
    prefix = Path(vertex_info.get_prefix())
    vertex_dir = graph_path / prefix
    
    # Read vertex count file
    vertex_count_file = vertex_dir / "vertex_count"
    if vertex_count_file.exists():
        vertex_count = int(vertex_count_file.read_text())
    else:
        vertex_count = 0
    
    # Find property group directories
    prop_groups = []
    for item in vertex_dir.iterdir():
        if item.is_dir() and not item.name.startswith("_"):
            prop_groups.append(item)
    
    # Read all vertex data
    vertex_data = []
    for prop_dir in prop_groups:
        for chunk_file in prop_dir.glob("*.parquet"):
            table = pq.read_table(str(chunk_file))
            for i in range(table.num_rows):
                row = [table.column(j)[i].as_py() for j in range(table.num_columns)]
                vertex_data.append(row)
    
    vertices[vertex_type] = vertex_data
    return vertices


def read_graphar_edges(edge_info, graph_path: Path, vertex_indices: dict) -> dict:
    """
    Read edges from GraphAr parquet files.

    Returns:
        Dictionary mapping edge name to list of edge data
    """
    edges = {}
    edge_name = edge_info.get_edge_type()
    
    # Find the edge directory
    prefix = Path(edge_info.get_prefix())
    edge_dir = graph_path / prefix
    
    # Read edge count file
    edge_count_file = edge_dir / "edge_count0"
    if edge_count_file.exists():
        edge_count = int(edge_count_file.read_text())
    else:
        edge_count = 0
    
    # Find adjacency list directory
    adj_list_dir = None
    prop_dirs = []
    
    for item in edge_dir.iterdir():
        if item.is_dir() and item.name.startswith("ordered_by_"):
            adj_list_dir = item
        elif item.is_dir() and item.name.startswith("_"):
            prop_dirs.append(item)
    
    if not adj_list_dir:
        return edges
    
    # Read adjacency list files
    adj_list_data = []
    for chunk_file in (adj_list_dir / "adj_list").glob("*.parquet"):
        table = pq.read_table(str(chunk_file))
        for i in range(table.num_rows):
            src_idx = table.column(0)[i].as_py()
            dst_idx = table.column(1)[i].as_py()
            adj_list_data.append((src_idx, dst_idx))
    
    # Read property files
    prop_data = {}
    for prop_dir in prop_dirs:
        prop_name = prop_dir.name.lstrip("_")
        prop_data[prop_name] = []
        for chunk_file in prop_dir.glob("*.parquet"):
            table = pq.read_table(str(chunk_file))
            for i in range(table.num_rows):
                row = [table.column(j)[i].as_py() for j in range(table.num_columns)]
                prop_data[prop_name].append(row)
    
    edges[edge_name] = {
        'adj_list': adj_list_data,
        'properties': prop_data
    }
    return edges


def convert_graphar_to_graph_std(
    graphar_dir: str,
    output_db_path: str,
    csr_table_name: str = "graph",
    directed: bool = False,
) -> None:
    """
    Convert GraphAr format to icebug-format.

    Args:
        graphar_dir: Path to directory with GraphAr data
        output_db_path: Path to output DuckDB database
        csr_table_name: Name prefix for CSR tables
        directed: Whether graph is directed
    """
    print("\n=== Converting GraphAr to Graph-Std Format ===")
    
    import graphar
    
    # Load graph info
    # Find the .graph.yml file in the directory
    graphar_path = Path(graphar_dir)
    yaml_files = list(graphar_path.glob("*.graph.yml"))
    if not yaml_files:
        raise ValueError(f"No .graph.yml file found in {graphar_dir}")
    yaml_path = yaml_files[0]
    graph_info = graphar.GraphInfo.load(str(yaml_path.absolute()))
    
    graph_path = graphar_path
    
    # Connect to output DuckDB database
    con = duckdb.connect(output_db_path)
    
    # Drop all existing tables
    result = con.execute("SHOW TABLES").fetchall()
    existing_tables = [row[0] for row in result]
    for table in existing_tables:
        con.execute(f"DROP TABLE IF EXISTS {table}")
    
    # Read vertices
    print("\nStep 1: Reading vertices...")
    vertex_type_to_table = {}
    vertex_indices = {}
    
    for i in range(graph_info.vertex_info_num()):
        vertex_info = graph_info.get_vertex_info_by_index(i)
        vertex_type = vertex_info.get_type()
        
        # Read vertex data
        vertex_dir = graph_path / Path(vertex_info.get_prefix())
        
        # Find property group directories and read data (directories start with _)
        vertex_rows = []
        prop_group_names = []
        schemas = {}
        
        for item in vertex_dir.iterdir():
            if item.is_dir() and item.name.startswith("_"):
                for chunk_file in sorted(item.iterdir()):
                    if chunk_file.is_file():
                        table = pq.read_table(str(chunk_file))
                        if not vertex_rows:
                            prop_group_names = [table.schema.field(j).name for j in range(table.num_columns)]
                            schemas = {table.schema.field(j).name: str(table.schema.field(j).type) for j in range(table.num_columns)}
                        for i in range(table.num_rows):
                            row = {table.schema.field(j).name: table.column(j)[i].as_py() for j in range(table.num_columns)}
                            vertex_rows.append(row)
        
        # Create node table
        node_table_name = f"nodes_{vertex_type}"
        
        if vertex_rows:
            col_defs = []
            for col_name in vertex_rows[0].keys():
                col_defs.append(f'"{col_name}" VARCHAR')
            
            con.execute(f"CREATE TABLE {node_table_name} ({', '.join(col_defs)})")
            
            for row in vertex_rows:
                # Properly escape single quotes in string values
                escaped_values = []
                for v in row.values():
                    if v is None:
                        escaped_values.append('NULL')
                    else:
                        str_v = str(v).replace("'", "''")
                        escaped_values.append(f"'{str_v}'")
                con.execute(f"INSERT INTO {node_table_name} VALUES ({', '.join(escaped_values)})")
            
            print(f"  Created {node_table_name} with {len(vertex_rows)} vertices")
            
            # Create mapping table
            pk_col = prop_group_names[0]
            mapping_table_name = f"{csr_table_name}_mapping_{vertex_type}"
            con.execute(f"""
                CREATE TABLE {mapping_table_name} AS
                SELECT row_number() OVER (ORDER BY "{pk_col}") - 1 AS csr_index,
                       "{pk_col}" AS original_node_id
                FROM {node_table_name}
                ORDER BY csr_index
            """)
            print(f"  Created {mapping_table_name}")
            
            vertex_type_to_table[vertex_type] = node_table_name
            vertex_indices[vertex_type] = mapping_table_name
        else:
            print(f"  Warning: No vertices found for type {vertex_type}")
    
    # Read edges and convert to CSR format
    print("\nStep 2: Reading edges and building CSR format...")
    
    for i in range(graph_info.edge_info_num()):
        edge_info = graph_info.get_edge_info_by_index(i)
        edge_type = edge_info.get_edge_type()
        src_type = edge_info.get_src_type()
        dst_type = edge_info.get_dst_type()
        
        print(f"\n  Processing edge {edge_type}: {src_type} -> {dst_type}")
        
        # Get source and destination mapping tables
        src_mapping = vertex_indices.get(src_type)
        dst_mapping = vertex_indices.get(dst_type)
        
        if not src_mapping or not dst_mapping:
            print(f"    Warning: Missing mapping tables for {src_type} -> {dst_type}")
            continue
        
        # Get vertex counts
        src_table = vertex_type_to_table.get(src_type)
        num_src_nodes = con.execute(f"SELECT COUNT(*) FROM {src_table}").fetchone()[0] if src_table else 0
        print(f"    Source nodes: {num_src_nodes}")
        
        # Read edge data
        edge_dir = graph_path / Path(edge_info.get_prefix())
        
        # Find adjacency list and property directories
        adj_list_dir = None
        prop_dirs = {}
        
        for item in edge_dir.iterdir():
            if item.is_dir() and item.name.startswith("ordered_by_"):
                adj_list_dir = item
            elif item.is_dir() and item.name.startswith("_"):
                prop_name = item.name.lstrip("_")
                prop_dirs[prop_name] = item
        
        if not adj_list_dir:
            print(f"    Warning: No adjacency list directory found")
            continue
        
        # Read adjacency list
        edges_list = []
        adj_list_path = adj_list_dir / "adj_list"
        for chunk_file in sorted(adj_list_path.rglob("*")):
            if chunk_file.is_file():
                table = pq.read_table(str(chunk_file))
                for i in range(table.num_rows):
                    src_idx = table.column(0)[i].as_py()
                    dst_idx = table.column(1)[i].as_py()
                    edges_list.append((src_idx, dst_idx))
        
        print(f"    Edges found: {len(edges_list)}")
        
        # Read properties
        edge_props = {}
        for prop_name, prop_dir in prop_dirs.items():
            prop_data = []
            for chunk_file in sorted(prop_dir.rglob("*")):
                if chunk_file.is_file():
                    table = pq.read_table(str(chunk_file))
                    for i in range(table.num_rows):
                        row = [table.column(j)[i].as_py() for j in range(table.num_columns)]
                        prop_data.append(row)
            if prop_data:
                edge_props[prop_name] = prop_data
                print(f"    Property '{prop_name}': {len(prop_data)} rows")
        
        # Create relations table
        rel_table_name = f"relations_{edge_type}"
        
        prop_cols = []
        if edge_props:
            prop_names = list(edge_props.values())[0][0]
            # Skip first column (_rank) from properties
            prop_cols = prop_names[1:] if prop_names[0] == "_rank" else prop_names
            
            col_defs = "csr_source BIGINT, csr_target BIGINT"
            for prop in prop_cols:
                col_defs += f", {prop} BIGINT"
            
            con.execute(f"CREATE TABLE {rel_table_name} ({col_defs})")
            
            # Insert edges with properties
            for i, (src_idx, dst_idx) in enumerate(edges_list):
                values = [src_idx, dst_idx]
                for prop_data in edge_props.values():
                    if i < len(prop_data):
                        # Skip _rank column
                        prop_vals = prop_data[i][1:] if prop_data[i][0] == "_rank" else prop_data[i]
                        values.extend(prop_vals)
                
                con.execute(f"INSERT INTO {rel_table_name} VALUES ({', '.join(str(v) for v in values)})")
        else:
            con.execute(f"CREATE TABLE {rel_table_name} (csr_source BIGINT, csr_target BIGINT)")
            for src_idx, dst_idx in edges_list:
                con.execute(f"INSERT INTO {rel_table_name} VALUES ({src_idx}, {dst_idx})")
        
        # Build CSR indptr
        indptr_table = f"{csr_table_name}_indptr_{edge_type}"
        con.execute(f"""
            CREATE TABLE {indptr_table} AS
            WITH node_range AS (
                SELECT unnest(range(0, {num_src_nodes})) AS node_id
            ),
            degrees AS (
                SELECT csr_source AS src, COUNT(*) AS deg
                FROM {rel_table_name}
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
            ORDER BY node_id
        """)
        
        # Add leading zero
        temp_table = f"{indptr_table}_temp"
        con.execute(f"DROP TABLE IF EXISTS {temp_table}")
        con.execute(f"CREATE TABLE {temp_table} (ptr BIGINT)")
        con.execute(f"INSERT INTO {temp_table} VALUES (CAST(0 AS BIGINT))")
        con.execute(f"INSERT INTO {temp_table} SELECT ptr FROM {indptr_table}")
        con.execute(f"DROP TABLE {indptr_table}")
        con.execute(f"ALTER TABLE {temp_table} RENAME TO {indptr_table}")
        
        # Build CSR indices
        indices_table = f"{csr_table_name}_indices_{edge_type}"
        
        col_defs = "target BIGINT"
        for prop in prop_cols:
            col_defs += f", {prop} BIGINT"
        
        con.execute(f"""
            CREATE TABLE {indices_table} AS
            SELECT csr_target AS target{', ' + ', '.join(prop_cols) if prop_cols else ''}
            FROM {rel_table_name}
            ORDER BY csr_source, csr_target
        """)
        
        print(f"    Created {indptr_table} and {indices_table}")
        
        # Drop temporary relations table
        con.execute(f"DROP TABLE IF EXISTS {rel_table_name};")
    
    # Copy all node tables with prefix
    for src_type, src_table in vertex_type_to_table.items():
        src_table_prefixed = f"{csr_table_name}_{src_table}"
        con.execute(f"CREATE OR REPLACE TABLE {src_table_prefixed} AS SELECT * FROM {src_table}")
    
    # Count total nodes and edges
    total_nodes = 0
    for src_type, src_table in vertex_type_to_table.items():
        count = con.execute(f"SELECT COUNT(*) FROM {src_table}").fetchone()[0]
        total_nodes += count
    
    total_edges = 0
    for i in range(graph_info.edge_info_num()):
        edge_info = graph_info.get_edge_info_by_index(i)
        edge_type = edge_info.get_edge_type()
        indices_table = f"{csr_table_name}_indices_{edge_type}"
        result = con.execute(f"SELECT COUNT(*) FROM {indices_table}").fetchone()
        if result:
            total_edges += result[0]
    
    # Create global metadata
    con.execute(f"""
        CREATE TABLE {csr_table_name}_metadata AS
        SELECT {total_nodes}::BIGINT AS n_nodes, {total_edges}::BIGINT AS n_edges, {directed}::BOOLEAN AS directed
    """)
    
    print(f"\n✅ Conversion complete: {total_nodes} nodes, {total_edges} edges")
    
    # Export to parquet and generate schema.cypher
    print("\nStep 3: Exporting to Parquet and generating schema.cypher...")
    
    output_path = Path(output_db_path)
    parquet_dir = output_path.parent / output_path.stem
    parquet_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Parquet output directory: {parquet_dir}")
    
    # Get all tables
    result = con.execute("SHOW TABLES").fetchall()
    all_tables = [row[0] for row in result]
    
    # Export each table
    for table_name in all_tables:
        parquet_file = parquet_dir / f"{table_name}.parquet"
        con.execute(f"COPY {table_name} TO '{parquet_file}' (FORMAT 'parquet')")
        print(f"  Exported: {table_name} -> {parquet_file.name}")
    
    # Generate schema.cypher
    schema_lines = []
    
    # Compute storage path
    storage_path = f"./{parquet_dir.name}/{csr_table_name}"
    
    # Generate NODE TABLE definitions
    schema_lines = []
    for vertex_type, src_table in vertex_type_to_table.items():
        table_name = f"{csr_table_name}_{src_table}"
        try:
            cols = con.execute(f"DESCRIBE {table_name}").fetchall()
            col_defs = []
            pk_col = None
            for col in cols:
                col_name, col_type = col[0], col[1]
                cypher_type = duckdb_type_to_cypher_type(col_type)
                col_defs.append(f"{col_name} {cypher_type}")
                if pk_col is None:
                    pk_col = col_name
            
            cols_str = ", ".join(col_defs)
            schema_lines.append(f"CREATE NODE TABLE {vertex_type}({cols_str}, PRIMARY KEY({pk_col})) WITH (storage = '{storage_path}');")
        except Exception as e:
            print(f"Warning: Could not generate schema for node table {table_name}: {e}")
    
    # Generate REL TABLE definitions
    for i in range(graph_info.edge_info_num()):
        edge_info = graph_info.get_edge_info_by_index(i)
        edge_type = edge_info.get_edge_type()
        src_type = edge_info.get_src_type()
        dst_type = edge_info.get_dst_type()
        rel_name = edge_type
        
        src_table = vertex_type_to_table.get(src_type, f"nodes_{src_type}")
        dst_table = vertex_type_to_table.get(dst_type, f"nodes_{dst_type}")
        
        indices_table = f"{csr_table_name}_indices_{edge_type}"
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
            schema_lines.append(
                f"CREATE REL TABLE {rel_name}(FROM {src_type} TO {dst_type}"
                f"{', ' + props_str if props_str else ''}) WITH (storage = '{storage_path}');"
            )
        except Exception as e:
            print(f"Warning: Could not generate schema for rel table {rel_name}: {e}")
    
    schema_cypher = "\n".join(schema_lines) + "\n"
    schema_file = parquet_dir / "schema.cypher"
    schema_file.write_text(schema_cypher)
    print(f"  Generated: schema.cypher")
    
    con.close()
    
    print(f"\n✅ All data saved to: {output_db_path}")
    print(f"✅ Parquet files saved to: {parquet_dir}")


def main():
    """Main function to convert GraphAr to icebug-format."""
    parser = argparse.ArgumentParser(
        description="Convert GraphAr format to icebug-format"
    )
    parser.add_argument(
        "--graphar-dir",
        type=str,
        required=True,
        help="Path to directory with GraphAr data",
    )
    parser.add_argument(
        "--output-db",
        type=str,
        required=True,
        help="Output DuckDB database path",
    )
    parser.add_argument(
        "--csr-table",
        type=str,
        default="graph",
        help="Table name prefix for CSR data (default: graph)",
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        help="Treat graph as directed (default: undirected)",
    )
    
    args = parser.parse_args()
    
    print("=== GraphAr to Graph-Std Converter ===\n")
    print(f"GraphAr directory: {args.graphar_dir}")
    print(f"CSR output database: {args.output_db}")
    print(f"CSR table prefix: {args.csr_table}")
    print(f"Directed: {args.directed}")
    
    convert_graphar_to_graph_std(
        graphar_dir=args.graphar_dir,
        output_db_path=args.output_db,
        csr_table_name=args.csr_table,
        directed=args.directed,
    )
    
    print("\n=== Conversion Completed Successfully! ===")


if __name__ == "__main__":
    main()
