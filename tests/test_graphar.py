import yaml
from icebug_format.graphar import YamlGraphInfo

def test_yaml_parser_fallback_loads_simple_graph(tmp_path):
    """Test that the custom YAML parser correctly reads a multi-file graph configuration."""
    
    # 1. Create a fully compliant vertex YAML file
    vertex_content = {
        "type": "person",
        "prefix": "vertex/person/",
        "chunk_size": 100,
        "version": "gar/v1",
        "property_groups": [
            {
                "file_type": "parquet",
                "properties": [
                    {"name": "id", "data_type": "int64", "is_primary": True}
                ]
            }
        ]
    }
    vertex_file = tmp_path / "person.vertex.yml"
    with open(vertex_file, "w") as f:
        yaml.dump(vertex_content, f)

    # 2. Create a fully compliant edge YAML file
    edge_content = {
        "src_type": "person",
        "edge_type": "knows",
        "dst_type": "person",
        "prefix": "edge/person_knows_person/",
        "chunk_size": 1024,
        "version": "gar/v1",
        "directed": True,
        "adj_lists": [
            {
                "ordered": False,
                "aligned_by": "src",
                "file_type": "parquet"
            }
        ]
    }
    edge_file = tmp_path / "person_knows_person.edge.yml"
    with open(edge_file, "w") as f:
        yaml.dump(edge_content, f)

    # 3. Create the main graph YAML file
    graph_content = {
        "name": "simple_graph",
        "version": "gar/v1",
        "vertices": ["person.vertex.yml"],
        "edges": ["person_knows_person.edge.yml"]
    }
    graph_file = tmp_path / "test.graph.yml"
    with open(graph_file, "w") as f:
        yaml.dump(graph_content, f)

    # 4. Load it using your custom parser
    graph_info = YamlGraphInfo.load(str(graph_file))

    # 5. Verify the vertex data matches exactly
    assert graph_info.vertex_info_num() == 1
    v_info = graph_info.get_vertex_info_by_index(0)
    assert v_info.get_type() == "person"
    assert v_info.get_prefix() == "vertex/person/"

    # 6. Verify the edge data matches exactly
    assert graph_info.edge_info_num() == 1
    e_info = graph_info.get_edge_info_by_index(0)
    assert e_info.get_edge_type() == "knows"
    assert e_info.get_src_type() == "person"
    assert e_info.get_dst_type() == "person"
    assert e_info.get_prefix() == "edge/person_knows_person/"