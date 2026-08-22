from icebug_format.graphar import YamlGraphInfo

def test_yaml_parser_fallback_loads_simple_graph(tmp_path):
    # 1. Write the exact validated Vertex YAML
    vertex_file = tmp_path / "person.vertex.yml"
    vertex_file.write_text("""type: person
version: gar/v1
chunk_size: 100
prefix: vertex/person/
property_groups:
  - prefix: id/
    file_type: parquet
    properties:
      - name: id
        data_type: int64
        is_primary: true
""")

    # 2. Write the exact validated Edge YAML
    edge_file = tmp_path / "person_knows_person.edge.yml"
    edge_file.write_text("""src_type: person
edge_type: knows
dst_type: person
version: gar/v1
chunk_size: 1024
prefix: edge/person_knows_person/
directed: true
adj_lists:
  - ordered: false
    aligned_by: src
    prefix: unordered_by_src/
    file_type: parquet
""")

    # 3. Write the exact validated Graph YAML
    graph_file = tmp_path / "test.graph.yml"
    graph_file.write_text("""name: simple_graph
version: gar/v1
vertices:
  - person.vertex.yml
edges:
  - person_knows_person.edge.yml
""")

    graph_info = YamlGraphInfo.load(str(graph_file))

    assert graph_info.vertex_info_num() == 1
    v_info = graph_info.get_vertex_info_by_index(0)
    assert v_info.get_type() == "person"
    assert v_info.get_prefix() == "vertex/person/"

    assert graph_info.edge_info_num() == 1
    e_info = graph_info.get_edge_info_by_index(0)
    assert e_info.get_edge_type() == "knows"
    assert e_info.get_src_type() == "person"
    assert e_info.get_dst_type() == "person"
    assert e_info.get_prefix() == "edge/person_knows_person/"