import yaml
from icebug_format.graphar import YamlGraphInfo

def test_yaml_parser_fallback_loads_empty_graph(tmp_path):
    """Test that our fallback YAML parser can read a basic graph configuration."""
    yaml_content = {
        "vertices": [],
        "edges": []
    }
    yaml_file = tmp_path / "test.graph.yml"

    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    graph_info = YamlGraphInfo.load(str(yaml_file))

    assert graph_info.vertex_info_num() == 0
    assert graph_info.edge_info_num() == 0