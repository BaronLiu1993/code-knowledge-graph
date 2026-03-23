import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from utils.graph.builder import (
    build_structural_graph,
    add_symbol_node,
    add_import_edge,
    resolve_imports,
    group_files_by_module,
    serialize_graph,
)


SAMPLE_TREE = {
    "tree": [
        {"path": "service", "type": "tree"},
        {"path": "service/userService.py", "type": "blob"},
        {"path": "service/authService.py", "type": "blob"},
        {"path": "models", "type": "tree"},
        {"path": "models/userModel.py", "type": "blob"},
        {"path": "router", "type": "tree"},
        {"path": "router/userRouter.py", "type": "blob"},
    ]
}


def test_build_structural_graph_creates_all_nodes():
    nodes, _ = build_structural_graph(SAMPLE_TREE)
    assert "service" in nodes
    assert "service/userService.py" in nodes
    assert "models/userModel.py" in nodes


def test_build_structural_graph_node_types():
    nodes, _ = build_structural_graph(SAMPLE_TREE)
    assert nodes["service"].node_type == "folder"
    assert nodes["service/userService.py"].node_type == "file"


def test_build_structural_graph_contains_edges():
    nodes, adj_list = build_structural_graph(SAMPLE_TREE)
    service_edges = adj_list["service"]
    targets = [e.target for e in service_edges]
    assert "service/userService.py" in targets
    assert "service/authService.py" in targets


def test_build_structural_graph_all_nodes_have_adj_key():
    nodes, adj_list = build_structural_graph(SAMPLE_TREE)
    for path in nodes:
        assert path in adj_list


def test_build_structural_graph_missing_parent_created():
    tree = {"tree": [{"path": "deep/nested/file.py", "type": "blob"}]}
    nodes, adj_list = build_structural_graph(tree)
    assert "deep/nested" in nodes
    assert nodes["deep/nested"].node_type == "folder"


def test_add_symbol_node_creates_symbol():
    nodes, adj_list = build_structural_graph(SAMPLE_TREE)
    symbol_id = add_symbol_node("service/userService.py", "UserService", "class", nodes, adj_list)
    assert symbol_id in nodes
    assert nodes[symbol_id].node_type == "class"
    assert nodes[symbol_id].label == "UserService"


def test_add_symbol_node_creates_defines_edge():
    nodes, adj_list = build_structural_graph(SAMPLE_TREE)
    symbol_id = add_symbol_node("service/userService.py", "getUser", "function", nodes, adj_list)
    targets = [e.target for e in adj_list["service/userService.py"]]
    assert symbol_id in targets


def test_add_import_edge_appended():
    nodes, adj_list = build_structural_graph(SAMPLE_TREE)
    add_import_edge("service/userService.py", "../models/userModel", adj_list)
    predicates = [e.predicate for e in adj_list["service/userService.py"]]
    assert "imports" in predicates


def test_resolve_imports_matches_existing_node():
    nodes, adj_list = build_structural_graph(SAMPLE_TREE)
    add_import_edge("service/userService.py", "userModel", adj_list)
    resolve_imports(nodes, adj_list)
    import_edges = [e for e in adj_list["service/userService.py"] if e.predicate == "imports"]
    assert any(e.target == "models/userModel.py" for e in import_edges)


def test_group_files_by_module():
    nodes, _ = build_structural_graph(SAMPLE_TREE)
    groups = group_files_by_module(nodes)
    assert "service" in groups
    assert "models" in groups
    assert "service/userService.py" in groups["service"]
    assert "models/userModel.py" in groups["models"]


def test_group_files_excludes_folders():
    nodes, _ = build_structural_graph(SAMPLE_TREE)
    groups = group_files_by_module(nodes)
    for paths in groups.values():
        for p in paths:
            assert nodes[p].node_type == "file"


def test_serialize_graph_structure():
    nodes, adj_list = build_structural_graph(SAMPLE_TREE)
    result = serialize_graph(nodes, adj_list)
    assert "nodes" in result
    assert isinstance(result["nodes"], list)


def test_serialize_graph_node_fields():
    nodes, adj_list = build_structural_graph(SAMPLE_TREE)
    result = serialize_graph(nodes, adj_list)
    node = result["nodes"][0]
    assert "name" in node
    assert "type" in node
    assert "label" in node
    assert "description" in node
    assert "pagerank" in node
    assert "relationships" in node


def test_serialize_graph_relationship_fields():
    nodes, adj_list = build_structural_graph(SAMPLE_TREE)
    result = serialize_graph(nodes, adj_list)
    folder_node = next(n for n in result["nodes"] if n["type"] == "folder" and n["relationships"])
    rel = folder_node["relationships"][0]
    assert "predicate" in rel
    assert "target" in rel
    assert "target_pagerank" in rel
