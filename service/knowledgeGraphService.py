import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.github.client import fetch_repo_tree, fetch_file_content_by_sha, is_code_file
from utils.graph.builder import build_structural_graph, resolve_imports, serialize_graph
from utils.graph.pagerank import compute_pagerank
from utils.ai.symbol_extractor import extract_symbols_from_file


def build_knowledge_graph(owner, repository):
    raw_tree = fetch_repo_tree(owner, repository)
    nodes, adj_list = build_structural_graph(raw_tree)
    code_files = [path for path, node in nodes.items() if node.node_type == "file" and is_code_file(path)]

    for path in code_files:
        content = fetch_file_content_by_sha(owner, repository, nodes[path].sha)
        extract_symbols_from_file(path, content, nodes, adj_list)
    
    resolve_imports(nodes, adj_list)
    compute_pagerank(nodes, adj_list)
    graph = serialize_graph(nodes, adj_list)
    nodes_sha_lookup = {path: node.sha for path, node in nodes.items() if node.sha}
    return graph, nodes_sha_lookup
