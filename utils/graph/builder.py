from collections import defaultdict


class GraphNode:
    def __init__(self, name, node_type, label="", sha=""):
        self.name = name
        self.node_type = node_type
        self.label = label or name.split("/")[-1]
        self.sha = sha
        self.description = ""
        self.pagerank = 0.0


class GraphEdge:
    def __init__(self, source, target, predicate, confidence=1.0):
        self.source = source
        self.target = target
        self.predicate = predicate
        self.confidence = confidence


def build_structural_graph(raw_tree):
    nodes = {}
    adj_list = defaultdict(list)

    for item in raw_tree.get("tree", []):
        path = item["path"]
        node_type = "folder" if item["type"] == "tree" else "file"
        nodes[path] = GraphNode(path, node_type, sha=item.get("sha", ""))
        adj_list[path]

        if "/" in path:
            parent = path.rsplit("/", 1)[0]
            if parent not in nodes:
                nodes[parent] = GraphNode(parent, "folder")
            adj_list[parent].append(GraphEdge(parent, path, "contains", confidence=1.0))

    return nodes, adj_list


def add_symbol_node(file_path, symbol_name, symbol_type, nodes, adj_list):
    symbol_id = f"{file_path}::{symbol_name}"
    nodes[symbol_id] = GraphNode(symbol_id, symbol_type, label=symbol_name)
    adj_list[symbol_id]
    adj_list[file_path].append(GraphEdge(file_path, symbol_id, "defines", confidence=1.0))
    return symbol_id


def add_import_edge(file_path, from_module, adj_list):
    adj_list[file_path].append(
        GraphEdge(file_path, from_module, "imports", confidence=0.9)
    )


def add_semantic_edge(source, target, predicate, adj_list, confidence=0.8):
    adj_list[source].append(GraphEdge(source, target, predicate, confidence=confidence))


def resolve_imports(nodes, adj_list):
    for edges in adj_list.values():
        for edge in edges:
            if edge.predicate != "imports":
                continue
            resolved = _find_matching_node(edge.target, nodes)
            if resolved:
                edge.target = resolved


def _find_matching_node(module_ref, nodes):
    normalized = module_ref.replace("./", "").replace("../", "").replace("\\", "/")
    for path in nodes:
        if path.endswith(normalized) or normalized in path:
            return path
    stem = normalized.split("/")[-1]
    for path in nodes:
        if path.split("/")[-1].startswith(stem):
            return path
    return None


def group_files_by_module(nodes):
    groups = {}
    for path, node in nodes.items():
        if node.node_type != "file":
            continue
        module = path.split("/")[0] if "/" in path else "root"
        groups.setdefault(module, []).append(path)
    return groups


def serialize_graph(nodes, adj_list):
    sorted_nodes = sorted(nodes.values(), key=lambda n: n.pagerank, reverse=True)
    return {
        "nodes": [
            {
                "name": n.name,
                "type": n.node_type,
                "label": n.label,
                "description": n.description,
                "pagerank": round(n.pagerank, 6),
                "relationships": [
                    {
                        "predicate": e.predicate,
                        "target": e.target,
                        "target_pagerank": round(nodes[e.target].pagerank, 6) if e.target in nodes else 0.0,
                    }
                    for e in adj_list.get(n.name, [])
                ],
            }
            for n in sorted_nodes
        ],
    }
