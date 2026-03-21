import networkx as nx

def compute_pagerank(nodes, adj_list, damping=0.85):
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes.keys())

    for source, edges in adj_list.items():
        for edge in edges:
            graph.add_edge(source, edge.target, weight=edge.confidence)

    scores = nx.pagerank(graph, alpha=damping, weight="weight")

    for name, node in nodes.items():
        node.pagerank = scores.get(name, 0.0)

    return scores
