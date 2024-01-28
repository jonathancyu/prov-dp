import networkx as nx
from .wrappers import Subgraph, GraphWrapper, IN, OUT

def to_nx(graph: Subgraph | GraphWrapper) -> nx.DiGraph:
    digraph: nx.DiGraph = nx.DiGraph()

    # NetworkX node IDs must index at 0
    node_ids = {node.get_id(): i
                for i, node in enumerate(graph.nodes)}
    for node in graph.nodes:
        digraph.add_node(node_ids[node.get_id()],
                         feature=node.get_token()
                         )
    for edge in graph.edges:
        digraph.add_edge(node_ids[edge.node_ids[IN]],
                         node_ids[edge.node_ids[OUT]],
                         feature=edge.get_token()
                         )
    return digraph
