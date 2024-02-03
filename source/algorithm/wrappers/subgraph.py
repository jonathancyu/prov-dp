from .edge_wrapper import EdgeWrapper
from .graph_wrapper import GraphWrapper
from .node_wrapper import NodeWrapper


class Subgraph:
    graph: GraphWrapper
    edges: list[EdgeWrapper]
    nodes: list[NodeWrapper]
    depth: int

    def __init__(self,
                 parent_graph: GraphWrapper,
                 edges: list[EdgeWrapper],
                 depth: int = None
                 ):
        self.graph = parent_graph
        self.depth = depth
        self.edges = edges
        nodes = set()
        for edge in self.edges:
            for node_id in edge.node_ids.values():
                nodes.add(parent_graph.get_node(node_id))
        self.nodes = list(nodes)
        self.depth = depth
