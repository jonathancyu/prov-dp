from .edge_wrapper import EdgeWrapper
from .node_wrapper import NodeWrapper
from .graph_wrapper import GraphWrapper


class Subgraph:
    parent_graph: GraphWrapper
    root_edge_id: int
    edges: list[EdgeWrapper]
    nodes: list[NodeWrapper]
    depth: int

    def __init__(self,
                 parent_graph: GraphWrapper,
                 root_edge_id: int,
                 direction: str,
                 depth: int
                 ):
        self.parent_graph = parent_graph
        self.root_edge_id = root_edge_id
        self.edges = parent_graph.get_subtree(root_edge_id, direction)
        nodes = set()
        for edge in self.edges:
            for node_id in edge.node_ids.values():
                nodes.add(parent_graph.get_node(node_id))
        self.nodes = list(nodes)
        self.depth = depth
