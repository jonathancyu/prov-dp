from .edge_wrapper import EdgeWrapper
from .graph_wrapper import GraphWrapper
from .node_wrapper import NodeWrapper


class Subgraph:
    graph: GraphWrapper
    root_edge_id: int
    direction: str
    edges: list[EdgeWrapper]
    nodes: list[NodeWrapper]
    depth: int

    def __init__(self,
                 parent_graph: GraphWrapper,
                 root_edge_id: int,
                 direction: str,
                 depth: int
                 ):
        self.graph = parent_graph
        self.root_edge_id = root_edge_id
        self.direction = direction
        self.edges = parent_graph.get_subtree(root_edge_id, direction)
        nodes = set()
        for edge in self.edges:
            for node_id in edge.node_ids.values():
                nodes.add(parent_graph.get_node(node_id))
        self.nodes = list(nodes)
        self.depth = depth
