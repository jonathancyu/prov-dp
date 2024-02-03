from .edge_wrapper import EdgeWrapper
from .node_wrapper import NodeWrapper


class Subtree:
    edges: list[EdgeWrapper]
    nodes: list[NodeWrapper]
    depth: int

    def __init__(self,
                 edges: list[EdgeWrapper],
                 nodes: list[NodeWrapper],
                 depth: int = None
                 ):
        self.edges = edges
        self.nodes = nodes
        self.depth = depth
