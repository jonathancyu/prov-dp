from graphson import Graph, Edge, NodeType, EdgeType
from .node_wrapper import NodeWrapper
from .edge_wrapper import EdgeWrapper
class GraphWrapper:
    graph: Graph
    nodes: list[NodeWrapper]
    edges: list[EdgeWrapper]

    _node_lookup: dict[int, NodeWrapper]
    _edge_lookup: dict[tuple[int,int], Edge]
    def __init__(self, graph: Graph):
        self.graph = graph
        self.nodes = [ NodeWrapper(node) for node in self.graph.nodes ]
        self.edges = [ EdgeWrapper(edge) for edge in self.graph.edges ]
        self._node_lookup = { node.get_id(): node for node in self.nodes }
        self._edge_lookup = {
            (edge.get_src_id(), edge.get_dst_id()): edge
            for edge in self.edges}
        self._set_node_times()

    def _set_node_times(self) -> None:
        included_nodes: set[int] = set()
        sorted_edges = sorted(self.edges, key=lambda e: e.get_time(), reverse=True)
        for edge in sorted_edges:
            src_node = self.get_node(edge.get_src_id())
            dst_node = self.get_node(edge.get_dst_id())
            if src_node is None or dst_node is None:
                continue
            src_node.add_outgoing()
            dst_node.add_incoming()

            # TODO: Is this how we should set time?
            for node_id in [edge.get_src_id(), edge.get_dst_id()]:
                if node_id in included_nodes:
                    continue
                node = self.get_node(node_id)
                node.set_time(edge.get_time())
    def get_node(self, id: int) -> NodeWrapper:
        return self._node_lookup.get(id)

    def get_edge(self, id: int) -> EdgeWrapper:
        return self._edge_lookup.get(id)

    def get_node_type(self, id: int) -> NodeType:
        return self.get_node(id)

    def get_edge_type(self, edge: EdgeWrapper) -> EdgeType:
        return EdgeType(
            edge = edge.edge,
            src_type = self.get_node(edge.get_src_id()).get_type(),
            dst_type = self.get_node(edge.get_dst_id()).get_type()
        )
