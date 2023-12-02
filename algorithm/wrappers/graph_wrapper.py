from pathlib import Path

from graphviz import Digraph

from graphson import Graph, NodeType, Node, EdgeType, Edge
from utility import get_edge_id
from .node_wrapper import NodeWrapper, IN, OUT
from .edge_wrapper import EdgeWrapper


class GraphWrapper:
    graph: Graph
    nodes: list[NodeWrapper]
    edges: list[EdgeWrapper]
    json_path: Path
    source_edge_id: int

    _node_lookup: dict[int, NodeWrapper]
    _edge_lookup: dict[EdgeWrapper, EdgeWrapper]

    _edge_id_lookup: dict[int, EdgeWrapper]

    _subtree_lookup: dict[str, dict[int, list[EdgeWrapper]]]

    def __init__(self, json_path: Path = None):
        if json_path is None:
            graph = Graph()
        else:
            graph = Graph.load_file(json_path)
            self.json_path = json_path
            self.source_edge_id = get_edge_id(str(json_path.stem))
        self.graph = graph
        self.nodes = [NodeWrapper(node) for node in self.graph.nodes]
        self.edges = [EdgeWrapper(edge) for edge in self.graph.edges]

        self._node_lookup = {node.get_id(): node for node in self.nodes}
        self._edge_lookup = {
            (edge.get_src_id(), edge.get_dst_id()): edge
            for edge in self.edges}
        self._set_node_times()

        self._edge_id_lookup = {edge.get_ref_id(): edge for edge in self.edges}

        self._subtree_lookup = {
            IN: {}, OUT: {}
        }

    # TODO: split this func, too many responsibilities
    def _set_node_times(self) -> None:
        included_nodes: set[int] = set()
        sorted_edges = sorted(self.edges, key=lambda e: e.get_time(), reverse=True)
        for edge in sorted_edges:
            src_node = self.get_node(edge.get_src_id())
            dst_node = self.get_node(edge.get_dst_id())
            if src_node is None or dst_node is None:  # TODO why does this occur?
                continue
            edge_id = edge.get_ref_id()
            edge_type = self.get_edge_type(edge)
            src_node.add_outgoing(edge_id, edge_type)
            dst_node.add_incoming(edge_id, edge_type)

            # TODO: Is this how we should set time?
            for node_id in [edge.get_src_id(), edge.get_dst_id()]:
                if node_id in included_nodes:
                    continue
                node = self.get_node(node_id)
                node.time = edge.get_time()

    def get_node(self, node_id: int) -> NodeWrapper:
        return self._node_lookup.get(node_id)

    def get_edge(self, edge: EdgeWrapper) -> EdgeWrapper:
        return self._edge_lookup.get(edge)

    def get_edge_by_id(self, edge_id: int) -> EdgeWrapper:
        return self._edge_id_lookup.get(edge_id)

    def get_node_type(self, node_id: int) -> NodeType:
        return self.get_node(node_id).get_type()

    def get_edge_type(self, edge: EdgeWrapper) -> EdgeType:
        return EdgeType(
            edge=edge.edge,
            src_type=self.get_node(edge.get_src_id()).get_type(),
            dst_type=self.get_node(edge.get_dst_id()).get_type()
        )

    def get_subtree(self, root_edge_id: int, direction: str) -> list[EdgeWrapper]:
        subtree = self._subtree_lookup[direction].get(root_edge_id)
        if subtree is not None:
            return subtree

        root_edge = self.get_edge_by_id(root_edge_id)
        subtree: list[EdgeWrapper] = [root_edge]

        root_node_id = root_edge.node_ids[direction]
        root_node = self.get_node(root_node_id)
        #  Base case: no outgoing edges, assuming we don't have a cycle (famous last words)
        for edge_id in root_node.edge_ids[direction]:
            subtree.extend(self.get_subtree(edge_id, direction))

        self._subtree_lookup[direction][root_edge_id] = subtree
        return subtree

    def get_tree_size(self, root_edge_id: int, direction: str) -> int:
        return len(self.get_subtree(root_edge_id, direction))

    def add_edge(self, edge: EdgeWrapper) -> None:
        self.edges.append(edge)

    def add_node(self, node: NodeWrapper) -> None:
        self.nodes.append(node)

    def to_dot(self) -> Digraph:
        graph = Graph()

        return Graph(
            vertices=[node.node for node in self.nodes],
            edges=[edge.edge for edge in self.edges]
        ).to_dot()