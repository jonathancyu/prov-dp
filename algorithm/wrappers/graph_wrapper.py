import itertools
from pathlib import Path

import networkx as nx
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

    _edge_lookup: dict[int, EdgeWrapper]

    _subtree_lookup: dict[str, dict[int, list[EdgeWrapper]]]

    def __init__(self, json_path: Path = None):
        if json_path is None:
            graph = Graph()
        else:
            graph = Graph.load_file(json_path)
            self.json_path = json_path
            self.source_edge_id = get_edge_id(str(json_path.stem))
        self.graph = graph
        self.tree_sizes = {
            IN: 0, OUT: 0
        }
        self.nodes = []
        self.edges = []
        self._node_lookup = {}
        self._edge_lookup = {}
        self._add_nodes(self.graph.nodes)
        self._add_edges(self.graph.edges)

        self._set_node_times()

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

    def get_edge(self, edge_id: int) -> EdgeWrapper:
        return self._edge_lookup.get(edge_id)

    def get_node_type(self, node_id: int) -> NodeType:
        return self.get_node(node_id).get_type()

    def get_edge_type(self, edge: EdgeWrapper) -> EdgeType:
        return EdgeType(
            edge=edge.edge,
            src_type=self.get_node(edge.get_src_id()).get_type(),
            dst_type=self.get_node(edge.get_dst_id()).get_type()
        )

    def get_subtree(self,
                    root_edge_id: int,
                    direction: str,
                    visited_node_ids=None) -> list[EdgeWrapper]:
        subtree = self._subtree_lookup[direction].get(root_edge_id)
        if subtree is not None:
            return subtree

        root_edge = self.get_edge(root_edge_id)
        subtree: list[EdgeWrapper] = [root_edge]

        root_node_id = root_edge.node_ids[direction]
        if visited_node_ids is None:
            visited_node_ids = []
        if root_node_id in visited_node_ids:
            return []
        visited_node_ids.append(root_node_id)

        root_node = self.get_node(root_node_id)
        for edge_id in root_node.edge_ids[direction]:
            subtree.extend(self.get_subtree(edge_id, direction, visited_node_ids))
        self._subtree_lookup[direction][root_edge_id] = subtree
        return subtree

    def get_tree_size(self, direction: str, root_edge_id: int = None) -> int:
        if root_edge_id is None:
            root_edge_id = self.source_edge_id
        return len(self.get_subtree(root_edge_id, direction))

    def add_edge(self, edge: EdgeWrapper) -> None:
        self.edges.append(edge)

    def add_node(self, node: NodeWrapper) -> None:
        self.nodes.append(node)

    def to_dot(self) -> Digraph:
        return Graph(
            vertices=[node.node for node in self.nodes],
            edges=[edge.edge for edge in self.edges]
        ).to_dot()

    def to_nx(self) -> nx.DiGraph:
        G: nx.DiGraph = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node.get_id(),
                       token=node.get_token()
                       )
        for edge in self.edges:
            G.add_edge(edge.node_ids[IN],
                       edge.node_ids[OUT],
                       token=edge.get_token()
                       )
        return G

    def _add_nodes(self, nodes: list[Node]):
        for node in nodes:
            node_wrapper = NodeWrapper(node)
            self.nodes.append(node_wrapper)
            self._node_lookup[node.id] = node_wrapper

    def _add_edges(self, edges: list[Edge]):
        for edge in edges:
            edge_wrapper = EdgeWrapper(edge)
            self.edges.append(edge_wrapper)
            self._edge_lookup[edge_wrapper.get_ref_id()] = edge_wrapper

    def get_paths(self) -> list[list[EdgeWrapper]]:
        paths: dict[str, list[list[EdgeWrapper]]] = {
            IN: [], OUT: []
        }
        for direction in [IN, OUT]:
            paths[direction].extend(self._get_paths_in_direction(self.get_edge(self.source_edge_id), direction))

        # Invert the IN (backtrack) paths
        paths[IN] = [path[::-1] for path in paths[IN]]
        # Trim source edge so that it's not included twice
        paths[OUT] = [path[:-1] for path in paths[OUT]]

        # Combine the paths into a single list and return
        path_combinations = itertools.product(paths[IN], paths[OUT])
        return [list(itertools.chain.from_iterable(path_pair)) for path_pair in path_combinations]

    def _get_paths_in_direction(self,
                                source: EdgeWrapper,
                                direction: str,
                                current_path: list[EdgeWrapper] = None,
                                visited_ids: list[int] = None
                                ) -> list[list[EdgeWrapper]]:
        # copy current path to avoid mutation
        current_path = current_path or []
        visited_ids = visited_ids or []
        source_id = source.get_ref_id()
        if source in current_path or source_id in visited_ids:
            return []
        visited_ids.append(source.get_ref_id())
        current_path = (current_path or []) + [source]

        node_id = source.node_ids[direction]
        node = self.get_node(node_id)

        if len(node.edge_ids[direction]) == 0:
            return [current_path]

        paths = []
        for edge_id in node.edge_ids[direction]:
            edge = self.get_edge(edge_id)
            new_paths = self._get_paths_in_direction(edge, direction, current_path, visited_ids)
            paths.extend(new_paths)

        return paths
