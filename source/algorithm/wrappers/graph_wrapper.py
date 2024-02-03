import itertools
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
from graphviz import Digraph

from source.graphson import Graph, NodeType, Node, EdgeType, Edge
from utility import get_edge_ref_id, logistic_function
from .node_wrapper import NodeWrapper, IN, OUT
from .edge_wrapper import EdgeWrapper


class GraphWrapper:
    graph: Graph
    nodes: list[NodeWrapper]
    edges: list[EdgeWrapper]
    source_edge_id: int | None

    _node_lookup: dict[int, NodeWrapper]
    _edge_lookup: dict[int, EdgeWrapper]
    _subtree_lookup: dict[str, dict[int, list[EdgeWrapper]]]
    _marked_edges: set[int]

    @staticmethod
    def load_file(json_path: Path) -> 'GraphWrapper':
        return GraphWrapper(
            Graph.load_file(json_path),
            get_edge_ref_id(str(json_path.stem))
        )

    def __init__(self,
                 graph: Graph = None,
                 source_edge_ref_id: int = None):
        if graph is None:
            graph = Graph()
        self.graph = graph

        self.nodes = []
        self.edges = []
        self._init_nodes(self.graph.nodes)
        self._init_edges(self.graph.edges)

        # todo: this is convoluted
        if source_edge_ref_id is not None:
            source_edge = [edge for edge in self.edges if edge.get_ref_id() == source_edge_ref_id]
            assert len(source_edge) == 1
            self.source_edge_id = source_edge[0].get_id()
        else:
            self.source_edge_id = None

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
            edge_id = edge.get_id()
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

    def remove_node(self, node: NodeWrapper) -> None:
        self.nodes.remove(node)

    def to_dot(self) -> Digraph:
        return Graph(
            vertices=[node.node for node in self.nodes],
            edges=[edge.edge for edge in self.edges]
        ).to_dot()

    def _init_nodes(self, nodes: list[Node]):
        self._node_lookup = {}
        for node in nodes:
            node_wrapper = NodeWrapper(node)
            self.nodes.append(node_wrapper)
            self._node_lookup[node.id] = node_wrapper

    def _init_edges(self, edges: list[Edge]):
        self._edge_lookup = {}
        for edge in edges:
            edge_wrapper = EdgeWrapper(edge)
            self.edges.append(edge_wrapper)
            self._edge_lookup[edge_wrapper.get_id()] = edge_wrapper

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
        source_ref_id = source.get_id()
        if source in current_path or source_ref_id in visited_ids:
            return []
        visited_ids.append(source.get_id())
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

    def get_next_node_id(self) -> int:
        return max([node.id for node in self.graph.nodes]) + 1

    def get_next_edge_id(self) -> int:
        return max([edge.get_id() for edge in self.edges]) + 1

    def _invert_edge(self, edge_id: int) -> None:
        edge = self.get_edge(edge_id)
        src_id, dst_id = edge.node_ids[IN], edge.node_ids[OUT]
        edge.invert()

        src_node, dst_node = self.get_node(src_id), self.get_node(dst_id)
        src_node.edge_ids[OUT].remove(edge_id)
        src_node.edge_ids[IN].append(edge_id)
        dst_node.edge_ids[IN].remove(edge_id)
        dst_node.edge_ids[OUT].append(edge_id)

    # Step 1. Original graph
    def original_graph(self) -> None:
        pass

    # Step 2. Invert all outgoing edges from files/IPs
    def _invert_outgoing_file_edges(self) -> None:
        edges_to_invert = []
        for node in self.nodes:
            if node.get_type() == NodeType.PROCESS_LET:
                continue
            edges_to_invert.extend(node.edge_ids[OUT])

        for edge_id in edges_to_invert:
            self._invert_edge(edge_id)

    # Step 3. Duplicate file/IP nodes for each incoming edge
    def _duplicate_file_ip_leaves(self) -> None:
        nodes_to_remove = []
        nodes_to_add = []
        for node in self.nodes:
            if node.get_type() == NodeType.PROCESS_LET:
                continue
            # Mark original node for removal, then create a duplicate node for each edge
            nodes_to_remove.append(node)
            for edge_id in node.edge_ids[IN]:
                # Create new node
                new_node = deepcopy(node)
                new_node_id = self.get_next_node_id() + len(nodes_to_add)
                new_node.node.id = new_node_id
                new_node.edge_ids = {IN: [edge_id], OUT: []}
                nodes_to_add.append(new_node)

                # Move edge to new node
                edge = self.get_edge(edge_id)
                edge.node_ids[OUT] = new_node_id
                edge.edge.dst_id = new_node_id
        # Apply changes
        for node in nodes_to_remove:
            self.remove_node(node)
        for node in nodes_to_add:
            self.add_node(node)

    # Step 4
    def _add_ephemeral_root(self) -> None:
        # Create root node
        raw_root_node = Node(
            _id=9999,
            TYPE=NodeType.EPHEMERAL
        )
        root_node = NodeWrapper(raw_root_node)
        self.add_node(root_node)

        # Create root edge for BFS
        source_edge = EdgeWrapper(Edge(
                _id=self.get_next_edge_id(),
                _outV=root_node.get_id(),
                _inV=root_node.get_id(),
                OPTYPE='EPHEMERAL',
                _label='EPHEMERAL',
                EVENT_START=-1
        ))
        self.add_edge(source_edge)
        self.source_edge_id = source_edge.get_id()

        # Add disjoint trees to root's children
        for node in self.nodes:
            if len(node.edge_ids[IN]) > 0 or node == root_node:
                continue

            # Create edge from ephemeral root to subtree root
            raw_edge = Edge(
                _id=self.get_next_edge_id(),
                _outV=root_node.get_id(),
                _inV=node.get_id(),
                OPTYPE='EPHEMERAL',
                _label='EPHEMERAL',
                EVENT_START=0
            )

            # Add edge to the graph
            edge = EdgeWrapper(raw_edge)
            self.add_edge(edge)
            root_node.edge_ids[OUT].append(edge.get_id())

    preprocess_steps: list[callable] = [
        original_graph,
        _invert_outgoing_file_edges,
        _duplicate_file_ip_leaves,
        _add_ephemeral_root
    ]

    def preprocess(self, output_dir: Path = None) -> 'GraphWrapper':
        for i, step in enumerate(self.preprocess_steps):
            step(self)
            if output_dir is not None:
                self.to_dot().save(output_dir / f'{i+1}_{step.__name__.strip("_")}.dot')

        return self

    def prune(self, alpha: float, epsilon: float) -> 'GraphWrapper':
        local_sensitivity: float = 1 / alpha
        # (height, edge_id) tuples
        queue = deque([(0, self.source_edge_id)])
        while len(queue) > 0:
            depth, edge_id = queue.popleft()
            edge = self.get_edge(edge_id)

            subtree_size = self.get_tree_size(OUT, edge_id)
            distance = alpha * subtree_size
            epsilon_prime = epsilon * distance

            p = logistic_function(epsilon_prime / local_sensitivity)
            prune_edge = np.random.choice([True, False], p=[p, 1 - p])
            # If we prune, don't add children to queue
            if prune_edge:
                self._marked_edges.add(edge_id)
                continue

            # Otherwise, continue adding children to queue
            node_id = edge.node_ids[OUT]
            node = self.get_node(node_id)
            next_edge_ids = node.edge_ids[OUT]
            queue.extend([
                (depth + 1, next_edge_id)
                for next_edge_id in next_edge_ids
            ])

        return self

    def get_train_data(self) -> list[tuple[str, 'GraphWrapper']]:
        """
        Returns a list of training data
        :return: List of tuples of the form (tokenized path, root edge ID of subtree)
        """
        pass

