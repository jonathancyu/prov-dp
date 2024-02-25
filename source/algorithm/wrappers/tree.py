import json
from collections import deque
from copy import deepcopy, copy
from pathlib import Path

import graphviz as gv
import networkx as nx
import numpy as np

from .edge import Edge
from .node import Node
from ..utility import logistic_function
from ...graphson import RawEdge, RawNode, RawGraph, NodeType


class Tree:
    source_edge_ref_id: int | None
    source_edge_id: int | None
    root_node_id: int | None

    marked_node_paths: dict[int, str]  # node_id: path

    __nodes: dict[int, Node]
    __edges: dict[int, Edge]
    __incoming_lookup: dict[int, set[int]]  # node_id: set[edge_id]
    __outgoing_lookup: dict[int, set[int]]  # node_id: set[edge_id]
    __subtree_lookup: dict[int, 'Tree']
    __training_data: list[tuple[list[int], 'Tree']]  # (path, subtree) tuples

    @staticmethod
    def load_file(json_path: Path) -> 'Tree':
        split = str(json_path.stem).split('-')
        ref_id = -1
        if len(split) == 3:
            ref_id = int(split[1])
        unprocessed_tree = Tree(
            RawGraph.load_file(json_path),
            ref_id
        )
        return unprocessed_tree.preprocess()

    def __init__(self,
                 graph: RawGraph = None,
                 source_edge_ref_id: int = None):
        graph = graph or RawGraph()
        self.__incoming_lookup = {}
        self.__outgoing_lookup = {}
        self.__init_nodes(graph.nodes)
        self.__init_edges(graph.edges)
        self.__init_source_edge(source_edge_ref_id)

        # Algorithm-specific fields
        self.__subtree_lookup = {}
        self.marked_node_paths = {}
        self.__training_data: list[tuple[list[int], Tree]] = []

    def __init_nodes(self, nodes: list[RawNode]):
        # Create a lookup by node ID
        self.__nodes = {}
        for raw_node in nodes:
            self.add_node(Node(raw_node))

    def __init_edges(self, edges: list[RawEdge]):
        # Create a lookup by edge ID and add edge references to nodes
        self.__edges = {}
        for raw_edge in edges:
            self.add_edge(Edge(raw_edge))

    def __init_source_edge(self, source_edge_ref_id: int | None) -> None:
        # Set the source_edge_ref_id to keep track of the original graph
        self.source_edge_ref_id = source_edge_ref_id
        if source_edge_ref_id is not None:
            # Ref ID is not the same as graphson ID, so we need to find the edge with the matching ref ID
            matches = [edge for edge in self.__edges.values()
                       if edge.get_ref_id() == source_edge_ref_id]
            assert len(matches) == 1
            self.source_edge_id = matches[0].get_id()
        else:
            self.source_edge_id = None

    def get_subtree(self,
                    root_node_id: int,
                    visited_node_ids: set[int] = None) -> 'Tree':
        """
        :param root_node_id: ID of the root node
        :param visited_node_ids: Accumulating list of node IDs that have already been visited
        :return: Subtree rooted at the given node
        """
        # Check if we've already computed this subtree
        subtree = self.__subtree_lookup.get(root_node_id)
        if subtree is not None:
            return subtree
        visited_node_ids = visited_node_ids or set()

        # Create a new GraphWrapper object to store the accumulating tree
        subtree = Tree()
        root_node = self.get_node(root_node_id)
        subtree_root_node = deepcopy(root_node)
        subtree.add_node(subtree_root_node)

        # Mark the node as visited
        visited_node_ids.add(root_node_id)

        # BFS recursively
        for edge_id in self.get_outgoing_edge_ids(root_node_id):
            edge = self.get_edge(edge_id)
            next_node_id = edge.get_dst_id()
            if next_node_id in visited_node_ids:
                continue
            # Add edge to the accumulating subgraph

            # Get the next subgraph, then add the connecting edge, and subgraph to the accumulating subgraph
            next_subgraph = self.get_subtree(edge.get_dst_id(), visited_node_ids)

            # Deep copy the graph components into the accumulating subgraph
            for new_node in next_subgraph.get_nodes():  # Nodes need to be added first
                subtree.add_node(deepcopy(new_node))

            subtree.add_edge(deepcopy(edge))
            for new_edge in next_subgraph.get_edges():
                subtree.add_edge(deepcopy(new_edge))

        # Cache result
        self.__subtree_lookup[root_node_id] = subtree

        return subtree

    def get_tree_size(self, root_edge_id) -> int:
        # Get size from the destination node
        root_node_id = self.get_edge(root_edge_id).get_dst_id()
        # Return the size of the subtree rooted at that node
        return len(self.get_subtree(root_node_id))

    # Wrapper functions
    def get_edges(self):
        return list(self.__edges.values())

    def get_nodes(self):
        return list(self.__nodes.values())

    def add_edge(self,
                 edge: Edge) -> None:
        assert self.__edges.get(edge.get_id()) is None
        assert self.get_node(edge.get_src_id()) is not None, f'Edge {edge.get_id()} has no source in graph'
        assert self.get_node(edge.get_dst_id()) is not None, f'Edge {edge.get_id()} has no destination in graph'
        edge_id = edge.get_id()

        # Add edge to graph and lookup
        self.__edges[edge_id] = edge
        self.__incoming_lookup[edge.get_dst_id()].add(edge_id)
        self.__outgoing_lookup[edge.get_src_id()].add(edge_id)

    def add_node(self,
                 node: Node) -> None:
        node_id = node.get_id()
        assert self.__nodes.get(node_id) is None
        self.__nodes[node_id] = node
        self.__incoming_lookup[node_id] = set()
        self.__outgoing_lookup[node_id] = set()

    def remove_node(self, node: Node) -> None:
        # Removes node from graph and lookup
        node_id = node.get_id()
        assert self.__nodes.get(node_id) is not None
        self.__nodes.pop(node_id)
        self.__incoming_lookup.pop(node_id)
        self.__outgoing_lookup.pop(node_id)

    def remove_edge(self, edge: Edge) -> None:
        # Removes edge from graph and lookup
        edge_id = edge.get_id()
        self.__edges.pop(edge_id)
        self.__incoming_lookup[edge.get_dst_id()].remove(edge_id)
        self.__outgoing_lookup[edge.get_src_id()].remove(edge_id)

    def get_next_node_id(self) -> int:
        return max([node_id for node_id in self.__nodes.keys()]) + 1

    def get_next_edge_id(self) -> int:
        return max([edge_id for edge_id in self.__edges.keys()]) + 1

    def get_outgoing_edge_ids(self, node_id: int) -> list[int]:
        return list(self.__outgoing_lookup[node_id])

    def get_incoming_edge_ids(self, node_id: int) -> list[int]:
        return list(self.__incoming_lookup[node_id])

    # Preprocessing functions
    def __invert_edge(self, edge_id: int) -> None:
        edge = self.get_edge(edge_id)
        src_id, dst_id = edge.get_src_id(), edge.get_dst_id()
        self.get_edge(edge_id).invert()
        self.__outgoing_lookup[src_id].remove(edge_id)
        self.__incoming_lookup[dst_id].remove(edge_id)
        self.__outgoing_lookup[dst_id].add(edge_id)
        self.__incoming_lookup[src_id].add(edge_id)

    # Step 1. Original graph
    def original_graph(self) -> None:
        pass

    # Step 2. Invert all outgoing edges from files/IPs
    def __invert_outgoing_file_edges(self) -> None:
        edges_to_invert = []
        for node in self.get_nodes():
            if node.get_type() == NodeType.PROCESS_LET:
                continue
            edges_to_invert.extend(self.get_outgoing_edge_ids(node.get_id()))

        for edge_id in edges_to_invert:
            self.__invert_edge(edge_id)

    # Step 3. Duplicate file/IP nodes for each incoming edge
    def __duplicate_file_ip_leaves(self) -> None:
        nodes = self.get_nodes().copy()
        for node in nodes:
            if node.get_type() == NodeType.PROCESS_LET:
                continue
            # Get coming edges of original node
            incoming_edge_ids = self.get_incoming_edge_ids(node.get_id()).copy()
            # Duplicate node for each incoming edge
            for edge_id in incoming_edge_ids:
                # Create new node
                new_node_id = self.get_next_node_id()
                new_node = deepcopy(node)
                new_node.set_id(new_node_id)
                self.add_node(new_node)

                # Point edge to the new node ID
                edge = self.get_edge(edge_id)
                self.remove_edge(edge)
                edge.set_dst_id(new_node_id)
                self.add_edge(edge)

            # Remove original node
            self.remove_node(node)

    # Step 4
    def __add_ephemeral_root(self) -> None:
        # Create root node
        raw_root_node = RawNode(
            _id=9999,
            TYPE=NodeType.EPHEMERAL,
        )
        raw_root_node.model_extra['_label'] = 'EPHEMERAL'
        raw_root_parent_node = RawNode(
            _id=10000,
            TYPE=NodeType.EPHEMERAL
        )
        raw_root_parent_node.model_extra['_label'] = 'EPHEMERAL'
        root_node, root_parent_node = Node(raw_root_node), Node(raw_root_parent_node)
        self.add_node(root_node)
        self.add_node(root_parent_node)

        # Create root edge for BFS
        source_edge = Edge(RawEdge(
            _id=self.get_next_edge_id(),
            _outV=root_parent_node.get_id(),
            _inV=root_node.get_id(),
            OPTYPE='EPHEMERAL',
            _label='EPHEMERAL',
            EVENT_START=-1
        ))
        self.add_edge(source_edge)
        self.source_edge_id = source_edge.get_id()

        # Add disjoint trees to root's children
        for node in self.get_nodes():
            # If this is an ephemeral node, or if it's not a root node, skip
            if len(self.get_incoming_edge_ids(node.get_id())) > 0 or node in [root_node, root_parent_node]:
                continue

            # Create edge from ephemeral root to subtree root
            self.add_edge(
                Edge(RawEdge(
                    _id=self.get_next_edge_id(),
                    _outV=root_node.get_id(),
                    _inV=node.get_id(),
                    OPTYPE='EPHEMERAL',
                    _label='EPHEMERAL',
                    EVENT_START=0
                ))
            )

    __preprocess_steps: list[callable] = [
        original_graph,
        __invert_outgoing_file_edges,
        __duplicate_file_ip_leaves,
        __add_ephemeral_root
    ]

    def preprocess(self, output_dir: Path = None) -> 'Tree':
        for i, step in enumerate(self.__preprocess_steps):
            step(self)
            if output_dir is not None:
                with open(output_dir / f'{i + 1}_{step.__name__.strip("_")}.json', 'w',
                          encoding='utf-8') as output_file:
                    output_file.write(self.to_json())
                self.to_dot().save(output_dir / f'{i + 1}_{step.__name__.strip("_")}.dot')

        return self

    def __prune_tree(self,
                     root_node_id: int,
                     path: str) -> 'Tree':
        # Mark the node so we can replace it later
        assert self.get_node(root_node_id) is not None
        self.marked_node_paths[root_node_id] = path

        # Create subtree graph
        subtree: Tree = self.get_subtree(root_node_id)
        subtree.source_edge_ref_id = self.source_edge_ref_id
        num_roots = 0
        # Remove all subtree nodes and elements from the parent graph
        for edge in subtree.get_edges():
            self.remove_edge(edge)
        for node in subtree.get_nodes():
            if node.get_id() == root_node_id:
                num_roots += 1
                continue  # We want to keep this node, so we can replace later
            self.remove_node(node)
        assert num_roots == 1, f'Expected 1 root, got {num_roots}'

        assert len(self.get_incoming_edge_ids(root_node_id)) == 1, \
            f'Expected 1 outgoing edge, got {len(self.get_outgoing_edge_ids(root_node_id))}'
        subtree_root = subtree.get_node(root_node_id)
        assert subtree_root is not None
        assert len(subtree.get_incoming_edge_ids(root_node_id)) == 0
        assert self.get_node(root_node_id) is not None

        return subtree

    def prune(self, alpha: float, epsilon: float) -> 'Tree':
        sizes = []
        depths = []
        num_leaves = 0
        local_sensitivity: float = 1 / alpha
        # Breadth first search through the graph, keeping track of the path to the current node
        # (edge_id, list[edge_id_path]) tuples
        queue: deque[tuple[int, list[int]]] = deque([(self.source_edge_id, [])])
        visited_edge_ids: set[int] = set()
        while len(queue) > 0:
            # Standard BFS operations
            edge_id, path = queue.popleft()  # Could change to `queue.pop()` if you want a DFS
            if edge_id in visited_edge_ids:
                continue
            visited_edge_ids.add(edge_id)
            edge = self.get_edge(edge_id)

            # Calculate the probability of pruning a given tree
            subtree_size = self.get_tree_size(edge_id)
            distance = alpha * subtree_size  # Big tree -> big distance
            epsilon_prime = epsilon * distance  # Big distance -> big epsilon
            p = logistic_function(epsilon_prime / local_sensitivity)  # Big epsilon -> lower probability of pruning
            prune_edge: bool = np.random.choice([True, False],
                                                p=[p, 1 - p])
            # If we prune, don't add children to queue
            if prune_edge and len(path) > 1:  # Don't prune ephemeral root by restricting depth to > 1
                # Remove the tree rooted at this edge's dst_id from the graph
                pruned_tree = self.__prune_tree(edge.get_dst_id(), self.__path_to_string(path))

                # Add tree, and path to the tree to the training data
                self.__training_data.append((path, pruned_tree))

                # Ensure we don't try to BFS into the pruned tree
                visited_edge_ids.update(e.get_id() for e in pruned_tree.get_edges())

                # Track statistics
                sizes.append(subtree_size)
                depths.append(len(path))
                continue

            # Otherwise, continue adding children to queue
            node_id = edge.get_dst_id()
            node = self.get_node(node_id)
            next_edge_ids = self.get_outgoing_edge_ids(node_id)

            # If this isn't a leaf, then continue and add the next edges to the queue
            if len(next_edge_ids) > 0:
                queue.extend([
                    (next_edge_id, path + [edge_id])
                    for next_edge_id in next_edge_ids
                ])

            # If this is a leaf, add the path and current graph to the training data
            else:
                num_leaves += 1
                # Deep copy the leaf to modify it
                leaf_node = deepcopy(node)

                # Add the leaf to its own graph
                leaf_tree = Tree()
                leaf_tree.source_edge_ref_id = self.source_edge_ref_id
                leaf_tree.add_node(leaf_node)

                # Add the (path, graph) tuple to the training data
                self.__training_data.append((path, leaf_tree))
                continue

        # print(f'Pruned {len(self._marked_edges)} subgraphs, and added {num_leaves} leaf samples')
        return self, sizes, depths  # TODO: make this less hacky

    def __path_to_string(self, path: list[int]) -> str:
        tokens = []
        for edge_id in path:
            edge = self.get_edge(edge_id)
            node = self.get_node(edge.get_src_id())
            tokens.extend([
                node.get_token(),
                edge.get_token()
            ])

        return ' '.join(tokens)

    def get_train_data(self) -> list[tuple[str, 'Tree']]:
        """
        Returns a list of training data
        :return: List of tuples of the form (tokenized path, root edge ID of subtree)
        """
        return [
            (self.__path_to_string(path), graph)
            for path, graph in self.__training_data
        ]

    def get_node(self, node_id: int) -> Node:
        return self.__nodes.get(node_id)

    def get_edge(self, edge_id: int) -> Edge:
        return self.__edges.get(edge_id)

    def get_root_node_id(self) -> int:
        root_nodes = [node
                      for node in self.get_nodes()
                      if len(self.get_incoming_edge_ids(node.get_id())) == 0]
        if len(root_nodes) != 1:
            raise RuntimeError(f'Expected 1 root node, got {len(root_nodes)}: '
                               ', '.join([node.get_token() for node in root_nodes]))
        return root_nodes[0].get_id()

    def replace_node_with_tree(self,
                               node_id_to_replace: int,
                               graph: 'Tree') -> None:
        """
        Attach a subtree to the destination of the given edge
        @param node_id_to_replace: node to replace with subtree
        @param graph: subtree to replace with
        """
        node_id_translation = {}
        edge_id_translation = {}
        # Update node IDs to avoid collision in the current graph
        orphan_nodes = []
        for old_node in graph.get_nodes():
            # Copy the node, and give it a new ID
            old_node_id = old_node.get_id()
            node = deepcopy(old_node)
            node_id = self.get_next_node_id()
            node.set_id(node_id)

            # Add the ID to the lookup
            assert node_id_translation.get(old_node_id) is None
            node_id_translation[old_node_id] = node_id
            self.add_node(node)

            # If the node is an orphan, it's a root
            if len(graph.get_incoming_edge_ids(old_node_id)) == 0:
                orphan_nodes.append(node)

            # Mark the node to indicate it's been added after the fact
            node.marked = True

        # There should only be one orphan/root node
        assert len(orphan_nodes) == 1, f'Expected 1 orphan node, got {len(orphan_nodes)}/{len(graph.get_nodes())}'
        new_root_node = orphan_nodes[0]

        # Update edge IDs to avoid collision in the current graph, and bring up to date with node IDs
        for old_edge in graph.get_edges():
            # Copy the edge, and give it a new ID
            edge = deepcopy(old_edge)
            new_edge_id = self.get_next_edge_id()
            edge.set_id(new_edge_id)
            # Update the edge's node IDs to match the new graph
            assert node_id_translation.get(edge.get_src_id()) is not None
            assert node_id_translation.get(edge.get_dst_id()) is not None
            edge.translate_node_ids(node_id_translation)
            assert self.get_node(edge.get_src_id()) is not None
            assert self.get_node(edge.get_dst_id()) is not None

            # Add the ID to the lookup
            edge_id_translation[old_edge.get_id()] = new_edge_id
            self.add_edge(edge)
            # Mark the edge to indicate it's added after the fact
            edge.marked = True

        # Attach root node to root edge
        node_to_replace = self.get_node(node_id_to_replace)
        incoming_edges = self.get_incoming_edge_ids(node_id_to_replace)
        assert len(incoming_edges) == 1
        parent_edge_id = incoming_edges[0]
        parent_edge = self.get_edge(parent_edge_id)
        assert parent_edge is not None
        self.remove_edge(parent_edge)
        self.remove_node(node_to_replace)
        parent_edge.set_dst_id(new_root_node.get_id())
        self.add_edge(parent_edge)

    def __len__(self):
        return len(self.__nodes)

    # Exporter functions
    def to_dot(self) -> gv.Digraph:
        dot_graph = gv.Digraph()
        dot_graph.attr(rankdir='LR')
        included_nodes: set[Node] = set()
        sorted_edges = sorted(self.get_edges(), key=lambda e: e.get_time())

        def add_to_graph(new_node: Node):
            assert new_node is not None, 'Trying to add a null node to the graph'
            included_nodes.add(new_node)
            dot_graph.node(str(new_node.get_id()), **new_node.to_dot_args())

        num_missing = 0
        num_null = 0
        for edge in sorted_edges:
            src_id, dst_id = edge.get_src_id(), edge.get_dst_id()
            assert src_id is not None, f'Edge {edge.get_id()} has no source'
            assert dst_id is not None, f'Edge {edge.get_id()} has no destination'
            src, dst = self.get_node(src_id), self.get_node(dst_id)
            add_to_graph(src)
            add_to_graph(dst)

            dot_graph.edge(str(src_id), str(dst_id), **edge.to_dot_args())

        if num_missing > 0:
            print(f'Warn: {num_missing} MIA, {num_null} null out of {len(self.get_edges())}?')
        for node in self.get_nodes():
            if node not in included_nodes:
                add_to_graph(node)

        return dot_graph

    def to_nx(self) -> nx.DiGraph:
        digraph: nx.DiGraph = nx.DiGraph()
        # NetworkX node IDs must index at 0
        node_ids = {node.get_id(): i
                    for i, node in enumerate(self.get_nodes())}
        for node in self.get_nodes():
            digraph.add_node(node_ids[node.get_id()],
                             feature=node.get_token()
                             )
        for edge in self.get_edges():
            src, dst = edge.get_src_id(), edge.get_dst_id()
            if src is not None and dst is None:
                continue
            digraph.add_edge(node_ids[src],
                             node_ids[dst],
                             feature=edge.get_token())
        return digraph

    def assert_tree(self) -> None:
        for node in self.get_nodes():
            incoming_edges = self.get_incoming_edge_ids(node.get_id())
            assert len(incoming_edges) <= 1, f'Node {node.get_id()} has {len(incoming_edges)} incoming edges'

    def assert_complete(self) -> None:
        for edge in self.get_edges():
            assert edge.get_src_id() is not None, f'Edge {edge.get_id()} ({edge.get_token()} has None source'
            assert edge.get_dst_id() is not None, f'Edge {edge.get_id()} ({edge.get_token()} has None destination'
            if self.get_node(edge.get_src_id()) is None:
                print(f'Edge {edge.get_id()} ({edge.get_token()}) has no source')
            assert self.get_node(
                edge.get_dst_id()) is not None, f'Edge {edge.get_id()} ({edge.get_token()}) has no destination'
        for node in self.get_nodes():
            node_id = node.get_id()
            assert node.get_id() is not None, f'Node {node.get_token()} has None ID'
            for edge_id in self.get_incoming_edge_ids(node_id):
                edge = self.get_edge(edge_id)
                assert edge is not None, f'Node {node.get_token()} has no incoming edge {edge_id}'
                assert edge.get_dst_id() == node_id, \
                    (f'Node {node_id} has incoming edge {edge_id} '
                     f'with wrong destination ({edge.get_src_id()} -> {edge.get_dst_id()})')
            for edge_id in self.get_outgoing_edge_ids(node_id):
                edge = self.get_edge(edge_id)
                assert edge is not None, \
                    f'Node {node.get_token()} has no outgoing edge {edge_id}, {node.marked}'
                assert edge.get_src_id() == node_id, \
                    f'Node {node.get_token()} has outgoing edge {edge_id} with wrong source'

    def to_json(self) -> str:
        return json.dumps({
            'mode': 'EXTENDED',
            'vertices': [
                node.to_json_dict() for node in self.get_nodes()
            ],
            'edges': [
                edge.to_json_dict() for edge in self.get_edges()
            ],
        })
