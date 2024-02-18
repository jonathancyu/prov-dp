from collections import deque
from copy import deepcopy
from pathlib import Path

import graphviz as gv
import networkx as nx
import numpy as np

from source.graphson import Graph, NodeType, Node, EdgeType, Edge
from .edge_wrapper import EdgeWrapper
from .node_wrapper import NodeWrapper
from ..utility import logistic_function


# TODO: Should this be called "Tree", and preprocessing happens in the constructor: might be too complicated
class GraphWrapper:
    graph: Graph
    nodes: list[NodeWrapper]
    edges: list[EdgeWrapper]
    source_edge_ref_id: int | None
    source_edge_id: int | None
    root_node_id: int | None

    __node_lookup: dict[int, NodeWrapper]
    __edge_lookup: dict[int, EdgeWrapper]

    __subtree_lookup: dict[int, 'GraphWrapper']
    marked_edge_ids: dict[int, str]  # edge_id: path
    __training_data: list[tuple[list[int], 'GraphWrapper']]  # (path, subtree) tuples

    @staticmethod
    def load_file(json_path: Path) -> 'GraphWrapper':
        split = str(json_path.stem).split('-')
        ref_id = -1
        if len(split) == 3:
            ref_id = int(split[1])

        return GraphWrapper(
            Graph.load_file(json_path),
            ref_id
        )

    def __init__(self,
                 graph: Graph = None,
                 source_edge_ref_id: int = None):
        self.graph = graph or Graph()
        self.nodes = []
        self.edges = []
        self.__init_nodes(self.graph.nodes)
        self.__init_edges(self.graph.edges)
        self.__init_source_edge(source_edge_ref_id)

        # Algorithm-specific fields
        self.__subtree_lookup = {}
        self.marked_edge_ids = {}
        self.__training_data: list[tuple[list[int], GraphWrapper]] = []

    def __init_nodes(self, nodes: list[Node]):
        # Create a lookup by node ID
        self.__node_lookup = {}
        for node in nodes:
            node_wrapper = NodeWrapper(node)
            self.nodes.append(node_wrapper)
            self.__node_lookup[node.id] = node_wrapper

    def __init_edges(self, edges: list[Edge]):
        # Create a lookup by edge ID and add edge references to nodes
        self.__edge_lookup = {}
        for edge in edges:
            edge_wrapper = EdgeWrapper(edge)
            self.edges.append(edge_wrapper)
            self.__edge_lookup[edge_wrapper.get_id()] = edge_wrapper

            # Update attached node's edge_ids
            edge_id = edge.get_id()
            src_node = self.get_node(edge.get_src_id())
            if src_node is not None:
                src_node.add_outgoing(edge_id)

            dst_node = self.get_node(edge.get_dst_id())
            if dst_node is not None:
                dst_node.add_incoming(edge_id)

    def __init_source_edge(self, source_edge_ref_id: int | None) -> None:
        # Set the source_edge_ref_id to keep track of the original graph
        self.source_edge_ref_id = source_edge_ref_id
        if source_edge_ref_id is not None:
            # Ref ID is not the same as graphson ID, so we need to find the edge with the matching ref ID
            matches = [edge for edge in self.edges
                       if edge.get_ref_id() == source_edge_ref_id]
            assert len(matches) == 1
            self.source_edge_id = matches[0].get_id()
        else:
            self.source_edge_id = None

    def get_subtree(self,
                    root_node_id: int,
                    visited_node_ids: list[int] = None) -> 'GraphWrapper':
        """
        :param root_node_id: ID of the root node
        :param visited_node_ids: Accumulating of node IDs that have already been visited
        :return: Subtree rooted at the given node
        """
        # Check if we've already computed this subtree
        subtree = self.__subtree_lookup.get(root_node_id)
        if subtree is not None:
            return subtree
        visited_node_ids = visited_node_ids or []

        # Create a new GraphWrapper object to store the tree
        subgraph = GraphWrapper()
        root_node = self.get_node(root_node_id)
        visited_node_ids.append(root_node_id)
        subgraph.add_node(root_node)

        # BFS recursively
        for edge_id in root_node.get_outgoing():
            edge = self.get_edge(edge_id)
            next_node_id = edge.get_dst_id()
            if next_node_id in visited_node_ids:
                continue
            # Add edge to subgraph
            subgraph.add_edge(edge)
            next_subgraph = self.get_subtree(edge.get_dst_id(), visited_node_ids)
            if next_subgraph is not None:
                # TODO: is it possible for this to cause issues down the line by not deep copying?
                subgraph.edges.extend(next_subgraph.edges)
                subgraph.nodes.extend(next_subgraph.nodes)

        # Cache result
        self.__subtree_lookup[root_node_id] = subgraph

        return subgraph

    def get_tree_size(self, root_edge_id: int = None) -> int:
        root_edge_id = root_edge_id or self.source_edge_id
        if root_edge_id is None:
            return 0
        # Get size from the destination node
        root_node_id = self.get_edge(root_edge_id).get_dst_id()
        # Return the size of the subtree rooted at that node
        return len(self.get_subtree(root_node_id))

    def add_edge(self, edge: EdgeWrapper) -> None:
        edge_id = edge.get_id()
        assert self.get_edge(edge_id) is None

        # Add edge to graph and lookup
        self.edges.append(edge)
        self.graph.edges.append(edge.edge)
        self.__edge_lookup[edge_id] = edge

        # Add edge to src node's outgoing list
        src_node = self.get_node(edge.get_src_id())  # TODO: can simplify logic if get_outgoing is a set
        if edge_id not in src_node.get_outgoing():
            src_node.add_outgoing(edge_id)

        # Add edge to dst node's incoming list
        dst_node = self.get_node(edge.get_dst_id())
        if edge_id not in dst_node.get_incoming():
            dst_node.add_incoming(edge_id)

    def add_node(self, node: NodeWrapper) -> None:
        self.nodes.append(node)
        self.graph.nodes.append(node.node)
        self.__node_lookup[node.get_id()] = node

    def remove_node(self, node: NodeWrapper) -> None:
        # Removes node from
        self.nodes.remove(node)
        self.__node_lookup.pop(node.get_id())

    def remove_edge(self, edge: EdgeWrapper) -> None:
        # This does NOT update the node's edge_ids
        edge_id = edge.get_id()
        edge.set_src_id(None)
        self.edges.remove(edge)
        self.__edge_lookup.pop(edge_id)

    def get_next_node_id(self) -> int:
        return max([node.get_id() for node in self.nodes]) + 1

    def get_next_edge_id(self) -> int:
        return max([edge.get_id() for edge in self.edges]) + 1

    def __invert_edge(self, edge_id: int) -> None:
        edge = self.get_edge(edge_id)
        src_id, dst_id = edge.get_src_id(), edge.get_dst_id()
        edge.invert()

        src_node, dst_node = self.get_node(src_id), self.get_node(dst_id)
        src_node.remove_outgoing(edge_id)
        src_node.add_incoming(edge_id)
        dst_node.remove_incoming(edge_id)
        dst_node.add_outgoing(edge_id)

    # Step 1. Original graph
    def original_graph(self) -> None:
        pass

    # Step 2. Invert all outgoing edges from files/IPs
    def __invert_outgoing_file_edges(self) -> None:
        edges_to_invert = []
        for node in self.nodes:
            if node.get_type() == NodeType.PROCESS_LET:
                continue
            edges_to_invert.extend(node.get_outgoing())

        for edge_id in edges_to_invert:
            self.__invert_edge(edge_id)

    # Step 3. Duplicate file/IP nodes for each incoming edge
    def __duplicate_file_ip_leaves(self) -> None:
        nodes_to_remove = []
        nodes_to_add = []
        for node in self.nodes:
            if node.get_type() == NodeType.PROCESS_LET:
                continue
            # Mark original node for removal
            nodes_to_remove.append(node)
            # Create a duplicate node for each incoming edge
            for edge_id in node.get_incoming():
                # Point edge to a new node ID
                new_node_id = self.get_next_node_id() + len(nodes_to_add)
                edge = self.get_edge(edge_id)
                edge.set_dst_id(new_node_id)
                # Create new node with that ID
                new_node = deepcopy(node)
                new_node.set_id(new_node_id)
                new_node.set_incoming([edge_id])
                nodes_to_add.append(new_node)

        # Apply node changes
        for node in nodes_to_remove:
            self.remove_node(node)
        for node in nodes_to_add:
            self.add_node(node)

    # Step 4
    def __add_ephemeral_root(self) -> None:
        # Create root node
        raw_root_node = Node(
            _id=9999,
            TYPE=NodeType.EPHEMERAL,
        )
        raw_root_node.model_extra['_label'] = 'EPHEMERAL'
        raw_root_parent_node = Node(
            _id=10000,
            TYPE=NodeType.EPHEMERAL
        )
        raw_root_parent_node.model_extra['_label'] = 'EPHEMERAL'
        root_node, root_parent_node = NodeWrapper(raw_root_node), NodeWrapper(raw_root_parent_node)
        self.add_node(root_node)
        self.add_node(root_parent_node)

        # Create root edge for BFS
        source_edge = EdgeWrapper(Edge(
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
        for node in self.nodes:
            # If this is an ephemeral node, or if it's not a root node, skip
            if len(node.get_incoming()) > 0 or node in [root_node, root_parent_node]:
                continue

            # Create edge from ephemeral root to subtree root
            self.add_edge(
                EdgeWrapper(Edge(
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

    def preprocess(self, output_dir: Path = None) -> 'GraphWrapper':
        for i, step in enumerate(self.__preprocess_steps):
            step(self)
            if output_dir is not None:
                self.to_dot().save(output_dir / f'{i + 1}_{step.__name__.strip("_")}.dot')

        return self

    def __prune_tree(self,
                     root_edge_id: int,
                     path: str) -> 'GraphWrapper':
        # Mark the edge so we can append to it later
        self.marked_edge_ids[root_edge_id] = path
        # Create subtree graph
        root_edge = self.get_edge(root_edge_id)
        subtree: GraphWrapper = self.get_subtree(root_edge.get_dst_id())
        subtree.source_edge_ref_id = self.source_edge_ref_id
        # Detach root node from parent graph
        root_edge.set_dst_id(None)
        root_node = self.get_node(root_edge.get_dst_id())
        root_node.set_incoming([])

        # Remove all subtree nodes and elements from the parent graph
        for edge in subtree.edges:
            self.remove_edge(edge)
        for node in subtree.nodes:
            self.remove_node(node)
        return subtree

    def prune(self, alpha: float, epsilon: float) -> 'GraphWrapper':
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
            distance = alpha * subtree_size
            epsilon_prime = epsilon * distance
            p = logistic_function(epsilon_prime / local_sensitivity)
            prune_edge: bool = np.random.choice([True, False],
                                                 p=[p, 1 - p])
            # If we prune, don't add children to queue
            if prune_edge and len(path) > 1:  # Don't prune ephemeral root by restricting depth to > 1
                # Remove the tree rooted at this edge from the graph
                pruned_tree = self.__prune_tree(edge_id, self.__path_to_string(path))

                # Add tree, and path to the tree to the training data
                self.__training_data.append((path, pruned_tree))

                # Ensure we don't try to BFS into the pruned tree
                visited_edge_ids.update(e.get_id() for e in pruned_tree.edges)
                # Track statistics
                sizes.append(subtree_size)
                depths.append(len(path))
                continue

            # Otherwise, continue adding children to queue
            node_id = edge.get_dst_id()
            node = self.get_node(node_id)
            next_edge_ids = node.get_outgoing()

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
                leaf_node.set_incoming([])

                # Add the leaf to its own graph
                leaf_graph = GraphWrapper()
                leaf_graph.source_edge_ref_id = self.source_edge_ref_id
                leaf_graph.add_node(leaf_node)

                # Add the (path, graph) tuple to the training data
                self.__training_data.append((path, leaf_graph))
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

    def get_train_data(self) -> list[tuple[str, 'GraphWrapper']]:
        """
        Returns a list of training data
        :return: List of tuples of the form (tokenized path, root edge ID of subtree)
        """
        return [
            (self.__path_to_string(path), graph)
            for path, graph in self.__training_data
        ]

    def get_node(self, node_id: int) -> NodeWrapper:
        return self.__node_lookup.get(node_id)

    def get_edge(self, edge_id: int) -> EdgeWrapper:
        return self.__edge_lookup.get(edge_id)

    def get_node_type(self, node_id: int) -> NodeType:
        return self.get_node(node_id).get_type()

    def get_edge_type(self, edge: EdgeWrapper) -> EdgeType:
        return EdgeType(
            edge=edge.edge,
            src_type=self.get_node(edge.get_src_id()).get_type(),
            dst_type=self.get_node(edge.get_dst_id()).get_type()
        )

    def get_root_node_id(self) -> int:
        root_nodes = [node for node in self.nodes if len(node.get_incoming()) == 0]
        if len(root_nodes) != 1:
            raise RuntimeError(f'Expected 1 root node, got {len(root_nodes)}: '
                               ', '.join([node.get_token() for node in root_nodes]))
        return root_nodes[0].get_id()

    def insert_subgraph(self,
                        root_edge_id: int,
                        graph: 'GraphWrapper') -> None:
        """
        Attach a subgraph to the destination of the given edge
        @param root_edge_id: edge to attach the subgraph to
        @param graph: subgraph to attach
        """
        new_edge_ids = {}
        new_node_ids = {}
        # Update node IDs to avoid collision in the current graph
        for node in graph.nodes:
            # Copy the node, and give it a new ID
            node_id = node.get_id()
            new_node = deepcopy(node)
            new_node_id = self.get_next_node_id()
            new_node.node.id = new_node_id
            # Add the ID to the lookup
            new_node_ids[node_id] = new_node_id
            self.add_node(new_node)
            # Mark the node to indicate it's added after the fact
            new_node.node.marked = True

        # Update edge IDs to avoid collision in the current graph, and bring up to date with node IDs
        for edge in graph.edges:
            # Copy the edge, and give it a new ID
            new_edge = deepcopy(edge)
            new_edge_id = self.get_next_edge_id()
            new_edge.set_id(new_edge_id)
            # Update the edge's node IDs to match the new graph
            new_edge.node_ids = {
                direction: new_node_ids.get(edge.node_ids[direction], None)
                for direction in edge.node_ids
            }

            # Add the ID to the lookup
            new_edge_ids[edge.get_id()] = new_edge_id
            self.add_edge(new_edge)
            # Mark the edge to indicate it's added after the fact
            new_edge.edge.marked = True

        # Attach root node to root edge
        root_edge = self.get_edge(root_edge_id)
        # TODO: BUG - for some reason, this edge has a destination ID
        # if root_edge.get_dst_id() is not None:
        #     raise RuntimeError(f'Edge {root_edge_id} already has a destination ID ({root_edge.get_dst_id()})')
        root_edge.set_dst_id(new_node_ids[graph.get_root_node_id()])

    def __len__(self):
        return len(self.nodes)

    # Exporter functions
    def to_dot(self) -> gv.Digraph:
        return Graph(
            vertices=[node.node for node in self.nodes],
            edges=[edge.edge for edge in self.edges]
        ).to_dot()

    def to_nx(self) -> nx.DiGraph:

        digraph: nx.DiGraph = nx.DiGraph()

        # NetworkX node IDs must index at 0
        node_ids = {node.get_id(): i
                    for i, node in enumerate(self.nodes)}
        for node in self.nodes:
            digraph.add_node(node_ids[node.get_id()],
                             feature=node.get_token()
                             )
        for edge in self.edges:
            # If an edge has a null node, don't include it.
            # This occurs when we've pruned a subgraph
            if None in edge.node_ids.values():
                continue
            digraph.add_edge(node_ids[edge.get_src_id()],
                             node_ids[edge.get_dst_id()],
                             feature=edge.get_token()
                             )
        return digraph
