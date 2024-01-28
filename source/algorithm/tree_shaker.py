from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np

from utility import logistic_function
from source.graphson import NodeType, Node, Edge
from .graph_processor import GraphProcessor
from .wrappers import GraphWrapper, EdgeWrapper, NodeWrapper, Subgraph, IN, OUT


def invert_outgoing_file_edges(graph: GraphWrapper) -> None:
    # 2. Invert all outgoing edges from files/IPs
    edges_to_invert = []
    for node in graph.nodes:
        if node.get_type() == NodeType.PROCESS_LET:
            continue

        edges_to_invert.extend(node.edge_ids[OUT])
    for edge_id in edges_to_invert:
        graph.invert_edge(edge_id)


def duplicate_file_ip_leaves(graph: GraphWrapper) -> None:
    nodes_to_remove = []
    nodes_to_add = []
    for node in graph.nodes:
        if node.get_type() == NodeType.PROCESS_LET:
            continue

        # Mark original for removal

        # Mark original node for removal, then create a duplicate node for each edge
        nodes_to_remove.append(node)
        for edge_id in node.edge_ids[IN]:
            # Create new node
            new_node = deepcopy(node)
            new_node_id = graph.get_next_node_id() + len(nodes_to_add)
            new_node.node.id = new_node_id
            new_node.edge_ids = {IN: [edge_id], OUT: []}
            nodes_to_add.append(new_node)

            # Move edge to new node
            edge = graph.get_edge(edge_id)
            edge.node_ids[OUT] = new_node_id
            edge.edge.dst_id = new_node_id
    # Apply changes
    for node in nodes_to_remove:
        graph.remove_node(node)
    for node in nodes_to_add:
        graph.add_node(node)

def add_ephemeral_root(graph: GraphWrapper) -> None:
    raw_root_node = Node(
        _id=9999,
        TYPE=NodeType.EPHEMERAL
    )
    root_node = NodeWrapper(raw_root_node)
    graph.add_node(root_node)
    for node in graph.nodes:
        if len(node.edge_ids[IN]) > 0 or node == root_node:
            continue

        # Create edge from ephemeral root to subtree root
        raw_edge = Edge(
            _id=graph.get_next_edge_id(),
            _outV=root_node.get_id(),
            _inV=node.get_id(),
            OPTYPE='EPHEMERAL',
            _label='EPHEMERAL',
            EVENT_START=0
        )

        # Add edge to the graph
        edge = EdgeWrapper(raw_edge)
        graph.add_edge(edge)
        root_node.edge_ids[OUT].append(edge.get_id())


class TreeShaker(GraphProcessor):
    epsilon_p: float  # structural budget = delta * epsilon
    epsilon_m: float  # edge count budget = (1-delta) * epsilon
    alphas: dict[str, float]
    pruned_subgraphs: dict[str, list[Subgraph]]

    def __init__(self,
                 epsilon: float,
                 delta: float,
                 alpha: float):
        super().__init__()
        self.epsilon_p = delta * epsilon
        self.epsilon_m = (1-delta) * epsilon
        self.delta = delta
        self.alphas = {IN: alpha, OUT: alpha}
        self.pruned_subgraphs = {IN: [], OUT: []}

    def preprocess_graph(self, graph: GraphWrapper) -> GraphWrapper:
        output_dir = Path('../data/output/nd-52809777-processletevent')
        # 1. Original graph
        graph.to_dot().save(output_dir / '1_original_graph.dot')

        # 2. Invert all outgoing edges from files/IPs
        invert_outgoing_file_edges(graph)
        graph.to_dot().save(output_dir / '2_inverted_edges.dot')

        # 3. Duplicate file/IP nodes for each incoming edge
        duplicate_file_ip_leaves(graph)
        graph.to_dot().save(output_dir / '3_duplicate_leaves.dot')

        # 4. Add ephemeral root node, with edges to all root nodes (in degree == 0)
        add_ephemeral_root(graph)
        graph.to_dot().save(output_dir / '4_ephemeral_root.dot')

        return graph

    def perturb_graphs(self, input_graphs: list[GraphWrapper]) -> list[GraphWrapper]:
        # Graph preprocessing: Invert all read edges
        output_graphs = [self.preprocess_graph(input_graph) for input_graph in input_graphs]
        """ 
        # Prune all graphs (using epsilon_p budget)
        output_graphs: list[GraphWrapper] = []

        for input_graph in input_graphs:
            new_graph = GraphWrapper()
            new_graph.nodes = deepcopy(input_graph.nodes)

            # From both forward/back, prune subgraphs based on their side.

            # Create new graph with kept edges, and keep track of pruned subgraphs
            for direction in [IN, OUT]:
                kept_edges, pruned_subgraphs = self.prune_trees(input_graph, direction)
                for edge in kept_edges:
                    new_graph.add_edge(edge)
                for subgraph in pruned_subgraphs:
                    self.pruned_subgraphs[direction].append(subgraph)

            output_graphs.append(new_graph)
        ic(self.pruned_subgraphs)
        with open('pruned_subgraphs.pkl', 'wb') as f:
            pickle.dump(self.pruned_subgraphs, f)
        # Add edges to graphs (epsilon_m)
        """
        return output_graphs

    def prune_trees(self,
                    graph: GraphWrapper,
                    direction: str
                    ) -> tuple[list[EdgeWrapper], list[Subgraph]]:
        """
        Prunes edges off the input graph in a given direction
        :param graph: Input graph
        :param direction: Direction of pruning
        :return new_edges, pruned_subgraphs: List of edges kept, list of subgraphs pruned
        """
        visited_edges: set[EdgeWrapper] = set()
        kept_edges: list[EdgeWrapper] = []
        pruned_edges: list[tuple[int, int]] = []  # (depth, root_edge_id)
        local_sensitivity: float = 1 / self.alphas[direction]

        queue = deque([(0, graph.source_edge_id)])
        while len(queue) > 0:
            depth, edge_id = queue.popleft()
            edge: EdgeWrapper = graph.get_edge(edge_id)
            if edge in visited_edges:
                continue
            visited_edges.add(edge)

            subtree_size = graph.get_tree_size(direction, edge_id)
            distance = self.alphas[direction] * subtree_size
            epsilon_prime = self.epsilon_p * distance

            p = logistic_function(epsilon_prime / local_sensitivity)
            prune_edge = np.random.choice([True, False], p=[p, 1 - p])
            if prune_edge:
                pruned_edges.append((depth, edge.get_id()))
                continue

            kept_edges.append(edge)

            node_id: int = edge.node_ids[direction]
            node: NodeWrapper = graph.get_node(node_id)
            next_edge_ids: list[int] = node.edge_ids[direction]

            queue.extend([(depth + 1, next_edge_id)
                          for next_edge_id in next_edge_ids])

        pruned_subgraphs = [
            Subgraph(
                parent_graph=graph,
                edges=graph.get_subtree(pruned_edge_id, direction),
                depth=depth
            )
            for depth, pruned_edge_id in pruned_edges
        ]
        return kept_edges, pruned_subgraphs

    def add_trees(self,
                  input_graph: GraphWrapper,
                  output_graph: GraphWrapper,
                  direction: str) -> None:
        # There's an off-by-one error here - forward/backward both include source edge
        m = input_graph.get_tree_size(direction)
        m_perturbed = m + int(np.round(np.random.laplace(0, 1.0 / self.epsilon_m)))
        start_tree_size = output_graph.get_tree_size(direction)
        new_edges: list[EdgeWrapper] = []
        while (start_tree_size + len(new_edges)) < m_perturbed:
            pass

