from collections import deque
from copy import deepcopy

import numpy as np

from utility import logistic_function
from source.graphson import NodeType
from .graph_processor import GraphProcessor
from .wrappers import GraphWrapper, EdgeWrapper, NodeWrapper, Subgraph, IN, OUT


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

    def step2(self, graph: GraphWrapper) -> GraphWrapper:
        new_graph = deepcopy(graph)
        # 2. Invert all outgoing edges from files/IPs
        for node in new_graph.nodes:
            if node.get_type() == NodeType.PROCESS_LET:
                continue

            edge_ids = node.edge_ids
            expected = len(edge_ids[IN]) + len(edge_ids[OUT])

            outgoing_edges = edge_ids[OUT].copy()  # Copy list to prevent modification during iteration
            for edge_id in outgoing_edges:
                new_graph.invert_edge(edge_id)
                edge = new_graph.get_edge(edge_id)

            assert len(edge_ids[OUT]) == 0
            assert len(edge_ids[IN]) == expected
        return new_graph


    def preprocess_graph(self, input_graph: GraphWrapper) -> GraphWrapper:
        steps = {}
        # 1. Original graph
        graph = deepcopy(input_graph)
        steps['1_original_graph'] = graph

        # 2. Invert all outgoing edges from files/IPs
        graph = self.step2(graph)
        steps['2_inverted_edges'] = graph

        # Assert that all non-process nodes have out-degree 0
        for node in graph.nodes:
            print
            if str(node.get_type()) == str(NodeType.PROCESS_LET):
                continue
            print(f'{node.get_token()}: {len(node.edge_ids[OUT])}')
            assert len(node.edge_ids[OUT]) == 0


        # 3. Duplicate file/IP nodes for each incoming edge
        for node in graph.nodes:
            if node.get_type() == NodeType.PROCESS_LET:
                continue
            print(node)

            # Create a duplicate node for each edge, then delete original
            edge_ids = node.edge_ids
            assert len(edge_ids[OUT]) == 0  # In the previous step, we should have inverted all outgoing edges
            print(len(edge_ids[IN]))
            original_node_count = len(graph.nodes)
            new_node_count = 0
            if len(edge_ids[IN]) > 1:
                print(node)

            for edge_id in edge_ids[IN]:
                new_node_id = graph.get_next_node_id()
                new_node = deepcopy(node)
                new_node.node.id = new_node_id
                new_node.edge_ids = {IN: [edge_id], OUT: []}

                graph.add_node(new_node)
                new_node_count += 1

                edge = graph.get_edge(edge_id)
                edge.node_ids[OUT] = new_node_id
                edge.edge.dst_id = new_node_id

            graph.nodes.remove(node)

            for node in graph.nodes:
                if node.get_type() == NodeType.PROCESS_LET:
                    continue

                edge_ids = node.edge_ids
                assert len(edge_ids[OUT]) == 0
                if len(edge_ids[IN]) > 1:
                    print(node.get_token())

            assert new_node_count == len(node.edge_ids[IN])
            assert len(graph.nodes) == original_node_count + new_node_count - 1
        steps['3_duplicate_leaves'] = deepcopy(graph)

        #
        # # 4. Add ephemeral root node, with edges to all root nodes (in degree == 0)
        # raw_root_node = Node()
        # raw_root_node.id = new_graph.get_next_node_id()
        # raw_root_node.type = NodeType.EPHEMERAL
        # root_node = NodeWrapper(raw_root_node)
        # for node in new_graph.nodes:
        #     if len(node.edge_ids[IN]) > 0:
        #         continue
        #
        #     # Create edge from ephemeral root to subtree root
        #     raw_edge = Edge()
        #     raw_edge.id = new_graph.get_next_edge_id()
        #
        #     edge = EdgeWrapper(raw_edge)
        #     edge.node_ids[IN] = raw_root_node.id
        #     edge.node_ids[OUT] = node.get_id()
        #     root_node.edge_ids[OUT].append(edge.get_id())
        #
        #     new_graph.add_edge(edge)

        return steps
    def perturb_graphs(self, input_graphs: list[GraphWrapper]) -> list[GraphWrapper]:
        # Graph preprocessing: Invert all read edges
        output_graphs = []
        input_graphs = [self.preprocess_graph(input_graph) for input_graph in input_graphs]
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

