from collections import deque
from copy import deepcopy
from datetime import datetime

import numpy as np
from icecream import ic

from graphson import Graph
from utility import logistic_function
from .graph_processor import GraphProcessor, EDGES_PROCESSED, EDGES_FILTERED, PRUNED_SUBTREE_SIZES, PRUNED_AT_DEPTH
from .wrappers import GraphWrapper, EdgeWrapper, NodeWrapper, IN, OUT


class TreeShaker(GraphProcessor):
    epsilon: float
    alphas: dict[str, float]

    def __init__(self, epsilon: float, alpha: float):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = {IN: alpha, OUT: alpha}

    def perturb_graphs(self, input_graphs: list[GraphWrapper]):
        # Prune all graphs
        new_graphs: dict[int, GraphWrapper] = {}

        pruned_subtrees: dict[str, dict[int, list[list[EdgeWrapper]]]] = {
            IN: {}, OUT: {}
        }

        for input_graph in input_graphs:
            new_graph = GraphWrapper()
            new_graph.nodes = deepcopy(input_graph.nodes)
            for direction in [IN, OUT]:
                kept_edges, subtrees_pruned = self.prune_tree(input_graph, direction)
                for edge in kept_edges:
                    new_graph.add_edge(edge)
                for subtree in subtrees_pruned:
                    size = len(subtree)
                    pruned_subtrees[direction].setdefault(size, []).append(subtree)
            new_graphs[input_graph.source_edge_id] = new_graph

        # Sample from pruned subtrees to add edges back to graphs
        x = {
            key: np.average([len(tree) for tree in value])
            for key, value in pruned_subtrees
        }
        pass

    def prune_tree(self,
                   graph: GraphWrapper,
                   direction: str
                   ) -> tuple[list[EdgeWrapper], list[list[EdgeWrapper]]]:
        """
        Prunes edges off the input graph in a given direction
        :param graph: Input graph
        :param direction: Direction of pruning
        :return new_edges, pruned_subtrees: List of edges kept, list of subtrees pruned
        """
        visited_edges: set[EdgeWrapper] = set()
        kept_edges: list[EdgeWrapper] = []
        pruned_edges: list[EdgeWrapper] = []
        local_sensitivity: float = 1 / self.alphas[direction]

        queue = deque([(0, graph.source_edge_id)])
        while len(queue) > 0:
            depth, edge_id = queue.popleft()
            edge: EdgeWrapper = graph.get_edge_by_id(edge_id)
            if edge in visited_edges:
                continue
            visited_edges.add(edge)

            subtree_size = graph.get_tree_size(edge_id, direction)
            distance = self.alphas[direction] * subtree_size
            epsilon_prime = self.epsilon * distance

            p = logistic_function(epsilon_prime / local_sensitivity)
            prune_edge = np.random.choice([True, False], p=[p, 1 - p])
            if prune_edge:
                pruned_edges.append(edge)
                continue

            kept_edges.append(edge)

            node_id: int = edge.node_ids[direction]
            node: NodeWrapper = graph.get_node(node_id)
            next_edge_ids: list[int] = node.edge_ids[direction]

            queue.extend([(depth + 1, next_edge_id)
                          for next_edge_id in next_edge_ids])

        pruned_subtrees = [graph.get_subtree(pruned_edge.get_ref_id(), direction)
                           for pruned_edge in pruned_edges]
        return kept_edges, pruned_subtrees
