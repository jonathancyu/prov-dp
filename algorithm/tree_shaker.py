import random
from collections import deque
from copy import deepcopy
from datetime import datetime

import numpy as np
from icecream import ic
from tqdm import tqdm

from graphson import EdgeType
from utility import logistic_function
from .graph_processor import GraphProcessor, EDGES_PROCESSED, EDGES_FILTERED, PRUNED_SUBTREE_SIZES, PRUNED_AT_DEPTH
from .wrappers import GraphWrapper, EdgeWrapper, NodeWrapper, IN, OUT


class Tree:
    root_edge: EdgeWrapper
    edges: list[EdgeWrapper]

    def __init__(self, edges: list[EdgeWrapper]):
        self.root_edge = edges[0]
        self.edges = edges


class TreeCollection:
    tree_by_edge_type: dict[EdgeType, dict[int, list[Tree]]]

    def __init__(self):
        self.tree_by_edge_type = {

        }

    def add(self,
            parent_graph: GraphWrapper,
            subtree: Tree,
            depth: int
            ) -> None:
        edge_type = parent_graph.get_edge_type(subtree.root_edge)
        edge_type_dict = self.tree_by_edge_type.get(edge_type)
        if edge_type_dict is None:
            self.tree_by_edge_type[edge_type] = {}
        depth_list = self.tree_by_edge_type.get(depth)
        if depth_list is None:
            edge_type_dict[depth] = []
        depth_list.append(subtree)

    # TODO: We can introduce A LOT of complexity here if we wanted to (GAN?)
    def sample(self,
               edge_type: EdgeType,
               depth: int,
               budget: int) -> Tree | None:
        edge_type_dict = self.tree_by_edge_type.get(edge_type)
        if edge_type_dict is None:
            print(f"No tree found for {edge_type}")
            return
        depth_list = edge_type_dict.get(depth)
        if depth_list is None:
            print(f"No tree found for {edge_type}, depth {depth}")
            return

        # TODO: We need to somehow pick this based on a budget
        return random.choice(depth_list)


class TreeShaker(GraphProcessor):
    epsilon_p: float  # structural budget = delta * epsilon
    epsilon_m: float  # edge count budget = (1-delta) * epsilon
    alphas: dict[str, float]
    pruned_subtrees: dict[str, TreeCollection]

    def __init__(self,
                 epsilon: float,
                 delta: float,
                 alpha: float):
        super().__init__()
        self.epsilon_p = delta * epsilon
        self.epsilon_m = (1-delta) * epsilon
        self.delta = delta
        self.alphas = {IN: alpha, OUT: alpha}
        self.pruned_subtrees = {
            IN: TreeCollection(), OUT: TreeCollection()
        }

    def perturb_graphs(self, input_graphs: list[GraphWrapper]) -> list[GraphWrapper]:
        # Prune all graphs (epsilon_p)
        output_graphs: list[GraphWrapper] = []

        for input_graph in input_graphs:
            new_graph = GraphWrapper()
            new_graph.nodes = deepcopy(input_graph.nodes)
            for direction in [IN, OUT]:
                kept_edges, subtrees_pruned = self.prune_trees(input_graph, direction)
                for edge in kept_edges:
                    new_graph.add_edge(edge)
                for subtree in subtrees_pruned:
                    self.pruned_subtrees[direction].add(input_graph, subtree)

            output_graphs.append(new_graph)

        # Add edges to graphs (epsilon_m)
        for i, input_graph in input_graphs:
            output_graph = output_graphs[i]
            root_edge_id = output_graph.source_edge_id
            for direction in [IN, OUT]:
                self.add_trees(output_graph, direction)

        return output_graphs

    def prune_trees(self,
                    graph: GraphWrapper,
                    direction: str
                    ) -> tuple[list[EdgeWrapper], list[Tree]]:
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
                pruned_edges.append(edge)
                continue

            kept_edges.append(edge)

            node_id: int = edge.node_ids[direction]
            node: NodeWrapper = graph.get_node(node_id)
            next_edge_ids: list[int] = node.edge_ids[direction]

            queue.extend([(depth + 1, next_edge_id)
                          for next_edge_id in next_edge_ids])

        pruned_subtrees = [Tree(graph.get_subtree(pruned_edge.get_ref_id(), direction))
                           for pruned_edge in pruned_edges]
        return kept_edges, pruned_subtrees

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

