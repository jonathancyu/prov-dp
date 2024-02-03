from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np

from utility import logistic_function
from source.graphson import NodeType, Node, Edge
from .graph_processor import GraphProcessor
from .wrappers import GraphWrapper, EdgeWrapper, NodeWrapper, Subgraph, IN, OUT


class TreeShaker(GraphProcessor):
    epsilon_p: float  # structural budget = delta * epsilon
    epsilon_m: float  # edge count budget = (1-delta) * epsilon
    alpha: float
    pruned_subgraphs: dict[str, list[Subgraph]]
    pruned_subtrees: list[Subgraph]

    def __init__(self,
                 epsilon: float,
                 delta: float,
                 alpha: float):
        super().__init__()
        self.epsilon_p = delta * epsilon
        self.epsilon_m = (1-delta) * epsilon
        self.delta = delta
        self.alpha = alpha
        self.pruned_subgraphs = {IN: [], OUT: []}

    def perturb_graphs(self, input_graphs: list[GraphWrapper]) -> list[GraphWrapper]:
        # Graph preprocessing: Invert all read edges
        for graph in input_graphs:
            graph.preprocess()

        # Prune graphs and create training dataset
        train_data: list[tuple[str, GraphWrapper]] = []
        for graph in input_graphs:
            # Prune tree (using epsilon_p budget)
            graph.prune(self.alpha, self.epsilon_p)



        # Add paths and trees to dataset
        train_data = []

        # with open('pruned_subgraphs.pkl', 'wb') as f:
        #     pickle.dump(self.pruned_subgraphs, f)
        # Add edges to graphs (epsilon_m)

        return []

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

