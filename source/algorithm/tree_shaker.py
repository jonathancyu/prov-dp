import pickle
from itertools import product
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utility import logistic_function
from source.graphson import NodeType, Node, Edge
from .graph_processor import GraphProcessor
from .utility import map_pool
from .wrappers import GraphWrapper, EdgeWrapper, NodeWrapper, Subgraph, IN, OUT


class TreeShaker(GraphProcessor):
    epsilon_p: float  # structural budget = delta * epsilon
    epsilon_m: float  # edge count budget = (1-delta) * epsilon
    delta: float
    alpha: float

    def __init__(self,
                 epsilon: float,
                 delta: float,
                 alpha: float):
        super().__init__()
        self.epsilon_p = delta * epsilon
        self.epsilon_m = (1-delta) * epsilon
        self.delta = delta
        self.alpha = alpha

    def perturb_graphs(self, paths: list[Path]) -> None:
        # Load graphs
        graphs = list(map_pool(
            GraphWrapper.load_file,
            paths,
            'Loading graphs'
        ))
        # Preprocess graphs
        preprocessed_graphs = map_pool(
            GraphWrapper.preprocess,
            graphs,
            'Preprocessing graphs'
        )
        del graphs

        # Prune graphs
        pruned_graphs = list(map_pool(
            GraphWrapper.prune,
            list(product(preprocessed_graphs, [self.alpha], [self.epsilon_p])),
            'Pruning graphs'
        ))
        del preprocessed_graphs

        # Create training data
        train_data: list[tuple[str, GraphWrapper]] = list(map_pool(
            GraphWrapper.get_train_data,
            pruned_graphs,
            'Creating training data'
        ))
        del pruned_graphs

        # Dump so we can pick up in a notebook
        with open('train_data.pkl', 'wb') as f:
            pickle.dump(train_data, f)
            print(f'Wrote {len(train_data)} training examples to train_data.pkl')

        # Add edges to graphs (epsilon_m)

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

