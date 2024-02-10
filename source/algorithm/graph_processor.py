import pickle
from itertools import product
from pathlib import Path

import numpy as np

from .utility import map_pool
from .wrappers import GraphWrapper, EdgeWrapper


class GraphProcessor:
    epsilon_p: float  # structural budget = delta * epsilon
    epsilon_m: float  # edge count budget = (1-delta) * epsilon
    delta: float
    alpha: float

    def __init__(self,
                 epsilon: float,
                 delta: float,
                 alpha: float):
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
