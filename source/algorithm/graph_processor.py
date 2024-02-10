import pickle
from itertools import product
from pathlib import Path

import numpy as np

from .graphmodel import GraphModel
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
                 alpha: float,
                 model_path: Path = Path('.')):
        self.epsilon_p = delta * epsilon
        self.epsilon_m = (1-delta) * epsilon
        self.delta = delta
        self.alpha = alpha
        self.base_model_path = model_path

    def perturb_graphs(self, paths: list[Path]) -> None:
        # Load graphs
        graphs = list(map_pool(
            GraphWrapper.load_file,
            paths[:200],
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
        # pruned_graphs = list(map_pool(
        #     GraphWrapper.prune,
        #     list(product(preprocessed_graphs, [self.alpha], [self.epsilon_p])),
        #     'Pruning graphs'
        # ))
        # del preprocessed_graphs
        pruned_graphs = []
        for graph in preprocessed_graphs:
            pruned_graphs.append(GraphWrapper.prune(graph, self.alpha, self.epsilon_p))

        # Create training data
        train_data: list[tuple[str, GraphWrapper]] = list(map_pool(
            GraphWrapper.get_train_data,
            pruned_graphs,
            'Creating training data'
        ))

        # Save training data
        with open('train_data.pkl', 'wb') as f:
            pickle.dump(train_data, f)
            print(f'Wrote {len(train_data)} training examples to train_data.pkl')

        # Train model
        paths = []
        graphs = []
        for path, graph in train_data:
            paths.append(path)
            graphs.append(graph)

        model = GraphModel(
             paths=paths,
             graphs=graphs,
             context_length=8,
             base_model_path=self.base_model_path
        )
        model.train(epochs=10000)

        # Add edges to graphs (epsilon_m)

