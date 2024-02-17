import pickle
from itertools import product
from pathlib import Path

import numpy as np

from . import IN
from .graph_model import GraphModel
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
                 output_path: Path = Path('.')):
        self.epsilon_p = delta * epsilon
        self.epsilon_m = (1-delta) * epsilon
        self.delta = delta
        self.alpha = alpha
        self.output_path = output_path

    def perturb_graphs(self, paths: list[Path]) -> list[GraphWrapper]:
        # Load graphs
        graphs = list(map_pool(
            GraphWrapper.load_file,
            paths,
            'Loading graphs'
        ))
        # === Preprocess graphs === #
        preprocessed_graphs = map_pool(
            GraphWrapper.preprocess,
            graphs,
            'Preprocessing graphs'
        )
        del graphs

        # === Prune graphs === #
        pruned_graphs_and_sizes_and_depths = list(map_pool(
            GraphWrapper.prune,
            list(product(preprocessed_graphs, [self.alpha], [self.epsilon_p])),
            'Pruning graphs'
        ))
        pruned_graphs = []
        sizes = []
        num_pruned = []
        for graph, size, x in pruned_graphs_and_sizes_and_depths:
            num_pruned.append(len(graph.edges))
            pruned_graphs.append(graph)
            sizes.extend(size)
        print(f'Pruned graph size (N={len(sizes)}) - avg: {np.mean(sizes):.2f}, min: {np.min(sizes)}, max: {np.max(sizes)}, std: {np.std(sizes):.2f}')
        print(f'Num pruned edges (N={len(num_pruned)}) - avg: {np.mean(num_pruned):.2f}, min: {np.min(num_pruned)}, max: {np.max(num_pruned)}, std: {np.std(num_pruned):.2f}')
        # del preprocessed_graphs
        # pruned_graphs = []
        # for graph in preprocessed_graphs:
        #     pruned_graphs.append(GraphWrapper.prune(graph, self.alpha, self.epsilon_p))

        # === Create training data === #
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
        for graph_data in train_data:
            for path, graph in graph_data:
                paths.append(path)
                graphs.append(graph)

        model = GraphModel(
             paths=paths,
             graphs=graphs,
             context_length=8,
             base_model_path=self.output_path / 'models'
        )
        model.train(epochs=1000)

        # Add graphs back (epsilon_m)
        sizes = []
        num_marked_edges = []
        for graph in pruned_graphs:
            print(f'Processing graph {graph.source_edge_ref_id}')
            num_marked_edges.append(len(graph.marked_edge_ids))
            for marked_edge_id, path in graph.marked_edge_ids.items():
                print(f'    {marked_edge_id}')
                subgraph = model.predict(path)
                sizes.append(len(subgraph))
                graph.insert_subgraph(marked_edge_id, subgraph)

        print(f'Subgraph size - avg: {np.mean(sizes):.2f}, min: {np.min(sizes)}, max: {np.max(sizes)}, '
              f'std: {np.std(sizes):.2f}')
        print(f'Num marked edges - avg: {np.mean(num_marked_edges):.2f}, min: {np.min(num_marked_edges)}, ' +
              f'max: {np.max(num_marked_edges)}, std: {np.std(num_marked_edges):.2f}')

        return pruned_graphs
