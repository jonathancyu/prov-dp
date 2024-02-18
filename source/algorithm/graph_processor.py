import pickle
from itertools import product
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import IN
from .graph_model import GraphModel
from .utility import map_pool, print_stats
from .wrappers import GraphWrapper, EdgeWrapper


class GraphProcessor:
    epsilon_p: float  # structural budget = (  delta) * epsilon
    epsilon_m: float  # edge count budget = (1-delta) * epsilon
    delta: float
    alpha: float

    def __init__(self,
                 epsilon: float,
                 delta: float,
                 alpha: float,
                 args: any):
        self.args = args

        self.epsilon_p = delta * epsilon
        self.epsilon_m = (1-delta) * epsilon
        self.delta = delta
        self.alpha = alpha
        self.output_path = args.output_dir

    __process_steps: list[callable] = [
        GraphWrapper.load_file,
        GraphWrapper.preprocess,
        GraphWrapper.prune
    ]

    def prune_from_paths(self, paths: list[Path]) -> list[GraphWrapper]:
        previous = None
        result: list = paths

        for i, step in enumerate(self.__process_steps):
            if self.args.single_threaded:
                # Do a simple loop
                result = [
                    step(graph)
                    for graph in tqdm(result, GraphProcessor.__format_step(i, step))
                ]
            else:
                # Using map_pool pickles the arguments and results, which consumes a lot of memory
                result = list(map_pool(
                    step,
                    previous,
                    GraphProcessor.__format_step(i, step)
                ))
            del previous  # Release memory from de-pickled results
            previous = result
        return result

    @staticmethod
    def __format_step(i, step):
        return f'({i+1}) {step.__name__.strip("_")}'

    def perturb_graphs(self, paths: list[Path]) -> list[GraphWrapper]:
        # Load graphs
        graphs = map_pool(
            GraphWrapper.load_file,
            paths,
            'Loading graphs'
        )
        # === Preprocess graphs === #
        preprocessed_graphs = map_pool(
            GraphWrapper.preprocess,
            list(graphs),
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
        print_stats('Pruned graph size', sizes)
        print_stats('Num pruned edges', num_pruned)
        del preprocessed_graphs

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
        model.train(epochs=10)

        # Add graphs back (epsilon_m) # TODO: diff privacy here
        sizes = []
        num_marked_edges = []
        count_from_same_graph = []
        for graph in tqdm(pruned_graphs):
            # Stats
            from_same_graph = 0
            num_marked_edges.append(len(graph.marked_edge_ids))
            # Re-attach a random to each marked edge
            for marked_edge_id, path in graph.marked_edge_ids.items():
                # Predict a subgraph
                subgraph = model.predict(path)
                # Attach to the marked edge
                graph.insert_subgraph(marked_edge_id, subgraph)

                # Stats
                sizes.append(len(subgraph))
                if subgraph.source_edge_ref_id == graph.source_edge_ref_id:
                    from_same_graph += 1
            count_from_same_graph.append(from_same_graph)

        print_stats('Subgraph size', sizes)
        print_stats('# marked edges', num_marked_edges)
        print_stats('# unmoved subgraphs', count_from_same_graph)
        from_same_graph_percentages = [x / y for x, y in zip(count_from_same_graph, num_marked_edges)]
        print_stats('% unmoved subgraphs', from_same_graph_percentages)
        print()
        model.print_stats()
        return pruned_graphs
