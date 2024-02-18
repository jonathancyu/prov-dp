import pickle
from itertools import chain
from pathlib import Path

from tqdm import tqdm

from .graph_model import GraphModel
from .utility import map_pool, print_stats
from .wrappers import GraphWrapper


class GraphProcessor:
    epsilon_p: float  # structural budget = (  delta) * epsilon
    epsilon_m: float  # edge count budget = (1-delta) * epsilon
    delta: float
    alpha: float

    # Stats
    __sizes: list[int]
    __depths: list[int]

    # Processing pipeline
    __process_steps: list[callable]

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

        self.__sizes = []
        self.__depths = []

        self.__process_steps = [
            GraphWrapper.load_file,
            GraphWrapper.preprocess,
            self.__prune_graphs
        ]

    def __prune_graphs(self, graph: GraphWrapper) -> GraphWrapper:
        x = graph.prune(self.alpha, self.epsilon_p)
        return x

    def prune_from_paths(self, paths: list[Path]) -> list[tuple[GraphWrapper, int, int]]:
        previous = paths
        result: list = []
        for i, step in enumerate(self.__process_steps):
            step_label = f'({i+1}) {step.__name__.strip("_")}'
            if self.args.single_threaded:
                # Do a simple loop
                result = [
                    step(graph)
                    for graph in tqdm(previous, step_label)
                ]
            else:
                # Using map_pool pickles the arguments and results, which consumes a lot of memory
                result = list(map_pool(
                    step,
                    previous,
                    step_label
                ))
            del previous  # Release memory from de-pickled results
            previous = result
        return result

    def perturb_graphs(self, paths: list[Path]) -> list[GraphWrapper]:
        # Prune graphs
        pruned_graphs_and_sizes_and_depths = self.prune_from_paths(paths)
        # TODO: the fact that we're tracking stats IN the GraphWrapper
        #       is a red flag that graph.prune should be in this class
        pruned_graphs = []
        sizes = []
        num_pruned = []
        # TODO: we should be able to do this in the pipeline
        for graph, size, x in pruned_graphs_and_sizes_and_depths:
            num_pruned.append(len(graph.edges))
            pruned_graphs.append(graph)
            assert isinstance(size, list)
            sizes.extend(size)

        print_stats('Pruned graph size', sizes)
        print_stats('Num pruned edges', num_pruned)

        # Create training data and flatten into a single list
        train_data: list[tuple[str, GraphWrapper]] = list(
            chain.from_iterable(
                map_pool(
                    GraphWrapper.get_train_data,
                    pruned_graphs,
                    '(4) Creating training data'
                )
            )
        )

        # Save a (pruned_graphs, training_data) tuple
        with open('pruned_graphs_and_train_data.pkl', 'wb') as f:
            pickle.dump((pruned_graphs, train_data), f)
            print(f'  Wrote {len(pruned_graphs)} graphs and {len(train_data)} training samples to train_data.pkl')

        # Train model
        print('(5) Training model')
        paths = []
        graphs = []
        for path, graph in train_data:
            paths.append(path)
            graphs.append(graph)

        model = GraphModel(
             paths=paths,
             graphs=graphs,
             context_length=8,
             base_model_path=self.output_path / 'models'
        )
        model.train(epochs=100)

        # Add graphs back (epsilon_m) # TODO: diff privacy here
        sizes = []
        num_marked_edges = []
        count_from_same_graph = []
        for graph in tqdm(pruned_graphs, desc='(6) Re-attaching subgraphs'):
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
        model.print_distance_stats()
        return pruned_graphs
