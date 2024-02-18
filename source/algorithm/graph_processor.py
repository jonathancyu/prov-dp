import pickle
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from pathlib import Path
from typing import TypeVar, Generator

from tqdm import tqdm

from .graph_model import GraphModel
from .utility import print_stats
from .wrappers import GraphWrapper

T = TypeVar('T')


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

    # Step labels
    __step_number: int

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
        self.output_path.mkdir(exist_ok=True, parents=True)

        self.__sizes = []
        self.__depths = []

        self.__process_steps = [
            GraphWrapper.load_file,
            GraphWrapper.preprocess,
            self.prune_graph
        ]

        self.__step_number = 0

    def __step(self) -> str:
        # Step counter for tqdm bars
        self.__step_number += 1
        return f'({self.__step_number})'

    def prune_from_paths(self, paths: list[Path]) -> list[tuple[GraphWrapper, int, int]]:
        previous = paths
        result: list = []
        for i, step in enumerate(self.__process_steps):
            step_label = f'({i+1}) {step.__name__.strip("_")}'
            result = list(self.map(
                step,
                previous,
                step_label
            ))
            del previous  # Release memory from de-pickled results
            previous = result
        return result

    def process_graph(self, path: Path) -> tuple[list[GraphWrapper], list[tuple[str, GraphWrapper]]]:
        # Apply n
        result = path
        for step in self.__process_steps:
            result = step(result)
        return result

    def prune_graph(self, graph: GraphWrapper) -> GraphWrapper:
        return graph.prune(self.alpha, self.epsilon_p)

    def load_and_prune_graphs(self, paths: list[Path]) -> tuple[list[GraphWrapper], list[tuple[str, GraphWrapper]]]:
        pruned_graphs_and_sizes_and_depths = list(self.map(
            self.process_graph,
            paths,
            f'{self.__step()} Pruning graphs'
        ))
        # TODO: the fact that we're tracking stats in the GraphWrapper
        #       is a red flag that graph.prune should be in this class
        pruned_graphs = []
        sizes = []
        num_pruned = []
        # TODO: we should be able to do this in the pipeline
        for graph, size, x in pruned_graphs_and_sizes_and_depths:
            num_pruned.append(len(graph.edges))
            pruned_graphs.append(graph)
            sizes.extend(size)

        print_stats('Pruned graph size', sizes)
        print_stats('Num pruned edges', num_pruned)

        # Create training data and flatten into a single list
        train_data: list[tuple[str, GraphWrapper]] = list(
            chain.from_iterable(
                self.map(
                    GraphWrapper.get_train_data,
                    pruned_graphs,
                    f'{self.__step()} Creating training data'
                )
            )
        )
        return pruned_graphs, train_data

    def perturb_graphs(self, paths: list[Path]) -> list[GraphWrapper]:
        pruned_graph_path = self.output_path / 'pruned_graphs.pkl'
        if self.args.load_perturbed_graphs and pruned_graph_path.exists():
            # Load graphs and training data from file
            print(f'{self.__step()} Loading pruned graphs and training data from {pruned_graph_path}')
            with open(pruned_graph_path, 'rb') as f:

                pruned_graphs, train_data = pickle.load(f)
                print(f'  Loaded {len(pruned_graphs)} graphs and {len(train_data)} training samples')
        else:
            # Perturb graphs and write graphs and training data to file
            pruned_graphs, train_data = self.load_and_prune_graphs(paths)
            with open(pruned_graph_path, 'wb') as f:
                # Save a (pruned_graphs, training_data) tuple
                pickle.dump((pruned_graphs, train_data), f)
                print(f'  Wrote {len(pruned_graphs)} graphs and {len(train_data)} training samples to {pruned_graph_path}')

        # Train model
        print(f'{self.__step()} Training model')
        paths = []
        graphs = []
        for path, graph in train_data:
            paths.append(path)
            graphs.append(graph)

        model = GraphModel(
             paths=paths,
             graphs=graphs,
             context_length=8,
             base_model_path=self.output_path / 'models',
             load_graph2vec=self.args.load_graph2vec
        )
        model.train(epochs=10)

        # Add graphs back (epsilon_m) # TODO: diff privacy here
        sizes = []
        num_marked_edges = []
        count_from_same_graph = []
        for graph in tqdm(pruned_graphs, desc=f'{self.__step()} Re-attaching subgraphs'):
            # Stats
            num_marked_edges.append(len(graph.marked_edge_ids))
            from_same_graph = 0
            edge_ids = list(graph.marked_edge_ids.keys())
            # Re-attach a random to each marked edge
            PREDICTION_BATCH_SIZE = 10
            for batch_start in range(0, len(edge_ids), PREDICTION_BATCH_SIZE):
                batch_ids = edge_ids[batch_start:batch_start + PREDICTION_BATCH_SIZE]
                predictions = model.predict([graph.marked_edge_ids[batch_id] for batch_id in batch_ids])

                for i, subgraph in enumerate(predictions):
                    graph.insert_subgraph(edge_ids[i], subgraph)
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

    def map(self,
            func: callable,
            items: list[T],
            desc: str = '') -> Generator:
        if self.args.single_threaded:
            # Do a simple loop
            for graph in tqdm(items, desc=desc):
                yield func(graph)
        else:
            # Using this pickles the arguments and results, which consumes a lot of memory
            with ProcessPoolExecutor() as executor:
                # When we multiprocess, objects are pickled and copied in the child process
                # instead of using the same object, so we have to return objects from the
                # function to get the changes back
                futures = [
                    executor.submit(func, *item) if isinstance(item, tuple)
                    else executor.submit(func, item)
                    for item in items
                ]
                with tqdm(total=len(futures), desc=desc) as pbar:
                    for future in futures:
                        yield future.result()
                        pbar.update(1)
