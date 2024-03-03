import pickle
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from pathlib import Path
from typing import Generator

from tqdm import tqdm

from .graph_model import GraphModel
from .utility import print_stats, batch_list
from .wrappers import Tree


class GraphProcessor:
    __epsilon_p: float  # structural budget = (  delta) * epsilon
    __epsilon_m: float  # edge count budget = (1-delta) * epsilon
    __delta: float
    __alpha: float

    # Stats
    __sizes: list[int]
    __depths: list[int]

    # Processing pipeline
    __single_threaded: bool
    __process_steps: list[callable]

    # Step labels
    __step_number: int

    # Checkpoint flags
    __load_perturbed_graphs: bool
    __load_graph2vec: bool
    __load_model: bool

    # Model parameters
    __num_epochs: int
    __prediction_batch_size: int

    # Stats
    stats: dict[str, list[float]]

    def __init__(self,
                 epsilon: float = 1,
                 delta: float = 1,
                 alpha: float = 0.5,
                 output_dir: Path = Path('.'),
                 single_threaded: bool = False,
                 load_perturbed_graphs: bool = False,
                 load_graph2vec: bool = False,
                 load_model: bool = False,
                 num_epochs: int = 10,
                 prediction_batch_size: int = 10):

        self.__epsilon_p = delta * epsilon
        self.__epsilon_m = (1 - delta) * epsilon
        self.__delta = delta
        self.__alpha = alpha
        self.__sizes = []
        self.__depths = []

        self.__step_number = 0

        # TODO: there's gotta be a cleaner way to do this...
        # argparse args
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Algorithm configuration
        self.__single_threaded = single_threaded
        self.__num_epochs = num_epochs
        self.__prediction_batch_size = prediction_batch_size

        # Model parameters
        self.__num_epochs = num_epochs
        self.__prediction_batch_size = prediction_batch_size

        # Checkpoint flags
        self.__load_perturbed_graphs = load_perturbed_graphs
        self.__load_graph2vec = load_graph2vec
        self.__load_model = load_model

        # Stats
        self.stats = {}

    def __step(self) -> str:  # Step counter for pretty logging
        self.__step_number += 1
        return f'({self.__step_number})'

    def preprocess_graphs(self, paths: list[Path]) -> list[Tree]:
        return list(self.__map(
            Tree.load_file,
            paths,
            f'Preprocessing graphs'
        ))

    def process_graph(self, path: Path) -> tuple[list[Tree], list[tuple[str, Tree]]]:
        # Load and prune a graph
        tree = Tree.load_file(path)
        return tree.prune(self.__alpha, self.__epsilon_p)

    def load_and_prune_graphs(self, paths: list[Path]) -> tuple[list[Tree], list[tuple[str, Tree]]]:
        pruned_graphs_and_sizes_and_depths = list(self.__map(
            self.process_graph,
            paths,
            f'{self.__step()} Pruning graphs'
        ))
        # TODO: the fact that we're tracking stats in the Tree
        #       is a red flag 🚩🚩 that graph.prune should be in this class
        pruned_graphs = []
        sizes = []
        num_pruned = []
        # TODO: we should be able to do this in the pipeline
        for graph, size, x in pruned_graphs_and_sizes_and_depths:
            num_pruned.append(len(graph.get_edges()))
            pruned_graphs.append(graph)
            sizes.extend(size)

        print_stats('Pruned graph size', sizes)
        self.stats['pruned_graph_size'] = sizes
        print_stats('Num pruned edges', num_pruned)
        self.stats['#pruned_edges'] = num_pruned

        # Create training data and flatten into a single list
        train_data: list[tuple[str, Tree]] = list(
            chain.from_iterable(
                self.__map(
                    Tree.get_train_data,
                    pruned_graphs,
                    f'{self.__step()} Creating training data'
                )
            )
        )
        return pruned_graphs, train_data

    def perturb_graphs(self, paths: list[Path]) -> list[Tree]:
        pruned_graph_path = self.output_dir / 'pruned_graphs.pkl'
        if self.__load_perturbed_graphs and pruned_graph_path.exists():
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
                print(f'  Wrote {len(pruned_graphs)} graphs and {len(train_data)} '
                      f'training samples to {pruned_graph_path}')

        # Train model
        print(f'{self.__step()} Training model')
        paths = []
        graphs = []
        for path, tree in train_data:
            paths.append(path)
            graphs.append(tree)

        model = GraphModel(
             paths=paths,
             graphs=graphs,
             context_length=8,
             base_model_path=self.output_dir / 'models',
             load_graph2vec=self.__load_graph2vec,
             load_model=self.__load_model
        )
        model.train(epochs=self.__num_epochs)

        # Add graphs back (epsilon_m) # TODO: diff privacy here
        sizes = []
        num_marked_nodes = []
        num_unmoved_subtrees = []
        for tree in tqdm(pruned_graphs, desc=f'{self.__step()} Re-attaching subgraphs'):
            # Stats
            num_marked_nodes.append(len(tree.marked_node_paths))
            unmoved_subtrees = 0

            # Re-attach a random to each marked edge in batches
            node_ids = list(tree.marked_node_paths.keys())
            total = 0
            for batch in batch_list(node_ids, self.__prediction_batch_size):
                # Get a prediction for each edge in the batch
                predictions = model.predict(
                    [tree.marked_node_paths[node_id]
                     for node_id in batch]
                )
                # Attach predicted subgraph to the corresponding edge
                for i, subgraph in enumerate(predictions):
                    total += 1
                    assert tree.get_node(batch[i]) is not None
                    tree.replace_node_with_tree(batch[i], subgraph)
                    # Stats
                    sizes.append(len(subgraph))
                    if subgraph.source_edge_ref_id == tree.source_edge_ref_id:
                        unmoved_subtrees += 1
            assert total == len(node_ids)

            # Stats
            num_unmoved_subtrees.append(unmoved_subtrees)

        print_stats('Subgraph size', sizes)
        self.stats['subgraph_size'] = sizes
        print_stats('# marked nodes', num_marked_nodes)
        self.stats['#marked_nodes'] = num_marked_nodes
        print_stats('# unmoved subtrees', num_unmoved_subtrees)
        self.stats['#unmoved_subtrees'] = num_unmoved_subtrees
        percent_unmoved = [(x / y) * 100 for x, y in zip(num_unmoved_subtrees, num_marked_nodes)]
        print_stats('% unmoved subtrees', percent_unmoved)
        self.stats['% unmoved_subtrees'] = percent_unmoved

        print()
        model.print_distance_stats()
        return pruned_graphs

    def __map(self,
              func: callable,
              items: list,
              desc: str = '') -> Generator:
        if self.__single_threaded:
            # Do a simple loop
            for graph in tqdm(items, desc=desc):
                yield func(graph)
            return

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
