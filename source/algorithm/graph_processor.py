import pickle
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import Generator

import networkx as nx
import numpy as np
from tqdm import tqdm

from .graph_model import GraphModel
from .utility import print_stats, batch_list, logistic_function
from .wrappers import Tree

PRUNED_TREE_SIZE = 'pruned tree size (#nodes)'
PRUNED_TREE_HEIGHT = 'pruned tree height'
PRUNED_TREE_DEPTH = 'pruned tree depth'
NUM_MARKED_NODES = '# marked nodes'
ATTACHED_TREE_SIZE = 'attached tree size (#nodes)'
NUM_UNMOVED_SUBTREES = '# unmoved subtrees'
PERCENT_UNMOVED_SUBTREES = '% unmoved subtrees'


class GraphProcessor:
    # Pruning parameters
    __epsilon_1: float
    __epsilon_2: float
    __alpha: float
    __beta: float
    __gamma: float

    # List to aggregate training data (path: str, subtree: Tree) tuples
    __training_data: list[tuple[str, Tree]]

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
                 epsilon_1: float = 1,
                 epsilon_2: float = 0,
                 alpha: float = 0.5,
                 beta: float = 0,
                 gamma: float = 0,
                 output_dir: Path = Path('.'),
                 single_threaded: bool = False,
                 load_perturbed_graphs: bool = False,
                 load_graph2vec: bool = False,
                 load_model: bool = False,
                 num_epochs: int = 10,
                 prediction_batch_size: int = 10):
        # Pruning parameters
        self.__epsilon_1 = epsilon_1
        self.__epsilon_2 = epsilon_2
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma

        # List to aggregate training data
        self.__training_data = []

        # Logging
        self.__step_number = 0

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

    def preprocess_graphs(self, paths: list[Path]) -> list[Tree]:
        trees = list(self.__map(
            Tree.load_file,
            paths,
            f'Preprocessing graphs'
        ))
        self.print_tree_stats(trees)
        return trees

    def load_and_prune_graphs(self, paths: list[Path]) -> list[Tree]:
        trees = list(self.__map(
            Tree.load_file,
            paths,
            f'{self.__step()} Loading graphs'
        ))
        pruned_trees = list(self.__map(
            self.prune,
            trees,
            f'{self.__step()} Pruning graphs'
        ))
        self.__training_data = list(chain.from_iterable(
            [tree.training_data for tree in pruned_trees]
        ))
        self.__print_stats()

        return pruned_trees

    def prune(self, tree: Tree) -> Tree:
        # Returns tuple (pruned tree, list of training data)
        # Breadth first search through the graph, keeping track of the path to the current node
        # (node_id, list[edge_id_path]) tuples
        queue: deque[tuple[int, list[int]]] = deque([(tree.root_node_id, [])])
        visited_node_ids: set[int] = set()
        while len(queue) > 0:
            # Standard BFS operations
            src_node_id, path = queue.popleft()

            if src_node_id in visited_node_ids:
                continue
            visited_node_ids.add(src_node_id)

            # Dr. De: more metrics to consider: height/level at which a node is sitting (distance from leaf).
            #         can we use this along with the size to get a better result?
            # calculate the probability of pruning a given tree
            subtree_size = tree.get_tree_size(src_node_id)
            height = tree.get_tree_height(src_node_id)
            depth = len(path)
            distance = (self.__alpha * subtree_size) + (self.__beta * height) + (self.__gamma * depth)
            p = logistic_function(self.__epsilon_1 * distance)  # big distance -> lower probability of pruning
            prune_edge: bool = np.random.choice([True, False],
                                                p=[p, 1 - p])
            # if we prune, don't add children to queue
            if prune_edge and len(path) > 1:  # don't prune ephemeral root by restricting depth to > 1
                # remove the tree rooted at this edge's dst_id from the graph
                pruned_tree = tree.prune_tree(src_node_id)
                # Keep track of the node and its path, so we can attach to it later
                path_string = tree.path_to_string(path)
                tree.marked_node_paths[src_node_id] = path_string

                # add tree, and path to the tree to the training data
                tree.training_data.append((tree.path_to_string(path), pruned_tree))

                # ensure we don't try to bfs into the pruned tree
                visited_node_ids.update(node.get_id() for node in pruned_tree.get_nodes())

                # track statistics
                self.__add_stat(PRUNED_TREE_SIZE, subtree_size)
                self.__add_stat(PRUNED_TREE_HEIGHT, height)
                self.__add_stat(PRUNED_TREE_DEPTH, depth)

                continue

            # otherwise, continue adding children to queue
            next_edge_ids = tree.get_outgoing_edge_ids(src_node_id)

            for edge_id in next_edge_ids:
                src_node = tree.get_node(src_node_id)
                edge = tree.get_edge(edge_id)
                dst_node_id = edge.get_dst_id()
                dst_node = tree.get_node(dst_node_id)
                # if this isn't a leaf, then continue and add the next edges to the queue
                if len(next_edge_ids) > 0:
                    queue.append((dst_node_id, path + [edge_id]))

                # if this is a leaf, add the path and current graph to the training data
                else:
                    # deep copy the leaf and its parent to modify them
                    parent_node = deepcopy(src_node)
                    leaf_node = deepcopy(dst_node)

                    # add the leaf (and parent) to its own graph
                    leaf_tree = Tree()
                    leaf_tree.graph_id = tree.graph_id
                    leaf_tree.add_node(parent_node)
                    leaf_tree.add_node(leaf_node)
                    leaf_tree.add_edge(deepcopy(edge))
                    # add the (path, graph) tuple to the training data
                    tree.training_data.append((tree.path_to_string(path), leaf_tree))
                    continue

        return tree

    def perturb_graphs(self, paths: list[Path]) -> list[Tree]:
        pruned_graph_path = self.output_dir / 'pruned_graphs.pkl'
        if self.__load_perturbed_graphs and pruned_graph_path.exists():
            # Load graphs and training data from file
            print(f'{self.__step()} Loading pruned graphs and training data from {pruned_graph_path}')
            with open(pruned_graph_path, 'rb') as f:
                pruned_graphs, train_data = pickle.load(f)
                self.__training_data = train_data
                print(f'  Loaded {len(pruned_graphs)} graphs and {len(train_data)} training samples')
        else:
            # Perturb graphs and write graphs and training data to file
            pruned_graphs = self.load_and_prune_graphs(paths)
            with open(pruned_graph_path, 'wb') as f:
                # Save a (pruned_graphs, training_data) tuple
                pickle.dump((pruned_graphs, self.__training_data), f)
                print(f'  Wrote {len(pruned_graphs)} graphs and {len(self.__training_data)} '
                      f'training samples to {pruned_graph_path}')

        return pruned_graphs  # TODO: REMOVE THIS, its a hack to get quick stats
        # Train model
        print(f'{self.__step()} Training model')
        paths = []
        graphs = []
        for path, tree in self.__training_data:
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

        # Add graphs back (epsilon_2)
        for tree in tqdm(pruned_graphs, desc=f'{self.__step()} Re-attaching subgraphs'):
            # Stats
            self.__add_stat(NUM_MARKED_NODES, len(tree.marked_node_paths))
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
                    self.__add_stat(ATTACHED_TREE_SIZE, len(subgraph))
                    if subgraph.graph_id == tree.graph_id:
                        unmoved_subtrees += 1

            assert total == len(node_ids)

            # Stats
            self.__add_stat(NUM_UNMOVED_SUBTREES, unmoved_subtrees)

        if len(self.stats.get(NUM_UNMOVED_SUBTREES, [])) > 0:
            num_unmoved_subtrees = self.stats[NUM_UNMOVED_SUBTREES]
            num_marked_nodes = self.stats[NUM_MARKED_NODES]
            self.stats[PERCENT_UNMOVED_SUBTREES] = [(x / max(y, 0.0001)) * 100
                                                    for x, y in zip(num_unmoved_subtrees, num_marked_nodes)]
        self.__print_stats()
        model.print_distance_stats()
        self.print_tree_stats(pruned_graphs)
        return pruned_graphs

    @staticmethod
    def get_single_tree_stats(tree: Tree):
        outdegrees = []
        max_degree = 0
        for node in tree.get_nodes():
            node_id = node.get_id()
            outgoing_edges = tree.get_outgoing_edge_ids(node_id)
            outdegree = len(outgoing_edges)
            max_degree = max(outdegree, max_degree)
            outdegrees.append(outdegree)
        height = tree.get_tree_height(tree.root_node_id, )
        G = tree.to_nx().to_undirected()
        diameter = max([max(j.values()) for (i, j) in nx.shortest_path_length(G)])
        return outdegrees, max_degree, height, diameter

    def print_tree_stats(self, trees: list[Tree]) -> None:
        # avg/std of degree, height, diameter
        outdegrees: list[int] = []
        max_degrees: list[int] = []
        heights: list[int] = []
        diameters: list[int] = []
        results = list(self.__map(self.get_single_tree_stats, trees, 'Calculating tree stats'))

        for outdegree, max_degree, height, diameter in results:
            outdegrees.extend(outdegree)
            max_degrees.append(max_degree)
            heights.append(height)
            diameters.append(diameter)

        print_stats('outdegree: ', outdegrees)
        print_stats('max_degree: ', max_degrees)
        print_stats('height: ', heights)
        print_stats('diameter: ', diameters)

    def __print_stats(self):
        for stat, values in self.stats.items():
            print_stats(stat, values)

    def __add_stat(self, stat: str, value: float):
        if stat not in self.stats:
            self.stats[stat] = []
        self.stats[stat].append(value)
