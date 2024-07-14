import pickle
from collections import deque
from pathlib import Path
from typing import Callable, Generator

import numpy as np
from tqdm import tqdm

from src.algorithm.wrappers.tree import Marker, TreeStats

from .utility import print_stats, logistic_function, smart_map
from .wrappers import Tree

PRUNED_TREE_SIZE = "pruned tree size (#nodes)"
PRUNED_TREE_HEIGHT = "pruned tree height"
PRUNED_TREE_DEPTH = "pruned tree depth"
NUM_MARKED_NODES = "# marked nodes"
ATTACHED_TREE_SIZE = "attached tree size (#nodes)"
NUM_UNMOVED_SUBTREES = "# unmoved subtrees"
PERCENT_UNMOVED_SUBTREES = "% unmoved subtrees"


# TODO: SRP..?
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

    # Step labels
    __step_number: int

    # Checkpoint flags
    __load_perturbed_graphs: bool
    __load_graph2vec: bool
    __load_model: bool

    # Model parameters
    __reattach_mode: str
    __num_epochs: int
    __prediction_batch_size: int

    # Stats
    stats: dict[str, list[float]]

    def __init__(
        self,
        epsilon_1: float = 1,
        epsilon_2: float = 0,
        alpha: float = 0.5,
        beta: float = 0,
        gamma: float = 0,
        output_dir: Path = Path("."),
        single_threaded: bool = False,
        load_perturbed_graphs: bool = False,
        load_graph2vec: bool = False,
        load_model: bool = False,
        reattach_mode: str = "bucket",
        num_epochs: int = 10,
        prediction_batch_size: int = 10,
    ):
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

        # Reattach mode
        self.__reattach_mode = reattach_mode

        # Checkpoint flags
        self.__load_perturbed_graphs = load_perturbed_graphs
        self.__load_graph2vec = load_graph2vec
        self.__load_model = load_model

        # Stats
        self.stats = {}

    def __step(self) -> str:  # Step counter for pretty logging
        self.__step_number += 1
        return f"({self.__step_number})"

    def __map(self, func: Callable, items: list, desc: str = "") -> Generator:
        generator = smart_map(
            func=func, items=items, single_threaded=self.__single_threaded, desc=desc
        )
        for item in generator:
            yield item

    def preprocess_graphs(self, paths: list[Path]) -> list[Tree]:
        trees = list(self.__map(Tree.load_file, paths, f"Preprocessing graphs"))
        print("Data1 stats:")
        self.print_tree_stats(trees)
        return trees

    def get_tree_stats(
        self, trees: list[Tree]
    ) -> tuple:  # TODO: return ANYTHING other than this..
        stats: list[TreeStats] = list(
            smart_map(
                func=Tree.get_stats,
                items=trees,
                single_threaded=self.__single_threaded,
                desc="Calculating stats",
            )
        )

        node_stats = {"heights": [], "depths": [], "sizes": [], "degrees": []}

        tree_stats = {"heights": [], "sizes": [], "degrees": [], "diameters": []}
        for stat in tqdm(stats, desc="Aggregating stats"):
            # Node stats
            node_stats["heights"].extend(stat.heights)
            node_stats["depths"].extend(stat.depths)
            node_stats["sizes"].extend(stat.sizes)
            node_stats["degrees"].extend(stat.degrees)

            # Tree stats
            tree_stats["heights"].append(stat.height)
            tree_stats["sizes"].append(stat.size)
            tree_stats["degrees"].append(stat.degree)
            tree_stats["diameters"].append(stat.diameter)

        return node_stats, tree_stats

    def print_tree_stats(self, trees: list[Tree]):
        _, tree_stats = self.get_tree_stats(trees)
        print_stats("Tree height", tree_stats["heights"])
        print_stats("Tree size", tree_stats["sizes"])
        print_stats("Degrees", tree_stats["degrees"])
        print_stats("Diameters", tree_stats["diameters"])

    def load_and_prune_graphs(self, paths: list[Path]) -> list[Tree]:
        # Try to load checkpoint if one exists
        pruned_graph_path = self.output_dir / "pruned_graphs.pkl"
        if self.__load_perturbed_graphs and pruned_graph_path.exists():
            # Load graphs and training data from file
            print(
                f"{self.__step()} Loading pruned graphs and training data from {pruned_graph_path}"
            )
            with open(pruned_graph_path, "rb") as f:
                pruned_graphs, train_data = pickle.load(f)
                self.__training_data = train_data
                print(
                    f"  Loaded {len(pruned_graphs)} graphs and {len(train_data)} training samples"
                )
                return pruned_graphs

        # Load and convert input graphs to trees
        trees = self.preprocess_graphs(paths)

        pruned_trees = list(
            self.__map(self.prune, trees, f"{self.__step()} Pruning graphs")
        )

        # Aggregate training data from trees
        self.__training_data = []
        for tree in pruned_trees:
            self.__training_data.extend(tree.training_data)
            self.__add_stats(tree.stats)

        self.__print_stats()

        # Write result to checkpoint
        with open(pruned_graph_path, "wb") as f:
            # Save a (pruned_graphs, training_data) tuple
            pickle.dump((pruned_trees, self.__training_data), f)
            print(
                f"  Wrote {len(pruned_trees)} graphs and {len(self.__training_data)} "
                f"training samples to {pruned_graph_path}"
            )

        return pruned_trees

    def prune(self, tree: Tree) -> Tree:

        # Returns tuple (pruned tree, list of training data)
        # Breadth first search through the graph, keeping track of the path to the current node
        # (node_id, list[edge_id_path]) tuples
        root_node_id = tree.get_root()
        tree.init_node_stats(root_node_id, 0)
        queue: deque[tuple[int, list[int]]] = deque([(root_node_id, [])])
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
            node_stats = tree.get_node_stats(src_node_id)
            subtree_size, height, depth = (
                node_stats.size,
                node_stats.height,
                node_stats.depth,
            )
            # assert depth == len(path)
            distance = (
                (self.__alpha * subtree_size)
                + (self.__beta * height)
                + (self.__gamma * depth)
            )
            p = logistic_function(
                self.__epsilon_1 * distance
            )  # big distance -> lower probability of pruning
            prune_edge: bool = np.random.choice([True, False], p=[p, 1 - p])
            # if we prune, don't add children to queue
            if (
                prune_edge and len(path) > 1
            ):  # don't prune ephemeral root by restricting depth to > 1
                # remove the tree rooted at this edge's dst_id from the graph
                pruned_tree = tree.prune_tree(src_node_id)
                # Keep track of the node and its path, so we can attach to it later
                path_string = tree.path_to_string(path)
                tree.marked_node_paths[src_node_id] = path_string

                # Mark the node and keep its stats
                tree.marked_nodes[src_node_id] = Marker(
                    height=height, size=subtree_size, path=path_string, tree=pruned_tree
                )

                # add tree, and path to the tree to the training data
                tree.training_data.append(
                    (tree.path_to_string(path), pruned_tree)
                )  # TODO: add to Marker class?

                # ensure we don't try to bfs into the pruned tree
                visited_node_ids.update(
                    node.get_id() for node in pruned_tree.get_nodes()
                )

                # track statistics
                tree.add_stat(PRUNED_TREE_SIZE, subtree_size)
                tree.add_stat(PRUNED_TREE_HEIGHT, height)
                tree.add_stat(PRUNED_TREE_DEPTH, depth)

                continue

            # otherwise, continue adding children to the BFS queue
            for edge_id in tree.get_outgoing_edge_ids(src_node_id):
                edge = tree.get_edge(edge_id)
                dst_node_id = edge.get_dst_id()
                queue.append((dst_node_id, path + [edge_id]))

        return tree

    def perturb_graphs(self, paths: list[Path]) -> list[Tree]:
        pruned_graphs = self.load_and_prune_graphs(paths)

        model_type = self.__reattach_mode
        if model_type == "bucket":
            self.__re_add_with_bucket(pruned_graphs)
        else:
            raise ValueError(f"Unexpected model type {model_type}")

        for tree in pruned_graphs:
            tree.assert_valid_tree()

        if len(self.stats.get(NUM_UNMOVED_SUBTREES, [])) > 0:
            num_unmoved_subtrees = self.stats[NUM_UNMOVED_SUBTREES]
            num_marked_nodes = self.stats[NUM_MARKED_NODES]
            # TODO: this is broken
            self.stats[PERCENT_UNMOVED_SUBTREES] = [
                (x / max(y, 0.0001)) * 100
                for x, y in zip(num_unmoved_subtrees, num_marked_nodes)
            ]
        self.__print_stats()
        print("Data2 stats:")
        self.print_tree_stats(pruned_graphs)
        return pruned_graphs

    def __re_add_with_bucket(self, pruned_trees: list[Tree]):
        paths = []
        bucket = []
        for path, tree in self.__training_data:
            paths.append(path)
            bucket.append(tree)

        indices = np.arange(len(bucket))
        size_array = np.array([tree.size() for tree in bucket], dtype=int)
        assert len(indices) == len(size_array) == len(bucket)

        for tree in tqdm(pruned_trees, desc=f"{self.__step()} Re-attaching subgraphs"):
            # Stats
            self.__add_stat(NUM_MARKED_NODES, len(tree.marked_node_paths))
            unmoved_subtrees = 0

            for node_id, marker in tree.marked_nodes.items():
                distances = size_array - marker.size

                # Not the actual stdev - this controls the "tightness" of the distribution
                # TODO: not set in stone
                spread = (
                    1 / self.__epsilon_2
                )  # low epsilon -> high stdev -> less likely to choose tree w/ matching size
                weights = (1 / distances) ** spread
                probabilities = weights / sum(weights)

                if np.isnan(probabilities).any():
                    # TODO: this occurs when all sizes are the same equal
                    choice = np.random.choice(indices)
                else:
                    choice = np.random.choice(indices, p=probabilities)
                subtree: Tree = bucket[choice]

                tree.replace_node_with_tree(node_id, subtree)

                # Stats
                # Recall: pruned subtrees have an additional node
                # TODO: same node id or not
                self.__add_stat(ATTACHED_TREE_SIZE, (subtree.size() - 1))
                if subtree.graph_id == tree.graph_id:
                    unmoved_subtrees += 1

            # Stats
            self.__add_stat(NUM_UNMOVED_SUBTREES, unmoved_subtrees)

    def __print_stats(self):
        for stat, values in self.stats.items():
            print_stats(stat, values)

    def __add_stat(self, stat: str, value: float):
        if stat not in self.stats:
            self.stats[stat] = []
        self.stats[stat].append(value)

    def __add_stats(self, stats: dict[str, list[float]]):
        for stat, values in stats.items():
            if stat not in self.stats:
                self.stats[stat] = []
            self.stats[stat].extend(values)
