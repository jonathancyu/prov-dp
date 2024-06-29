import argparse
from pathlib import Path
from typing import TypeVar

from graphviz import Digraph

T = TypeVar("T")


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-i", "--input_dir", type=Path, help="Path to input graph directory"
    )

    # GraphProcessor arguments
    arg_parser.add_argument(
        "-N",
        "--num_graphs",
        type=int,
        default=None,
        help="Limit the number of graphs to process",
    )
    arg_parser.add_argument(
        "-o", "--output_dir", type=Path, help="Path to output graph directory"
    )

    # Differential privacy parameters
    arg_parser.add_argument(
        "-e1",
        "--epsilon1",
        type=float,
        default=1,
        help="Differential privacy budget for pruning",
    )
    arg_parser.add_argument(
        "-e2",
        "--epsilon2",
        type=float,
        default=1,
        help="Differential privacy budget for reattaching",
    )

    arg_parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=1,
        help="Weight of subtree size on pruning probability",
    )
    arg_parser.add_argument(
        "-b",
        "--beta",
        type=float,
        default=1,
        help="Weight of subtree height on pruning probability",
    )
    arg_parser.add_argument(
        "-c",
        "--gamma",
        type=float,
        default=1,
        help="Weight of subtree depth on pruning probability",
    )

    # Algorithm configuration
    arg_parser.add_argument(
        "-s",
        "--single_threaded",
        action="store_true",
        help="Disable multiprocessing (for debugging)",
    )
    arg_parser.add_argument(
        "-r",
        "--reattach-mode",
        choices=["mlp", "bucket"],
        help="Mode to use when reattaching pruned subtrees",
    )

    # Model parameters
    arg_parser.add_argument(
        "-Ne", "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    arg_parser.add_argument(
        "-pb",
        "--prediction_batch_size",
        type=int,
        default=10,
        help="Batch size for path -> graph predictions",
    )

    # Checkpoint flags
    arg_parser.add_argument(
        "-p",
        "--load_perturbed_graphs",
        action="store_true",
        help="Load perturbed graphs from output directory",
    )
    arg_parser.add_argument(
        "-g",
        "--load_graph2vec",
        action="store_true",
        help="Load graph2vec model from output directory",
    )
    arg_parser.add_argument(
        "-m",
        "--load_model",
        action="store_true",
        help="Load parameters from output directory",
    )
    return arg_parser.parse_args()


def save_dot(dot_graph: Digraph, file_path: Path, pdf=False) -> None:
    file_path = file_path.with_suffix(".dot")
    file_path.parent.mkdir(exist_ok=True, parents=True)
    dot_graph.save(file_path)
    if pdf:
        dot_graph.render(file_path, format="pdf")
