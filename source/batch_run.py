import argparse
import contextlib
from copy import deepcopy
from pathlib import Path

from source.utility import run_processor


def batch_run(args):
    args.delta = 1.0  # Allocate all privacy budget to pruning
    for epsilon_1 in [15, 20, 25, 30, 35, 40, 45]:
        for alpha in [0.1, 0.5, 0.9]:
            for beta in [0.1, 0.5, 0.9]:
                for gamma in [0.1, 0.5, 0.9]:
                    current_args = deepcopy(args)
                    print(f'(0) beginning epsilon_1={epsilon_1}, alpha={alpha}, beta={beta}, gamma={gamma}')
                    current_args.epsilon = epsilon_1
                    current_args.alpha = alpha
                    current_args.beta = beta
                    current_args.gamma = gamma
                    run_processor(current_args)
                    print()
                    print()


def main(args):
    with open("../output/output.txt", "w") as f:
        with contextlib.redirect_stdout(f):
            batch_run(args)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input_dir', type=Path,
                            help='Path to input graph directory')

    # GraphProcessor arguments
    arg_parser.add_argument('-N', '--num_graphs', type=int, default=None,
                            help='Limit the number of graphs to process')
    arg_parser.add_argument('-o', '--output_dir', type=Path,
                            help='Path to output graph directory')

    # Differential privacy parameters
    arg_parser.add_argument('-e1', '--epsilon1', type=float, default=1,
                            help='Differential privacy budget for pruning')
    arg_parser.add_argument('-e2', '--epsilon2', type=float, default=1,
                            help='Differential privacy budget for reattaching')

    arg_parser.add_argument('-a', '--alpha', type=float, default=1,
                            help='Weight of subtree size on pruning probability')
    arg_parser.add_argument('-b', '--beta', type=float, default=1,
                            help='Weight of subtree height on pruning probability')
    arg_parser.add_argument('-c', '--gamma', type=float, default=1,
                            help='Weight of subtree depth on pruning probability')

    # Algorithm configuration
    arg_parser.add_argument('-s', '--single_threaded', action='store_true',
                            help='Disable multiprocessing (for debugging)')

    # Model parameters
    arg_parser.add_argument('-Ne', '--num_epochs', type=int, default=10,
                            help='Number of training epochs')
    arg_parser.add_argument('-pb', '--prediction_batch_size', type=int, default=10,
                            help='Batch size for path -> graph predictions')

    # Checkpoint flags
    arg_parser.add_argument('-p', '--load_perturbed_graphs', action='store_true',
                            help='Load perturbed graphs from output directory')
    arg_parser.add_argument('-g', '--load_graph2vec', action='store_true',
                            help='Load graph2vec model from output directory')
    arg_parser.add_argument('-m', '--load_model', action='store_true',
                            help='Load parameters from output directory')

    main(arg_parser.parse_args())
