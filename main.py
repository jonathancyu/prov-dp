import argparse
import contextlib
import gc
import inspect
import pickle
import random
import numpy as np
from copy import deepcopy
from pathlib import Path

from tqdm import tqdm

from source import Tree
from source.algorithm import GraphProcessor
from utility import save_dot


def to_processor_args(args):
    # Map args to GraphProcessor constructor
    parameters = inspect.signature(GraphProcessor.__init__).parameters
    processor_args = {}
    for arg, value in vars(args).items():
        if arg not in parameters:
            continue
        processor_args[arg] = value

    return processor_args


def run_processor(args):
    input_paths = list(args.input_dir.glob('*.json'))
    # Apply graph limit
    if args.num_graphs is not None:
        random.seed(args.num_graphs)
        input_paths = random.sample(input_paths, args.num_graphs)
        args.output_dir = args.output_dir.with_stem(f'{args.output_dir.stem}_N={args.num_graphs}')
    args.output_dir = args.output_dir.with_stem(f'{args.output_dir.stem}'
                                                f'_e1={args.epsilon1}'
                                                f'_e2={args.epsilon2}'
                                                f'_a={args.alpha}'
                                                f'_b={args.beta}'
                                                f'_c={args.gamma}')

    # Run graph processor
    graph_processor = GraphProcessor(**to_processor_args(args))
    perturbed_graphs: list[Tree] = graph_processor.perturb_graphs(input_paths)

    # Save final graph objects
    with open(args.output_dir / 'perturbed_graphs.pkl', 'wb') as f:
        pickle.dump(perturbed_graphs, f)

    # Save dot files
    for graph in tqdm(perturbed_graphs, desc='Saving graphs'):
        base_file_name = f'nd_{graph.graph_id}_processletevent'
        file_path = args.output_dir / base_file_name / f'{base_file_name}.json'
        save_dot(graph.to_dot(), file_path)

        with open(file_path, 'w') as f:
            f.write(graph.to_json())

    # Clean up for the next run
    with open('stats.csv', 'a') as f:
        header = ['N', 'epsilon1', 'epsilon2', 'alpha', 'beta', 'gamma']
        f.write(f'{args.num_graphs},{args.epsilon1},{args.epsilon2},{args.alpha},{args.beta},{args.gamma}')
        for key, value in graph_processor.stats.items():
            mean, std, min_val, max_val = 'x', 'x', 'x', 'x'
            if (len(value)) > 0:
                mean = np.mean(value)
                std = np.std(value)
                min_val = np.min(value)
                max_val = np.max(value)
            f.write(f'{mean},{std},{min_val},{max_val},')
            header.extend([f'{key}_{label}' for label in ['mean', 'std', 'min', 'max']])
        print(','.join(header))
        f.write('\n')
    del graph_processor
    del perturbed_graphs
    gc.collect()


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
    run_processor(args)
    # with open("output.txt", "w") as f:
    #     with contextlib.redirect_stdout(f):
    #         batch_run(args)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input_dir', type=Path, help='Path to input graph directory')

    # GraphProcessor arguments
    arg_parser.add_argument('-N', '--num_graphs', type=int, default=None,
                            help='Limit the number of graphs to process')
    arg_parser.add_argument('-o', '--output_dir', type=Path, help='Path to output graph directory')

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
