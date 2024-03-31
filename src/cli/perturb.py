import gc
import inspect
import pickle
import random

import numpy as np
from tqdm import tqdm

from src import GraphProcessor, Tree
from src.cli.utility import parse_args, save_dot


def run_processor(args):
    input_paths = list(args.input_dir.rglob('nd*.json'))
    # Apply graph limit
    if args.num_graphs is not None:
        random.seed(args.num_graphs)
        input_paths = random.sample(input_paths, args.num_graphs)
        args.output_dir = args.output_dir.with_stem(
            f'{args.output_dir.stem}_N={args.num_graphs}')
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
    with open(args.output_dir / 'stats.csv', 'a') as f:
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
        print(','.join(header))  # TODO this is way messy
        f.write('\n')
    del graph_processor
    del perturbed_graphs
    gc.collect()


def to_processor_args(args):
    # Map args to GraphProcessor constructor
    parameters = inspect.signature(GraphProcessor.__init__).parameters
    processor_args = {}
    for arg, value in vars(args).items():
        if arg not in parameters:
            continue
        processor_args[arg] = value

    return processor_args


if __name__ == '__main__':
    run_processor(parse_args())
