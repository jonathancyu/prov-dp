import argparse
import inspect
import pickle
import random
from pathlib import Path

from tqdm.auto import tqdm

from source.algorithm import GraphProcessor
from utility import save_dot


def main(args):
    # Apply graph limit
    input_paths = list(args.input_dir.glob('*.json'))
    if args.num_graphs is not None:
        random.seed(args.num_graphs)
        input_paths = random.sample(input_paths, args.num_graphs)
        args.output_dir = args.output_dir.with_stem(f'{args.output_dir.stem}_N={args.num_graphs}')

    # Map args to GraphProcessor constructor
    parameters = inspect.signature(GraphProcessor.__init__).parameters
    processor_args = {}
    for arg, value in vars(args).items():
        if arg not in parameters:
            print(f'Warning: {arg} is not a valid GraphProcessor argument')
            continue
        processor_args[arg] = value

    # Run graph processor
    tree_shaker = GraphProcessor(**processor_args)
    trees = tree_shaker.preprocess_graphs(input_paths)
    for tree in trees:
        tree.to_json()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input_dir', type=Path, help='Path to input graph directory')

    # GraphProcessor arguments
    arg_parser.add_argument('-N', '--num_graphs', type=int, default=None,
                            help='Limit the number of graphs to process')
    arg_parser.add_argument('-o', '--output_dir', type=Path, help='Path to output graph directory')

    main(arg_parser.parse_args())
