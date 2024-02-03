import argparse
import random
from pathlib import Path

from tqdm import tqdm

from source.algorithm import TreeShaker, GraphWrapper
from utility import save_dot


def main(args):
    random.seed(123)
    input_paths = list(args.input_dir.glob('*.json'))
    random.shuffle(input_paths)
    input_paths = input_paths[:10]
    graphs = [GraphWrapper.load_file(path) for path in tqdm(input_paths, desc='Loading graphs')]
    tree_shaker = TreeShaker(epsilon=1, delta=0.5, alpha=1)
    tree_shaker.perturb_graphs(graphs)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input_dir', type=Path, help='Path to input graph directory')
    arg_parser.add_argument('-o', '--output_dir', type=Path, help='Path to output graph directory')
    main(arg_parser.parse_args())
