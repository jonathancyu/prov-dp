import argparse
from pathlib import Path

from source.algorithm import GraphProcessor


def main(args):
    input_paths = list(args.input_dir.glob('*.json'))
    tree_shaker = GraphProcessor(epsilon=1, delta=0.5, alpha=1)
    tree_shaker.perturb_graphs(input_paths)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input_dir', type=Path, help='Path to input graph directory')
    arg_parser.add_argument('-o', '--output_dir', type=Path, help='Path to output graph directory')
    main(arg_parser.parse_args())
