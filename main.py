import argparse
from pathlib import Path

from source.algorithm import TreeShaker, GraphWrapper
from utility import save_dot


def main(args):
    input_graphs = [GraphWrapper.load_file(args.input)]
    tree_shaker = TreeShaker(epsilon=0.1, delta=0.1, alpha=0.1)

    output_graph = tree_shaker.perturb_graphs(input_graphs)[0]


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input', type=Path, help='Path to input graph')
    arg_parser.add_argument('-o', '--output_dir', type=Path, help='Path to output graph directory')
    main(arg_parser.parse_args())