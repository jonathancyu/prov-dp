import argparse
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from source.algorithm import TreeShaker, GraphWrapper


def main(args):
    input_paths = list(args.input_dir.glob('*.json'))[:200]
    with ProcessPoolExecutor() as executor:
        graphs = list(
            tqdm(
                executor.map(
                    GraphWrapper.load_file,
                    input_paths
                ),
                total=len(input_paths),
                desc='Loading graphs'
            ))

    tree_shaker = TreeShaker(epsilon=1, delta=0.5, alpha=1)
    tree_shaker.perturb_graphs(graphs)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input_dir', type=Path, help='Path to input graph directory')
    arg_parser.add_argument('-o', '--output_dir', type=Path, help='Path to output graph directory')
    main(arg_parser.parse_args())
