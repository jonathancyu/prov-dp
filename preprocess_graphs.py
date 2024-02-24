import argparse
from pathlib import Path

from tqdm import tqdm

from source.algorithm import GraphProcessor


def main(args):
    input_paths = list(args.input_dir.glob('*.json'))
    tree_graph_dir = args.output_dir / 'preprocessed_trees'
    tree_graph_dir.mkdir(exist_ok=True, parents=True)

    # Run graph processor
    tree_shaker = GraphProcessor()
    trees = tree_shaker.preprocess_graphs(input_paths)
    for path, tree in tqdm(zip(input_paths, trees), total=len(trees), desc='Exporting to json'):
        file_name = path.name
        output_path = tree_graph_dir / file_name
        with open(output_path, 'w') as f:
            f.write(tree.to_json())


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input_dir', type=Path, help='Path to input graph directory')

    # GraphProcessor arguments
    arg_parser.add_argument('-N', '--num_graphs', type=int, default=None,
                            help='Limit the number of graphs to process')
    arg_parser.add_argument('-o', '--output_dir', type=Path, help='Path to output graph directory')

    main(arg_parser.parse_args())
