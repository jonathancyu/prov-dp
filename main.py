import argparse
import pickle
import random
from pathlib import Path

from source.algorithm import GraphProcessor
from utility import save_dot


def main(args):
    random.seed(42)
    input_paths = list(args.input_dir.glob('*.json'))[:10]
    tree_shaker = GraphProcessor(epsilon=1, delta=0.5, alpha=1, args=args)
    perturbed_graphs = tree_shaker.perturb_graphs(input_paths)
    with open(args.output_dir / 'perturbed_graphs.txt', 'wb') as f:
        pickle.dump(perturbed_graphs, f)

    for graph in perturbed_graphs:
        save_dot(graph.to_dot(), args.output_dir / f'nd-{graph.source_edge_ref_id}-processletevent.dot')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input_dir', type=Path, help='Path to input graph directory')
    arg_parser.add_argument('-o', '--output_dir', type=Path, help='Path to output graph directory')
    arg_parser.add_argument('-g', '--load_graph2vec', action='store_true',
                            help='Load graph2vec model from output directory')
    arg_parser.add_argument('--single_threaded', action='store_true',
                            help='Disable multiprocessing (for debugging)')
    main(arg_parser.parse_args())
