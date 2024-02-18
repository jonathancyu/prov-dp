import argparse
import pickle
import random
from pathlib import Path

from source.algorithm import GraphProcessor
from utility import save_dot


def main(args):
    input_paths = list(args.input_dir.glob('*.json'))
    if args.num_graphs is not None:
        random.seed(args.num_graphs)
        input_paths = random.sample(input_paths, args.num_graphs)

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
    arg_parser.add_argument('-s', '--single_threaded', action='store_true',
                            help='Disable multiprocessing (for debugging)')
    arg_parser.add_argument('-N', '--num_graphs', type=int, default=None,
                            help='Limit the number of graphs to process')
    arg_parser.add_argument('-pb', '--prediction_batch_size', type=int, default=10,
                            help='Batch size for path -> graph predictions')

    # Checkpoint flags
    arg_parser.add_argument('-p', '--load_perturbed_graphs', action='store_true',
                            help='Load perturbed graphs from output directory')
    arg_parser.add_argument('-g', '--load_graph2vec', action='store_true',
                            help='Load graph2vec model from output directory')
    arg_parser.add_argument('-m', '--load_model', action='store_true',  # TODO: not implemented
                            help='Load parameters from output directory')

    main(arg_parser.parse_args())
