import argparse
import inspect
import pickle
import random
from pathlib import Path

from tqdm import tqdm

from source import Tree
from source.algorithm import GraphProcessor
from utility import save_dot


def main(args):
    # Apply graph limit
    input_paths = list(args.input_dir.glob('*.json'))
    if args.num_graphs is not None:
        random.seed(args.num_graphs)
        input_paths = random.sample(input_paths, args.num_graphs)
        args.output_dir = args.output_dir.with_stem(f'{args.output_dir.stem}_N={args.num_graphs}')
    args.output_dir = args.output_dir.with_stem(f'{args.output_dir.stem}_a={args.alpha}_d={args.delta}_e={args.epsilon}')

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
    perturbed_graphs: list[Tree] = tree_shaker.perturb_graphs(input_paths)

    # Save final graph objects
    with open(args.output_dir / 'perturbed_graphs.pkl', 'wb') as f:
        pickle.dump(perturbed_graphs, f)

    # Save dot files
    for graph in tqdm(perturbed_graphs, desc='Saving graphs'):
        base_file_name = f'nd_{graph.source_edge_ref_id}_processletevent'
        file_path = args.output_dir / base_file_name / f'{base_file_name}.json'
        save_dot(graph.to_dot(), file_path)

        with open(file_path, 'w') as f:
            f.write(graph.to_json())


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input_dir', type=Path, help='Path to input graph directory')

    # GraphProcessor arguments
    arg_parser.add_argument('-N', '--num_graphs', type=int, default=None,
                            help='Limit the number of graphs to process')
    arg_parser.add_argument('-o', '--output_dir', type=Path, help='Path to output graph directory')

    # Differential privacy parameters
    arg_parser.add_argument('-e', '--epsilon', type=float, default=1,
                            help='Differential privacy budget')
    arg_parser.add_argument('-d', '--delta', type=float, default=0.5,
                            help='Portion of privacy budget to allocate to pruning')
    arg_parser.add_argument('-a', '--alpha', type=float, default=1,
                            help='Weight of subtree size on pruning probability (high delta, big tree -> don\'t prune')

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
    arg_parser.add_argument('-m', '--load_model', action='store_true',  # TODO: not implemented
                            help='Load parameters from output directory')

    main(arg_parser.parse_args())
