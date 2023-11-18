import argparse
import os
import shutil
from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import pandas as pd
from icecream import ic
from tqdm import tqdm

from algorithm import perturb_graph, etmf_stats
from graphson import Graph
from utility import save_dot, get_stats, count_disconnected_nodes


def evaluate(input_path: Path, output_dir: Path,
             num_samples: int, epsilons: list[float]
             ) -> pd.DataFrame:
    results: list[pd.Series] = []

    configurations = [(input_path, output_dir, num_samples, epsilon) for epsilon in epsilons]
    results = [evaluate_for_epsilon(*configuration)
               for configuration in configurations ]
    # with Pool(processes=8) as pool:
    #     results = pool.starmap(evaluate_for_epsilon, configurations)
    #     pool.close()
    #     pool.join()

    return pd.concat(results, axis=1).T

def evaluate_for_epsilon(input_path: Path, output_dir: Path, 
                         num_samples: int, epsilon_1: float
                         ) -> pd.Series:
    metrics: dict[str, Callable[[Graph,Graph],float]] = {
        '#edges': lambda _, output_graph: len(output_graph.edges),
        '#edges kept': lambda input_graph, output_graph: len(set(input_graph.edges).intersection(output_graph.edges)),
        '#disconnected nodes': lambda _, output_graph: count_disconnected_nodes(output_graph)
    }
    metric_data = { key: [] for key, _ in metrics.items() }
    for i in tqdm(range(num_samples)):
        input_graph = Graph.load_file(input_path)
        output_graph = perturb_graph(input_graph, epsilon_1=epsilon_1)

        for metric, func in metrics.items():
            metric_data[metric].append(func(input_graph, output_graph))

        save_dot(output_graph.to_dot(), 
                 output_dir / f'{input_path.stem}_epsilon-{epsilon_1}_{i}',
                 dot=False, pdf=True)


    series_dict = OrderedDict()
    series_dict['epsilon'] = epsilon_1
    for metric, data in metric_data.items():
        for key, value in get_stats(metric, data).items():
            series_dict[key] = value
    return pd.Series(data=series_dict, index=series_dict.keys())


def main(args: dict) -> None:
    original_graph = Graph.load_file(args.input_path)
    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    save_dot(original_graph.to_dot(), 
             Path('input') / args.input_path.stem, 
             pdf=True)

    evaluate(input_path     = args.input_path,
             output_dir     = args.output_dir,
             num_samples    = 1,
             epsilons       = [0.5, 1, 3, 5, 10, 15, 20, 30]
            ).sort_values('epsilon').to_csv('output.csv', index=False)
    
    ic(etmf_stats.stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Graph perturber')
    parser.add_argument('-i', '--input_path', type=Path, 
                        required=True, help='Path to input graph')
    parser.add_argument('-o', '--output_dir', type=Path,
                        required=True, help='Path to output graph directory')
    parser.add_argument('-n', '--num-graphs', type=int, 
                        help='Number of perturbed graphs to generate')
    parser.add_argument('-e', '--epsilon', type=float)

    main(parser.parse_args())