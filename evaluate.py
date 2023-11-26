import argparse
import itertools
import os
import shutil
from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml
from icecream import ic
from tqdm import tqdm

from algorithm import TreeShaker, GraphWrapper, count_disconnected_nodes, PRUNED_AT_DEPTH
from graphson import Graph
from utility import save_dot, get_stats, get_edge_id

class Parameters:
    epsilon_1: float
    epsilon_2: float
    alpha: float
    def __init__(self,
                 epsilon_1: float,
                 epsilon_2: float,
                 alpha: float):
        self.epsilon_1 = float(epsilon_1)
        self.epsilon_2 = float(epsilon_2)
        self.alpha = float(alpha)

def evaluate(input_path: Path, output_path: Path,
             graph_name: str,
             num_samples: int,
             parameter_list: list[Parameters],
             parallel: bool
             ) -> pd.DataFrame:

    input_file_path = (input_path / graph_name / graph_name).with_suffix('.json')

    configurations = [
        (input_file_path, output_path, num_samples, parameters.epsilon_1, parameters.epsilon_2, parameters.alpha)
        for parameters in parameter_list
    ]
    if parallel:
        with Pool(processes=8) as pool:
            results = pool.starmap(evaluate_for_epsilon, configurations)
    else:
        results = [evaluate_for_epsilon(*configuration)
                   for configuration in configurations]

    return pd.concat(results, axis=1).T


def evaluate_for_epsilon(input_path: Path, output_path: Path,
                         num_samples: int,
                         epsilon_1: float, epsilon_2: float, alpha: float
                         ) -> pd.Series:
    graph_name = input_path.stem
    edge_id = get_edge_id(graph_name)
    graph_output_dir = output_path / graph_name / f'epsilon_{epsilon_1}_{epsilon_2}-alpha_{alpha}'
    if os.path.isdir(graph_output_dir):
        shutil.rmtree(graph_output_dir)

    metrics: dict[str, Callable[[Graph, Graph], float]] = {
        '#edges': lambda _, output_graph: len(output_graph.edges),
        '#edges kept': lambda input_graph, output_graph: len(set(input_graph.edges).intersection(output_graph.edges)),
        '#disconnected nodes': lambda _, output_graph: count_disconnected_nodes(GraphWrapper(output_graph))
    }
    metric_data = {key: [] for key, _ in metrics.items()}

    # processor = ExtendedTopMFilter()
    processor = TreeShaker()
    for i in tqdm(range(num_samples), desc=f'({epsilon_1},{epsilon_2},{alpha})'):
        input_graph = Graph.load_file(input_path)
        # output_graph: Graph = processor.perturb_graph(input_graph, epsilon_1, epsilon_2)
        output_graph: Graph = processor.perturb_graph(input_graph, edge_id, epsilon_1, epsilon_2, alpha)

        for metric, func in metrics.items():
            metric_data[metric].append(func(input_graph, output_graph))

        save_dot(output_graph.to_dot(),
                 graph_output_dir / f'{graph_name}_{i}',
                 dot=True, pdf=True)

    with open(graph_output_dir / 'pruning-stats.txt', 'w', encoding='utf-8') as f:
        stats_str = processor.get_stats_str()
        f.write(stats_str)

    series_dict = OrderedDict()
    series_dict['epsilon_1'] = epsilon_1
    series_dict['epsilon_2'] = epsilon_2
    series_dict['alpha'] = alpha
    for stat, count in processor.stats.items():
        if stat.startswith(PRUNED_AT_DEPTH):
            series_dict[stat] = count
    for metric, data in metric_data.items():
        for key, value in get_stats(metric, data).items():
            series_dict[key] = value

    return pd.Series(data=series_dict, index=series_dict.keys())


# noinspection PyUnresolvedReferences
def main(input_directory: str,
         output_directory: str,
         graph_name: str,
         epsilon_1_values: list[float],
         epsilon_2_values: list[float],
         alpha_values: list[float],
         num_samples: int,
         parallel: bool) -> pd.DataFrame:
    input_path = Path(input_directory)
    output_path = Path(output_directory)

    parameter_combinations = itertools.product(
        epsilon_1_values,
        epsilon_2_values,
        alpha_values
    )
    result: pd.DataFrame = evaluate(
        input_path=input_path,
        output_path=output_path,
        graph_name=graph_name,
        parameter_list=[Parameters(*combination)
                        for combination in parameter_combinations],
        num_samples=num_samples,
        parallel=parallel
    )
    return result.sort_values(['epsilon_1', 'epsilon_2'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Provenance Graph')
    parser.add_argument('-c', '--config', type=Path, help='Path to config yaml file')
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
    result = main(**config)
    output_file_path = Path(config['output_directory']) / config['graph_name'] / 'stats.csv'
    output_file_path.parent.mkdir(exist_ok=True, parents=True)
    result.to_csv(output_file_path, index=False)
