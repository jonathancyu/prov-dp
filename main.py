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

from algorithm import TreeShaker, GraphWrapper, count_disconnected_nodes
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
        self.epsilon_1 = epsilon_1
        self.epsilon_2 = epsilon_2
        self.alpha = alpha

def evaluate(input_path: Path, output_path: Path,
             graph_name: str,
             num_samples: int,
             parameter_list: list[Parameters]
             ) -> pd.DataFrame:

    input_file_path = (input_path / graph_name / graph_name).with_suffix('.json')

    configurations = [
        (input_file_path, output_path, num_samples, parameters.epsilon_1, parameters.epsilon_2, parameters.alpha)
        for parameters in parameter_list
    ]
    # results = [evaluate_for_epsilon(*configuration)
    #            for configuration in configurations]
    with Pool(processes=8) as pool:
        results = pool.starmap(evaluate_for_epsilon, configurations)

    return pd.concat(results, axis=1).T


def evaluate_for_epsilon(
        input_path: Path, output_path: Path,
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
    for metric, data in metric_data.items():
        for key, value in get_stats(metric, data).items():
            series_dict[key] = value

    return pd.Series(data=series_dict, index=series_dict.keys())


# noinspection PyUnresolvedReferences
def main(args: argparse.Namespace) -> None:
    input_path = Path(args.input_directory)
    output_path = Path(args.output_directory)

    parameter_combinations = itertools.product(
        args.epsilon_1_values,
        args.epsilon_2_values,
        args.alpha_values
    )
    result: pd.DataFrame = evaluate(
        input_path=input_path,
        output_path=output_path,
        graph_name=args.graph_name,
        num_samples=args.num_samples,
        parameter_list=[Parameters(*combination)
                        for combination in parameter_combinations]
    )
    result = result.sort_values(['epsilon_1', 'epsilon_2'])
    output_file_path = output_path / args.graph_name / 'stats.csv'
    output_file_path.parent.mkdir(exist_ok=True, parents=True)
    result.to_csv(output_file_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Provenance Graph')
    parser.add_argument('-c', '--config', type=Path, help='Path to config yaml file')
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
    main(argparse.Namespace(**config))
