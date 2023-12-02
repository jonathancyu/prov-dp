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
from utility import save_dot, get_stats


class Parameters:
    epsilon: float
    alpha: float

    def __init__(self,
                 epsilon: float,
                 alpha: float):
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)


def evaluate(input_path: Path, output_path: Path,
             num_samples: int,
             parameter_list: list[Parameters],
             parallel: bool
             ) -> pd.DataFrame:

    configurations = [
        (input_path, output_path, num_samples, parameters.epsilon, parameters.alpha)
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
                         epsilon: float, alpha: float
                         ) -> pd.Series:
    metrics: dict[str, Callable[[GraphWrapper, GraphWrapper], float]] = {
        '#edges input': lambda input_graph, _: len(set(input_graph.edges)),
        '#edges': lambda _, output_graph: len(output_graph.edges),
        '#edges kept': lambda input_graph, output_graph: len(set(input_graph.edges).intersection(output_graph.edges)),
        '#disconnected nodes': lambda _, output_graph: count_disconnected_nodes(output_graph)
    }
    metric_data = {key: [] for key, _ in metrics.items()}

    # Process graphs
    input_paths = list(input_path.glob('*.json'))[:10]  # TODO remove limit
    input_graphs = [GraphWrapper(input_path) for input_path in input_paths]

    graph_output_path = output_path / f'epsilon_{epsilon}-alpha_{alpha}'
    if os.path.isdir(graph_output_path):
        shutil.rmtree(graph_output_path)

    processor = TreeShaker(epsilon=epsilon, alpha=alpha)
    output_graphs: list[GraphWrapper] = input_graphs  # processor.perturb_graphs(input_graphs)
    assert len(input_graphs) == len(output_graphs)

    for i in range(len(input_graphs)):
        input_graph: GraphWrapper = input_graphs[i]
        output_graph: GraphWrapper = output_graphs[i]
        for metric, func in metrics.items():
            metric_data[metric].append(func(input_graph, output_graph))
        save_dot(output_graph.to_dot(),
                 output_path / input_graph.json_path.stem,
                 pdf=True)

    with open(output_path / 'processor-stats.txt', 'w', encoding='utf-8') as f:
        stats_str = processor.get_stats_str()
        f.write(stats_str)

    series_dict = OrderedDict()
    series_dict['epsilon'] = epsilon
    series_dict['alpha'] = alpha

    for metric, data in metric_data.items():
        for key, value in get_stats(metric, data).items():
            series_dict[key] = value

    return pd.Series(data=series_dict, index=series_dict.keys())


# noinspection PyUnresolvedReferences
def main(input_directory: str,
         output_directory: str,
         epsilon_values: list[float],
         alpha_values: list[float],
         num_samples: int,
         parallel: bool) -> pd.DataFrame:
    input_path = Path(input_directory)
    output_path = Path(output_directory)

    parameter_combinations = itertools.product(
        epsilon_values,
        alpha_values
    )
    result: pd.DataFrame = evaluate(
        input_path=input_path,
        output_path=output_path,
        parameter_list=[Parameters(*combination)
                        for combination in parameter_combinations],
        num_samples=num_samples,
        parallel=parallel
    )
    return result.sort_values(['epsilon', 'alpha'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Provenance Graph')
    parser.add_argument('-c', '--config', type=Path, help='Path to config yaml file')
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
    result_df = main(**config)
    output_file_path = Path(config['output_directory']) / 'stats.csv'
    output_file_path.parent.mkdir(exist_ok=True, parents=True)
    result_df.to_csv(output_file_path, index=False)
