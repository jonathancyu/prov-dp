import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from icecream import ic 
from graphviz import Digraph
from tqdm import tqdm

from graphson import Node, Edge, EdgeType, Graph
from utility import group_by_lambda, extended_top_m_filter, etmf_stats

warnings.filterwarnings('ignore', category=DeprecationWarning)


def perturb_graph(graph: Graph, epsilon_1=1) -> Graph:
    node_groups = group_by_lambda(graph.nodes, lambda node: node.type)
    edge_type_groups = group_by_lambda(graph.edges, lambda edge: EdgeType(edge, graph._node_lookup))

    new_edges: list[Edge] = []
    for edge_type, edges in edge_type_groups.items():
        perturbed_edges = extended_top_m_filter(
            src_nodes=node_groups[edge_type.src_type],
            dst_nodes=node_groups[edge_type.dst_type],
            existing_edges=edges,
            edge_type=edge_type,
            epsilon_1=epsilon_1
        )
        new_edges.extend(perturbed_edges)

    return Graph(vertices=graph.nodes, edges=new_edges)

def save_dot(dot_graph: Digraph, folder_name: str, file_path: Path, pdf: bool=False) -> None:
    output_path = (Path(folder_name) / file_path.stem).with_suffix('.dot')
    dot_graph.save(output_path)
    if pdf:
        dot_graph.render(output_path, format='pdf')

    
def main(args: dict) -> None:
    input_graph = Graph.load_file(args.input_path)
    save_dot(input_graph.to_dot(), 'input', args.input_path, pdf=True)

    results: list[pd.Series] = []
    N = 100
    epsilons = [0.5, 1, 5, 10]
    for epsilon_1 in epsilons:
        result: pd.Series = evaluate(N, input_graph, epsilon_1)
        ic(result)
        results.append(result)

    df = pd.concat(results, axis=1).T
    print(df)
    df.to_csv('output.csv', index=False)

def get_stats(stat: str, data: list[int]) -> dict:
    result = {
        'avg': np.average(data),
        'stdev': np.std(data),
        'min': min(data),
        'max': max(data)
    }
    return { f'{stat} {key}': value for key, value in result.items() }

def evaluate(num_samples: int, input_graph: Graph, epsilon_1: float) -> pd.Series:
    num_edges = []
    num_edges_kept = []
    num_disconnected_nodes = []
    for i in tqdm(range(num_samples)):
        output_graph = perturb_graph(input_graph, epsilon_1=epsilon_1)
        included_nodes: set[Node] = set()
        for edge in output_graph.edges:
            included_nodes.add(output_graph.get_node(edge.src_id))
            included_nodes.add(output_graph.get_node(edge.dst_id))
        num_disconnected_nodes.append(len(set(output_graph.nodes) - included_nodes))
        num_edges_kept.append(len(set(input_graph.edges).intersection(output_graph.edges)))
        num_edges.append(len(output_graph.edges))
    data = {
        'epsilon': epsilon_1,
        **get_stats('#edges', num_edges),
        **get_stats('#edges kept', num_edges_kept),
        **get_stats('#disconnected nodes', num_disconnected_nodes)
    }
    return pd.Series(data=data, index=data.keys())

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Graph perturber')
    parser.add_argument('-i', '--input_path', type=Path, 
                        required=True, help='Path to input graph')
    parser.add_argument('-n', '--num-graphs', type=int, 
                        help='Number of perturbed graphs to generate')
    parser.add_argument('-e', '--epsilon', type=float)

    main(parser.parse_args())