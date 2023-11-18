import warnings
import argparse
from pathlib import Path

import numpy as np
from icecream import ic 
from graphviz import Digraph
from tqdm import tqdm

from graphson import Node, Edge, EdgeType, Graph
from utility import group_by_lambda, extended_top_m_filter, etmf_stats

warnings.filterwarnings('ignore', category=DeprecationWarning)


class GraphProcessor:
    def process(self, graph: Graph) -> Graph:
        node_groups = group_by_lambda(graph.nodes, lambda node: node.type)
        edge_type_groups = group_by_lambda(graph.edges, lambda edge: EdgeType(edge, graph._node_lookup))

        new_edges: list[Edge] = []
        for edge_type, edges in edge_type_groups.items():
            perturbed_edges = extended_top_m_filter(
                src_nodes=node_groups[edge_type.src_type],
                dst_nodes=node_groups[edge_type.dst_type],
                existing_edges=edges,
                edge_type=edge_type,
                epsilon_1=5
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
    evaluate()

def evaluate(processor: GraphProcessor, input_graph: Graph) -> None:
    processor = GraphProcessor()
    num_excluded = []
    num_edges = []
    for i in tqdm(range(100)):
        graph = processor.process(input_graph)
        included_nodes: set[Node] = set()
        for edge in graph.edges:
            included_nodes.add(graph.get_node(edge.src_id))
            included_nodes.add(graph.get_node(edge.dst_id))
        num_excluded.append(len(set(graph.nodes) - included_nodes))
        num_edges.append(len(graph.edges))
    print('STATS')
    ic(etmf_stats.stats)
    print('EDGES')
    ic(np.average(num_edges))
    ic(np.std(num_edges))
    ic(min(num_edges), max(num_edges))
    print('NODES')
    ic(len(input_graph.nodes))
    ic(np.average(num_excluded))
    ic(np.std(num_excluded))
    ic(min(num_excluded), max(num_excluded))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Graph perturber')
    parser.add_argument('-i', '--input_path', type=Path, 
                        required=True, help='Path to input graph')
    parser.add_argument('-n', '--num-graphs', type=int, 
                        help='Number of perturbed graphs to generate')
    parser.add_argument('-e', '--epsilon', type=float)

    main(parser.parse_args())