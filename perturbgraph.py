import warnings
import argparse
from pathlib import Path

from icecream import ic 

from graphson import Node, NodeType, Edge, Graph, GraphsonObject
from utility import group_by_lambda, extended_top_m_filter

warnings.filterwarnings('ignore', category=DeprecationWarning)


class EdgeType:
    src_type: NodeType
    dst_type: NodeType
    optype: str
    def __init__(self, edge: Edge, node_lookup: dict[int, Node]):
        self.src_type = node_lookup[edge.src_id].type
        self.dst_type = node_lookup[edge.dst_id].type
        self.optype = edge.optype


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
                optype=edge_type.optype,
                epsilon_1=5
            )
            new_edges.extend(perturbed_edges)

        return Graph(vertices=graph.nodes, edges=new_edges)

    
def main(args: dict) -> None:
    input_graph = Graph.load_file(args.input_path)
    input_graph.to_dot('input-' + args.input_path.stem + '.dot', pdf=True)

    processor = GraphProcessor()
    output_graph = processor.process(input_graph)
    output_graph.to_dot(args.output_path.stem + '.dot', pdf=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Graph perturber')
    parser.add_argument('-i', '--input_path', type=Path, 
                        required=True, help='Path to input graph')
    parser.add_argument('-o', '--output_path', type=Path, 
                        required=True, help='Path to output graph')
    parser.add_argument('-n', '--num-graphs', type=int, 
                        help='Number of perturbed graphs to generate')
    parser.add_argument('-e', '--epsilon', type=float)

    main(parser.parse_args())