import math
import warnings
import argparse
import json
from typing import Callable

import numpy as np
from icecream import ic 

from graphson import Node, NodeType, Edge, Graph, GraphsonObject

warnings.filterwarnings('ignore', category=DeprecationWarning)



def group_by_lambda[T](objects: list[GraphsonObject], 
                       get_attribute: Callable[[GraphsonObject], T]
                       ) -> dict[T, list[GraphsonObject]]:
    grouped = {}
    for obj in objects:
        key = get_attribute(obj)
        if not key in grouped:
            grouped[key] = []
        grouped[key].append(obj)

    return grouped

def node_type_tuple(
        edge: Edge, 
        node_lookup: dict[int, Node]
        ) -> tuple[NodeType, NodeType]:
    src_node = node_lookup[edge.srcid]
    dst_node = node_lookup[edge.dstid]
    return (src_node.type, dst_node.type)

class GraphProcessor:
    graph: Graph
    node_groups:        dict[NodeType, list[Node]]
    edge_type_groups:   dict[str, list[Node]]
    edge_type_groups:   dict[tuple[NodeType, NodeType], list[Node]]


    def __init__(self, graph: Graph):
        self.graph = graph
        self.node_groups = group_by_lambda(graph.nodes, lambda node: node.type)
        self.edge_type_groups = group_by_lambda(graph.edges, lambda edge: edge.optype)
        self.edge_node_type_groups = group_by_lambda(graph.edges, lambda edge: node_type_tuple(edge, graph._node_lookup))
        ic(self.edge_node_type_groups.keys())


    def process(self) -> None:
        process_to_process_edges = self.edge_node_type_groups[
            (NodeType.PROCESS_LET, NodeType.PROCESS_LET)
            ]
        ic(len(process_to_process_edges))
        process_to_process_edges_perturbed = perturb(self.graph.nodes, process_to_process_edges, 'Start_Processlet')
        ic(process_to_process_edges_perturbed)

    
def main(args: dict) -> None:
    input_graph = Graph.load(args.input)
    input_graph.to_dot(args.output, pdf=True)
    processor = GraphProcessor(input_graph)


def perturb(nodes: list[Node], edges: list[Edge], optype: str) -> (list[Node], list[Edge]):
    # https://web.archive.org/web/20170921192428id_/https://hal.inria.fr/hal-01179528/document
    epsilon_1 = 0.5
    epsilon_2 = 1
    n: int = len(nodes)
    m: int = len(edges)

    m_perturbed: int = m + int(np.round(np.random.laplace(0, 1.0/epsilon_2)))
    epsilon_t: float = math.log( ((n*(n-1))/(2*m_perturbed)) - 1)

    if epsilon_1 < epsilon_t:
        theta = (1/(2*epsilon_1)) * epsilon_t
    else:
        theta = (1/epsilon_1) \
              * math.log(
                  (n*(n-1)/(4*m_perturbed)) + (1/2)*(math.exp(epsilon_1)-1)
                )
    n_1 = 0

    new_edge_tuples: list(tuple(int, int)) = []
    for edge in edges:
        weight = 1 + np.random.laplace(0, 1.0/epsilon_1)
        if weight > theta:
            new_edge_tuples.append((edge.srcid, edge.dstid))
            n_1 += 1
    
    while n_1 < (m_perturbed-1):
    
        # TODO: What if we sample from two different lists here? This will definitely change the calculations
        src = np.random.choice(nodes)
        dst = np.random.choice(nodes)
        edge = (src.id, dst.id)
        if edge not in new_edge_tuples:
            new_edge_tuples.append(edge)
            n_1 += 1
    
    return [
        Edge.of(srcid = edge_tuple[0], dstid = edge_tuple[1], optype=optype)
        for edge_tuple in new_edge_tuples
    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Graph perturber')
    parser.add_argument('-i', '--input', type=str, 
                        required=True, help='Path to input graph')
    parser.add_argument('-o', '--output', type=str, 
                        required=True, help='Path to output graph')
    parser.add_argument('-n', '--num-graphs', type=int, 
                        help='Number of perturbed graphs to generate')
    parser.add_argument('-e', '--epsilon', type=float)

    main(parser.parse_args())