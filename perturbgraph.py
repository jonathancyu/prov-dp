import argparse
import json
import numpy as np
import math
import warnings
from enum import Enum
from pydantic import BaseModel, Field, root_validator, Extra
from collections import defaultdict
from typing import Callable, TypeVar
from icecream import ic

warnings.filterwarnings("ignore", category=DeprecationWarning)


class GraphsonObject(BaseModel):
    @root_validator(pre=True)
    def extract_values(cls, values):
        result = {}
        for key, value in values.items():
            if key.startswith('_') or not isinstance(value, dict):
                result[key] = value
            else:
                result[key] = value['value']
        return result
    class Config:
        extra = 'allow'

class NodeType(Enum):
    ProcessLet = "ProcessNode"
    File = "FileNode"
    IpChannel = "SocketChannelNode"
class Node(GraphsonObject):
    id: int = Field(..., alias='_id')
    type: NodeType = Field(..., alias='TYPE')

    def __repr__(self):
        return f'{id}: {self.type}'

edge_id_sequence = 5000
class Edge(GraphsonObject):
    id: int = Field(alias='_id') # TODO: May be edge ID conflicts when we export the graph
    srcid: int = Field(..., alias='_inV')
    dstid: int = Field(..., alias='_outV')
    optype: str = Field(..., alias='OPTYPE')

    def __repr__(self):
        return f'{self.srcid}-{self.optype}-{self.dstid}'
    
    def of(srcid: int, dstid: int, optype: str, id:int=edge_id_sequence):
        global edge_id_sequence
        if id == edge_id_sequence:
            edge_id_sequence += 1
        return Edge(_id=id, _inV=srcid, _outV=dstid, OPTYPE=optype)

class Graph(BaseModel):
    nodes: list[Node] = Field(..., alias='vertices')
    edges: list[Edge] = Field(..., alias='edges')
    
T = TypeVar('T')
def group_by_lambda(
        objects: list[GraphsonObject], 
        get_attribute: Callable[[GraphsonObject], T]
        ) -> dict[T, list[GraphsonObject]]:
    grouped = {}
    for object in objects:
        key = get_attribute(object)
        if not key in grouped:
            grouped[key] = []
        grouped[key].append(object)

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
    node_lookup:        dict[int, Node]
    edge_lookup:        dict[list[int, int], Edge]

    node_groups:        dict[NodeType, list[Node]]
    edge_type_groups:   dict[str, list[Node]]
    edge_type_groups:   dict[tuple[NodeType, NodeType], list[Node]]


    def __init__(self, path_to_json: str):
        with open(path_to_json) as input_file:
            input_json = json.load(input_file)
            self.graph = Graph(**input_json)
        ic(len(self.graph.nodes))
        ic(len(self.graph.edges))
        self.node_lookup = {node.id: node for node in self.graph.nodes}
        self.edge_lookup = {(edge.srcid, edge.dstid): edge for edge in self.graph.edges}

        self.node_groups = group_by_lambda(self.graph.nodes, lambda node: node.type)
        self.edge_type_groups = group_by_lambda(self.graph.edges, lambda edge: edge.optype)
        self.edge_node_type_groups = group_by_lambda(self.graph.edges, lambda edge: node_type_tuple(edge, self.node_lookup))
        ic(self.edge_node_type_groups.keys())

    def process(self):
        process_to_process_edges = self.edge_node_type_groups[(NodeType.ProcessLet, NodeType.ProcessLet)]
        ic(len(process_to_process_edges))
        process_to_process_edges_perturbed = perturb(self.graph.nodes, process_to_process_edges, 'Start_Processlet')
        ic(process_to_process_edges_perturbed)

    
def main(args: dict) -> None:
    processor = GraphProcessor(args.input)

    processor.process()


    
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
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input graph')
    parser.add_argument('-o', '--output', type=str, help='Path to output graph')
    parser.add_argument('-n', '--num-graphs', type=int, help='Number of perturbed graphs to generate')
    parser.add_argument('-e', '--epsilon', type=float)

    main(parser.parse_args())


