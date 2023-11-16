import argparse
import json
import numpy as np
import math
import warnings
from enum import Enum
from pydantic import BaseModel, Field, root_validator, Extra
from collections import defaultdict
from typing import Callable, TypeVar
warnings.filterwarnings("ignore", category=DeprecationWarning)
class GraphsonObject(BaseModel):
    @root_validator(pre=True)
    def extract_values(cls, values):
        return {
            key: (value if key.startswith('_') else value['value']) 
            for key, value in values.items()
            }
    class Config:
        extra = 'allow'

class NodeType(Enum):
    ProcessLet = "ProcessNode"
    File = "FileNode"
    IpChannel = "SocketChannelNode"
class Node(GraphsonObject):
    id: int = Field(..., alias='_id')
    type: NodeType = Field(..., alias='TYPE')

edge_id_counter = 0
class Edge(GraphsonObject):
    id: int = Field(..., alias='_id') # TODO: May be edge ID conflicts when we export the graph
    srcid: int = Field(..., alias='_inV')
    dstid: int = Field(..., alias='_outV')
    optype: str = Field(alias='OPTYPE')

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

    
def main(args: dict) -> None:
    with open(args.input) as input_file:
        input_graph = json.load(input_file)
    
    graph = Graph(**input_graph)
    node_lookup = {node.id: node for node in graph.nodes }
    node_groups: dict[NodeType, Node] = group_by_lambda(graph.nodes, lambda node: node.type)

    edge_lookup = {(edge.srcid, edge.dstid): edge for edge in graph.edges}
    edge_groups: dict[str, Edge] = group_by_lambda(graph.edges, lambda edge: edge.optype)


    


    
def perturb(nodes: list[Node], edges: list[Edge]) -> (list[Node], list[Edge]):
    # https://web.archive.org/web/20170921192428id_/https://hal.inria.fr/hal-01179528/document
    epsilon_1: float
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

    new_edges: list(tuple(int, int)) = []
    for edge in edges:
        weight = 1 + np.random.laplace(0, 1.0/epsilon_1)
        if weight > theta:
            new_edges.append(edge)
            n_1 += 1
    
    while n_1 < (m_perturbed-1):
        src = np.random.choice(nodes)
        dst = np.random.choice(nodes)
        edge = (src, dst)
        if edge not in new_edges:
            new_edges.append(edge)
            n_1 += 1
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Graph perturber')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input graph')
    parser.add_argument('-o', '--output', type=str, help='Path to output graph')
    parser.add_argument('-n', '--num-graphs', type=int, help='Number of perturbed graphs to generate')
    parser.add_argument('-e', '--epsilon', type=float)

    main(parser.parse_args())


