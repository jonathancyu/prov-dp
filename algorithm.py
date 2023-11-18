import warnings
import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from icecream import ic 
from graphson import Node, NodeType, Edge, EdgeType, Graph
from utility import group_by_lambda, create_edge, uniform_generator, save_dot

def perturb_graph(graph: Graph, epsilon_1: float) -> Graph:
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

class Stats:
    stats: dict[str, Counter[EdgeType,int]] = {}
    def __init__(self, stats: list[str]):
        for stat in stats:
            self.stats[stat] = Counter()

    def increment(self, edge_type: EdgeType, stat: str) -> None:
        self.stats[stat].update([str(edge_type)])

PROCESSED = 'total processed'
SELF_REFERRING = 'self referring'
TIME_FILTERED = 'time filtered'
etmf_stats = Stats([PROCESSED, SELF_REFERRING, TIME_FILTERED])


def extended_top_m_filter(src_nodes: list[Node], dst_nodes: list[Node], 
                          existing_edges: list[Edge], 
                          edge_type: EdgeType,
                          epsilon_1: float, epsilon_2: float=1
                          ) -> list[Edge]:
    """Compute constants"""
    n_s, n_d = len(src_nodes), len(dst_nodes)
    m = len(existing_edges)
    m_perturbed = m + int(np.round(np.random.laplace(0, 1.0/epsilon_2)))
    if m_perturbed <= 0:
        return []
    
    num_possible_edges = n_s*n_d
    epsilon_t = math.log(
        (num_possible_edges/m) - 1
        )
    if epsilon_1 < epsilon_t:
        theta = epsilon_t / (2*epsilon_1)
    else:
        theta = math.log(
            (num_possible_edges / (2*m_perturbed))
            + (math.exp(epsilon_1)-1)/2
        ) / epsilon_1

    """Execute Extended Top-m Filter"""    
    new_edges: set[Edge] = set()
    for edge in existing_edges:
        weight = 1 + np.random.laplace(0, 1.0/epsilon_1)
        if weight > theta:
            new_edges.add(edge)
            
    uniform_time = uniform_generator(existing_edges)
    while len(new_edges) < m_perturbed:
        src_node: Node = np.random.choice(src_nodes)
        dst_node: Node = np.random.choice(dst_nodes)
        etmf_stats.increment(edge_type, PROCESSED)

        # Provenance-specific constraints
        # This filtering DEFINITELY affects our selection of theta
        if src_node == dst_node:
            etmf_stats.increment(edge_type, SELF_REFERRING)
            continue
        if src_node.type == NodeType.PROCESS_LET \
            and dst_node.type == NodeType.PROCESS_LET \
            and src_node.time >= dst_node.time:
            etmf_stats.increment(edge_type, SELF_REFERRING)
            continue
        
        new_edge = create_edge(src_node, dst_node, edge_type.optype, uniform_time)
        if new_edge not in new_edges: # TODO: Also filter self-referential edges?
            new_edges.add(new_edge)
    return new_edges