import math
from collections import Counter
from typing import Callable

import numpy as np
from icecream import ic 

from graphson import Node, NodeType, Edge, EdgeType, Graph
from utility import group_by_lambda, uniform_generator

PROCESSED = '#edges processed'
SELF_REFERRING = '#self referring edges pruned'
TIME_FILTERED = '#edges pruned by time'

class GraphProcessor:
    stats: dict[str, Counter[EdgeType,int]] = {}
    def __init__(self):
        for stat in [PROCESSED, SELF_REFERRING, TIME_FILTERED]:
            self.stats[stat] = Counter()

    def perturb_graph(self, input_graph: Graph, epsilon_1: float) -> Graph:
        node_groups = group_by_lambda(input_graph.nodes, lambda node: node.type)
        edge_type_groups = group_by_lambda(input_graph.edges, lambda edge: EdgeType(edge, input_graph._node_lookup))

        new_edges: list[Edge] = []
        for edge_type, edges in edge_type_groups.items():
            perturbed_edges = self.extended_top_m_filter(
                src_nodes=node_groups[edge_type.src_type],
                dst_nodes=node_groups[edge_type.dst_type],
                existing_edges=edges,
                edge_type=edge_type,
                epsilon_1=epsilon_1
            )
            new_edges.extend(perturbed_edges)

        return Graph(vertices=input_graph.nodes, edges=new_edges)

    def filter(self, 
               src_node: Node, dst_node: Node, 
               edge_type: EdgeType) -> bool:
        if src_node == dst_node:
            self.increment(SELF_REFERRING, edge_type)
            return True
        elif src_node.type == NodeType.PROCESS_LET \
                and dst_node.type == NodeType.PROCESS_LET \
                and src_node.time >= dst_node.time:
            self.increment(TIME_FILTERED, edge_type)
            return True
        return False


    def extended_top_m_filter(self, 
                              src_nodes: list[Node], dst_nodes: list[Node], 
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
            self.increment(PROCESSED, edge_type)

            # Provenance-specific constraints
            # This filtering DEFINITELY affects our selection of theta
            if self.filter(src_node, dst_node, edge_type):
                continue
            
            new_edge = create_edge(src_node, dst_node, edge_type.optype, uniform_time)
            if new_edge not in new_edges: # TODO: Also filter self-referential edges?
                new_edges.add(new_edge)
        return new_edges

    def increment(self, stat: str, edge_type: EdgeType) -> None:
        self.stats[stat].update([str(edge_type)])

    def get_stats_str(self) -> None:
        lines = []
        for stat, counter in self.stats.items():
            lines.append(stat)
            stat_str = {stat}
            for edge_type, value in counter.items():
                lines.append(f'  {edge_type}: {value}')
            lines.append('')
        return '\n'.join(lines)
    
def create_edge(src_node: Node, dst_node: Node,
                optype: str,
                time_func: Callable[[], int]):
    edge_time = time_func() # TODO: what should this value be?
    # I was thinking it'd be the avg of src_node and dst_node times, but nodes dont have time attributes
    return Edge.of(
        src_id=src_node.id, dst_id=dst_node.id, 
        optype=optype,
        time=edge_time
    )