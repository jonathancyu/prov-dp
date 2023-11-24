import math
from collections import Counter
from datetime import datetime
from typing import Callable

import numpy as np
from icecream import ic 

from graphson import Node, NodeType, Edge, EdgeType, Graph
from utility import group_by_lambda, uniform_generator



EDGES_PROCESSED = '#edges processed'
EDGES_FILTERED = '#edges filtered'
SELF_REFERRING = '#self referring edges pruned'
TIME_FILTERED = '#edges pruned by time'


class GraphProcessor:
    stats: dict[str, Counter[EdgeType,int]] = {}
    runtimes: list[float]

    def __init__(self):
        for stat in [EDGES_PROCESSED, SELF_REFERRING, TIME_FILTERED]:
            self.stats[stat] = Counter()
            self.runtimes = []
        
    
    def _set_node_times(self, graph: Graph) -> None:
        included_nodes: set[int] = set()
        sorted_edges = sorted(graph.edges, key=lambda e: e.time, reverse=True)
        for edge in sorted_edges:
            src_node = graph.get_node(edge.src_id)
            dst_node = graph.get_node(edge.dst_id)
            src_node.add_outgoing()
            dst_node.add_incoming()
            
            for node_id in [edge.src_id, edge.dst_id]:
                if node_id in included_nodes:
                    continue
                node = graph.get_node(node_id)
                node.set_time(edge.time)


    def perturb_graph(self, 
                      input_graph: Graph, epsilon_1: float, epsilon_2: float) -> Graph:
        start_time = datetime.now()

        node_groups = group_by_lambda(input_graph.nodes, lambda node: node.type)
        edge_type_groups = group_by_lambda(input_graph.edges, lambda edge: EdgeType(edge, input_graph._node_lookup))
        self._set_node_times(input_graph)


        new_edges: list[Edge] = []
        for edge_type, edges in edge_type_groups.items():
            perturbed_edges = self.extended_top_m_filter(
                src_nodes=node_groups[edge_type.src_type],
                dst_nodes=node_groups[edge_type.dst_type],
                existing_edges=edges,
                edge_type=edge_type,
                epsilon_1=epsilon_1,
                epsilon_2=epsilon_2
            )
            new_edges.extend(perturbed_edges)

        self.runtimes.append((datetime.now() - start_time).total_seconds())    
        
        return Graph(vertices=input_graph.nodes, edges=new_edges)

    def filter(self, 
               src_node: Node, dst_node: Node, 
               edge_type: EdgeType) -> bool:
        self.increment_counter(EDGES_PROCESSED, edge_type)
        if src_node == dst_node:
            self.increment_counter(SELF_REFERRING, edge_type)
            return True
        elif src_node.type == NodeType.PROCESS_LET \
                and dst_node.type == NodeType.PROCESS_LET \
                and src_node.time >= dst_node.time:
            self.increment_counter(TIME_FILTERED, edge_type)
            return True
        return False
    
    def pick_random_node(self, nodes: list[Node]) -> Node:
        return np.random.choice(nodes)
    
    def pick_nodes(self, src_nodes: list[Node], dst_nodes: list[Node]) -> tuple[Node,Node]:
        return self.pick_random_node(src_nodes), self.pick_random_node(dst_nodes)
    
    # Top-M Filter: https://doi.org/10.1145/2808797.2809385
    # Line numbers correspond to Algorithm 1
    def extended_top_m_filter(self, 
                              src_nodes: list[Node], dst_nodes: list[Node], 
                              existing_edges: list[Edge], 
                              edge_type: EdgeType,
                              epsilon_1: float, epsilon_2: float
                              ) -> list[Edge]:
        """Lines 1-8: Compute constants"""
        # 1
        new_edges: set[Edge] = set()
        # 2-3
        m = len(existing_edges)
        m_perturbed = m + int(np.round(np.random.laplace(0, 1.0/epsilon_2)))
        if m_perturbed <= 0:
            return []
        # 4
        n_s, n_d = len(src_nodes), len(dst_nodes)
        num_possible_edges = n_s*n_d
        epsilon_t = math.log(
            (num_possible_edges/m) - 1
            )
        #5-8
        if epsilon_1 < epsilon_t:
            theta = epsilon_t / (2*epsilon_1)
        else:
            theta = math.log(
                (num_possible_edges / (2*m_perturbed))
                + (math.exp(epsilon_1)-1)/2
            ) / epsilon_1

        """Line"""
        # 9-15
        for edge in existing_edges:
            weight = 1 + np.random.laplace(0, 1.0/epsilon_1)
            if weight > theta:
                new_edges.add(edge)
                
        # 16-22
        uniform_time = uniform_generator(existing_edges)
        while len(new_edges) < m_perturbed:
            # 19: random pick an edge (i, j)
            src_node, dst_node = self.pick_nodes(src_nodes, dst_nodes)

            # Provenance-specific constraints
            # This filtering DEFINITELY affects our selection of theta and the DP proof
            if self.filter(src_node, dst_node, edge_type):
                continue
            
            # 20-21
            new_edge = create_edge(src_node, dst_node, edge_type.optype, uniform_time)
            if new_edge not in new_edges:
                new_edges.add(new_edge)
        return new_edges

    def increment_counter(self, stat: str, edge_type: EdgeType) -> None:
        self.stats[stat].update([str(edge_type)])

    def get_stats_str(self) -> None:
        lines = []
        lines.append(f'runtime avg: {np.average(self.runtimes)}')
        lines.append(f'runtime stdev: {np.std(self.runtimes)}')
        lines.append(f'runtime (min, max): ({min(self.runtimes), max(self.runtimes)})')
        lines.append('')
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