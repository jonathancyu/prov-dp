import math
from collections import Counter
from datetime import datetime
from typing import Callable

import numpy as np
from graphviz import Digraph
from icecream import ic
from pydantic import Field

from graphson import Node, NodeType, Edge, EdgeType, Graph
from utility import group_by_lambda, uniform_generator



EDGES_PROCESSED = '#edges processed'
EDGES_FILTERED = '#edges filtered'
SELF_REFERRING = '#self referring edges pruned'
TIME_FILTERED = '#edges pruned by time'

class NodeWrapper:
    node: Node

    _time: int
    _in_degree: int = 0
    _out_degree: int = 0

    def __init__(self, node: Node):
        self.node = node

    def add_incoming(self) -> None:
        self._in_degree += 1

    def add_outgoing(self) -> None:
        self._out_degree += 1

    def get_time(self) -> int:
        return self._time

    def set_time(self, time: int) -> None:
        self._time = time

    def get_in_degree(self) -> int:
        return self._in_degree

    def get_out_degree(self) -> int:
        return self._out_degree

    def get_id(self):
        return self.node.id

    def get_type(self):
        return self.node.type

class EdgeWrapper:
    edge: Edge
    def __init__(self, edge: Edge):
        self.edge = edge
    def get_src_id(self):
        return self.edge.src_id

    def get_dst_id(self):
        return self.edge.dst_id

    def get_time(self):
        return self.edge.time

    def get_op_type(self):
        return self.edge.optype

    def __hash__(self):
        return hash(self.edge)

class GraphWrapper:
    graph: Graph
    nodes: list[NodeWrapper]
    edges: list[EdgeWrapper]

    _node_lookup: dict[int, NodeWrapper]
    _edge_lookup: dict[tuple[int,int], Edge]
    def __init__(self, graph: Graph):
        self.graph = graph
        self.nodes = [ NodeWrapper(node) for node in self.graph.nodes ]
        self.edges = [ EdgeWrapper(edge) for edge in self.graph.edges ]
        self._node_lookup = { node.get_id(): node for node in self.nodes }
        self._edge_lookup = {
            (edge.get_src_id(), edge.get_dst_id()): edge
            for edge in self.edges}
        self._set_node_times()

    def _set_node_times(self) -> None:
        included_nodes: set[int] = set()
        sorted_edges = sorted(self.edges, key=lambda e: e.get_time(), reverse=True)
        for edge in sorted_edges:
            src_node = self.get_node(edge.get_src_id())
            dst_node = self.get_node(edge.get_dst_id())
            if src_node is None or dst_node is None:
                continue
            src_node.add_outgoing()
            dst_node.add_incoming()

            # TODO: Is this how we should set time?
            for node_id in [edge.get_src_id(), edge.get_dst_id()]:
                if node_id in included_nodes:
                    continue
                node = self.get_node(node_id)
                node.set_time(edge.get_time())
    def get_node(self, id: int) -> NodeWrapper:
        return self._node_lookup.get(id)

    def get_edge(self, id: int) -> EdgeWrapper:
        return self._edge_lookup.get(id)

    def get_node_type(self, id: int) -> NodeType:
        return self.get_node(id)

    def get_edge_type(self, edge: EdgeWrapper) -> EdgeType:
        return EdgeType(
            edge = edge.edge,
            src_type = self.get_node(edge.get_src_id()).get_type(),
            dst_type = self.get_node(edge.get_dst_id()).get_type()
        )

class GraphProcessor:
    stats: dict[str, Counter[EdgeType,int]] = {}
    runtimes: list[float]

    def __init__(self):
        for stat in [EDGES_PROCESSED, SELF_REFERRING, TIME_FILTERED]:
            self.stats[stat] = Counter()
            self.runtimes = []

    def perturb_graph(self, 
                      input_graph_object: Graph,
                      epsilon_1: float,
                      epsilon_2: float
                      ) -> Graph:
        graph = GraphWrapper(input_graph_object)
        start_time = datetime.now()

        node_groups: dict[NodeType, list[NodeWrapper]] = group_by_lambda(graph.nodes, lambda node: node.get_type())
        edge_type_groups: dict[EdgeType, list[EdgeWrapper]] = group_by_lambda(graph.edges, lambda edge: graph.get_edge_type(edge))
        for edge in graph.edges:
            edge_type: EdgeType = graph.get_edge_type(edge)

        edge_type_frequence: dict[EdgeType, float] = {}
        for edge in graph.edges:
            edge_type = graph.get_edge_type(edge)
            edge_type_frequence[edge_type] = (len(edge_type_groups[edge_type]) / len(graph.edges))



        new_edges: list[EdgeWrapper] = []
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
        return Graph(vertices = [ node_wrapper.node for node_wrapper in graph.nodes],
                     edges = [ edge_wrapper.edge for edge_wrapper in new_edges])

    def filter(self,
               src_node: NodeWrapper,
               dst_node: NodeWrapper,
               edge_type: EdgeType) -> bool:
        self.increment_counter(EDGES_PROCESSED, edge_type)
        if src_node == dst_node:
            self.increment_counter(SELF_REFERRING, edge_type)
            return True
        elif src_node.get_type() == NodeType.PROCESS_LET \
                and dst_node.get_type() == NodeType.PROCESS_LET \
                and src_node.get_time() >= dst_node.get_time():
            self.increment_counter(TIME_FILTERED, edge_type)
            return True
        return False
    
    def pick_random_node(self, nodes: list[NodeWrapper], weights: list[float] = None) -> NodeWrapper:
        if weights is None or sum(weights) == 0:
            weights = [1.0 for node in nodes]
        total_weight = sum(weights)
        weights = [ weight/total_weight for weight in weights]
        return np.random.choice(nodes, 1, p=weights)[0]
    
    # Top-M Filter: https://doi.org/10.1145/2808797.2809385
    # Line numbers correspond to Algorithm 1
    def extended_top_m_filter(self,
                              src_nodes: list[NodeWrapper], dst_nodes: list[NodeWrapper],
                              existing_edges: list[Edge],
                              edge_type: EdgeType,
                              epsilon_1: float, epsilon_2: float
                              ) -> list[EdgeWrapper]:
        # 1
        new_edges: set[EdgeWrapper] = set()
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

        # 9-15
        for edge in existing_edges:
            weight = 1 + np.random.laplace(0, 1.0/epsilon_1)
            if weight > theta:
                new_edges.add(edge)
                
        # 16-22
        uniform_time = uniform_generator(existing_edges)
        while len(new_edges) < m_perturbed:
            # 19: random pick an edge (i, j)
            src_node = self.pick_random_node(src_nodes, [node.get_out_degree() for node in src_nodes])
            dst_node = self.pick_random_node(dst_nodes, [node.get_out_degree() for node in dst_nodes])
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


def count_disconnected_nodes(graph: GraphWrapper) -> float:
    included_nodes: set[Node] = set()
    for edge in graph.edges:
        included_nodes.add(graph.get_node(edge.get_src_id()))
        included_nodes.add(graph.get_node(edge.get_dst_id()))
    return len(set(graph.nodes) - included_nodes)

def create_edge(src_node: NodeWrapper, dst_node: NodeWrapper,
                optype: str,
                time_func: Callable[[], int]):
    edge_time = time_func() # TODO: what should this value be?
    # I was thinking it'd be the avg of src_node and dst_node times, but nodes dont have time attributes
    return EdgeWrapper(
        Edge.of(
            src_id=src_node.get_id(),
            dst_id=dst_node.get_id(),
            optype=optype,
            time=edge_time
        )
    )