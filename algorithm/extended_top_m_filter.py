from datetime import datetime

import numpy as np
import math

from utility import group_by_lambda, uniform_generator
from graphson import Graph, NodeType, Edge, EdgeType
from algorithm.wrappers.node_wrapper import NodeWrapper
from algorithm.wrappers.edge_wrapper import EdgeWrapper
from algorithm.wrappers.graph_wrapper import GraphWrapper
from .graph_processor import GraphProcessor

class ExtendedTopMFilter(GraphProcessor):
    def perturb_graph(self,
                      input_graph_object: Graph,
                      epsilon_1: float,
                      epsilon_2: float
                      ) -> Graph:
        graph = GraphWrapper(input_graph_object)
        start_time = datetime.now()

        node_groups: dict[NodeType, list[NodeWrapper]] \
            = group_by_lambda(graph.nodes, lambda node: node.get_type())
        edge_type_groups: dict[EdgeType, list[EdgeWrapper]] \
            = group_by_lambda(graph.edges, lambda edge: graph.get_edge_type(edge))

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
        return Graph(
            vertices   = [node.node for node in graph.nodes],
            edges      = [edge.edge for edge in new_edges]
        )

    def filter(self,
               src_node: NodeWrapper,
               dst_node: NodeWrapper,
               edge_type: EdgeType) -> bool:
        self.increment_counter(self.EDGES_PROCESSED, edge_type)
        if src_node == dst_node:
            self.increment_counter(self.SELF_REFERRING, edge_type)
            return True
        elif src_node.get_type() == NodeType.PROCESS_LET \
                and dst_node.get_type() == NodeType.PROCESS_LET \
                and src_node.time >= dst_node.time:
            self.increment_counter(self.TIME_FILTERED, edge_type)
            return True
        return False

    def pick_random_node(self, nodes: list[NodeWrapper], weights: list[float] = None) -> NodeWrapper:
        if weights is None or sum(weights) == 0:
            weights = [1.0 for node in nodes]
        total_weight = sum(weights)
        weights = [weight / total_weight for weight in weights]
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
        m_perturbed = m + int(np.round(np.random.laplace(0, 1.0 / epsilon_2)))
        if m_perturbed <= 0:
            return []
        # 4
        n_s, n_d = len(src_nodes), len(dst_nodes)
        num_possible_edges = n_s * n_d
        epsilon_t = math.log(
            (num_possible_edges / m) - 1
        )
        # 5-8
        if epsilon_1 < epsilon_t:
            theta = epsilon_t / (2 * epsilon_1)
        else:
            theta = math.log(
                (num_possible_edges / (2 * m_perturbed))
                + (math.exp(epsilon_1) - 1) / 2
            ) / epsilon_1

        # 9-15
        for edge in existing_edges:
            weight = 1 + np.random.laplace(0, 1.0 / epsilon_1)
            if weight > theta:
                new_edges.add(edge)

        # 16-22
        uniform_time = uniform_generator(existing_edges)
        while len(new_edges) < m_perturbed:
            # 19: random pick an edge (i, j)
            src_node = self.pick_random_node(src_nodes, [node.get_out_degree(edge_type) for node in src_nodes])
            dst_node = self.pick_random_node(dst_nodes, [node.get_in_degree(edge_type) for node in dst_nodes])
            # Provenance-specific constraints
            # This filtering DEFINITELY affects our selection of theta and the DP proof
            if self.filter(src_node, dst_node, edge_type):
                continue

            # 20-21
            new_edge = self.create_edge(src_node, dst_node, edge_type.optype, uniform_time)
            if new_edge not in new_edges:
                new_edges.add(new_edge)
        return new_edges