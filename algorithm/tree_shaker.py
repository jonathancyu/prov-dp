import warnings
from collections import deque
from datetime import datetime

import numpy as np
from icecream import ic

from graphson import Graph, Edge
from .graph_processor import GraphProcessor
from .wrappers import GraphWrapper, EdgeWrapper, NodeWrapper, IN, OUT

class TreeShaker(GraphProcessor):
    def __init__(self):
        super().__init__()
    def perturb_graph(self,
                    input_graph_object: Graph,
                    source_edge_id: int,
                    epsilon_1: float,
                    epsilon_2: float
                    ):
        start_time = datetime.now()

        graph = GraphWrapper(input_graph_object)
        assert graph.get_edge_by_id(source_edge_id) is not None

        # https://par.nsf.gov/servlets/purl/10191952 Relaxed Indistinguishability of Neighbors
        alpha: float = 2.0 # They use 0.5, 1, 2

        in_edges: list[EdgeWrapper] = self.perturb_tree(
            graph = graph,
            source_edge_ids = [source_edge_id],
            direction = IN,
            epsilon_1 = epsilon_1,
            epsilon_2 = epsilon_2,
            alpha = alpha)
        out_edges: list[EdgeWrapper]  = self.perturb_tree(
            graph = graph,
            source_edge_ids = [source_edge_id],
            direction = OUT,
            epsilon_1 = epsilon_1,
            epsilon_2 = epsilon_2,
            alpha = alpha)
        new_edges: list[EdgeWrapper] = in_edges + out_edges
        self.runtimes.append((datetime.now() - start_time).total_seconds())
        return Graph(
            vertices    = [node.node for node in graph.nodes],
            edges       = [edge.edge for edge in new_edges]
        )


    def perturb_tree(self,
                     graph: GraphWrapper,
                     source_edge_ids: list[int],
                     direction: str,
                     epsilon_1: float,
                     epsilon_2: float,
                     alpha: float
                     ) -> list[EdgeWrapper]:
        visited_edges: set[EdgeWrapper] = set()
        queue = deque(source_edge_ids)
        ic(f'Starting queue size: {len(queue)}')
        local_sensitivity: float = 1/alpha

        new_edges: list[EdgeWrapper] = []
        while len(queue) > 0:
            # BFS, so FIFO. Append to back, pop from front (left).
            edge_id: int = queue.popleft()
            edge: EdgeWrapper = graph.get_edge_by_id(edge_id)
            edge_type = graph.get_edge_type(edge)
            self.increment_counter(self.EDGES_PROCESSED, edge_type)

            # Higher distance -> high epsilon_prime -> p is lower
            subtree_size = graph.get_tree_size(edge_id, direction)
            distance = alpha * subtree_size
            epsilon_prime = epsilon_1 * distance
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=RuntimeWarning)
                try:
                    p = 1/(1 + np.exp(epsilon_prime / local_sensitivity))
                except RuntimeWarning:
                    p = 0
            prune_edge = np.random.choice([True, False], p=[p, 1-p])
            if edge in visited_edges or prune_edge:
                if prune_edge:
                    self.increment_counter(self.EDGES_FILTERED, edge_type)
                continue
            visited_edges.add(edge)

            node_id: int = edge.node_ids[direction]
            node: NodeWrapper = graph.get_node(node_id)

            new_edges.append(edge)
            next_edge_ids: list[int] = node.edge_ids[direction]
            queue.extend(next_edge_ids)

        return new_edges
