from collections import deque
from datetime import datetime

import numpy as np
from icecream import ic

from graphson import Graph
from utility import logistic_function
from .graph_processor import GraphProcessor, EDGES_PROCESSED, EDGES_FILTERED, SELF_REFERRING, TIME_FILTERED, PRUNED_AT_DEPTH
from .wrappers import GraphWrapper, EdgeWrapper, NodeWrapper, IN, OUT


class TreeShaker(GraphProcessor):
    def __init__(self):
        super().__init__()

    def perturb_graph(self,
                      input_graph_object: Graph,
                      source_edge_id: int,
                      epsilon_1: float,
                      epsilon_2: float,
                      alpha: float
                      ):
        start_time = datetime.now()

        graph = GraphWrapper(input_graph_object)
        source_edge = graph.get_edge_by_id(source_edge_id)
        assert source_edge is not None

        # TODO: make alpha larger for backward b/c startup sequence is often the same
        in_edges: list[EdgeWrapper] = self.perturb_tree(
            graph=graph,
            source_edge_id=source_edge_id,
            direction=IN,
            epsilon=epsilon_1,
            alpha=alpha)
        out_edges: list[EdgeWrapper] = self.perturb_tree(
            graph=graph,
            source_edge_id=source_edge_id,
            direction=OUT,
            epsilon=epsilon_2,
            alpha=alpha)
        new_edges: list[EdgeWrapper] = list(set(in_edges + out_edges))
        self.runtimes.append((datetime.now() - start_time).total_seconds())
        return Graph(
            vertices=[node.node for node in graph.nodes],
            edges=[edge.edge for edge in new_edges]
        )

    def perturb_tree(self,
                     graph: GraphWrapper,
                     source_edge_id: int,
                     direction: str,
                     epsilon: float,
                     alpha: float
                     ) -> list[EdgeWrapper]:
        visited_edges: set[EdgeWrapper] = set()
        queue = deque([(0, source_edge_id)])
        local_sensitivity: float = 1 / alpha

        new_edges: list[EdgeWrapper] = []
        while len(queue) > 0:
            # BFS, so FIFO. Append to back, pop from front (left).
            depth, edge_id = queue.popleft()
            edge: EdgeWrapper = graph.get_edge_by_id(edge_id)
            edge_type = graph.get_edge_type(edge)
            self.increment_counter(EDGES_PROCESSED + f' ({direction})', edge_type)

            # https://par.nsf.gov/servlets/purl/10191952 Relaxed Indistinguishability of Neighbors
            # Higher distance -> high epsilon_prime -> p is lower
            subtree_size = graph.get_tree_size(edge_id, direction)
            distance = alpha * subtree_size
            epsilon_prime = epsilon * distance
            p = logistic_function(epsilon_prime / local_sensitivity)
            prune_edge = np.random.choice([True, False], p=[p, 1 - p])
            if edge in visited_edges or prune_edge:
                if prune_edge:
                    self.increment_counter(PRUNED_AT_DEPTH + f'={depth}', edge_type)
                    self.increment_counter(EDGES_FILTERED + f' ({direction})', edge_type)
                continue
            visited_edges.add(edge)

            node_id: int = edge.node_ids[direction]
            node: NodeWrapper = graph.get_node(node_id)

            new_edges.append(edge)
            next_edge_ids: list[int] = node.edge_ids[direction]
            queue.extend([(depth + 1, next_edge_id) for next_edge_id in next_edge_ids])

        return new_edges
