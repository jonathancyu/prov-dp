from collections import deque
from datetime import datetime

import numpy as np
from icecream import ic

from graphson import Graph, Edge
from .graph_processor import GraphProcessor
from .wrappers import GraphWrapper, EdgeWrapper, NodeWrapper, IN, OUT

class TreeShaker(GraphProcessor):
    def perturb_graph(self,
                    input_graph_object: Graph,
                    source_edge_id: int,
                    epsilon_1: float,
                    epsilon_2: float
                    ):
        start_time = datetime.now()

        graph = GraphWrapper(input_graph_object)
        assert graph.get_edge_by_id(source_edge_id) is not None

        in_edges: list[EdgeWrapper] = self.perturb_tree(graph, [source_edge_id], IN)
        out_edges: list[EdgeWrapper]  = self.perturb_tree(graph, [source_edge_id], OUT)
        new_edges: list[EdgeWrapper] = in_edges + out_edges
        ic(len(in_edges), len(out_edges), len(new_edges))
        self.runtimes.append((datetime.now() - start_time).total_seconds())
        return Graph(
            vertices    = [node.node for node in graph.nodes],
            edges       = [edge.edge for edge in new_edges]
        )


    def perturb_tree(self,
                     graph: GraphWrapper,
                     source_edge_ids: list[int],
                     direction: str
                     ) -> list[EdgeWrapper]:
        queue = deque(source_edge_ids)
        ic(f'Starting queue size: {len(queue)}')

        new_edges: list[EdgeWrapper] = []
        while len(queue) > 0:
            # BFS, so FIFO. Append to back, pop from front (left).
            edge_id: int = queue.popleft()
            edge: EdgeWrapper = graph.get_edge_by_id(edge_id)

            threshold = 0.5
            weight = 1
            if edge.visited[direction] or weight < threshold:
                continue
            edge.visited[direction] = True

            node_id: int = edge.node_ids[direction]
            node: NodeWrapper = graph.get_node(node_id)

            new_edges.append(edge)
            next_edge_ids: list[int] = node.edge_ids[direction]
            queue.extend(next_edge_ids)
            ic(len(queue), len(new_edges))

        return new_edges

