from collections import Counter
from typing import Callable

import numpy as np

from algorithm.wrappers.edge_wrapper import EdgeWrapper
from algorithm.wrappers.graph_wrapper import GraphWrapper
from algorithm.wrappers.node_wrapper import NodeWrapper
from graphson import Edge, EdgeType


class GraphProcessor:
    stats: dict[str, Counter[EdgeType, int]] = {}
    runtimes: list[float]

    EDGES_PROCESSED = '#edges processed'
    EDGES_FILTERED = '#edges filtered'
    SELF_REFERRING = '#self referring edges pruned'
    TIME_FILTERED = '#edges pruned by time'

    def __init__(self):
        for stat in [self.EDGES_PROCESSED, self.EDGES_FILTERED, self.SELF_REFERRING, self.TIME_FILTERED]:
            self.stats[stat] = Counter()
            self.runtimes = []

    def increment_counter(self, stat: str, edge_type: EdgeType) -> None:
        self.stats[stat].update([str(edge_type)])

    def get_stats_str(self) -> str:
        lines = [f'runtime avg: {np.average(self.runtimes)}',
                 f'runtime stdev: {np.std(self.runtimes)}',
                 f'runtime (min, max): ({min(self.runtimes), max(self.runtimes)})']
        for stat, counter in self.stats.items():
            lines.append(stat)
            for edge_type, value in counter.items():
                lines.append(f'  {edge_type}: {value}')
            lines.append('')
        return '\n'.join(lines)

    @staticmethod
    def create_edge(src_node: NodeWrapper, dst_node: NodeWrapper,
                    optype: str,
                    time_func: Callable[[], int]
                    ) -> EdgeWrapper:
        edge_time = time_func()  # TODO: what should this value be?
        # I was thinking it'd be the avg of src_node and dst_node times, but nodes dont have time attributes
        return EdgeWrapper(
            Edge.of(
                src_id=src_node.get_id(),
                dst_id=dst_node.get_id(),
                optype=optype,
                time=edge_time
            )
        )


def count_disconnected_nodes(graph: GraphWrapper) -> float:
    included_nodes: set[NodeWrapper] = set()
    for edge in graph.edges:
        included_nodes.add(graph.get_node(edge.get_src_id()))
        included_nodes.add(graph.get_node(edge.get_dst_id()))
    return len(set(graph.nodes) - included_nodes)
