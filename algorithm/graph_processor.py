from collections import Counter
from typing import Callable

import numpy as np

from algorithm.wrappers.edge_wrapper import EdgeWrapper
from algorithm.wrappers.graph_wrapper import GraphWrapper
from algorithm.wrappers.node_wrapper import NodeWrapper
from graphson import Edge, EdgeType

EDGES_PROCESSED = '#edges processed'
EDGES_FILTERED = '#edges filtered'
PRUNED_AT_DEPTH = '#pruned at depth'
SELF_REFERRING = '#self referring edges pruned'
TIME_FILTERED = '#edges pruned by time'
PRUNED_SUBTREE_SIZES = 'sizes of subtree pruned'


class GraphProcessor:
    stats: dict[str, Counter[EdgeType, int]]
    lists: dict[str, list[any]]

    def __init__(self):
        self.stats = {}
        self.lists = {'runtimes': []}

    def increment_counter(self, key: str, edge_type: EdgeType) -> None:
        if self.stats.get(key) is None:
            self.stats[key] = Counter()
        self.stats[key].update([str(edge_type)])

    def get_stats_str(self) -> str:
        lines = []  # [f'runtime avg: {np.average(self.lists['runtimes'])}',
        #          f'runtime stdev: {np.std(self.lists['runtimes'])}',
        #          f'runtime (min, max): ({min(self.lists['runtimes']), max(self.lists['runtimes'])})']
        for stat, counter in self.stats.items():
            lines.append(stat)
            for edge_type, value in counter.items():
                lines.append(f'  {edge_type}: {value}')
            lines.append('')
        return '\n'.join(lines)


def count_disconnected_nodes(graph: GraphWrapper) -> float:
    included_nodes: set[NodeWrapper] = set()
    for edge in graph.edges:
        included_nodes.add(graph.get_node(edge.get_src_id()))
        included_nodes.add(graph.get_node(edge.get_dst_id()))
    return len(set(graph.nodes) - included_nodes)
