from pathlib import Path
from typing import Callable

import numpy as np
from graphviz import Digraph

from graphson import Graph, Node, Edge, GraphsonObject


def group_by_lambda[T](objects: list[GraphsonObject], 
                       get_attribute: Callable[[GraphsonObject], T]
                       ) -> dict[T, list[GraphsonObject]]:
    grouped = {}
    for obj in objects:
        key = get_attribute(obj)
        if not key in grouped:
            grouped[key] = []
        grouped[key].append(obj)

    return grouped


def node_from_list(node_id: int, node_list: list[Node]) -> Node:
    candidates = filter(lambda n: n.id==node_id, node_list)
    assert len(candidates) == 1
    return candidates[0]

def uniform_generator(edges: list[Edge]) -> Callable[[],int]:
    times = list(map(lambda x: x.time, edges))
    min_time, max_time = min(times), max(times)

    return lambda: int(np.round(np.random.uniform(min_time, max_time)))

def save_dot(dot_graph: Digraph, file_path: Path, 
             dot=True, pdf=False) -> None:
    file_path = file_path.with_suffix('.dot')
    file_path.parent.mkdir(exist_ok=True, parents=True)
    dot_graph.save(file_path)
    if pdf:
        dot_graph.render(file_path, format='pdf')

def get_stats(stat: str, data: list[int]) -> dict:
    result = {
        'avg': np.average(data),
        'stdev': np.std(data),
        'min': min(data),
        'max': max(data)
    }
    return { f'{stat} {key}': value for key, value in result.items() }

def count_disconnected_nodes(graph: Graph) -> float:
    included_nodes: set[Node] = set()
    for edge in graph.edges:
        included_nodes.add(graph.get_node(edge.src_id))
        included_nodes.add(graph.get_node(edge.dst_id))
    return len(set(graph.nodes) - included_nodes)