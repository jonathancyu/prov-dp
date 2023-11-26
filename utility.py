from pathlib import Path
from typing import Callable

import numpy as np
from graphviz import Digraph

from graphson import GraphsonObject, Node, Edge, Graph
from algorithm.wrappers import EdgeWrapper


def group_by_lambda[T](objects: list[GraphsonObject], 
                       get_attribute: Callable[[GraphsonObject], T]
                       ) -> dict[T, list[GraphsonObject]]:
    grouped: dict[T, list[GraphsonObject]] = {}
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

def uniform_generator(edges: list[EdgeWrapper]) -> Callable[[],int]:
    times = list(map(lambda x: x.get_time(), edges))
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

def get_edge_id(graph_name: str) -> int:
    split = graph_name.split('-')
    try:
        assert len(split) == 3
        return int(split[1])
    except:
        return ''