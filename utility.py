from pathlib import Path
from typing import Callable

import numpy as np
from graphviz import Digraph

from graphson import Node, Edge, GraphsonObject


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

def save_dot(dot_graph: Digraph, folder_name: str, file_path: Path, pdf: bool=False) -> None:
    output_path = (Path(folder_name) / file_path.stem).with_suffix('.dot')
    dot_graph.save(output_path)
    if pdf:
        dot_graph.render(output_path, format='pdf')

def get_stats(stat: str, data: list[int]) -> dict:
    result = {
        'avg': np.average(data),
        'stdev': np.std(data),
        'min': min(data),
        'max': max(data)
    }
    return { f'{stat} {key}': value for key, value in result.items() }