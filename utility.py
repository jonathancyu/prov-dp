import warnings
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
from graphviz import Digraph

from source.graphson import GraphsonObject

T = TypeVar('T')


def group_by_lambda(objects: list[GraphsonObject],
                       get_attribute: Callable[[GraphsonObject], T]
                       ) -> dict[T, list[GraphsonObject]]:
    grouped: dict[T, list[GraphsonObject]] = {}
    for obj in objects:
        key = get_attribute(obj)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(obj)

    return grouped


def save_dot(dot_graph: Digraph,
             file_path: Path,
             pdf=False) -> None:
    file_path = file_path.with_suffix('.dot')
    file_path.parent.mkdir(exist_ok=True, parents=True)
    dot_graph.save(file_path)
    if pdf:
        dot_graph.render(file_path, format='pdf')