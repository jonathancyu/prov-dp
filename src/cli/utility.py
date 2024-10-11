from pathlib import Path
from typing import TypeVar

from graphviz import Digraph

T = TypeVar("T")


def save_dot(dot_graph: Digraph, file_path: Path, pdf=False) -> None:
    file_path = file_path.with_suffix(".dot")
    file_path.parent.mkdir(exist_ok=True, parents=True)
    dot_graph.save(file_path)
    if pdf:
        dot_graph.render(file_path, format="pdf")
