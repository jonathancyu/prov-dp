import json
from pathlib import Path
from typing import Callable

from graphviz import Digraph
from pydantic import BaseModel, Field

from .edge import Edge
from .node import Node


# noinspection PyArgumentList
class Graph(BaseModel):
    nodes: list[Node] = Field(alias='vertices', default_factory=list)
    edges: list[Edge] = Field(alias='edges', default_factory=list)

    _included_nodes: set[Node]
    _node_lookup: dict[int, Node]

    @staticmethod
    def load_file(path_to_json: Path):
        with open(path_to_json, 'r', encoding='utf-8') as input_file:
            input_json = json.load(input_file)
            return Graph(**input_json)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._node_lookup = {
            node.id: node
            for node in self.nodes
        }

    def to_dict(self) -> dict:
        model = self.model_dump(by_alias=True)
        model['mode'] = 'EXTENDED'
        model['vertices'] = [node.to_dict() for node in self.nodes]
        model['edges'] = [edge.to_dict() for edge in self.edges]
        return model

    def to_dot(self) -> Digraph:
        dot_graph = Digraph()
        dot_graph.attr(rankdir='LR')
        included_nodes: set[Node] = set()
        sorted_edges = sorted(self.edges, key=lambda e: e.time)

        def add_to_graph(node: Node):
            dot_graph.node(str(node.id), **node.to_dot_args())

        for edge in sorted_edges:
            if edge.dst_id is None or edge.src_id is None:
                print(edge.id)
                continue
            add_to_graph(self.get_node(edge.src_id))
            dot_graph.edge(str(edge.src_id), str(edge.dst_id), **edge.to_dot_args())
            add_to_graph(self.get_node(edge.dst_id))

        for node in self.nodes:
            if node not in included_nodes:
                add_to_graph(node)
        return dot_graph

    def get_node(self, node_id: int):
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise ValueError(f'Node with ID {node_id} not found')

    def _add_node(self,
                  node_id: int,
                  callback: Callable[[Node], None]):
        node = self._node_lookup.get(node_id)
        if node not in self._included_nodes:
            self._included_nodes.add(node)
            callback(node)
