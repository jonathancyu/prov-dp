from pathlib import Path
from typing import Callable
import json

from graphviz import Digraph
from pydantic import BaseModel, Field

from .node import Node
from .edge import Edge

class Graph(BaseModel):
    nodes:              list[Node] = Field(alias='vertices', default_factory=list)
    edges:              list[Edge] = Field(alias='edges', default_factory=list)

    _node_lookup:        dict[int, Node]
    _edge_lookup:        dict[list[int, int], Edge]

    @staticmethod
    def load_file(path_to_json: Path):
        with open(path_to_json, 'r', encoding='utf-8') as input_file:
            input_json = json.load(input_file)
            return Graph(**input_json)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._node_lookup = {node.id: node for node in self.nodes}
        self._edge_lookup = {(edge.src_id, edge.dst_id): edge for edge in self.edges}

    def _add_node(self, 
                  node_id: int, included_nodes: set[Node], 
                  callback: Callable[[Node],None]
                  ) -> None:
        if node_id not in included_nodes:
            included_nodes.add(node_id)
            node = self._node_lookup[node_id]
            callback(node)

    def get_node(self, node_id: int) -> Node:
        return self._node_lookup[node_id]
    
    def get_edge(self, edge_id: int) -> Edge:
        return self._node_lookup[edge_id]

    def to_dict(self) -> dict:
        model = self.model_dump(by_alias=True)
        model['mode'] = 'EXTENDED'
        model['vertices'] = [ node.to_dict() for node in self.nodes ]
        model['edges'] = [ edge.to_dict() for edge in self.edges ]
        return model

    def to_dot(self) -> Digraph:
        dot_graph = Digraph()
        dot_graph.attr(rankdir='LR')
        included_nodes: set[Node] = set()
        sorted_edges = sorted(self.edges, key=lambda e: e.time)
        add_to_graph = lambda node: dot_graph.node(str(node.id), **node.to_dot_args())
        for edge in sorted_edges:
            self._add_node(edge.src_id, included_nodes, add_to_graph)
            dot_graph.edge(str(edge.src_id), str(edge.dst_id), **edge.to_dot_args())
            self._add_node(edge.dst_id, included_nodes, add_to_graph)

        disconnected_nodes = 0
        for node in self.nodes:
            if node not in included_nodes:
                disconnected_nodes += 1
                add_to_graph(node)
        return dot_graph