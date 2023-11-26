from collections import Counter

from graphson import Node, NodeType, EdgeType

IN = 'IN'
OUT = 'OUT'


class NodeWrapper:
    node: Node
    edge_ids: dict[str, list[int]]
    time: int

    _in_degree: dict[EdgeType, int]
    _out_degree: dict[EdgeType, int]

    def __init__(self, node: Node):
        self.node = node
        self._degree_by_type = {
            IN: Counter(), OUT: Counter
        }
        self.edge_ids = {
            IN: [], OUT: []
        }

    def add_incoming(self, edge_id: int, edge_type: EdgeType) -> None:
        self._degree_by_type[IN].update([edge_type])
        self.edge_ids[IN].append(edge_id)

    def add_outgoing(self, edge_id: int, edge_type: EdgeType) -> None:
        self._degree_by_type[OUT].update([edge_type])
        self.edge_ids[OUT].append(edge_id)

    def get_in_degree(self, edge_type: EdgeType) -> int:
        return self._degree_by_type[IN].get(edge_type, 0)

    def get_out_degree(self, edge_type: EdgeType) -> int:
        return self._degree_by_type[OUT].get(edge_type, 0)

    def get_id(self) -> int:
        return self.node.id

    def get_type(self) -> NodeType:
        return self.node.type

    def get_degree(self):
        return self._in_degree
