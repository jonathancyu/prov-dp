from graphson import Node, NodeType, EdgeType
class NodeWrapper:
    node: Node

    _time: int
    _in_degree: dict[EdgeType, int]
    _out_degree: dict[EdgeType, int]

    def __init__(self, node: Node):
        self.node = node
        self._in_degree = {}
        self._out_degree = {}

    def get_time(self) -> int:
        return self._time

    def set_time(self, time: int) -> None:
        self._time = time

    def add_incoming(self, edge_type: EdgeType) -> None:
        self._in_degree[edge_type] = self._in_degree.get(edge_type, 0) + 1

    def add_outgoing(self, edge_type: EdgeType) -> None:
        self._out_degree[edge_type] = self._out_degree.get(edge_type, 0) + 1

    def get_in_degree(self, edge_type: EdgeType) -> int:
        return self._in_degree.get(edge_type, 0)

    def get_out_degree(self, edge_type: EdgeType) -> int:
        return self._out_degree.get(edge_type, 0)

    def get_id(self) -> int:
        return self.node.id

    def get_type(self) -> NodeType:
        return self.node.type

    def get_degree(self):
        return self._in_degree