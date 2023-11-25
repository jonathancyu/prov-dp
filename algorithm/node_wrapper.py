from graphson import Node, NodeType
class NodeWrapper:
    node: Node

    _time: int
    _in_degree: int = 0
    _out_degree: int = 0

    def __init__(self, node: Node):
        self.node = node

    def add_incoming(self) -> None:
        self._in_degree += 1

    def add_outgoing(self) -> None:
        self._out_degree += 1

    def get_time(self) -> int:
        return self._time

    def set_time(self, time: int) -> None:
        self._time = time

    def get_in_degree(self) -> int:
        return self._in_degree

    def get_out_degree(self) -> int:
        return self._out_degree

    def get_id(self) -> int:
        return self.node.id

    def get_type(self) -> NodeType:
        return self.node.type