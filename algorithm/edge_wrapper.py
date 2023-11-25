from graphson import Edge
class EdgeWrapper:
    edge: Edge
    def __init__(self, edge: Edge):
        self.edge = edge
    def get_src_id(self) -> int:
        return self.edge.src_id

    def get_dst_id(self) -> int:
        return self.edge.dst_id

    def get_time(self) -> int:
        return self.edge.time

    def get_op_type(self) -> str:
        return self.edge.optype

    def __hash__(self):
        return hash(self.edge)