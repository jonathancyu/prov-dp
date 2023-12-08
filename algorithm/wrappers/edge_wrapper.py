from graphson import Edge

from .node_wrapper import IN, OUT


class EdgeWrapper:
    edge: Edge

    def __init__(self, edge: Edge):
        self.edge = edge
        self.node_ids = {
            IN: edge.src_id,
            OUT: edge.dst_id
        }

    def get_ref_id(self) -> int:
        return int(self.edge.model_extra['REF_ID'])

    def get_src_id(self) -> int:
        return self.edge.src_id

    def get_dst_id(self) -> int:
        return self.edge.dst_id

    def get_time(self) -> int:
        return self.edge.time

    def get_op_type(self) -> str:
        return self.edge.optype

    def get_token(self) -> str:
        model = self.edge.model_dump()
        return '_'.join([model['_label'],  self.edge.optype])

    def __hash__(self):
        return hash(self.edge)
