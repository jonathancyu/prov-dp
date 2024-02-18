from source.graphson import Edge

from .directions import IN, OUT


class EdgeWrapper:
    edge: Edge
    node_ids: dict[str, int | None]

    def __init__(self, edge: Edge):
        self.edge = edge
        self.node_ids = {
            IN: edge.src_id,
            OUT: edge.dst_id
        }

    def invert(self) -> None:
        # Swap the src and dst ids
        src_id, dst_id = self.node_ids[IN], self.node_ids[OUT]
        self.set_src_id(dst_id)
        self.set_dst_id(src_id)

    def get_id(self) -> int:
        return self.edge.id

    def set_id(self, new_id: int) -> None:
        self.edge.id = new_id

    def get_ref_id(self) -> int:
        return int(self.edge.model_extra['REF_ID'])

    def get_src_id(self) -> int:
        assert self.edge.src_id == self.node_ids[IN]
        return self.edge.src_id

    def get_dst_id(self) -> int:
        assert self.edge.dst_id == self.node_ids[OUT]
        return self.edge.dst_id

    def set_src_id(self, src_id: int | None) -> None:
        self.edge.src_id = src_id
        self.node_ids[IN] = src_id

    def set_dst_id(self, dst_id: int | None) -> None:
        self.edge.dst_id = dst_id
        self.node_ids[OUT] = dst_id

    def get_time(self) -> int:
        return self.edge.time

    def get_op_type(self) -> str:
        return self.edge.optype

    def get_token(self) -> str:
        model = self.edge.model_dump(by_alias=True)
        return '_'.join([model['_label'],  self.edge.optype])

    def __hash__(self):
        return hash(self.edge)
