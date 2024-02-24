from ...graphson import RawEdge


class Edge:
    edge: RawEdge
    marked: bool = False

    def __init__(self, edge: RawEdge):
        self.edge = edge

    def invert(self) -> None:
        # Swap the src and dst ids
        src_id, dst_id = self.get_src_id(), self.get_dst_id()
        self.set_src_id(dst_id)
        self.set_dst_id(src_id)

    def get_id(self) -> int:
        return self.edge.id

    def set_id(self, new_id: int) -> None:
        self.edge.id = new_id

    def get_ref_id(self) -> int:
        return int(self.edge.model_extra['REF_ID'])

    def get_src_id(self) -> int:
        return self.edge.src_id

    def get_dst_id(self) -> int:
        return self.edge.dst_id

    def set_src_id(self, src_id: int | None) -> None:
        self.edge.src_id = src_id

    def set_dst_id(self, dst_id: int | None) -> None:
        self.edge.dst_id = dst_id

    def get_time(self) -> int:
        return self.edge.time

    def get_op_type(self) -> str:
        return self.edge.optype

    def get_token(self) -> str:
        model = self.edge.model_dump(by_alias=True)
        return '_'.join([model['_label'],  self.edge.optype])

    def translate_node_ids(self, translation: dict[int, int]) -> None:
        self.set_src_id(translation[self.get_src_id()])
        self.set_dst_id(translation[self.get_dst_id()])

    def __eq__(self, other: 'Edge') -> bool:
        return self.get_id() == other.get_id()

    def __hash__(self):
        return hash(self.get_id())

    def to_dot_args(self) -> dict[str, any]:
        # model = self.model_dump(by_alias=True, exclude={'time'})
        args = {
            'color': 'black',
            'label': ''
        }
        if self.get_op_type() == 'EPHEMERAL':
            args['color'] = 'blue'
        if self.marked:
            args['color'] = 'green'
        # if self.time is not None:
        #     args['label'] += format_timestamp(self.time)
        args['label'] += self.edge.label
        return args
