from pydantic import Field

from .graphsonobject import GraphsonObject
from .node import NodeType
from .utility import format_timestamp

EDGE_ID_SEQUENCE = 5000


class Edge(GraphsonObject):
    id: int = Field(alias='_id')  # TODO: May be edge ID conflicts when we export the graph
    src_id: int = Field(..., alias='_outV')
    dst_id: int = Field(..., alias='_inV')
    optype: str = Field(..., alias='OPTYPE')
    label: str = Field(..., alias='_label')
    time: int = Field(..., alias='EVENT_START')

    def __repr__(self):
        return f'{self.src_id}-{self.optype}-{self.dst_id}'

    @staticmethod
    def of(src_id: int, dst_id: int,
           optype: str,
           time: int,
           new_id: int = None, ):
        global EDGE_ID_SEQUENCE
        if new_id is None:
            new_id = EDGE_ID_SEQUENCE
            EDGE_ID_SEQUENCE += 1
        return Edge(_id=new_id, _inV=src_id, _outV=dst_id, OPTYPE=optype, EVENT_START=time)

    def to_dot_args(self) -> dict[str, any]:
        model = self.model_dump(by_alias=True, exclude={'time'})
        args = {
            'color': 'black',
            'label': ''
        }
        if self.optype == 'EPHEMERAL':
            args['color'] = 'blue'
        # if self.time is not None:
        #     args['label'] += format_timestamp(self.time)
        args['label'] += self.label
        return args

    def __hash__(self):
        return hash((self.src_id, self.dst_id))


class EdgeType:
    src_type: NodeType
    dst_type: NodeType
    optype: str

    def __init__(self,
                 edge: Edge,
                 src_type: NodeType,
                 dst_type: NodeType
                 ):
        self.src_type = src_type
        self.dst_type = dst_type
        self.optype = edge.optype

    def __hash__(self):
        return hash((self.src_type,
                     self.optype,
                     self.dst_type))

    def __eq__(self, other):
        return self.src_type == other.src_type \
            and self.optype == other.optype \
            and self.dst_type == other.dst_type

    def __str__(self):
        return f'[{self.src_type.name}]-{self.optype}-[{self.dst_type.name}]'
