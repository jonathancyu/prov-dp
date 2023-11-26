from enum import Enum

from pydantic import Field

from .graphsonobject import GraphsonObject
from .utility import format_label, sanitize


class NodeType(Enum):
    PROCESS_LET = 'ProcessNode'
    FILE = 'FileNode'
    IP_CHANNEL = 'SocketChannelNode'

    def __str__(self):
        return self.name

    def __int__(self):
        raise ValueError


class Node(GraphsonObject):
    id: int = Field(..., alias='_id')
    type: NodeType = Field(..., alias='TYPE')

    def __hash__(self) -> int:
        return hash(self.id)

    def to_dot_args(self) -> dict[str, any]:
        model = self.model_dump(by_alias=True, exclude=['time'])
        args = {}
        match self.type:
            case NodeType.PROCESS_LET:
                args = {
                    'color': 'black',
                    'shape': 'box',
                    'style': 'solid',
                    'label': format_label(model, [('exe_name', 'EXE_NAME'),
                                                  ('cmd', 'CMD')])
                    # + f'\nfirst_event: {format_timestamp(self.time)}'
                }
            case NodeType.FILE:
                args = {
                    'color': 'pink',
                    'shape': 'oval',
                    'style': 'filled',
                    'label': 'path: ' + model['FILENAME_SET'][0]['value']
                }
            case NodeType.IP_CHANNEL:
                args = {
                    'color': 'yellow',
                    'shape': 'box',
                    'style': 'filled',
                    'label': format_label(model, [
                        ('srcIP', 'LOCAL_INET_ADDR'),
                        ('dstIP', 'REMOTE_INET_ADDR'),
                        ('type', 'CHANEL_STATE')
                    ])
                }
        return {key: sanitize(value) for key, value in args.items()}
