import json
import warnings; warnings.filterwarnings('ignore', category=DeprecationWarning)
from datetime import datetime
from enum import Enum
from pathlib import Path
from time import time_ns
from typing import Callable

from pydantic import BaseModel, Field, root_validator
from icecream import ic
from graphviz import Digraph


def format_timestamp(timestamp: int) -> str:
    return datetime.fromtimestamp(int(timestamp/1e9)).strftime('%Y-%m-%d %H:%M:%S')

def string_to_field(value: str):
    try:
        return {
            'type': 'long',
            'value': int(value)
        }
    except ValueError:
        return {
            'type': 'string',
            'value': value
        }

def format_label(model: dict, label_key_list: list[tuple[str, str]]) -> str:
    return ' '.join(
        [ f'{label}: {model.get(key)}' for label, key in label_key_list ]
    )


class GraphsonObject(BaseModel):
    @root_validator(pre=True)
    def extract_values(cls, values):
        result = {}
        for key, value in values.items():
            if isinstance(value, list):
                assert len(value) in [0, 1]
                value = value[0]
            if key.startswith('_') or not isinstance(value, dict):
                result[key] = value
            else:
                result[key] = value['value']
        return result
    class Config:
        extra = 'allow'
    
    def to_dict(self):
        new_dict = {}
        model = self.model_dump(by_alias=True)
        for key, value in model.items():
            str_value = str(value)
            if key.startswith('_'):
                new_dict[key] = str_value
            else:
                new_dict[key] = string_to_field(str_value)
        return new_dict


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
    _time: int

    def get_time(self) -> int:
        return self._time
    
    def set_time(self, new_time: int) -> None:
        self.time = new_time

    def __hash__(self):
        return self.id

    def to_dot_args(self) -> dict[str, any]:
        model = self.model_dump(by_alias=True, exclude=['time'])
        args = {}
        match self.type:
            case NodeType.PROCESS_LET:
                args = {
                    'color': 'black',
                    'shape': 'box',
                    'style': 'solid',
                    'label': format_label(model, [
                        ('exe_name', 'EXE_NAME'),
                        ('cmd', 'CMD')
                    ]) + f'\nfirst_event: {format_timestamp(self.time)}'
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
        return args

EDGE_ID_SEQUENCE = 5000
class Edge(GraphsonObject):
    id: int = Field(alias='_id') # TODO: May be edge ID conflicts when we export the graph
    src_id: int = Field(..., alias='_outV')
    dst_id: int = Field(..., alias='_inV')
    optype: str = Field(..., alias='OPTYPE')
    time: int = Field(..., alias='EVENT_START')

    def __repr__(self):
        return f'{self.src_id}-{self.optype}-{self.dst_id}'
    
    @staticmethod
    def of(src_id: int, dst_id: int, 
           optype: str,
           time: int,
           new_id: int = None,) :
        global EDGE_ID_SEQUENCE
        if new_id is None:
            new_id = EDGE_ID_SEQUENCE
            EDGE_ID_SEQUENCE += 1
        return Edge(_id=new_id, _inV=src_id, _outV=dst_id, OPTYPE=optype, EVENT_START=time)
    
    def to_dot_args(self) -> dict[str, any]: 
        model = self.model_dump(by_alias=True)
        args = {
            'color': 'black'
        }
        if self.time is not None:
            args['label'] = format_timestamp(self.time)
        return args
    
    def __hash__(self):
        return hash((self.src_id, self.dst_id))

class EdgeType:
    src_type: NodeType
    dst_type: NodeType
    optype: str
    def __init__(self, edge: Edge, node_lookup: dict[int, Node]):
        self.src_type = node_lookup[edge.src_id].type
        self.dst_type = node_lookup[edge.dst_id].type
        self.optype = edge.optype

    def __hash__(self):
        return hash((self.src_type, self.dst_type))

    def __str__(self):
        return f'[{self.src_type.name}]-{self.optype}-[{self.dst_type.name}]'
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
        self._init_node_times()

    def _add_node(self, 
                  node_id: int, included_nodes: set[Node], 
                  callback: Callable[[Node],None]
                  ) -> None:
        if node_id not in included_nodes:
            included_nodes.add(node_id)
            node = self._node_lookup[node_id]
            callback(node)
    
    def _init_node_times(self) -> None:
        included_nodes: set[int] = set()
        sorted_edges = sorted(self.edges, key=lambda e: e.time)
        for edge in sorted_edges:
            self._add_node(edge.src_id, included_nodes, lambda node: node.set_time(edge.time))
            self._add_node(edge.dst_id, included_nodes, lambda node: node.set_time(edge.time))


        
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