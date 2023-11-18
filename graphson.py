from datetime import datetime
from enum import Enum
from pathlib import Path
import json

from pydantic import BaseModel, Field, root_validator
from icecream import ic
from graphviz import Digraph


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
        ic(model)
        for key, value in model.items():
            str_value = str(value)
            if key.startswith('_'):
                new_dict[key] = str_value
            else:
                new_dict[key] = string_to_field(str_value)
        ic(new_dict)
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

    def __repr__(self):
        return f'{id}: {self.type}'

    def to_dot_args(self) -> dict[str, any]:
        model = self.model_dump(by_alias=True)
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
                    ])
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
            timestamp = self.time/1e9
            args['label'] = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        return args
    
    def __hash__(self):
        return hash((self.src_id, self.dst_id))


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

    def _add_node(self, node_id: int, digraph: Digraph, included_nodes: set[Node]) -> None:
        if node_id not in included_nodes:
            included_nodes.add(node_id)
            node = self._node_lookup[node_id]
            digraph.node(str(node_id), **node.to_dot_args())

    def to_dot(self, output_path: str, pdf: bool=False) -> None:
        dot_graph = Digraph()
        dot_graph.attr(rankdir='LR')
        included_nodes: set[int] = set()
        sorted_edges = sorted(self.edges, key=lambda e: e.time)
        for edge in sorted_edges:
            self._add_node(edge.src_id, dot_graph, included_nodes)
            dot_graph.edge(str(edge.src_id), str(edge.dst_id), **edge.to_dot_args())
            self._add_node(edge.dst_id, dot_graph, included_nodes)

        for node in self.nodes:
            if node.id not in included_nodes:
                ic(f'skipped node {node.id}')
        dot_graph.save(output_path)
        if pdf:
            dot_graph.render(output_path, format='pdf')
        