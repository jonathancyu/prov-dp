from source.graphson import Node, NodeType, EdgeType
from .directions import IN, OUT


class NodeWrapper:
    node: Node
    edge_ids: dict[str, list[int]]

    _in_degree: dict[EdgeType, int]
    _out_degree: dict[EdgeType, int]

    def __init__(self, node: Node):
        self.node = node
        self.edge_ids = {
            IN: [], OUT: []
        }

    def __add_edge(self, edge_id: int, direction: str) -> None:
        if edge_id is None or edge_id in self.edge_ids[direction]:
            return
        self.edge_ids[direction].append(edge_id)

    def add_incoming(self, edge_id: int) -> None:
        self.__add_edge(edge_id, IN)

    def add_outgoing(self, edge_id: int) -> None:
        self.__add_edge(edge_id, OUT)

    def __remove_edge(self, edge_id: int, direction: str) -> None:
        if edge_id in self.edge_ids[direction]:
            self.edge_ids[direction].remove(edge_id)

    def remove_incoming(self, edge_id: int):
        self.__remove_edge(edge_id, IN)

    def remove_outgoing(self, edge_id: int):
        self.__remove_edge(edge_id, OUT)

    def set_incoming(self, edge_ids: list[int]) -> None:
        self.edge_ids[IN] = edge_ids

    def set_outgoing(self, edge_ids: list[int]) -> None:
        self.edge_ids[OUT] = edge_ids

    def get_incoming(self) -> list[int]:
        return self.edge_ids[IN]

    def get_outgoing(self) -> list[int]:
        return self.edge_ids[OUT]

    def get_in_degree(self) -> int:
        return len(self.edge_ids[IN])

    def get_out_degree(self) -> int:
        return len(self.edge_ids[OUT])

    def get_id(self) -> int:
        return self.node.id

    def set_id(self, new_id: int) -> None:
        self.node.id = new_id

    def get_type(self) -> NodeType:
        return self.node.type

    def get_token(self) -> str:
        model = self.node.model_dump()
        token = f'{self.node.type}_'
        if self.node.type == NodeType.PROCESS_LET:
            token += model['EXE_NAME']
        elif self.node.type == NodeType.FILE:
            token += model["FILENAME_SET"][0]["value"].replace(' ', '_')
        elif self.node.type == NodeType.IP_CHANNEL:
            src_ip = model['LOCAL_INET_ADDR']
            dst_ip = model['REMOTE_INET_ADDR']
            token += f'{src_ip}_{dst_ip}'
        elif self.node.type == NodeType.EPHEMERAL:
            token = '.'
        else:
            raise ValueError(f'Unknown node type: {self.node.type}')

        return token
