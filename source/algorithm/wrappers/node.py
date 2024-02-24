from ...graphson import NodeType, RawNode


class Node:
    node: RawNode
    incoming_edges: list[int]
    outgoing_edges: list[int]
    marked: bool = False

    def __init__(self, node: RawNode):
        self.node = node
        self.incoming_edges, self.outgoing_edges = [], []

    # Wrapper functions
    @staticmethod
    def __add_edge(edge_id: int, collection: list[int]) -> None:
        if edge_id is None or edge_id in collection:
            return
        collection.append(edge_id)

    @staticmethod
    def __remove_edge(edge_id: int, collection: list[int]) -> None:
        if edge_id in collection:
            collection.remove(edge_id)

    def add_incoming(self, edge_id: int) -> None:
        Node.__add_edge(edge_id, self.incoming_edges)

    def add_outgoing(self, edge_id: int) -> None:
        Node.__add_edge(edge_id, self.outgoing_edges)

    def remove_incoming(self, edge_id: int):
        Node.__remove_edge(edge_id, self.incoming_edges)

    def remove_outgoing(self, edge_id: int):
        Node.__remove_edge(edge_id, self.outgoing_edges)

    def set_incoming_edges(self, edge_ids: list[int]) -> None:
        self.incoming_edges = edge_ids

    def set_outgoing_edges(self, edge_ids: list[int]) -> None:
        self.outgoing_edges = edge_ids

    def get_incoming_edges(self) -> list[int]:
        return self.incoming_edges

    def get_outgoing_edges(self) -> list[int]:
        return self.outgoing_edges

    # Adapter functions (reach into Node object)
    def get_id(self) -> int:
        return self.node.id

    def set_id(self, new_id: int) -> None:
        self.node.id = new_id

    def get_type(self) -> NodeType:
        return self.node.type

    # Comparison functions
    def __eq__(self, other: 'Node') -> bool:
        return self.get_id() == other.get_id()

    def __hash__(self) -> int:
        return hash(self.get_id())

    # Exporter functions
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

    @staticmethod
    def __format_label(model: dict, label_key_list: list[tuple[str, str]]) -> str:
        return ' '.join(
            [f'{label}: {model.get(key)}' for label, key in label_key_list]
        )

    @staticmethod
    def __sanitize(label: str) -> str:
        # Replace backslashes with forward slashes and double quotes with single quotes
        return label.replace('\\', '/') \
            .replace('\\"', '\'')

    def to_dot_args(self) -> dict[str, any]:
        model = self.node.model_dump(by_alias=True, exclude={'time'})
        match self.get_type():
            case NodeType.PROCESS_LET:
                args = {
                    'color': 'black',
                    'shape': 'box',
                    'style': 'solid',
                    'label': Node.__format_label(model, [('exe_name', 'EXE_NAME'),
                                                         ('cmd', 'CMD')])
                }
            case NodeType.FILE:
                filename = 'no file name'
                if model.get('FILENAME_SET') is not None:
                    filename = model['FILENAME_SET'][0]['value']
                args = {
                    'color': 'pink',
                    'shape': 'oval',
                    'style': 'filled',
                    'label': 'path: ' + filename
                }
            case NodeType.IP_CHANNEL:
                args = {
                    'color': 'yellow',
                    'shape': 'box',
                    'style': 'filled',
                    'label': Node.__format_label(model, [
                        ('srcIP', 'LOCAL_INET_ADDR'),
                        ('dstIP', 'REMOTE_INET_ADDR'),
                        ('type', 'CHANEL_STATE')
                    ])
                }
            case NodeType.EPHEMERAL:
                args = {
                    'color': 'blue',
                    'shape': 'oval',
                    'style': 'solid',
                    'label': 'ephemeral'
                }
            case _:
                raise ValueError(f'Unknown node type: {self.get_type()}')
        if self.marked:
            args['color'] = 'greenyellow'
        return {key: Node.__sanitize(value) for key, value in args.items()}
