from src.algorithm.extended_top_m_filter import EdgeType
from src.graphson.raw_node import NodeType


OPTYPE_LOOKUP: dict[EdgeType, str] = {
    EdgeType(NodeType.PROCESS_LET, NodeType.PROCESS_LET): "Start_Processlet",
    EdgeType(NodeType.PROCESS_LET, NodeType.FILE): "Write",
    EdgeType(NodeType.PROCESS_LET, NodeType.IP_CHANNEL): "Write",
    EdgeType(NodeType.FILE, NodeType.PROCESS_LET): "Read",
    EdgeType(NodeType.IP_CHANNEL, NodeType.PROCESS_LET): "Read",
}
