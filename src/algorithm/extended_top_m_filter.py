from dataclasses import dataclass
import itertools
from random import shuffle

import numpy as np
import math

from src.algorithm.wrappers.edge import Edge
from src.algorithm.wrappers.graph import Graph
from src.algorithm.wrappers.node import Node
from src.graphson.raw_edge import RawEdge
from src.graphson.raw_node import NodeType


@dataclass(frozen=True)  # set frozen to create hash method
class EdgeType:
    src_type: NodeType
    dst_type: NodeType
    op_type: str


class ExtendedTopMFilter:
    # Pruning parameters
    __epsilon: float
    __delta: float
    __single_threaded: bool

    def __init__(
        self, epsilon: float = 1, delta: float = 0.5, single_threaded: bool = False
    ):
        self.__epsilon = epsilon
        self.__delta = delta
        self.__single_threaded = single_threaded

    def filter_graph(self, graph: Graph) -> None:
        """
        Modifies the input graph according to the top-m filter
        """
        # Get distinct edge types
        # TODO: should this happen before across the whole dataset?
        edges = graph.get_edges()
        edge_lookup: dict[EdgeType, list[Edge]] = {}

        for edge in edges:
            src = graph.get_node(edge.get_src_id())
            dst = graph.get_node(edge.get_dst_id())
            edge_type = EdgeType(
                src_type=src.get_type(),
                dst_type=dst.get_type(),
                op_type=edge.get_op_type(),
            )
            if edge_type not in edge_lookup:
                edge_lookup[edge_type] = []
            edge_lookup[edge_type].append(edge)

        for edge_type, edges in edge_lookup.items():
            # TODO: how to allocate epsilon
            self.__run_filter(
                graph,
                edge_type,
                edges,
                epsilon_1=self.__epsilon * self.__delta,
                epsilon_2=self.__epsilon * (1 - self.__delta),
            )

        # TODO: update algorithm to reflect this happens only after all filters are ran
        # [28-36] Update graph. Keep only component containing original node if result is disconnected
        graph.remove_disconnected_components()

    def __run_filter(
        self,
        graph: Graph,
        edge_type: EdgeType,
        edges: list[Edge],
        epsilon_1: float,
        epsilon_2: float,
    ) -> None:
        # Start by removing all edges
        print(f"Filtering edge type: {edge_type}")
        for edge in edges:
            graph.remove_edge(edge)

        # [3-4]
        m = len(edges)
        # TODO: breaks if leq 0 since we divide by it.
        m_perturbed = max(1, np.ceil(m + np.random.laplace(0, 1 / epsilon_2)))

        # [5-6]
        V_s, V_d = [], []
        for node in graph.get_nodes():
            if node.get_type() == edge_type.src_type:
                V_s.append(node)
            if node.get_type() == edge_type.dst_type:
                V_d.append(node)
        E_possible = list(itertools.product(V_s, V_d))
        E_valid: list[tuple[Node, Node]] = [
            edge
            for edge in E_possible
            if ExtendedTopMFilter.__is_valid(edge, edge_type.op_type)
        ]
        print(f"possible: {len(list(E_possible))}, valid: {len(E_valid)}")
        # [7]
        val = (len(E_valid) / m_perturbed) - 1
        assert val > 0, f"<{val}>"
        epsilon_t = math.log(val)

        # [8-13] Set filter bound
        if epsilon_1 < epsilon_t:
            theta = epsilon_t / (2 * epsilon_1)
        else:
            theta = (1 / epsilon_1) * math.log(
                (len(E_valid) / (2 * m_perturbed)) + (1 / 2) * (math.exp(epsilon_1 - 1))
            )

        # [14-20] Filter existing edges
        for edge in edges:
            A_ij = 1  # Edge already exists thus A_ij = 1
            A_ij_perturbed = A_ij + np.random.laplace(0, 1 / epsilon_1)
            # If edge passes the filter, add it. Otherwise filter it
            if A_ij_perturbed > theta:
                # Add the edge
                graph.add_edge(edge)

        # [21-27] Add new edges
        # using for loop instead of while to guarantee termination.
        # functionally the same though.
        shuffle(E_valid)  # [23]
        for src, dst in E_valid:
            src_id, dst_id = src.get_id(), dst.get_id()
            # [22]
            if len(graph.get_edges()) >= m_perturbed:
                break
            # [24]
            if graph.has_edge(src_id, dst_id):
                continue
            # [25] Add edge
            new_edge = Edge(
                RawEdge(
                    _id=graph.get_next_edge_id(),
                    _outV=src_id,
                    _inV=dst_id,
                    OPTYPE=edge_type.op_type,
                    # TODO: determine how to get these values
                    _label="TODO",
                    EVENT_START=-1,
                )
            )
            graph.add_edge(new_edge)

    @classmethod
    def __is_valid(cls, edge: tuple[Node, Node], op_type: str) -> bool:
        # TODO: Add further validations
        src, dst = edge
        return src != dst
