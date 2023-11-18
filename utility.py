import math
from typing import Callable

import numpy as np

from graphson import Node, Edge, GraphsonObject


def group_by_lambda[T](objects: list[GraphsonObject], 
                       get_attribute: Callable[[GraphsonObject], T]
                       ) -> dict[T, list[GraphsonObject]]:
    grouped = {}
    for obj in objects:
        key = get_attribute(obj)
        if not key in grouped:
            grouped[key] = []
        grouped[key].append(obj)

    return grouped


def node_from_list(node_id: int, node_list: list[Node]) -> Node:
    candidates = filter(lambda n: n.id==node_id, node_list)
    assert len(candidates) == 1
    return candidates[0]

def uniform_generator(edges: list[Edge]) -> Callable[[],int]:
    times = list(map(lambda x: x.time, edges))
    min_time, max_time = min(times), max(times)

    return lambda: int(np.round(np.random.uniform(min_time, max_time)))

def create_edge(src_node: Node, dst_node: Node,
                optype: str,
                time_func: Callable[[], int]):
    edge_time = time_func() # TODO: what should this value be?
    # I was thinking it'd be the avg of src_node and dst_node times, but nodes dont have time attributes
    return Edge.of(
        src_id=src_node.id, dst_id=dst_node.id, 
        optype=optype,
        time=edge_time
    )

def extended_top_m_filter(src_nodes: list[Node], dst_nodes: list[Node], 
                          existing_edges: list[Edge], 
                          optype: str,
                          epsilon_1: float, epsilon_2: float=1
                          ) -> list[Edge]:
    n_s, n_d = len(src_nodes), len(dst_nodes)
    m = len(existing_edges)
    m_perturbed = m + int(np.round(np.random.laplace(0, 1.0/epsilon_2)))
    if m_perturbed <= 0:
        return []
    
    num_possible_edges = n_s*n_d
    epsilon_t = math.log(
        (num_possible_edges/m) - 1
        )
    if epsilon_1 < epsilon_t:
        theta = epsilon_t / (2*epsilon_1)
    else:
        theta = math.log(
            (num_possible_edges / (2*m_perturbed))
            + (math.exp(epsilon_1)-1)/2
        ) / epsilon_1
    
    new_edges: set[Edge] = set()
    for edge in existing_edges:
        # TODO: could we tweak this initial weight based on time?
        weight = 1 + np.random.laplace(0, 1.0/epsilon_1) # add to 1 b/c A_ij = 1 for this edge
        if weight > theta:
            new_edges.add(edge)
    # at this point, len(edge_tuples) = n_1
    uniform_time = uniform_generator(existing_edges)
    while len(new_edges) < m_perturbed:
        src_node = np.random.choice(src_nodes)
        dst_node = np.random.choice(dst_nodes)
        new_edge = create_edge(src_node, dst_node, optype, uniform_time)
        if new_edge not in new_edges: # TODO: Also filter self-referential edges?
            new_edges.add(new_edge)

    return new_edges
