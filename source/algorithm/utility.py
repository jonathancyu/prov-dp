import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar, Generator

import networkx as nx
import numpy as np
import torch
import tqdm

from .wrappers import Subgraph, GraphWrapper, IN, OUT


def get_edge_ref_id(graph_name: str) -> int:
    split = graph_name.split('-')
    try:
        assert len(split) == 3
        return int(split[1])
    except AssertionError:
        return -1


def to_nx(graph: Subgraph | GraphWrapper) -> nx.DiGraph:
    digraph: nx.DiGraph = nx.DiGraph()

    # NetworkX node IDs must index at 0
    node_ids = {node.get_id(): i
                for i, node in enumerate(graph.nodes)}
    for node in graph.nodes:
        digraph.add_node(node_ids[node.get_id()],
                         feature=node.get_token()
                         )
    for edge in graph.edges:
        # If an edge has a null node, don't include it.
        # This occurs when we've pruned a subgraph
        if None in edge.node_ids.values():
            continue
        digraph.add_edge(node_ids[edge.node_ids[IN]],
                         node_ids[edge.node_ids[OUT]],
                         feature=edge.get_token()
                         )
    return digraph


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def tokenize(path):
    assert len(path) % 2 == 0
    return [f'{path[idx]}|{path[idx + 1]}' for idx in range(0, len(path), 2)]


def build_vocab(paths: list[str]) -> tuple[list[str], dict[str, int]]:
    """
    Builds the vocabulary for the model.
    :param paths: list of paths
    :return: integer to string, and string to integer mappings
    """
    token_set = set()
    distinct_paths = set()
    for path in paths:
        path = tokenize(path.split(' '))
        token_set.update(path)
        distinct_paths.add(' '.join(path))
    tokens = ['.'] + list(token_set)
    print(f'  Found {len(tokens)} tokens and {len(distinct_paths)} distinct paths in {len(paths)} entries')
    return tokens, {token: i for i, token in enumerate(tokens)}


def print_stats(name: str, samples: list) -> None:
    print(f'  {name} (N={len(samples)}) - mean: {np.mean(samples)}, std: {np.std(samples)}, '
          f'min: {np.min(samples)}, max: {np.max(samples)}')


def logistic_function(x: float) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=RuntimeWarning)
        try:
            return 1 / (1 + np.exp(x))
        except RuntimeWarning:
            return 0
