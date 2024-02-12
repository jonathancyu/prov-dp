from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar, Generator

import networkx as nx
import torch
from tqdm import tqdm

from .wrappers import Subgraph, GraphWrapper, IN, OUT

T = TypeVar('T')


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
        digraph.add_edge(node_ids[edge.node_ids[IN]],
                         node_ids[edge.node_ids[OUT]],
                         feature=edge.get_token()
                         )
    return digraph


def map_pool(func: callable,
             items: list[T],
             desc: str = '') -> Generator:
    with ProcessPoolExecutor() as executor:
        # When we multiprocess, objects are pickled and passed to the child process
        # So we have to return objects from the function to get the changes back
        futures = [
            executor.submit(func, *item) if isinstance(item, tuple)
            else executor.submit(func, item)
            for item in items
        ]
        with tqdm(total=len(futures), desc=desc) as pbar:
            for future in futures:
                yield future.result()
                pbar.update(1)


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
    print(f'Found {len(tokens)} tokens and {len(distinct_paths)} distinct paths in {len(paths)} entries')
    return tokens, {token: i for i, token in enumerate(tokens)}
