from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar, Generator

import networkx as nx
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
