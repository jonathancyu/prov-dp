import pytest
from pathlib import Path
import os

from algorithm import GraphWrapper, IN, OUT
from graphson import NodeType


@pytest.fixture(autouse=True)
def graph(request) -> GraphWrapper:
    current_directory = Path(request.fspath.dirname)

    return GraphWrapper(
        current_directory / '..' / 'resources' / 'nd-52809777-processletevent.json'
    )


def test_step2__non_process_out_degree_is_0(graph):
    print(os.getcwd())
        # assert len(node.edge_ids[OUT]) == 0

