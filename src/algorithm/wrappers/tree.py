import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import graphviz as gv
import networkx as nx

from src.algorithm.utility import get_cycle

from .edge import Edge
from .node import Node
from ...graphson import RawEdge, RawNode, RawGraph, NodeType


@dataclass
class Marker:
    height: int
    size: int
    path: str
    tree: "Tree"


@dataclass
class NodeStats:
    height: int
    size: int
    depth: int


@dataclass
class TreeStats:
    # Aggregates
    heights: list[int]
    depths: list[int]
    sizes: list[int]
    degrees: list[int]

    # Totals
    height: int
    size: int
    degree: int
    diameter: int


class Tree:
    graph_id: int | None
    root_node_id: int | None

    training_data: list[tuple[str, "Tree"]]
    marked_nodes: dict[int, Marker]  # node_id: data
    marked_node_paths: dict[int, str]  # node_id: path
    stats: dict[
        str, list[float]
    ]  # Used to keep track of stats from within forked processes

    __nodes: dict[int, Node]
    __edges: dict[int, Edge]
    __incoming_lookup: dict[int, set[int]]  # node_id: set[edge_id]
    __outgoing_lookup: dict[int, set[int]]  # node_id: set[edge_id]
    __node_stats: dict[int, NodeStats]

    @staticmethod
    def load_file(json_path: Path) -> "Tree":
        file_name = str(json_path.stem)
        if "-" in file_name:
            split = file_name.split("-")
        elif "_" in file_name:
            split = file_name.split("_")
        else:
            raise ValueError(f"Invalid file name: {file_name}")
        ref_id = -1
        if len(split) == 3:
            ref_id = int(split[1])
        unprocessed_tree = Tree(RawGraph.load_file(json_path), ref_id)
        tree = unprocessed_tree.preprocess()
        tree.assert_valid_tree()

        return tree

    def __init__(
        self, graph: RawGraph | None = None, source_edge_ref_id: int | None = None
    ):
        graph = graph or RawGraph()
        self.__incoming_lookup = {}
        self.__outgoing_lookup = {}
        self.__init_nodes(graph.nodes)
        self.__init_edges(graph.edges)
        self.__init_source(source_edge_ref_id)

        # Algorithm-specific fields
        self.__node_stats = {}
        self.training_data = []
        self.marked_nodes = {}
        self.marked_node_paths = {}
        self.stats = {}

    def __init_nodes(self, nodes: list[RawNode]):
        # Create a lookup by node ID
        self.__nodes = {}
        for raw_node in nodes:
            self.add_node(Node(raw_node))

    def __init_edges(self, edges: list[RawEdge]):
        # Create a lookup by edge ID and add edge references to nodes
        self.__edges = {}
        for raw_edge in edges:
            self.add_edge(Edge(raw_edge))

    def __init_source(self, source_edge_ref_id: int | None) -> None:
        # Set the graph_id to keep track of the original graph
        self.graph_id = source_edge_ref_id
        if source_edge_ref_id is not None:
            # Ref ID is not the same as graphson ID, so we need to find the edge with the matching ref ID
            matches = [
                edge
                for edge in self.__edges.values()
                if edge.get_ref_id() == source_edge_ref_id
            ]
            if len(matches) == 1:
                self.root_node_id = matches[0].get_src_id()
                return
        self.root_node_id = None

    def get_subtree(
        self, root_node_id: int, visited_node_ids: set[int] | None = None
    ) -> "Tree":
        """
        :param root_node_id: ID of the root node
        :param visited_node_ids: Accumulating list of node IDs that have already been visited
        :return: Subtree rooted at the given node
        """
        # Check if we've already computed this subtree
        visited_node_ids = visited_node_ids or set()

        # Create a new GraphWrapper object to store the accumulating tree
        subtree = Tree()
        root_node = self.get_node(root_node_id)
        subtree_root_node = deepcopy(root_node)
        subtree.add_node(subtree_root_node)

        # Mark the node as visited
        visited_node_ids.add(root_node_id)

        # BFS recursively
        for edge_id in self.get_outgoing_edge_ids(root_node_id):
            edge = self.get_edge(edge_id)
            next_node_id = edge.get_dst_id()
            if next_node_id in visited_node_ids:
                continue

            # Get the next subgraph, then add the connecting edge, and subgraph to the accumulating subgraph
            next_subgraph = self.get_subtree(edge.get_dst_id(), visited_node_ids)

            # Deep copy the graph components into the accumulating subgraph
            for new_node in next_subgraph.get_nodes():  # Nodes need to be added first
                subtree.add_node(deepcopy(new_node))

            # Add edge to the accumulating subgraph
            subtree.add_edge(deepcopy(edge))
            for new_edge in next_subgraph.get_edges():
                subtree.add_edge(deepcopy(new_edge))

        return subtree

    def init_node_stats(self, root_node_id: int, depth: int) -> None:
        # initialize tree stat lookup
        edges = self.get_outgoing_edge_ids(root_node_id)
        if len(edges) == 0:
            self.__node_stats[root_node_id] = NodeStats(height=0, size=1, depth=depth)
            return

        size = 1
        heights_of_subtrees = []
        for edge_id in edges:
            edge = self.get_edge(edge_id)
            dst_id = edge.get_dst_id()
            self.init_node_stats(dst_id, depth + 1)
            stats = self.get_node_stats(dst_id)
            size += stats.size
            heights_of_subtrees.append(stats.height)
        height = 1 + max(heights_of_subtrees)

        self.__node_stats[root_node_id] = NodeStats(
            height=height, size=size, depth=depth
        )

    def get_node_stats(self, node_id: int):
        return self.__node_stats[node_id]

    # Wrapper functions
    def get_edges(self) -> list[Edge]:
        return list(self.__edges.values())

    def get_nodes(self) -> list[Node]:
        return list(self.__nodes.values())

    def add_edge(self, edge: Edge) -> None:
        assert self.__edges.get(edge.get_id()) is None
        assert (
            self.get_node(edge.get_src_id()) is not None
        ), f"Edge {edge.get_id()} has no source in graph"
        assert (
            self.get_node(edge.get_dst_id()) is not None
        ), f"Edge {edge.get_id()} has no destination in graph"
        edge_id = edge.get_id()

        # Add edge to graph and lookup
        self.__edges[edge_id] = edge
        self.__incoming_lookup[edge.get_dst_id()].add(edge_id)
        self.__outgoing_lookup[edge.get_src_id()].add(edge_id)

    def add_node(self, node: Node) -> None:
        node_id = node.get_id()
        assert self.__nodes.get(node_id) is None
        self.__nodes[node_id] = node
        self.__incoming_lookup[node_id] = set()
        self.__outgoing_lookup[node_id] = set()

    def remove_node(self, node: Node) -> None:
        # Removes node from graph and lookup
        node_id = node.get_id()
        assert self.__nodes.get(node_id) is not None
        self.__nodes.pop(node_id)
        self.__incoming_lookup.pop(node_id)
        self.__outgoing_lookup.pop(node_id)

    def remove_edge(self, edge: Edge) -> None:
        # Removes edge from graph and lookup
        edge_id = edge.get_id()
        self.__edges.pop(edge_id)
        self.__incoming_lookup[edge.get_dst_id()].remove(edge_id)
        self.__outgoing_lookup[edge.get_src_id()].remove(edge_id)

    def get_next_node_id(self) -> int:
        return max([node_id for node_id in self.__nodes.keys()]) + 1

    def get_next_edge_id(self) -> int:
        return max([edge_id for edge_id in self.__edges.keys()]) + 1

    def get_outgoing_edge_ids(self, node_id: int) -> list[int]:
        return list(self.__outgoing_lookup[node_id])

    def get_incoming_edge_ids(self, node_id: int) -> list[int]:
        return list(self.__incoming_lookup[node_id])

    # Step 1. Original graph
    def original_graph(self) -> None:
        pass

    # Step 2. Break cycles: Invert all outgoing edges from files/IPs
    def __invert_outgoing_file_edges(self) -> None:
        edges_to_invert = []
        for node in self.get_nodes():
            if node.get_type() == NodeType.PROCESS_LET:
                continue
            edges_to_invert.extend(self.get_outgoing_edge_ids(node.get_id()))

        for edge_id in edges_to_invert:
            edge = self.get_edge(edge_id)
            assert edge.get_src_id() != edge.get_dst_id()

            self.remove_edge(
                edge
            )  # Remove then re-add edge to update adjacency lookups
            edge.invert()  # Flip source and destination
            self.add_edge(edge)

        # Assert there are no system resources w/ outgoing edges
        for node in self.get_nodes():
            if node.get_type() == NodeType.PROCESS_LET:
                continue
            assert len(self.get_outgoing_edge_ids(node.get_id())) == 0

    # The graph is now a directed acyclic graph - Dr. De
    # Step 3. Remove lattice structure: Duplicate file/IP nodes for each incoming edge
    def __duplicate_file_ip_leaves(self) -> None:
        nodes = self.get_nodes().copy()
        for node in nodes:
            # Get incoming edges of original node
            incoming_edge_ids = self.get_incoming_edge_ids(node.get_id()).copy()

            # If this is a process, or if the file/ip has 0/1 incoming, just skip.
            if node.get_type() == NodeType.PROCESS_LET or len(incoming_edge_ids) < 2:
                continue

            # Duplicate node for each incoming edge
            for edge_id in incoming_edge_ids:
                # Create new node
                new_node_id = (
                    self.get_next_node_id()
                )  # Modify node_id -> to keep track of the original node_id
                new_node = deepcopy(node)
                new_node.set_id(new_node_id)
                self.add_node(new_node)

                # Point edge to the new node ID
                edge = self.get_edge(edge_id)
                self.remove_edge(edge)
                edge.set_dst_id(new_node_id)
                self.add_edge(edge)  # This function modifies the graphs' adjacency maps

            # Remove original node
            self.remove_node(node)

    # Step 4. Convert forest to a tree
    def __add_virtual_root(self) -> None:
        agent_id = self.get_nodes()[0].node.model_extra[
            "AGENT_ID"
        ]  # AgentID is always the same for DARPA
        # Create root node
        raw_root_node = RawNode(
            _id=self.get_next_node_id(),
            TYPE=NodeType.VIRTUAL,
        )
        raw_root_node.model_extra["EXE_NAME"] = "VIRTUAL"
        raw_root_node.model_extra["CMD"] = "VIRTUAL"
        raw_root_node.model_extra["_label"] = "VIRTUAL"
        raw_root_node.model_extra["AGENT_ID"] = agent_id

        root_node = Node(raw_root_node)
        self.add_node(root_node)

        self.root_node_id = root_node.get_id()

        # Add disjoint trees to root's children
        for node in self.get_nodes():
            # If this is a virtual node, or if it's not a root node, skip

            non_self_cycle_incoming_edges = []
            for edge_id in self.get_incoming_edge_ids(node.get_id()):
                edge = self.get_edge(edge_id)
                if edge.get_src_id() == edge.get_dst_id():
                    continue
                non_self_cycle_incoming_edges.append(edge_id)

            if len(non_self_cycle_incoming_edges) > 0 or node is root_node:
                continue

            # Create edge from virtual root to subtree root
            self.add_edge(
                Edge(
                    RawEdge(
                        _id=self.get_next_edge_id(),
                        _outV=root_node.get_id(),
                        _inV=node.get_id(),
                        OPTYPE="FILE_EXEC",
                        _label="FILE_EXEC",
                        EVENT_START=0,
                    )
                )
            )

    # Sanity check for the tree: Verify it is a valid tree
    def assert_valid_tree(self):
        # A valid tree must have at least one node
        assert self.__nodes is not None and len(self.__nodes) > 0

        # Find the root: a node with no incoming edges
        root_candidates = [
            node_id
            for node_id, edges in self.__incoming_lookup.items()
            if len(edges) == 0
        ]

        # There must be exactly one root node
        assert len(root_candidates) == 1
        root = root_candidates[0]

        visited = set()

        def is_tree(node_id: int, path: list | None = None):
            if path is None:
                path = []
            node = self.get_node(node_id)
            path = path + [f"[{node_id}: {node.get_token()}]"]

            assert (
                node_id not in visited
            ), f"Found a cycle: {get_cycle(path)}. Incoming: {[self.get_edge(e).get_token() for e in self.get_incoming_edge_ids(node_id)]}"

            visited.add(node_id)
            for edge_id in self.get_outgoing_edge_ids(node_id):
                edge = self.get_edge(edge_id)
                next_node_id = edge.get_dst_id()

                if not is_tree(next_node_id, path + [f"--{edge.get_token()}->"]):
                    return False
            return True

        # Start DFS from root to check for cycles and if all nodes are reachable
        assert is_tree(root), "Not all nodes are reachable"

        # Check if all nodes were visited (tree is connected)
        assert len(visited) == len(
            self.__nodes
        ), f"Visited {len(visited)}/{len(self.__nodes)}, tree is NOT connected"

    __preprocess_steps: list[Callable] = [
        original_graph,
        __invert_outgoing_file_edges,
        __duplicate_file_ip_leaves,
        __add_virtual_root,
    ]

    def preprocess(self, output_dir: Path = None) -> "Tree":
        for i, step in enumerate(self.__preprocess_steps):
            step(self)
            if output_dir is not None:
                with open(
                    output_dir / f'{i + 1}_{step.__name__.strip("_")}.json',
                    "w",
                    encoding="utf-8",
                ) as output_file:
                    output_file.write(self.to_json())
                self.to_dot().save(
                    output_dir / f'{i + 1}_{step.__name__.strip("_")}.dot'
                )

        return self

    def prune_tree(self, root_node_id: int) -> "Tree":
        # Create subtree graph
        subtree: Tree = self.get_subtree(root_node_id)
        subtree.graph_id = self.graph_id
        num_roots = 0
        # Remove all subtree nodes and elements from the parent graph
        for edge in subtree.get_edges():
            self.remove_edge(edge)
        for node in subtree.get_nodes():
            if node.get_id() == root_node_id:
                num_roots += 1
                continue  # We want to keep this node, so we can replace later
            self.remove_node(node)

        # Add the root edge and parent to the subtree, so we can preserve the edge-node relationship
        # Make sure this happens after removing nodes, so we don't remove the edge and its parent
        incoming_edge_ids = self.get_incoming_edge_ids(root_node_id)
        assert (
            len(incoming_edge_ids) == 1
        ), f"Pruned tree should have 1 incoming edge, found {len(incoming_edge_ids)}"
        root_edge_id = incoming_edge_ids[0]
        root_edge = self.get_edge(root_edge_id)
        root_edge_source = self.get_node(root_edge.get_src_id())
        subtree.add_node(root_edge_source)
        subtree.add_edge(root_edge)

        # Sanity checks on the tree's state
        assert num_roots == 1, f"Expected 1 root, got {num_roots}"

        assert (
            len(self.get_incoming_edge_ids(root_node_id)) == 1
        ), f"Expected 1 outgoing edge, got {len(self.get_outgoing_edge_ids(root_node_id))}"
        subtree_root = subtree.get_node(root_node_id)
        assert subtree_root is not None
        assert len(subtree.get_incoming_edge_ids(root_node_id)) == 1

        return subtree

    def path_to_string(self, path: list[int]) -> str:
        tokens = []
        for edge_id in path:
            edge = self.get_edge(edge_id)
            node = self.get_node(edge.get_src_id())
            tokens.extend([node.get_token(), edge.get_token()])

        return " ".join(tokens)

    def get_node(self, node_id: int) -> Node:
        node = self.__nodes.get(node_id)
        assert node is not None, f"Node {node_id} does not exist"
        return node

    def get_edge(self, edge_id: int) -> Edge:
        edge = self.__edges.get(edge_id)
        assert edge is not None, f"Edge {edge_id} does not exist"
        return edge

    def replace_node_with_tree(self, node_id_to_replace: int, graph: "Tree") -> None:
        """
        Attach a subtree to the destination of the given edge.
        ex: X is the node to replace.
            A -e-> X
        The tree's root MUST have out-degree 0, so it can be represented as this:
            R -f-> T
        e is replaced with f, X with T, and A is ignored. Result:
            A -f-> T
        @param node_id_to_replace: node to replace with subtree
        @param graph: subtree to replace with
        """
        # self.assert_valid_tree()
        node_id_translation = {}
        edge_id_translation = {}
        # Update node IDs to avoid collision in the current graph
        orphan_nodes = []
        for old_node in graph.get_nodes():
            # Copy the node, and give it a new ID
            old_node_id = old_node.get_id()
            node = deepcopy(old_node)
            node_id = self.get_next_node_id()
            node.set_id(node_id)

            # If the node is an orphan, it's the root R, so keep track of it.
            if len(graph.get_incoming_edge_ids(old_node_id)) == 0:
                orphan_nodes.append(node)

            # Add the ID to the lookup
            assert node_id_translation.get(old_node_id) is None
            node_id_translation[old_node_id] = node_id
            self.add_node(node)

            # Mark the node to indicate it's been added after the fact
            node.marked = True

        # There should only be one orphan/root node (R)
        assert (
            len(orphan_nodes) == 1
        ), f"Expected 1 orphan node, got {len(orphan_nodes)}/{len(graph.get_nodes())}"
        R = orphan_nodes[0]

        # Update edge IDs in the subtree to avoid collision in the current graph, and bring up to date with node IDs
        for old_edge in graph.get_edges():
            # Copy the edge, and give it a new ID
            edge = deepcopy(old_edge)
            new_edge_id = self.get_next_edge_id()
            edge.set_id(new_edge_id)
            # Update the edge's node IDs to match the new graph
            assert node_id_translation.get(edge.get_src_id()) is not None
            assert node_id_translation.get(edge.get_dst_id()) is not None
            edge.translate_node_ids(node_id_translation)
            assert self.get_node(edge.get_src_id()) is not None
            assert self.get_node(edge.get_dst_id()) is not None

            # Add the ID to the lookup
            edge_id_translation[old_edge.get_id()] = new_edge_id
            self.add_edge(edge)
            assert new_edge_id in self.get_outgoing_edge_ids(edge.get_src_id())
            assert new_edge_id in self.get_incoming_edge_ids(edge.get_dst_id())
            # Mark the edge to indicate it's added after the fact
            edge.marked = True

        # A -e-> X  original self
        # R -f-> T  tree to add
        # A -f-> T  new self

        # find f
        outgoing_edges = [
            self.get_edge(e) for e in self.get_outgoing_edge_ids(R.get_id())
        ]
        assert len(outgoing_edges) == 1
        edge_f = outgoing_edges[0]

        # Remove -e-> X from the graph
        X = self.get_node(node_id_to_replace)
        incoming_edges = self.get_incoming_edge_ids(node_id_to_replace)
        assert len(incoming_edges) == 1
        edge_e_id = incoming_edges[0]
        edge_e = self.get_edge(edge_e_id)
        self.remove_edge(edge_e)  # Remove edge before node to preserve graph state
        self.remove_node(X)

        # Remove R from the graph
        self.remove_node(R)
        # -f-> T is already in the graph, so attach it to A and we're done
        A_id = edge_e.get_src_id()
        edge_f.set_src_id(A_id)
        self.__outgoing_lookup[A_id].add(edge_f.get_id())  # HACK: not good either
        assert edge_f.get_id() in self.get_outgoing_edge_ids(edge_e.get_src_id())
        self.assert_valid_tree()

    def size(self) -> int:
        return len(self.__nodes)

    def add_stat(self, stat: str, value: float):
        if stat not in self.stats:
            self.stats[stat] = []
        self.stats[stat].append(value)

    # Exporter functions
    def to_dot(self) -> gv.Digraph:
        dot_graph = gv.Digraph()
        dot_graph.attr(rankdir="LR")
        included_nodes: set[Node] = set()
        sorted_edges = sorted(self.get_edges(), key=lambda e: e.get_time())

        def add_to_graph(new_node: Node):
            assert new_node is not None, "Trying to add a null node to the graph"
            included_nodes.add(new_node)
            dot_graph.node(str(new_node.get_id()), **new_node.to_dot_args())

        num_missing = 0
        num_null = 0
        for edge in sorted_edges:
            src_id, dst_id = edge.get_src_id(), edge.get_dst_id()
            assert src_id is not None, f"Edge {edge.get_id()} has no source"
            assert dst_id is not None, f"Edge {edge.get_id()} has no destination"
            src, dst = self.get_node(src_id), self.get_node(dst_id)
            add_to_graph(src)
            add_to_graph(dst)

            dot_graph.edge(str(src_id), str(dst_id), **edge.to_dot_args())

        if num_missing > 0:
            print(
                f"Warn: {num_missing} MIA, {num_null} null out of {len(self.get_edges())}?"
            )
        for node in self.get_nodes():
            if node not in included_nodes:
                add_to_graph(node)

        return dot_graph

    def to_nx(self) -> nx.DiGraph:
        digraph: nx.DiGraph = nx.DiGraph()
        # NetworkX node IDs must index at 0
        node_ids = {node.get_id(): i for i, node in enumerate(self.get_nodes())}
        for node in self.get_nodes():
            digraph.add_node(node_ids[node.get_id()], feature=node.get_token())
        for edge in self.get_edges():
            src, dst = edge.get_src_id(), edge.get_dst_id()
            if src is not None and dst is None:
                continue
            digraph.add_edge(node_ids[src], node_ids[dst], feature=edge.get_token())
        return digraph

    def assert_complete(self) -> None:
        for edge in self.get_edges():
            assert (
                edge.get_src_id() is not None
            ), f"Edge {edge.get_id()} ({edge.get_token()} has None source"
            assert (
                edge.get_dst_id() is not None
            ), f"Edge {edge.get_id()} ({edge.get_token()} has None destination"
            if self.get_node(edge.get_src_id()) is None:
                print(f"Edge {edge.get_id()} ({edge.get_token()}) has no source")
            assert (
                self.get_node(edge.get_dst_id()) is not None
            ), f"Edge {edge.get_id()} ({edge.get_token()}) has no destination"
        for node in self.get_nodes():
            node_id = node.get_id()
            assert node.get_id() is not None, f"Node {node.get_token()} has None ID"
            for edge_id in self.get_incoming_edge_ids(node_id):
                edge = self.get_edge(edge_id)
                assert edge.get_dst_id() == node_id, (
                    f"Node {node_id} has incoming edge {edge_id} "
                    f"with wrong destination ({edge.get_src_id()} -> {edge.get_dst_id()})"
                )
            for edge_id in self.get_outgoing_edge_ids(node_id):
                edge = self.get_edge(edge_id)
                assert (
                    edge.get_src_id() == node_id
                ), f"Node {node.get_token()} has outgoing edge {edge_id} with wrong source"

    def to_json(self) -> str:
        return json.dumps(
            {
                "mode": "EXTENDED",
                "vertices": [node.to_json_dict() for node in self.get_nodes()],
                "edges": [edge.to_json_dict() for edge in self.get_edges()],
            }
        )

    def get_tree_height(self, root_node_id) -> int:
        return self.__node_stats[root_node_id].height

    def get_root(self) -> int:
        root_ids = [
            node_id
            for node_id in self.__nodes.keys()
            if len(self.get_incoming_edge_ids(node_id)) == 0
        ]
        assert len(root_ids) == 1, f"Expected only 1 root, got {len(root_ids)}"
        return root_ids[0]

    def get_stats(self) -> "TreeStats":
        self.__node_stats = {}  # HACK:  this is not good
        self.init_node_stats(self.get_root(), 0)
        self.assert_valid_tree()
        heights = []
        depths = []
        sizes = []
        degrees = []
        G = self.to_nx().to_undirected()
        diameter = max([max(j.values()) for (_, j) in nx.shortest_path_length(G)])
        del G

        for node_id in self.__nodes.keys():
            assert len(self.__node_stats) == len(
                self.__nodes
            ), f"{len(self.__node_stats)}, {len(self.__nodes)}"
            assert node_id in self.__nodes, f"Node {node_id} doesnt exist"
            assert node_id in self.__node_stats, self.get_incoming_edge_ids(node_id)
            stat = self.get_node_stats(node_id)
            heights.append(stat.height)
            depths.append(stat.depth)
            sizes.append(stat.size)
            degrees.append(len(self.get_outgoing_edge_ids(node_id)))

        return TreeStats(
            heights=heights,
            depths=depths,
            sizes=sizes,
            degrees=degrees,
            height=max(heights),
            size=max(sizes),
            degree=max(degrees),
            diameter=diameter,
        )
