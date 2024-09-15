from typing import Counter
from src.algorithm.graph_processor import GraphProcessor
from src.algorithm.utility import smart_map
from src.algorithm.wrappers.graph import Graph
from src.algorithm.wrappers.tree import Tree
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Result:
    path: Path
    success: bool
    message: str


def process(path: Path) -> Result:
    try:
        tree = GraphProcessor.load_tree_from_file(path)
        del tree
        return Result(path=path, success=True, message="Success")
    except AssertionError as e:
        return Result(path=path, success=False, message=str(e))


class TestTree:
    def test_convert_tree_and_back(self):
        data_path = (
            Path.home() / "workspace" / "SyssecLab" / "differential-privacy" / "data"
        )
        output_path = Path("./output")
        graph_pattern = "nd*json"
        benign_graphs: list[Path] = list(
            (data_path / "benign_graphs").rglob(graph_pattern)
        )
        graph_path = benign_graphs[0]
        print(graph_path.name)
        tree = GraphProcessor.load_tree_from_file(graph_path)
        graph: Graph = tree.revert_to_graph(output_path)
        with open(output_path / graph_path.name, "w") as f:
            f.write(graph.to_json())
        graph.assert_complete()

    def test_load_file_works_for_all_darpa_data(self):
        file = open("output.txt", "w")
        data_path = (
            Path.home() / "workspace" / "SyssecLab" / "differential-privacy" / "data"
        )
        graph_pattern = "nd*json"
        attack_graphs: list[Path] = list(
            (data_path / "attack_graphs").rglob(graph_pattern)
        )
        benign_graphs: list[Path] = list(
            (data_path / "benign_graphs").rglob(graph_pattern)
        )
        graph_paths: list[Path] = sorted(attack_graphs + benign_graphs)

        result_generator = smart_map(
            func=process, items=graph_paths, single_threaded=True
        )
        results = {True: [], False: []}
        dirs = {True: Counter(), False: Counter()}
        errors = Counter()

        for result in list(result_generator):
            success = result.success
            path = result.path.relative_to(data_path)

            if not success:
                file.write(f"Failed {path} with error\n  {result.message}\n")
                errors.update([result.message[:5]])

            results[success].append(result)
            dirs[success].update([path.parent])

        file.write(f"{len(results[True])} successes, {len(results[False])} failed\n\n")

        def format_counters(counter: Counter) -> str:
            items = [f"{k}: {v}" for k, v in counter.items()]
            return "\n".join(items)

        file.write("Errors:\n" + format_counters(errors))
        # file.write(f"\nSUCCESS:\n{format_counters(dirs[True])}\n")
        # file.write(f"\nFAILURE:\n{format_counters(dirs[False])}\n")

        file.close()

        assert len(results[False]) == 0
