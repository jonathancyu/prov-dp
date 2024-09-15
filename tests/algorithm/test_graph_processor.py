import json
import shutil
from pathlib import Path
from src.algorithm.graph_processor import GraphProcessor
from src.cli.add_csv import add_csv_to_json
from src.cli.utility import save_dot


class TestRunConfigurations:
    def test_convert_tree_and_back(self):
        data_path = (
            Path.home() / "workspace" / "SyssecLab" / "differential-privacy" / "data"
        )
        graph_pattern = "nd*json"
        benign_graphs: list[Path] = list(
            (data_path / "benign_graphs").rglob(graph_pattern)
        )
        graph_path = benign_graphs[0]
        print("PATH: " + str(graph_path))
        output_path = Path("./output") / graph_path.stem
        output_path.mkdir(parents=True, exist_ok=True)

        # Original
        shutil.copyfile(graph_path, output_path / "1_original_graph.json")
        tree = GraphProcessor.load_tree_from_file(graph_path)

        # Tree
        with open(output_path / "2_original_tree.json", "w") as f:
            f.write(tree.to_json())
        save_dot(tree.to_dot(), output_path / "2_original_tree.dot")

        # Perturbed
        processor = GraphProcessor()
        perturbed_graphs = processor.perturb_graphs([graph_path])
        with open(output_path / "3_modified_tree.json", "w") as f:
            f.write(perturbed_graphs[0].to_json())
        save_dot(perturbed_graphs[0].to_dot(), output_path / "3_modified_tree.dot")

        # Graph
        graph: Graph = perturbed_graphs[0].revert_to_graph()
        with open(output_path / "4_reverted_graph.json", "w") as f:
            f.write(graph.to_json())
        save_dot(graph.to_dot(), output_path / "4_reverted_graph.dot")
    def test_tc3_theia(self):
        data_path = (
            Path.home() / "workspace" / "SyssecLab" / "differential-privacy" / "data"
        )
        input_path: Path = data_path / "benign_graphs" / "tc3-theia" / "firefox" / "nd"
        output_path: Path = data_path / "output"

        graph_processor = GraphProcessor(
            output_dir=output_path / "tc3-theia" / "perturbed" / "benign",
            epsilon=1,
            delta=0.5,
            alpha=0,
            beta=1,
            gamma=0,
            single_threaded=True,
        )

        input_paths: list[Path] = list(input_path.rglob("nd*.json"))[:10]
        perturbed_graphs = list(graph_processor.perturb_graphs(input_paths))

        for tree in perturbed_graphs:
            tree.assert_valid_tree()
            base_file_name = f"nd_{tree.graph_id}_processletevent"
            file_path = output_path / base_file_name / f"{base_file_name}.json"
            save_dot(tree.to_dot(), file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f:
                f.write(tree.to_json())

        # Write stats to json
        stat_path = output_path / "processor_stats.json"
        print(f"Writing stats to {stat_path}")
        with open(stat_path, "w") as f:
            f.write(json.dumps(graph_processor.stats))

        # Json to csv
        # for json_path in list(output_path.rglob("nd*.json")):
        #     add_csv_to_json(json_path)
