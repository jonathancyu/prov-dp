import json
from pathlib import Path
from src.algorithm.graph_processor import GraphProcessor
from src.cli.add_csv import add_csv_to_json
from src.cli.utility import save_dot


class TestRunConfigurations:
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
        for json_path in list(output_path.rglob("nd*.json")):
            add_csv_to_json(json_path)
