import json
from pathlib import Path
from src.algorithm.graph_processor import GraphProcessor
from src.graphson.json_to_csv import mapJSONToCSV


class TestRunConfigurations:
    def test_tc3_theia(self):
        data_path = (
            Path.home() / "workspace" / "SyssecLab" / "differential-privacy" / "data"
        )
        input_path: Path = data_path / "benign_graphs" / "tc3-theia" / "firefox" / "nd"
        output_path: Path = data_path / "output"

        graph_processor = GraphProcessor(
            output_dir=output_path / "tc3-theia" / "data2" / "benign",
            reattach_mode="bucket",
            epsilon_1=0.01,
            epsilon_2=1,
            alpha=0,
            beta=1,
            gamma=0,
            single_threaded=True,
        )

        input_paths: list[Path] = list(input_path.rglob("nd*.json"))[:10]
        perturbed_graphs = list(graph_processor.perturb_graphs(input_paths))

        for tree in perturbed_graphs:
            tree.assert_valid_tree()

        # Write stats to json
        stat_path = output_path / "processor_stats.json"
        print(f"Writing stats to {stat_path}")
        with open(stat_path, "w") as f:
            f.write(json.dumps(graph_processor.stats))

        # Json to csv
        for json_path in list(output_path.rglob("nd*.json")):
            mapJSONToCSV(json_path, json_path.parent)
