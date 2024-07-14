from pathlib import Path
from src.algorithm.graph_processor import GraphProcessor


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
            epsilon_1=0.1,
            epsilon_2=0.1,
            alpha=0.5,
            beta=0.5,
            gamma=0.5,
            single_threaded=True,
        )

        input_paths: list[Path] = list(input_path.rglob("nd*.json"))[:10]
        perturbed_graphs = list(graph_processor.perturb_graphs(input_paths))
        for tree in perturbed_graphs:
            tree.assert_valid_tree()
