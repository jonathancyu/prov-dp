from pathlib import Path
from src.algorithm.extended_top_m_filter import ExtendedTopMFilter
from src.algorithm.wrappers.graph import Graph


class TestExtendedTopMFilter:
    def test_run_config(self):
        data_path = (
            Path.home()
            / "workspace"
            / "SyssecLab"
            / "differential-privacy"
            / "data"
            / "attack_graphs"
            / "tc3-fived"
            / "3.10"
        )
        output_dir = Path("./output")
        benign_graph_paths: list[Path] = list(data_path.rglob("nd*json"))

        processor = ExtendedTopMFilter(epsilon=1, delta=0.5, single_threaded=True)
        for input_path in benign_graph_paths:
            output_path = output_dir / f"1{input_path.stem}"
            graph = Graph.load_file(input_path)
            graph.write_dot(output_path)

            processor.filter_graph(graph)
            graph.write_dot(output_dir / f"2{input_path.stem}")
