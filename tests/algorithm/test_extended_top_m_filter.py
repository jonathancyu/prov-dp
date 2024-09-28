from pathlib import Path
from src.algorithm.extended_top_m_filter import ExtendedTopMFilter
from src.algorithm.wrappers.graph import Graph
from src.cli.utility import save_dot


class TestExtendedTopMFilter:
    def test_run_config(self):
        data_path = (
            Path.home()
            / "workspace"
            / "SyssecLab"
            / "differential-privacy"
            / "data"
            / "benign_graphs"
            / "tc3-trace"
            / "firefox"
            / "nd"
        )
        output_dir = Path("./output")
        benign_graph_paths: list[Path] = list(data_path.rglob("nd*json"))[:1]

        processor = ExtendedTopMFilter(epsilon=1, delta=0.5, single_threaded=True)
        for input_path in benign_graph_paths:
            output_path = output_dir / f"1{input_path.stem}"
            graph = Graph.load_file(input_path)
            save_dot(graph.to_dot(), output_path)

            processor.filter_graph(graph)
            save_dot(graph.to_dot(), output_dir / f"2{input_path.stem}")
