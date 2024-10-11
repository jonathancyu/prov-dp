from argparse import Namespace
import json
from pathlib import Path
from typing import Callable, List, Sequence

from src.algorithm.extended_top_m_filter import ExtendedTopMFilter
from src.algorithm.wrappers.graph import Graph
from src.cli.configs import Config

from .perturb import parse_args
from guppy import hpy
import cowsay


def batch_run(
    fn: Callable[[Namespace], None],
    base_args: Namespace,
    configurations: Sequence[Config],
):
    # Update the arguments for each configuration and run
    h = hpy()
    for config in configurations:
        print("#" * 100)
        current_args = config.merge(base_args)
        print(current_args)
        fn(current_args)
        print(h.heap())
        break
        # Clean up for the next run
        gc.collect()


def run_etmf(
    input_dir: Path,
    output_dir: Path,
    epsilon: float,
    delta: float,
    single_threaded: bool = False,
) -> None:
    benign_graph_paths: list[Path] = list(input_dir.rglob("nd*json"))
    processor = ExtendedTopMFilter(
        epsilon=epsilon, delta=delta, single_threaded=single_threaded
    )
    for input_path in benign_graph_paths:
        graph = Graph.load_file(input_path)
        processor.filter_graph(graph)
        with open(output_dir / input_path.stem) as f:
            f.write(json.dumps(graph.to_json()))


def main(base_args):
    performers = ["fived", "trace", "theia"]
    for performer in performers:
        cowsay.cow(f"Running {performer}!")
        input_dir = Path(f"/mnt/f/data/by_performer/{performer}/benign")
        output_dir = Path(f"/mnt/f/data/by_performer_output/{performer}/perturbed")

        base_args.input_dir = input_dir
        base_args.output_dir = output_dir
        # Run tree processor
        # batch_run(
        #     fn=run_processor, base_args=base_args, configurations=TREE_CONFIGURATIONS
        # )

    # Run ETMF
    for performer in performers:
        for epsilon in [0.1, 1, 10]:
            cowsay.cow(f"ETmF {performer} epsilon={epsilon}!")
            input_dir = Path(f"/mnt/f/data/by_performer/{performer}/benign")
            output_dir = Path(f"/mnt/f/data/by_performer_output/{performer}/perturbed")

            run_etmf(
                input_dir=input_dir, output_dir=output_dir, epsilon=epsilon, delta=0.5
            )


if __name__ == "__main__":
    main(parse_args())
