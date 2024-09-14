import gc
import inspect
import json
import pickle
import random

from tqdm import tqdm

from src import GraphProcessor, Tree
from src.cli.utility import parse_args, save_dot


def run_processor(args):
    input_paths = list(args.input_dir.rglob("nd*.json"))
    # Apply graph limit
    if args.num_graphs is not None:
        input_paths = random.sample(input_paths, args.num_graphs)
        args.output_dir = args.output_dir.with_stem(
            f"{args.output_dir.stem}_N={args.num_graphs}"
        )
    args.output_dir = args.output_dir.with_stem(
        f"{args.output_dir.stem}"
        f"_e={args.epsilon}"
        f"_d={args.delta}"
        f"__a={args.alpha}"
        f"_b={args.beta}"
        f"_c={args.gamma}"
    )
    print(
        f"Started run with input {args.input_dir}, and parameters {args.output_dir.name}"
    )

    # Run graph processor
    graph_processor = GraphProcessor(**to_processor_args(args))
    perturbed_graphs: list[Tree] = graph_processor.perturb_graphs(input_paths)

    # Save dot files
    for graph in tqdm(perturbed_graphs, desc="Saving graphs"):
        base_file_name = f"nd_{graph.graph_id}_processletevent"
        file_path = args.output_dir / base_file_name / f"{base_file_name}.json"
        save_dot(graph.to_dot(), file_path)

        with open(file_path, "w") as f:
            f.write(graph.to_json())

    # Write stats to json
    with open(args.output_dir / "processor_stats.json", "w") as f:
        f.write(json.dumps(graph_processor.stats))

    # Clean up for the next run
    del graph_processor
    del perturbed_graphs
    gc.collect()


def to_processor_args(args):
    # Map args to GraphProcessor constructor
    parameters = inspect.signature(GraphProcessor.__init__).parameters
    processor_args = {}
    for arg, value in vars(args).items():
        if arg not in parameters:
            print(f"WARNING: {arg} not in parameters")
            continue
        processor_args[arg] = value

    return processor_args


if __name__ == "__main__":
    run_processor(parse_args())
