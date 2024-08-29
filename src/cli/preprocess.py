import argparse
from pathlib import Path

from tqdm import tqdm

from src.algorithm import GraphProcessor
from .utility import save_dot


# TODO: Need to add a param limit so we can ensure the preprocessed uses the same
# random sample as the perturbed
def main(args):
    input_paths = list(args.input_dir.rglob("*.json"))

    # Run graph processor
    tree_shaker = GraphProcessor()
    trees = tree_shaker.preprocess_graphs(input_paths)
    if args.output_dir is None:
        return

    for path, tree in tqdm(
        zip(input_paths, trees), total=len(trees), desc="Exporting to json"
    ):
        file_name = path.name.replace("-", "_").replace(".json", "")
        if args.preserve_structure:
            parent_dir = path.parent.parent.parent.relative_to(args.input_dir)
            dir_name = path.parent.name.replace("-", "_").replace(".json", "")
            output_path = args.output_dir / parent_dir / dir_name / file_name
        else:
            output_path = args.output_dir / file_name / f"{file_name}.json"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, "w") as f:
            f.write(tree.to_json())
        save_dot(tree.to_dot(), output_path)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-i", "--input_dir", type=Path, help="Path to input graph directory"
    )
    arg_parser.add_argument(
        "-N",
        "--num_graphs",
        type=int,
        default=None,
        help="Limit the number of graphs to process",
    )
    arg_parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        required=False,
        default=None,
        help="Path to output graph directory",
    )
    arg_parser.add_argument(
        "-p",
        "--preserve-structure",
        action="store_true",
        default=False,
        help="Preserve the input directory structure",
    )

    main(arg_parser.parse_args())
