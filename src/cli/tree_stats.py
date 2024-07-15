import argparse
import json
from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from src import Tree
from src.algorithm.utility import smart_map
from src.algorithm.wrappers.tree import TreeStats


def graph(data: list, bins: int, stat: str, output_dir: Path):
    plt.figure()
    sns.histplot(data=data, bins=bins)
    plt.xlabel("Count")
    plt.ylabel(stat)
    plt.title(f"{stat} distribution")
    plt.savefig(str(output_dir / f"{stat}.pdf"))


def main(args):
    input_dir: Path = args.input_dir
    input_paths: list[Path] = list(input_dir.rglob("nd*.json"))
    trees: list[Tree] = list(
        smart_map(
            func=Tree.load_file,
            items=input_paths,
            single_threaded=args.single_threaded,
            desc="Loading trees",
        )
    )
    stats: list[TreeStats] = list(
        smart_map(
            func=Tree.get_stats,
            items=trees,
            single_threaded=args.single_threaded,
            desc="Calculating stats",
        )
    )

    node_stats = {"heights": [], "depths": [], "sizes": [], "degrees": []}

    tree_stats = {"heights": [], "sizes": [], "degrees": [], "diameters": []}
    for stat in tqdm(stats, desc="Aggregating stats"):
        # Node stats
        node_stats["heights"].extend(stat.heights)
        node_stats["depths"].extend(stat.depths)
        node_stats["sizes"].extend(stat.sizes)
        node_stats["degrees"].extend(stat.degrees)

        # Tree stats
        tree_stats["heights"].append(stat.height)
        tree_stats["sizes"].append(stat.size)
        tree_stats["degrees"].append(stat.degree)
        tree_stats["diameters"].append(stat.diameter)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for stat, values in tree_stats.items():
        graph(values, args.num_bins, stat, args.output_dir)

    with open(output_dir / "dataset_stats.json", "w") as file:
        file.write(json.dumps({"node_stats": node_stats, "tree_stats": tree_stats}))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-i", "--input_dir", help="Input directory", type=Path, required=True
    )
    args.add_argument(
        "-o",
        "--output_dir",
        help="Output directory for figures",
        type=Path,
        default=Path("./output"),
    )
    args.add_argument(
        "-n", "--num_bins", help="Number of bins for histogram", type=int, default=25
    )

    args.add_argument("-s", "--single-threaded", action="store_true")

    main(args.parse_args())
