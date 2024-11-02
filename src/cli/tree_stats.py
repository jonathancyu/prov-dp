import argparse
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


def calculate_stats(input_path: Path) -> TreeStats:
    tree = Tree.load_file(input_path)
    stats = tree.get_stats() # BUG: broken, but also we don't use this script ...
    del tree
    return stats


def main(args):
    input_dir: Path = args.input_dir
    print(f"Begining {input_dir}")
    input_paths: list[Path] = list(input_dir.rglob("nd*.json"))
    stats: list[TreeStats] = list(
        smart_map(
            func=calculate_stats,
            items=input_paths,
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

    for stat, values in tree_stats.items():
        print(
            f"tree {stat}: avg= {sum(values)/len(values):.4f}, min= {min(values):.4f}, max= {max(values):.4f}"
        )

    for stat, values in node_stats.items():
        print(
            f"node_stats {stat}: avg= {sum(values)/len(values):.4f}, min= {min(values):.4f}, max= {max(values):.4f}"
        )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-i", "--input_dir", help="Input directory", type=Path, required=True
    )

    args.add_argument("-s", "--single-threaded", action="store_true")

    main(args.parse_args())
