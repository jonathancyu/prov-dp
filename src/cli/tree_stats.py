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
    plt.xlabel('Count')
    plt.ylabel(stat)
    plt.title(f'{stat} distribution')
    plt.savefig(str(output_dir / f'{stat}.pdf'))


def main(args):
    input_dir: Path = args.input_dir
    input_dir.mkdir(parents=True, exist_ok=True)
    input_paths: list[Path] = list(input_dir.rglob('nd*.json'))
    trees: list[Tree] = list(smart_map(
        func=Tree.load_file,
        items=input_paths,
        single_threaded=args.single_threaded,
        desc='Loading trees'
    ))
    stats: list[TreeStats] = list(smart_map(
        func=Tree.get_stats,
        items=trees,
        single_threaded=args.single_threaded,
        desc='Calculating stats'
    ))
    stats: list[TreeStats] = [tree.get_stats() for tree in trees]
    heights = []
    depths = []
    sizes = []
    degrees = []
    diameters = []
    for stat in tqdm(stats, desc='Aggregating stats'):
        heights.append(stat.height)
        # depths.append(stat.depth)
        sizes.append(stat.size)
        degrees.append(stat.degree)
        diameters.append(stat.diameter)

    graph(heights, args.num_bins, 'height', args.output_dir)
    graph(sizes, args.num_bins, 'size', args.output_dir)
    graph(degrees, args.num_bins, 'degree', args.output_dir)
    graph(diameters, args.num_bins, 'diameter', args.output_dir)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input_dir', help='Input directory',
                      type=Path, required=True)
    args.add_argument('-o', '--output_dir', help='Output directory for figures',
                      type=Path, default=Path('./output'))
    args.add_argument('-n', '--num_bins', help='Number of bins for histogram',
                      type=int, default=25)

    args.add_argument('-s', '--single-threaded', action='store_true')

    main(args.parse_args())
