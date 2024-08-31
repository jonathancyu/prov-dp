import argparse
import json
from pathlib import Path
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd


def graph(data: list, bins: int, stat: str, output_dir: Path, base_name: str = ""):
    plt.figure()
    sns.histplot(
        data=data,
        bins=bins,
    )
    sns.despine(left=True, bottom=True)
    sns.set_theme(style="white")
    plt.ylabel("Count")
    plt.xlabel(stat)
    plt.yscale("log")
    plt.title(f"{stat} distribution")
    plt.savefig(str(output_dir / f"{stat}{base_name}.pdf"))


def to_df_row(data: dict) -> dict:
    row = {}
    for stat, samples in data.items():
        values = [float(x) for x in samples]
        row[f"avg {stat}"] = sum(values) / len(samples)
        row[f"std {stat}"] = np.std(values)
        row[f"min {stat}"] = min(values)
        row[f"max {stat}"] = max(values)
    return row


def main(input_dir: Path, output_dir: Path):
    rows = []
    for file_path in input_dir.rglob("processor_stats.json"):
        print(file_path)
        row = {}
        parent = file_path.parent
        parameter_str = parent.name[len("perturbed_") :]
        for param in parameter_str.split("_"):
            split = param.split("=")
            assert len(split) == 2
            row[split[0]] = float(split[1])
        row["delta"] = row["e1"] / (row["e1"] + row["e2"])
        print(row)

        # Load json
        with open(file_path, "r") as f:
            row_data = json.load(f)
            for key, value in row_data.items():
                row[key] = value

        # Add to df
        rows.append(to_df_row(row_data))

        # Pruned subtree distribution histogram
        subtree_sizes = row["pruned tree size (#nodes)"]
        assert subtree_sizes is not None
        graph(
            subtree_sizes,
            50,
            "pruned tree size (#nodes)",
            output_dir,
            base_name=f"{parent.parent.name}_{parameter_str}",
        )
    with open(input_dir / "all_processor_stats.json", "w") as f:
        f.write(json.dumps(rows))

    df = pd.DataFrame(rows)
    print(df.head())


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-i", "--input_dir", type=Path, help="Input directory path", required=True
    )
    arg_parser.add_argument(
        "-o", "--output_dir", type=Path, help="Output directory path", required=True
    )
    args = arg_parser.parse_args()
    main(args.input_dir, args.output_dir)
