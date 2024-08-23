# Overview
This repository contains the library and notebooks related to applying differentially private perturbation to 
provenance graphs.

## Quick Start
1. Create a new virtual environment: `pip -m venv venv`
2. Activate virtual environment: `source venv/bin/activate`
3. Install dependencies via `pip install -r requirements.txt`

Note: Scripts are ran using `python -m` to avoid having to manipulate the `PYTHONPATH` environment variable.

## Convert graphs to trees
To run the graph preprocessing pipeline, use the following.
The `-p` flag is used to make the output directory match the folder structure of the input directory.
```shell
python -m src.cli.preprocess \
  -i ../data/attack_graphs/ \
  -o ../data/output/data1/anomaly -p
```

## Perturb graphs
To run the graph perturbation pipeline, run the following. More information on the arguments can be found in the 
`parse_args` function in `source/cli/utility`
```shell
python -m src.cli.perturb \
  -i ../data/benign_graphs/tc3-theia/firefox/nd \
  -o output/tc3-theia/data2/benign \
  --epsilon_1 1 \
  --epsilon_2 1 \
  --alpha 0.1 \
  --beta 0.1 \
  --gamma 0.1
  ```

## Calculate tree statistics
This script calculates various statistics for trees inside the specified directory.
```shell
python -m src.cli.tree_stats \
  -i ../data/attack_graphs \
  -o output/tc3-theia/data1/benign -s
```

## Convert to Csv
Given an input directory, for each graph .json file, extract the graphs' features into csv files inside of the files' directory.
These csv files are then used as input to the GNN pipeline.
```shell
python -m src.cli.add_csv \
  -i ../data/output/tc3-theia/data2/
```

# Project overview
## src.algorithm
This package contains the core logic of the project.
- `graph_processor.py` is responsible for loading files, pruning trees, and then reattaching the pruned trees according to differential privacy. The core of the differential privacy pipeline can be called using `GraphProcessor.perturb_graphs`.
## src.algorithm.wrappers
- `edge.py, node.py` contains simple wrappers for `src.graphson.raw_edge` and `src.graphson.raw_node`, respectively.
- `tree.py` contains the graph-to-tree converson logic, as well as functions to help prune and re-attach subtrees.
## src.cli
This package contains CLI wrappers to interact with the `algorithm` package.
- `perturb.py` - Run the graph processing pipeline
- `batch_perturb.py` - Run the graph processing pipeline multiple times across different permutations of differential privacy settings.
- `preprocess.py` - Convert provenance graphs into trees
- `tree_stats.py` - Generate statistics and figures from the graph-to-tree conversion process.
- `add_csv.py` - Convert json graph to csv format for input into the GNN
## src.graphson
This package contains simple Pydantic models used to serialize graphs to and from json.

