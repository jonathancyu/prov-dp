# Overview
This repository contains the library and notebooks related to applying differentially private perturbation to 
provenance graphs.

## Quick Start
To begin, run `pip install -r requirements.txt`.

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
  --epsilon1 1 \
  --alpha 0.1 \
  --beta 0.1 \
  --gamma 0.1 \
  --num_epochs 1000 \
  --prediction_batch_size 5
  ```

## Calculate tree statistics
This script calculates various statistics for trees inside the specified directory.
```shell
python -m src.cli.tree_stats -i ../data/attack_graphs \
  -o output/tc3-theia/data1/benign -s
```
