# Overview
This repository contains the library and notebooks related to applying differentially private perturbation to 
provenance graphs.
## Quick Start
First, run `pip install -r requirements.txt`.
To run the graph perturbation pipeline, run the following. More information on the arguments can be found in `main.py`
```shell
python main.py -i ../data/benign_graphs/tc3-theia/firefox/nd -o ../data/output/tc3-theia/data2/benign --epsilon 1 --delta 0.75 --alpha 0.1 --num_epochs 1000 --prediction_batch_size 5
```
To run the graph preprocessing pipeline, use the following.
The `-p` flag is used to make the output directory match the folder structure of the input directory.
```shell
python preprocess_graphs.py -i ../data/attack_graphs/ -o ../data/prov_dp/output/anomaly -p
```
