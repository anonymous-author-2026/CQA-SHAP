# CQA-SHAP

This repository contains all the code and resources used in the evaluation of our paper "CQA-SHAP: Model-agnostic Explanation for Complex Query Answering over Knowledge Graphs".

## Prerequisites

### Environment Setup

We recommend using a conda environment with python `3.10`. You can use the following commands to set up the environment:

```bash
conda create -n xcqa python=3.10
```

To activate the environment, use:

```bash
conda activate xcqa
```

The list of required packages is provided in the `requirements.txt` file. You can install them using pip:

```bash
pip install -r requirements.txt
```

### Data Preparation

You can download the dataset from Google Drive using [gdown](https://github.com/wkentaro/gdown).

```bash
gdown https://drive.google.com/file/d/1JWjHSG6REn9TQ-dFAkqBjMIQSu9lptSv
```

**Note:** Our original data is based on the [CQD](https://github.com/uclnlp/cqd/) repository. However, we made a few changes to NELL dataset to have the same format as FB15k-237 for a unified data loading process. Furthermore, we enriched FB15k-237 with titles of entities based on [KNN-KG repository](https://github.com/zjunlp/KNN-KG/tree/main/dataset/FB15k-237).


After downloading, run the following command to extract the data (this will create a `data` directory):

```bash
unzip data.zip
```

## Models

Our implementation supports the following well-known CQA models. The first three models are under the same implementation known as KGReasoning. The pre-trained models for the CQD are provided by the original authors, so we use the same checkpoints. For the other three models, there are no publicly available pre-trained models, so we used the exact implementations from their respective repositories and trained them based on their guidelines to make sure the results are similar to the reported ones. 

| Model Name | Reference | 
|------------|------------|
|  BetaE | https://github.com/snap-stanford/KGReasoning |
|  Query2Box | https://github.com/snap-stanford/KGReasoning |
|  GQE | https://github.com/snap-stanford/KGReasoning |
|  CQD | https://github.com/uclnlp/cqd |

### CQD Checkpoints

To download the checkpoints for the CQD model, use the following commands:

```bash
cd models/CQD
```

```bash
gdown https://drive.google.com/file/d/1uEu9cm5tefNcYtuC383Uvswkv4bIDT9H
```

```bash
unzip models.zip
```

You should see the checkpoints (`pt` files) in the `models/CQD/models` directory.

### KGReasoning Checkpoints

To download the checkpoints for the BetaE, Query2Box, and GQE models, use the following commands:

```bash
cd models/KGReasoning
```

```bash
gdown https://drive.google.com/file/d/1hj53HoJG1Kw2NjwB7p0SzWAStRj1-eLS
```

```bash
unzip models.zip
```

After extraction, you will see two folders `FB15k-237` and `NELL` inside the `models/KGReasoning/logs` directory, each containing the model checkpoints for the respective datasets. Inside each dataset folder, you will find subfolders for each model (`betae`, `gqe`, `query2box`) containing their respective checkpoints.

Note that the training set of both the original datasets (FB15k-237 and NELL) and the harder versions (FB15k-237+H and NELL+H) are the same, so the checkpoints can be used for both versions.

## Evaluation

To reproduce the results of the necessary and sufficiency evaluation, you can use the `eval.py` script. This script takes the following arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset` | str | `FB` | Dataset name: `FB` (FB15k-237) or `NELL` (NELL995) |
| `--model_name` | str | `query2box` | Model: `query2box`, `gqe`, `betae`, `cqd`, or `all` |
| `--query_type` | str (one or more) | `2p` | Query type(s): `2p`, `3p`, `4p`, `2i`, `3i`, `4i`, `ip`, `pi`, `2u`, `up`, or `all` |
| `--output_dir` | str | `evaluations` | Directory where result CSV files are saved |
| `--iterations` | int | `3` | Number of random-substitution iterations per query/answer pair |
| `--benchmark` | int | `2` | Benchmark version: `1` for CQD benchmark, `2` for Hard benchmark |
| `--relation_selection` | str | `random` | Substitute-relation strategy: `random`, `score`, or `cosine` |
| `--similarity_file` | str | *(auto-detect)* | Path to a precomputed relation-similarity CSV. Auto-detected from the appropriate `relation_similarity/` subdirectory when omitted |
| `--top_r` | int | `3` | Number of top similar neighbours to use per source relation. In `score`/`cosine` mode the i-th iteration uses the i-th closest neighbour, so `--iterations` is automatically set to `--top_r` |

**Note:** The methodology for predicate substitution discussed in the paper is the `random` mode. The `score` and `cosine` modes are additional strategies we explored to investigate the effect of substituting with more similar relations, and they require precomputing relation similarities. The last two arguments (`similarity_file` and `top_r`) are only relevant for these modes.

Results are saved as CSV files under `--output_dir` with the naming convention:
```
evaluation_{dataset}_{benchmark}_{model}_{query_type}_{relation_selection}.csv
```

For generating the results of all models and query types for FB15k-237+H, you can run the following command:

```bash
python eval.py --dataset FB --model_name all --query_type all --benchmark 2 --iterations 3 --relation_selection random
```

For NELL+H, use:

```bash
python eval.py --dataset NELL --model_name all --query_type all --benchmark 2 --iterations 3 --relation_selection random
```