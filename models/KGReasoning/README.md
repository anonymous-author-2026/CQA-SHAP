# KGReasoning

This part is based on the code from [KGReasoning](https://github.com/snap-stanford/KGReasoning) which includes the official implementation of [BetaE](https://arxiv.org/abs/2010.11465) model, along with [Query2box](https://arxiv.org/abs/2002.05969) and [GQE](https://arxiv.org/abs/1806.01445) models.

## Requirements

Further than the main requirements in the root `README.md`, please install the following packages.

```bash
pip install tensorboard
pip install tensorboardX
```
## Dataset

The data used in the BetaE and Query2box papers can be downloaded from [here](http://snap.stanford.edu/betae/KG_data.zip). As mentioned in the original codebase, the BetaE data is more realistic due to the maximum number of answers in the validation/test sets. Therefore, we only use this part of the downloaded data in our experiments. Furthermore, we ignore `FB15k` dataset due to its known data leakage issues.

You can use the following commands to creat the data folder and download the data into it.

```bash
mkdir data
cd data
wget http://snap.stanford.edu/betae/KG_data.zip
unzip KG_data.zip
```

<details>
<summary><i>Data Files Description</i></summary>

Each folder in the `data` directory represents a specific KG, and includes the following files.

| **File Name** | **Description** |
| ------------- | --------------- |
| `train.txt`/`valid.txt`/`test.txt` | KG edges |
| `id2rel.pkl`/`rel2id.pkl`/`ent2id.pkl`/`id2ent.pkl` | KG entity relation dicts |
| `train-queries.pkl`/`valid-queries.pkl`/`test-queries.pkl` | `defaultdict(set)`, each key represents a query structure, and the value represents the instantiated queries |
| `train-answers.pkl` | `defaultdict(set)`, each key represents a query, and the value represents the answers obtained in the training graph (edges in `train.txt`) |
| `valid-easy-answers.pkl`/`test-easy-answers.pkl` | `defaultdict(set)`, each key represents a query, and the value represents the answers obtained in the training graph (edges in `train.txt`) / valid graph (edges in `train.txt`+`valid.txt`) |
| `valid-hard-answers.pkl`/`test-hard-answers.pkl` | `defaultdict(set)`, each key represents a query, and the value represents the **additional** answers obtained in the validation graph (edges in `train.txt`+`valid.txt`) / test graph (edges in `train.txt`+`valid.txt`+`test.txt`) |
</details>

## Query Formats

This codebase use tuples to represent different query structures. We've already provided a function to convert the unified query format we use in this repo to the desired format of this model in the `predict.py` file. Note that we only did this for those query types that are used in our experiments.

## Model Training/Checkpoints

Unfortunately, the original codebase does not provide pre-trained model checkpoints. Therefore, you need to train the models on the desired dataset before using them for predictions. However, we've done this step and shared the trained model checkpoints for all 3 models on both `FB15k-237` and `NELL995` datasets. You can download the checkpoints from [here](). After unzipping the downloaded file, you can access the model checkpoints in the `logs` folder for each dataset and model.

```bash
wget <download_link>
unzip KGReasoning_checkpoints.zip
```


<details>
<summary><i>Training the Models Yourself</i></summary>

You can use the following command to train a model on a specific dataset.

```bash
cd models
```

#### FB15k-237

BetaE:

```bash
CUDA_VISIBLE_DEVICES=0 python -m KGReasoning.main --cuda --do_train --do_valid --do_test \
  --data_path KGReasoning/data/FB15k-237-betae -n 128 -b 512 -d 400 -g 60 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo beta --valid_steps 15000 \
  -betam "(1600,2)" --print_on_screen
  ```

Query2box:

```bash
CUDA_VISIBLE_DEVICES=0 python -m KGReasoning.main --cuda --do_train --do_valid --do_test \
  --data_path KGReasoning/data/FB15k-237-betae -n 128 -b 512 -d 400 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo box --valid_steps 15000 \
  -boxm "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --print_on_screen
```

GQE:

```bash
CUDA_VISIBLE_DEVICES=0 python -m KGReasoning.main --cuda --do_train --do_valid --do_test \
  --data_path KGReasoning/data/FB15k-237-betae -n 128 -b 512 -d 800 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo vec --valid_steps 15000 \
  --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --print_on_screen
```

#### NELL995

BetaE:

```bash
CUDA_VISIBLE_DEVICES=0 python -m KGReasoning.main --cuda --do_train --do_valid --do_test \
  --data_path KGReasoning/data/NELL-betae -n 128 -b 512 -d 400 -g 60 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo beta --valid_steps 15000 \
  -betam "(1600,2)" --print_on_screen
```

Query2box:

```bash
CUDA_VISIBLE_DEVICES=0 python -m KGReasoning.main --cuda --do_train --do_valid --do_test \
  --data_path KGReasoning/data/NELL-betae -n 128 -b 512 -d 400 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo box --valid_steps 15000 \
  -boxm "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --print_on_screen
```

GQE:

```bash
CUDA_VISIBLE_DEVICES=0 python -m KGReasoning.main --cuda --do_train --do_valid --do_test \
  --data_path KGReasoning/data/NELL-betae -n 128 -b 512 -d 800 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo vec --valid_steps 15000 \
  --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --print_on_screen
```

#### Note

The hyperparameters used in the above commands are based on the original codebase recommendations (`example.sh` file).

We suggest that you rename the created `logs` folder after training to avoid potential conflicts with what we've used in our experiments. Thus, you can rename it as follows:

```bash
logs
|-- FB15k-237
|    |-- query2box
|         |-- checkpoint
|         |-- config.json
|         |-- events.out.tfevents...
|         |-- train.log
|    |-- betae
|         |-- ...
|    |-- gqe
|         |-- ...
|-- NELL995
|    |-- query2box
|         |-- ...
|    |-- betae
|         |-- ...
|    |-- gqe
|         |-- ...
```

</details>


## Usage

We implemented `predict.py` to execute the model on a single query and get the prediction results. You can use the same code for all 3 models by changing the model name in the arguments.

```python
from predict import KGR
q2b = KGR(method="query2box") # or "betae" or "gqe"
```

<details>
<summary><i>Changing Hyperparameters</i></summary>

In case you want to change the hyperparameters, you can pass them as additional arguments when initializing the model. For example:

```python
args = argparse.Namespace()
args.gamma = 12.0
args.cpu_num = 10
...
q2b = KGR(method="query2box", args=args)
```

</details>

Then, you can simply pass a query with the type of `Query` to the `predict` function to get the results.

```python
q2b.predict(query_sample)
```
 
This will return a dictionary that contains the the sorted scores and the ranked list of entities for the given query.

```python
{'scores': tensor([ 12.6836,   8.3422,   6.4791,  ..., -33.9443, -34.8359, -35.0932], device='cuda:0'),
 'ids': tensor([ 2865,  2120, 10548,  ...,  6916, 13430, 13718], device='cuda:0')}
```

<details>
<summary><i>Interpreting the Results</i></summary>

The first item is the predicted entity with the highest likelihood to answer the query, and the last item is the entity with the lowest likelihood. This means that the entity with the id of `2865` is the top-1 prediction for the given query with a score of `12.6836`.
