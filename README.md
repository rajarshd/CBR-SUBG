# Knowledge Base Question Answering using Case-based Reasoning over Subgraphs
This is the official implementation of the paper - [Knowledge Base Question Answering using Case-based Reasoning over Subgraphs](https://arxiv.org/abs/2202.10610).

## Installation
```bash
conda create -n pygnn python=3.9
conda activate pygnn
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia ## this installs pytorch 1.9
pip install transformers
pip install wandb
conda install pytorch-scatter -c rusty1s
conda install pytorch-sparse -c rusty1s
conda install pytorch-cluster -c rusty1s
conda install pytorch-spline-conv -c rusty1s
conda install pytorch-geometric -c rusty1s -c conda-forge
pip install ordered_set
```
Please refer to the [environment.yml](environment.yml) file for an exhaustitve list.

## Adaptive subgraphs
The subgraphs collected by our adaptive subgraph gathering strategy (Sec 3.2) of the paper for all the datasets used in the paper can be downloaded from this [link](https://drive.google.com/drive/folders/1UIiD-UCHQuDTIvascfvVW5X3mZDg3lfh?usp=sharing). Download and uncompress the file and you should be able to see a directory for each of the datasets. In each of the directory, you should be able to see the following files ``{train|valid|dev}_roberta-base_mean_pool_masked_cbr_subgraph_k={25|75}.json``. The subgraph for each query is present in these files. Note, that they are mapped to integers - to convert them to strings (i.e. KBids) refer to the ``{entities, relations}_roberta-base_mean_pool_masked_cbr_subgraph_k={25|75}.txt``

### Parsing the file name
Lets consider the filename ``train_roberta-base_mean_pool_masked_cbr_subgraph_k=25.json``.

``train`` - refers to the fact this file denotes the train set

``roberta-base_mean_pool_masked`` - refers to the fact that to obtain the KNN questions, the question representations were obtained by using a roberta-base model followed by a mean pool over the token representations in the last layer. Moreover masked denotes that the entity mentions in each question were replaced by the <mask> token (refer to sec 3.1 of the paper for details)
  
``k=25`` - means that to obtain the subgraphs, we used paths collected from 25 KNN queries (refer to sec 3.2 of the paper)


### Collecting your own subgraphs
Please refer to the ``Readme`` in the [adaptive_subgraph_collection](adaptive_subgraph_collection) directory. 
  
## Training
The [runner.py](src/runner.py) file is the main file that is needed to run the code. A few of the popular flags are listed below. 

  ```bash
  python runner.py 
  --do_train # flag for training
  --do_predict # flag for prediction
  --add_dist_feature=1 # add distance feature to node embeddings
  --add_inv_edges_to_edge_index=1 # add inverse KB edges
  --data_dir=<path_to_data_dir> 
  --data_file_suffix=roberta-base_mean_pool_masked_cbr_subgraph_k=25 # which subgraph file to use?
  --dataset_name=webqsp # dataset name
  --dist_metric=cosine # distance metric to use (l2 or cosine)
  --eval_batch_size=1 # batch size during eval
  --eval_steps=100 # how many gradient steps before evaluating on validation
  --gradient_accumulation_steps=4 # gradient accumulation steps
  --learning_rate=0.0024129869604528702 # learning rate
  --logging_steps=10 
  --loss_metric=txent # loss fn (tx-ent/margin)
  --max_grad_norm=1 
  --num_gcn_layers=3 # num GCN layers
  --num_neighbors_eval=5 # number of KNNs at eval
  --num_neighbors_train=10 # number of KNNs during train
  --num_train_epochs=30 
  --output_dir=<path_to_output_models> 
  --temperature=0.06453104622317246 # temperature in loss (txent)
  --train_batch_size=1 # train batch size
  --transform_input=1 # add linear layer after sparse input embedding layer
  --use_fast_rgcn=1 # use fast_rgcn
  --use_wandb=1 # use wandb?
  ```
 
 ## Pre-trained models and collected subgraphs
The pre-trained models and subgraphs can be downloaded from [here](https://drive.google.com/drive/folders/1UIiD-UCHQuDTIvascfvVW5X3mZDg3lfh?usp=sharing). The query specific subgraphs are present in the input json files. 

  The commands to reproduce the results from the paper are present [here](scripts_to_reproduce_paper_results.sh).

 ## Citation
  If you use the code, data (e.g. subgraphs) or models, we would be grateful if you cite
  ```
  @inproceedings{cbr_subg,
  title={Knowledge Base Question Answering by Case-based Reasoning over Subgraphs},
  author={Das, Rajarshi and Godbole, Ameya and Naik, Ankita and Tower, Elliot and Jia, Robin and Zaheer, Manzil and Hajishirzi, Hannaneh and McCallum, Andrew},
  booktitle={ICML},
  year={2022}
}
 ```
  
