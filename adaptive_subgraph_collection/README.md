# Adaptive Subgraph Collection

The adaptive subgraph collection procedure collects paths that join the query entities and answers of KNN queries and then traverses them in the neighborhood of the entities of the given query.
The set of facts gathered by this process form the adaptive query subgraph for the given query. In this page, you can download the query subgraphs used for our experiments in the paper. Additionally,
we also describe the code so that you can make the changes for your dataset.

## Requirements
1. A KB loaded in a server with a SPARQL endpoint. For our experiments, we used the Freebase KB. Follow the steps [here](https://github.com/dki-lab/Freebase-Setup) to set up Freebase and a SPARQL endpoint to Freebase via Virtuoso server. This was the setup used in our experiments.
2. Modify the server endpoint URLs at 2 places ([here](https://github.com/ameyagodbole/cbr-weak-supervision/blob/0e98052d9ad961273627935331782ffe7c6d328a/adaptive_subgraph_collection/adaptive_utils.py#L37) and [here](https://github.com/ameyagodbole/cbr-weak-supervision/blob/0e98052d9ad961273627935331782ffe7c6d328a/adaptive_subgraph_collection/adaptive_utils.py#L72))

## Download adaptive subgraphs


## Running your own adaptive subgraph collection procedure

1. Collect all chains around subgraphs of train questions. The script below collect 2-hop chains for each query entity in the question

```
python adaptive_subgraph_collection/find_chains_in_full_KB.py --use_gold_entities --dataset_name=<dataset_name> --output_dir <path_to_output_dir> --train_file <path_to_train_file>
```

2. Now for a query in either train/dev/test set, we retrieve its {K}-NN queries from the train set, gather the collected paths from step 1 and traverse them for the query entity
```
python  adaptive_subgraph_collection/adaptive_graph_collection.py --use_gold_entities --collected_chains_file=<path_to_collected_chains_file_in_step_1> --dataset_name=<dataset_name> --input_file=<path_to_train_or_dev_or_test_file> --k=<num_knn> --knn_file=<path_to_file_that_stores_knn_for_each_query> --out_dir=<path_to_output_dir> --split=train/dev/test --job_id=0 --total_jobs=N --use_wandb=1
```
3. This step creates the input file in the format required by our reasoning model. Hence this step is optional and required only if you want to use the reasoning model of CBR-SubG
```
python adaptive_subgraph_collection/create_input_with_cbr_subgraph.py --subgraph_dir <path_to_output_dir_of_step_2> --input_dir <path_to_input_data_dir> --dataset_name <dataset_name> --output_dir <path_to_output_dir> --knn_dir <path_to_dir_containing_knns_of_each_file> --k=<num_knn> --use_gold_entities
```
This method should output the train, dev and test file containing the KNNs and collected adaptive subgraphs which can be used as input to our model.
