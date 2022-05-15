import datetime
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field

import torch
import wandb
from transformers import TrainingArguments, HfArgumentParser

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(path, os.pardir)))

from data_loaders.kbqa_dataloader import KBQADataLoader
from models.rgcn.rgcn_model import RGCN, QueryAwareRGCN
from models.compgcn.compgcn_models import CompGCN_TransE
from text_handler import PrecomputedQueryEncoder, QueryEncoder
from model_trainer import ModelTrainer

from global_config import logger
from data_loaders.training_utils import *


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CBRTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.transform_input = (model_args.transform_input == 1)
    model_args.use_fast_rgcn = (model_args.use_fast_rgcn == 1)
    model_args.add_dist_feature = (model_args.add_dist_feature == 1)
    model_args.add_inv_edges_to_edge_index = (model_args.add_inv_edges_to_edge_index == 1)
    model_args.use_sparse_feats = (model_args.use_sparse_feats == 1)
    if model_args.use_scoring_head == "none":
        model_args.use_scoring_head = None
    training_args.use_wandb = (training_args.use_wandb == 1)
    training_args.load_best_model_at_end = True
    # if model_args.use_sparse_feats and not model_args.transform_input:
    #     raise ValueError("When use_sparse_feats is True, transform_input has to be True")
    if training_args.task == 'pt_match':
        project_tags = ["pt_match", "rgcn"]
        if data_args.dataset_name != 'synthetic':
            project_tags.append("kbqa")
    elif training_args.task == 'kbc':
        if model_args.gnn == 'CompGCN_TransE':
            project_tags = ['kbc', 'CompGCN_TransE']
        if model_args.gnn == 'RGCN':
            project_tags = ['kbc', 'RGCN']
    if training_args.use_wandb:
        wandb.init(project="cbr-weak-supervision", tags=project_tags)

    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S_")
    rand_str = str(uuid.uuid4())[:8]
    training_args.output_dir = os.path.join(training_args.output_dir, "out-" + suffix + rand_str)
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    # also log to a log file
    fileHandler = logging.FileHandler("{0}/{1}".format(training_args.output_dir, "log.txt"))
    logger.addHandler(fileHandler)
    logger.info("Output directory is {}".format(training_args.output_dir))
    logger.info("=========Config:============")
    logger.info(json.dumps(training_args.to_dict(), indent=4, sort_keys=True))
    logger.info(json.dumps(vars(model_args), indent=4, sort_keys=True))
    logger.info(json.dumps(vars(data_args), indent=4, sort_keys=True))
    logger.info("============================")
    if training_args.max_steps > 0:
        logger.info("max_steps is given, train will run till whichever is sooner of num_train_epochs and max_steps")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if training_args.task == 'pt_match' or training_args.task == 'kbc':
        dataset_obj = KBQADataLoader(data_args.data_dir, data_args.data_file_suffix, training_args.train_batch_size,
                                     training_args.eval_batch_size, model_args.add_dist_feature,
                                     model_args.add_inv_edges_to_edge_index, data_args.max_dist, 
                                     training_args.downsample_eval_frac, training_args.task, data_args.dataset_name,
                                     data_args.precomputed_query_encoding_dir, data_args.paths_file_kbc, data_args.kb_system_file)
    else:
        raise NotImplemented(f"training_args.task: {training_args.task}")

    query_encoder = None
    model_args.node_feat_dim = dataset_obj.node_feat_dim
    model_args.n_additional_feat = dataset_obj.n_additional_feat
    model_args.n_base_feat = dataset_obj.n_base_feat
    model_args.max_dist = data_args.max_dist
    model_args.num_relations = dataset_obj.n_relations
    # model_args.num_entities = dataset_obj.n_entities
    model_args.device = device
    if training_args.task == 'pt_match' and model_args.use_query_aware_gcn:
        if data_args.precomputed_query_encoding_dir is not None and training_args.train_query_encoder == 0:
            logger.info("query_encoder: using precomputed query encodings")
            query_encoder = PrecomputedQueryEncoder(dataset_obj)
        else:
            logger.info("query_encoder: creating query encoder model")
            if data_args.precomputed_query_encoding_dir is not None:
                logger.warning("query_encoder: ignoring precomputed query encodings")
            query_encoder = QueryEncoder(model_args.query_encoder_model, model_args.pooling_type,
                                         (training_args.train_query_encoder == 1)).to(device)
        # Set the query encoding dimension based on the chosen encoder
        model_args.query_dim = query_encoder.get_query_embedding_dim()
        if model_args.use_sparse_feats:
            solver_model = QueryAwareRGCN(model_args, dataset_obj.base_feature_matrix).to(device)
        else:
            solver_model = QueryAwareRGCN(model_args).to(device)
    else:
        if model_args.use_sparse_feats:
            solver_model = RGCN(model_args, dataset_obj.base_feature_matrix).to(device)
        else:
            solver_model = RGCN(model_args).to(device)

    if training_args.use_wandb:
        wandb.config.update(model_args)
        wandb.config.update(data_args)
        wandb.config.update(training_args)
        wandb.config.update({"final_output_dir": training_args.output_dir})

    if model_args.model_ckpt_path is not None and os.path.exists(model_args.model_ckpt_path):
        logger.info("Path to a checkpoint found; loading the checkpoint!!!")
        state_dict = torch.load(model_args.model_ckpt_path)
        solver_model.load_state_dict(state_dict)
    optim_state_dict = None
    if model_args.optim_ckpt_path is not None and os.path.exists(model_args.optim_ckpt_path):
        logger.info("Path to a OPTIMIZER checkpoint found; loading the checkpoint!!!")
        optim_state_dict = torch.load(model_args.optim_ckpt_path)
    global_step = None
    if model_args.model_args_ckpt_path is not None and os.path.exists(model_args.model_args_ckpt_path):
        logger.info("Path to a model_args checkpoint found; loading the global_step!!!")
        with open(model_args.model_args_ckpt_path) as fin:
            loaded_model_args = json.load(fin)
            # load the global step
            global_step = loaded_model_args["global_step"]

    if training_args.patience:
        early_stopping = EarlyStopping("Hits@1", patience=training_args.patience)
    else:
        early_stopping = None

    trainer = ModelTrainer(solver_model, query_encoder, dataset_obj, training_args=training_args, data_args=data_args,
                           model_args=model_args, optim_state_dict=optim_state_dict, global_step=global_step,
                           device=device, early_stopping=early_stopping)
    if training_args.do_train:
        trainer.train()

    if training_args.do_eval:
        if training_args.do_train:
            logger.warning("Evaluating current trained model...")
        elif model_args.model_ckpt_path is None or not os.path.exists(model_args.model_ckpt_path):
            logger.warning("No path to model found!!!, Evaluating with a random model...")
        trainer.evaluate(log_output=(training_args.log_eval_result == 1))

    if training_args.do_predict:
        if model_args.model_ckpt_path is None or not os.path.exists(model_args.model_ckpt_path):
            logger.warning("No path to model found!!!, Evaluating with a random model...")
        trainer.predict()


@dataclass
class CBRTrainingArguments(TrainingArguments):
    """
    subclass of HF training arguments.
    """
    use_wandb: int = field(default=0, metadata={"help": "use wandb if 1"})
    task: str = field(default='pt_match', metadata={"help": "Options: [kbc, pt_match]"})
    dist_metric: str = field(default='l2', metadata={"help": "Options for pt_match: [l2, cosine], "
                                                             "Currently no options for kbc"})
    dist_aggr1: str = field(default='mean', metadata={"help": "Distance aggregation function at each neighbor query. "
                                                              "Options: [none (no aggr), mean, sum]"})
    dist_aggr2: str = field(default='mean', metadata={"help": "Distance aggregation function across all neighbor "
                                                              "queries. Options: [mean, sum]"})
    loss_metric: str = field(default='margin', metadata={"help": "Options for pt_match: [margin, txent], "
                                                                 "Options for kbc: [bce, dist]"})
    margin: float = field(default=5.0, metadata={"help": "Margin for loss computation"})
    sampling: float = field(default=1.0, metadata={"help": "Fraction of negative samples used"})
    temperature: float = field(default=1.0, metadata={"help": "Temperature for temperature scaled cross-entropy loss"})
    log_eval_result: int = field(default=0, metadata={"help": "Whether to log distances and ranking during evaluation"})
    train_batch_size: int = field(default=8, metadata={"help": "Training batch size"})
    eval_batch_size: int = (field(default=8, metadata={"help": "Evaluation batch size"}))
    learning_rate: float = field(default=0.001, metadata={"help": "Starting learning rate"})
    train_query_encoder: int = field(default=0, metadata={"help": "Whether to train the query encoder model when "
                                                                  "training query-aware message passing networks"})
    encoder_learning_rate: float = field(default=5e-5, metadata={"help": "Initial learning rate for query encoder."})
    warmup_steps: int = (field(default=0, metadata={"help": "scheduler warm up steps"}))
    downsample_eval_frac: float = field(default=1.0, metadata={"help": "Fraction of dev set to use for evaluation. "
                                                                       "Currently only implemented for pt_match"})
    kbc_eval_type: str = field(default='both', metadata={"help": "head/tail/both"})
    patience: int = field(default=None, metadata={"help": "Early Stopping Patience"})

    # Arguments inherited from TrainingArguments that affect code:
    # output_dir: str = field(
    #     metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    # )
    # num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    # max_steps: int = field(
    #     default=-1,
    #     metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    # )
    # gradient_accumulation_steps: int = field(
    #     default=1,
    #     metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    # )
    # max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    # logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    # eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    # save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    # save_total_limit: Optional[int] = field(
    #     default=None,
    #     metadata={
    #         "help": (
    #             "Limit the total amount of checkpoints."
    #             "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
    #         )
    #     },
    # )
    # weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_name: str = field(metadata={"help": "synthetic is a special dataset. all other datasets are treated as "
                                                "kb completion datasets"})
    data_dir: str = field(metadata={"help": "The path to data directory (contains train.json, dev.json, test.json)."})
    data_file_suffix: str = field(default='roberta-base_mean_pool_masked_cbr_subgraph_k=25', metadata={"help": "The suffix s for using train_s.json, dev_s.json, "
                                                                "test_s.json instead of train.json, dev.json, "
                                                                "test.json."})
    kb_system_file: str = field(default=None, metadata={
        "help": "The path to KB system file containing the full list of relations."})
    precomputed_query_encoding_dir: str = field(default=None, metadata={
        "help": "The path to directory containing precomputed query encodings query_enc_{train,dev,test}.pt. "
                "Will raise an error if used with train_query_encoder=1"})
    max_dist: int = field(default=3, metadata={"help": "When using distance from seed node as feature, this is the "
                                                       "maximum distance expected (would be the radius of the graph "
                                                       "from seed entities)"})
    otf: bool = field(default=False,
                      metadata={"help": "Use on the fly subgraph sampling, otherwise load paths from pkl file"})
    otf_max_nodes: int = field(default=1000,
                               metadata={"help": "Maximum number of nodes per subgraph in on-the-fly sampling"})
    edge_dropout: float = field(default=0.0, metadata={"help": "Percentage of edges in subgraphs to randomly remove"})
    node_dropout: float = field(default=0.0, metadata={"help": "Percentage of nodes in subgraphs to randomly remove"})
    num_neighbors_train: int = field(default=1,
                                     metadata={
                                         "help": "Number of near-neighbor subgraphs, k, to train with. K number of graphs will be randomly sampled from a larger list"})
    num_neighbors_eval: int = field(default=5,
                                    metadata={"help": "Number of near-neighbor subgraphs, k, to eval with"})
    adaptive_subgraph_k: int = field(default=25,
                                     metadata={
                                         "help": "Number of nearest neighbors used for creating the subgraphs for each question."})
    label_smooth: float = field(default=0.0, metadata={"help": "label smoothing"})
    paths_file_kbc: str = field(default='paths_1000_len_3.pkl', metadata={"help": "Paths file name"})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    transform_input: int = field(default=0, metadata={"help": "Add linear transform over one-hot input encoding"})
    use_fast_rgcn: int = field(default=1, metadata={"help": "Choose between RGCNConv (GPU memory-efficient by"
                                                            " iterating over each individual relation type) and"
                                                            " FastRGCNConv"})
    use_query_aware_gcn: int = field(default=0, metadata={"help": "Choose between vanilla RGCN and question aware "
                                                                  "variation (only used by KBQA)"})
    transform_query: int = field(default=0, metadata={"help": "Add linear transform over query encoding"})
    query_proj_dim: int = field(default=32, metadata={"help": "When using transform_query, dim to project down to"})
    query_attn_type: str = field(default=None, metadata={"help": "Type of query-aware attention to implement. "
                                                                 "Options: ['full', 'dim', 'sep']"})
    query_attn_activation: str = field(default='softmax', metadata={"help": "Activation fn for query-aware attention. "
                                                                            "Options: ['softmax', 'sigmoid']"})
    query_encoder_model: str = field(default=None, metadata={"help": "Model card or ckpt path compatible with the"
                                                                     " transformers library. [Tested for "
                                                                     "`roberta-base`]"})
    pooling_type: str = field(default='pooler', metadata={"help": "Output pooling to use for query encoding. "
                                                                  "Options: ['pooler', 'cls', 'mean_pool']"})
    node_feat_dim: int = field(default=None, metadata={"help": "Dimension of node input features"})
    dense_node_feat_dim: int = field(default=512, metadata={
        "help": "If not using sparse features, dimension of input entity embedding"})
    use_sparse_feats: int = field(default=1, metadata={"help": "1 if using sparse_feats"})
    gcn_dim: int = field(default=32, metadata={"help": "GCN layer dimension"})
    num_bases: int = field(default=None, metadata={"help": "Number of bases for basis-decomposition of relation "
                                                           "embeddings"})
    num_gcn_layers: int = field(default=3, metadata={"help": "Number of GCN layers"})
    add_dist_feature: int = field(default=1, metadata={"help": "Add (one-hot) distance from seed node as feature to "
                                                               "entity repr"})
    add_inv_edges_to_edge_index: int = field(default=1, metadata={"help": "[SYNTHETIC DATA] Include inverse relations "
                                                                          "in message passing. By default, messages are"
                                                                          " only passed one way"})
    use_scoring_head: str = field(default=None, metadata={"help": "Options: [transe, none]"})
    model_ckpt_path: str = field(default=None, metadata={"help": "Checkpoint to load"})
    optim_ckpt_path: str = field(default=None, metadata={"help": "Optimizer checkpoint to load"})
    model_args_ckpt_path: str = field(default=None, metadata={"help": "Model args to load"})
    gnn: str = field(default="RGCN", metadata={"help": "Which GNN model to use on subgraphs"})
    drop_rgcn: float = field(default=0.0, metadata={"help": "Dropout probability for RGCN model"})


if __name__ == '__main__':
    main()
