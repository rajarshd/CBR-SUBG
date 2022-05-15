import json
import os
import shutil
import time

import numpy as np
import torch
import wandb
from dist_fns import L2Dist, CosineDist
from global_config import logger
from loss import MarginLoss, TXent, DistanceLoss, BCELoss
from neighbors import Nneighbors, NneighborsFromData
from subgraphs import NNSubgraphsFromData
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup
from data_loaders.training_utils import *

DIST_FN = {'l2': L2Dist,
           'cosine': CosineDist
           }
LOSS_FN = {('pt_match', 'margin'): MarginLoss,
           ('pt_match', 'txent'): TXent,
           ('kbc', 'bce'): BCELoss,
           ('kbc', 'dist'): DistanceLoss,
           ('kbc', 'margin'): MarginLoss,
           ('kbc', 'txent'): TXent,
           }


class ModelTrainer:
    def __init__(self, model, query_encoder, dataset_obj, training_args, model_args, data_args, optim_state_dict,
                 global_step, device,
                 early_stopping):

        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        self.model = model
        self.query_encoder = query_encoder
        self.dataset_obj = dataset_obj
        self.global_step = 0
        self.device = device
        self.best_ckpt_step = -1
        self.saved_ckpts = []
        self.best_metric = -np.inf
        self.weak_best_metric = -np.inf
        self.early_stopping = early_stopping
        if self.early_stopping:
            self.early_stopping(self.best_metric)

        self.dist_fn = DIST_FN[training_args.dist_metric](stage1_aggr=self.training_args.dist_aggr1,
                                                          stage2_aggr=self.training_args.dist_aggr2)
        self.loss = LOSS_FN[(training_args.task, training_args.loss_metric)](margin=training_args.margin,
                                                                             temperature=training_args.temperature)
        if self.training_args.task == 'kbc' and self.model_args.use_scoring_head:
            self.scorer_dist_fn = L2Dist(stage1_aggr='none')
            self.scorer_loss_fn = MarginLoss(margin=9.0)
        self.subgraphs = NNSubgraphsFromData(dataset_obj=dataset_obj)
        if self.training_args.task == 'pt_match':
            self.neighbors = NneighborsFromData(dataset_obj=dataset_obj)
        elif self.training_args.task == 'kbc':
            self.neighbors = Nneighbors(dataset_obj, self.device)

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': training_args.weight_decay, 'lr': training_args.learning_rate},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': training_args.learning_rate},
        ]
        self.trainable_params = list(model.parameters())
        if self.query_encoder is not None and self.query_encoder.is_trainable():
            grouped_parameters.extend([
                {'params': [p for n, p in query_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': training_args.weight_decay, 'lr': training_args.encoder_learning_rate},
                {'params': [p for n, p in query_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': training_args.encoder_learning_rate},
            ])
            self.trainable_params.extend(query_encoder.parameters())
        self.optimizer = torch.optim.AdamW(grouped_parameters, lr=training_args.learning_rate,
                                           weight_decay=training_args.weight_decay)
        total_num_steps = int(self.training_args.num_train_epochs * (
                len(self.dataset_obj.train_dataloader) / self.training_args.gradient_accumulation_steps))
        logger.info("Total number of gradient steps would be {}".format(total_num_steps))
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 1000, total_num_steps)
        if global_step is not None:
            logger.info("Global step is not 0 but {}; probably resuming training from a checkpoint".format(global_step))
            self.global_step = global_step
        if optim_state_dict is not None:
            logger.info("Optimizer checkpoint found, loading the state dict from the checkpoint!!!")
            self.optimizer.load_state_dict(optim_state_dict)
            # reinit the scheduler with new warmup have to do this way because of a bug?
            # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, self.training_args.warmup_steps,
                                                             total_num_steps, last_epoch=self.global_step)

    def train(self):
        # self.evaluate(from_train=True)
        # if self.best_ckpt_step == self.global_step:
        #     self.save()
        running_loss = []  # Tracks loss between gradient accumulation steps
        local_step = 0
        stop_training, exit_by_error = False, False
        self.optimizer.zero_grad()
        loss_scale = [0.0, 0.0]
        for epoch in trange(int(self.training_args.num_train_epochs), desc=f"[Full Loop]"):
            if stop_training:
                break
            if hasattr(self.dataset_obj, 'lazy_load_ctr'):
                logger.info(f"lazy_load_ctr: {self.dataset_obj.lazy_load_ctr}")
            for batch_ctr, batch in enumerate(tqdm(self.dataset_obj.train_dataloader, desc=f"[Train]")):
                curr_batch_loss = 0.0
                # with torch.autograd.detect_anomaly():
                batch_start_time = time.time()
                self.model.train()
                if self.query_encoder is not None and self.query_encoder.is_trainable():
                    self.query_encoder.train()

                # First entry in the batch is the query, rest are nearest neighbors
                nn_list, nn_slices = self.neighbors(batch, k=self.data_args.num_neighbors_train)
                nn_batch, nn_slices = self.subgraphs(query_and_knn_list=nn_list, nn_slices=nn_slices)
                if nn_batch is None:
                    logger.info("The current batch was returned empty!!!!")
                    continue
                new_batch_len = len(nn_slices) - 1
                nn_batch.x = nn_batch.x.to(self.device)
                nn_batch.edge_index = nn_batch.edge_index.to(self.device)
                nn_batch.edge_attr = nn_batch.edge_attr.to(self.device)
                if self.model_args.add_dist_feature:
                    nn_batch.dist_feats = nn_batch.dist_feats.to(self.device)
                if self.model_args.use_query_aware_gcn:
                    query_embeddings = self.query_encoder(ex_ids=nn_batch.ex_id, split=nn_batch.split,
                                                          text_batch=nn_batch.query_str, device=self.device)
                    if torch.isnan(query_embeddings).any():
                        stop_training = True
                        exit_by_error = True
                        logger.warning("NaN observed in query embedding. Terminating run.")
                        if self.training_args.use_wandb:
                            wandb.log({"abort_code": "query_nan"})
                        break
                    sub_batch_repr = self.model(nn_batch.x, nn_batch.edge_index, nn_batch.edge_attr,
                                                query_embeddings, nn_batch.x_batch, nn_batch.edge_attr_batch,
                                                nn_batch.dist_feats)
                else:
                    sub_batch_repr = self.model(nn_batch.x, nn_batch.edge_index, nn_batch.edge_attr,
                                                nn_batch.dist_feats)
                if torch.isnan(sub_batch_repr).any():
                    stop_training = True
                    exit_by_error = True
                    logger.warning("NaN observed in batch representation. Terminating run.")
                    if self.training_args.use_wandb:
                        wandb.log({"abort_code": "batch_repr_nan"})
                    break
                loss = None
                for i in range(new_batch_len):
                    # s corresponds to query
                    # t corresponds to neighbors
                    repr_s = sub_batch_repr.narrow(0, nn_batch.__slices__['x'][nn_slices[i] + 0],
                                                   nn_batch.__slices__['x'][nn_slices[i] + 1] -
                                                   nn_batch.__slices__['x'][nn_slices[i] + 0])
                    labels_s = nn_batch.label_node_ids.narrow(0, nn_batch.__slices__['x'][nn_slices[i] + 0],
                                                              nn_batch.__slices__['x'][nn_slices[i] + 1] -
                                                              nn_batch.__slices__['x'][nn_slices[i] + 0])
                    repr_t = sub_batch_repr.narrow(0, nn_batch.__slices__['x'][nn_slices[i] + 1],
                                                   nn_batch.__slices__['x'][nn_slices[i + 1]] -
                                                   nn_batch.__slices__['x'][nn_slices[i] + 1])
                    labels_t = nn_batch.label_node_ids.narrow(0, nn_batch.__slices__['x'][nn_slices[i] + 1],
                                                              nn_batch.__slices__['x'][nn_slices[i + 1]] -
                                                              nn_batch.__slices__['x'][nn_slices[i] + 1])
                    label_identifiers = nn_batch.x_batch.narrow(0, nn_batch.__slices__['x'][nn_slices[i] + 1],
                                                                nn_batch.__slices__['x'][nn_slices[i + 1]] -
                                                                nn_batch.__slices__['x'][nn_slices[i] + 1])[labels_t]
                    assert label_identifiers.min() >= 0
                    label_identifiers = label_identifiers - label_identifiers.min()
                    label_identifiers = label_identifiers.to(repr_s.device)
                    dists = self.dist_fn(repr_s, repr_t[labels_t], target_identifiers=label_identifiers)
                    mask = ((labels_s == 1.0) + (
                                torch.FloatTensor(len(dists)).uniform_() < self.training_args.sampling)).to(self.device)
                    contrast_loss = self.loss(dists[mask], labels_s[mask]) / new_batch_len
                    loss_scale[0] += contrast_loss.detach().cpu().item()
                    if loss is None:
                        loss = contrast_loss
                    else:
                        loss += contrast_loss
                    if self.training_args.task == 'kbc' and self.model_args.use_scoring_head:
                        seeds_s = nn_batch.seed_node_ids.narrow(0, nn_batch.__slices__['x'][nn_slices[i] + 0],
                                                                nn_batch.__slices__['x'][nn_slices[i] + 1] -
                                                                nn_batch.__slices__['x'][nn_slices[i] + 0])
                        rel_ids = torch.LongTensor([nn_batch.query_str[i]]).to(repr_s.device)
                        head_repr = self.model.run_scoring_head(repr_s[seeds_s], rel_ids)
                        l2_dist = self.scorer_dist_fn(repr_s, head_repr)
                        supervised_loss = self.scorer_loss_fn(l2_dist[mask], labels_s[mask]) / new_batch_len
                        loss += supervised_loss
                        loss_scale[1] += supervised_loss.detach().cpu().item()
                if torch.isnan(loss):
                    stop_training = True
                    exit_by_error = True
                    logger.warning("NaN observed in loss. Terminating run.")
                    if self.training_args.use_wandb:
                        wandb.log({"abort_code": "loss_nan"})
                    break
                if self.training_args.gradient_accumulation_steps > 1:
                    loss = loss / self.training_args.gradient_accumulation_steps
                loss.backward()
                curr_batch_loss += loss.item()
                running_loss.append(curr_batch_loss)
                local_step += 1
                if self.training_args.use_wandb:
                    # tracks loss on current batch
                    wandb.log({'loss': curr_batch_loss}, commit=False)
                if local_step % self.training_args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, self.training_args.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    # Bookkeeping
                    self.global_step += 1
                    if 0 < self.training_args.max_steps <= self.global_step:
                        stop_training = True
                    if self.training_args.use_wandb:
                        # total_loss tracks accumulated loss between update steps
                        wandb.log({'total_loss': np.sum(running_loss), 'global_step': self.global_step}, commit=False)
                    if self.global_step % self.training_args.logging_steps == 0 or stop_training:
                        if len(loss_scale) == 0:
                            loss_scale = [1.0, 1.0]
                        logger.info(f"Epoch: {epoch}, Batch: {batch_ctr}, Loss: {np.sum(running_loss):0.6g}, "
                                    f"Loss Scale: {loss_scale}")
                    running_loss = []
                    loss_scale = [0.0, 0.0]
                    if self.global_step % self.training_args.eval_steps == 0:
                        self.evaluate(from_train=True)
                        if self.early_stopping:
                            try:
                                self.early_stopping(self.best_metric)
                            except EarlyStoppingException as e:
                                logger.error(str(e))
                                stop_training = True
                    if self.best_ckpt_step == self.global_step or self.global_step % self.training_args.save_steps == 0:
                        self.save()
                if self.training_args.use_wandb:
                    wandb.log({'local_step': local_step})
                if stop_training:
                    logger.info('max_steps reached. Stop training')
                    break

        if not exit_by_error:
            self.evaluate(from_train=True)
            if self.best_ckpt_step == self.global_step:
                self.save()

    def evaluate_on_dataloader(self, dataloader, log_output=False):
        output_log = {}
        results = {}
        for batch_ctr, batch in tqdm(enumerate(dataloader), desc=f"[Evaluate]"):
            # First entry in the batch is the query, rest are nearest neighbors
            nn_list, nn_slices = self.neighbors(batch, k=self.data_args.num_neighbors_eval)
            nn_batch, nn_slices = self.subgraphs(query_and_knn_list=nn_list, nn_slices=nn_slices)
            nn_batch.x = nn_batch.x.to(self.device)
            nn_batch.edge_index = nn_batch.edge_index.to(self.device)
            nn_batch.edge_attr = nn_batch.edge_attr.to(self.device)
            if self.model_args.add_dist_feature:
                nn_batch.dist_feats = nn_batch.dist_feats.to(self.device)
            if self.model_args.use_query_aware_gcn:
                query_embeddings = self.query_encoder(ex_ids=nn_batch.ex_id, split=nn_batch.split,
                                                      text_batch=nn_batch.query_str, device=self.device)
                sub_batch_repr = self.model(nn_batch.x, nn_batch.edge_index, nn_batch.edge_attr,
                                            query_embeddings, nn_batch.x_batch, nn_batch.edge_attr_batch,
                                            nn_batch.dist_feats)
            else:
                sub_batch_repr = self.model(nn_batch.x, nn_batch.edge_index, nn_batch.edge_attr, nn_batch.dist_feats)
            for i in range(len(batch)):
                # s corresponds to query
                # t corresponds to neighbors
                ex_id = int(batch[i].ex_id)
                repr_s = sub_batch_repr.narrow(0, nn_batch.__slices__['x'][nn_slices[i] + 0],
                                               nn_batch.__slices__['x'][nn_slices[i] + 1] -
                                               nn_batch.__slices__['x'][nn_slices[i] + 0])
                repr_t = sub_batch_repr.narrow(0, nn_batch.__slices__['x'][nn_slices[i] + 1],
                                               nn_batch.__slices__['x'][nn_slices[i + 1]] -
                                               nn_batch.__slices__['x'][nn_slices[i] + 1])
                labels_t = nn_batch.label_node_ids.narrow(0, nn_batch.__slices__['x'][nn_slices[i] + 1],
                                                          nn_batch.__slices__['x'][nn_slices[i + 1]] -
                                                          nn_batch.__slices__['x'][nn_slices[i] + 1])
                label_identifiers = nn_batch.x_batch.narrow(0, nn_batch.__slices__['x'][nn_slices[i] + 1],
                                                            nn_batch.__slices__['x'][nn_slices[i + 1]] -
                                                            nn_batch.__slices__['x'][nn_slices[i] + 1])[labels_t]
                if len(label_identifiers) == 0:
                    # all the neighbors have zero answers, which is unfortunate
                    logger.info("None of the nearest neighbors have answer nodes...ignoring")
                    results['count'] = 1 + results.get('count', 0.0)  # penalize
                    if log_output:
                        output_log[ex_id] = {}
                        output_log[ex_id]["qid"] = batch[i].query
                        output_log[ex_id]["query_str"] = batch[i].query_str
                        output_log[ex_id]["gold_answer"] = batch[i].answers
                        output_log[ex_id]["KNN"] = [nn_batch[k].query_str for k in
                                                    range(nn_slices[i] + 1, nn_slices[i + 1])]
                        output_log[ex_id]["KNN_ids"] = [nn_batch[k].query for k in
                                                        range(nn_slices[i] + 1, nn_slices[i + 1])]
                        output_log[ex_id][
                            "comment"] = "None of the nearest neighbors have answer nodes...ignoring and penalizing"
                    continue
                assert label_identifiers.min() >= 0
                label_identifiers = label_identifiers - label_identifiers.min()
                label_identifiers = label_identifiers.to(repr_s.device)
                dists = self.dist_fn(repr_s, repr_t[labels_t], target_identifiers=label_identifiers).cpu().numpy()
                if log_output:
                    output_log[ex_id] = {}
                    output_log[ex_id]["qid"] = batch[i].query
                    output_log[ex_id]["query_str"] = batch[i].query_str
                    output_log[ex_id]["gold_answer"] = batch[i].answers
                    output_log[ex_id]["KNN"] = [nn_batch[k].query_str for k in
                                                range(nn_slices[i] + 1, nn_slices[i + 1])]
                    output_log[ex_id]["KNN_ids"] = [nn_batch[k].query for k in
                                                    range(nn_slices[i] + 1, nn_slices[i + 1])]
                pred_ranks = np.argsort(dists)
                if self.training_args.task == 'pt_match':
                    ex_labels = set(np.where(batch[i].label_node_ids == 1)[0])
                    if len(batch[i].answers) == 0:
                        # answer was an empty list; happens in WebQSP
                        logger.info("Answer is empty list, ignoring...")
                        results['count'] = 1 + results.get('count', 0.0)
                        for at_k in [1, 3, 5, 10]:
                            results[f'avg_hits@{at_k}'] = results.get(f'avg_hits@{at_k}', 0.0) + 1.0
                            results[f'avg_weak_hits@{at_k}'] = results.get(f'avg_weak_hits@{at_k}', 0.0) + 1.0
                        if log_output:
                            output_log[ex_id][
                                "comment"] = "Answer list is empty. Getting a +1 for this following graftnet style evaluation"
                        continue
                    example_ranks = []
                    pred_global_ids = batch[i].x[pred_ranks].numpy()  # map to global ids
                    curr_rank = 1
                    if len(ex_labels) > 0:
                        for node_id in pred_ranks:
                            if node_id in ex_labels:
                                example_ranks.append(curr_rank)
                                if len(example_ranks) == len(ex_labels):
                                    break
                            else:
                                curr_rank += 1
                        assert len(ex_labels) == len(example_ranks)
                    if self.data_args.dataset_name != 'synthetic':
                        # Penalize ranking for answer entities outside the k-hop subgraph
                        for _ in range(batch[i].penalty):
                            example_ranks.append((batch[i].num_nodes + self.dataset_obj.n_entities) / 2)
                    results['count'] = 1 + results.get('count', 0.0)
                    results.setdefault('mean_ranks', []).append(np.mean(example_ranks))
                    results.setdefault('max_ranks', []).append(np.max(example_ranks))
                    results.setdefault('hits@1', []).append(np.mean(np.array(example_ranks) == 1))
                    results['avg_mean_ranks'] = np.mean(example_ranks) + results.get('avg_mean_ranks', 0.0)
                    results['avg_max_ranks'] = np.max(example_ranks) + results.get('avg_max_ranks', 0.0)
                    for at_k in [1, 3, 5, 10]:
                        results[f'avg_hits@{at_k}'] = np.mean(np.array(example_ranks) <= at_k) + \
                                                      results.get(f'avg_hits@{at_k}', 0.0)
                        results[f'avg_weak_hits@{at_k}'] = results.get(f'avg_weak_hits@{at_k}', 0.0) + \
                                                           (1.0 if np.any(np.array(example_ranks) <= at_k) else 0.0)
                    if log_output:
                        # output_log[ex_id]['ranks'] = example_ranks
                        output_log[ex_id]["is_hits@1_correct"] = 1.0 if np.any(np.array(example_ranks) <= 1) else 0.0
                        output_log[ex_id]["is_hits@3_correct"] = 1.0 if np.any(np.array(example_ranks) <= 3) else 0.0
                        output_log[ex_id]["is_hits@5_correct"] = 1.0 if np.any(np.array(example_ranks) <= 5) else 0.0
                        output_log[ex_id]["top_5_preds"] = [self.dataset_obj.id2ent[pred_id] for pred_id in
                                                            pred_global_ids[:5]]
                elif self.training_args.task == 'kbc':
                    # kbc eval is a bit different since each missing edge (e1, r, e2) is considered
                    # a separate query, even if there are multiple missing (e1, r, [e2..]) edges
                    e1, r = batch[i].query
                    assert self.dataset_obj.all_kg_map is not None
                    all_gold_answers = self.dataset_obj.all_kg_map[(e1, r)]
                    gold_answers = batch[i].answers
                    # map pred_ranks from subgraph to global ids
                    pred_ranks = batch[i].x[pred_ranks].numpy()
                    for gold_answer in gold_answers:
                        # remove all other gold answers from prediction
                        filtered_answers = []
                        for pred in pred_ranks:
                            pred = self.dataset_obj.id2ent[pred]
                            if pred in all_gold_answers and pred != gold_answer:
                                continue
                            else:
                                filtered_answers.append(pred)
                        rank = None
                        for i, e_to_check in enumerate(filtered_answers):
                            if gold_answer == e_to_check:
                                rank = i + 1
                                break
                        results['count'] = 1 + results.get('count', 0.0)
                        if rank is not None:
                            if rank <= 10:
                                results["avg_hits@10"] = 1 + results.get("avg_hits@10", 0.0)
                                if rank <= 5:
                                    results["avg_hits@5"] = 1 + results.get("avg_hits@5", 0.0)
                                    if rank <= 3:
                                        results["avg_hits@3"] = 1 + results.get("avg_hits@3", 0.0)
                                        if rank <= 1:
                                            results["avg_hits@1"] = 1 + results.get("avg_hits@1", 0.0)
                            results["avg_rr"] = (1.0 / rank) + results.get("avg_rr", 0.0)
                            if log_output:
                                # output_log[ex_id]['ranks'] = example_ranks
                                output_log[ex_id]["is_hits@1_correct"] = 1.0 if rank <= 1 else 0.0
                                output_log[ex_id]["is_hits@3_correct"] = 1.0 if rank <= 3 else 0.0
                                output_log[ex_id]["is_hits@5_correct"] = 1.0 if rank <= 5 else 0.0
                                output_log[ex_id]["is_hits@10_correct"] = 1.0 if rank <= 10 else 0.0
        # logger.info("None counter is {}".format(none_ctr))
        final_results = {}
        normalizer = results.pop('count')
        for k, v in results.items():
            if k.startswith('avg'):
                final_results[k] = v / normalizer
            else:
                assert isinstance(v, list)
                final_results[k] = np.asarray(v)
        return final_results, output_log

    @torch.no_grad()
    def evaluate(self, from_train=False, log_output=False):
        logger.info("Starting eval...")
        st_time = time.time()
        self.model.eval()
        if self.query_encoder is not None:
            self.query_encoder.eval()
        output_log = {}
        results = {}
        if self.training_args.task == 'pt_match' or self.training_args.task == 'kbc':
            eval_loaders = [("dev", self.dataset_obj.dev_dataloader)]
        else:
            eval_loaders = None
        for split, loader in eval_loaders:
            init_results, init_output_log = self.evaluate_on_dataloader(loader, log_output)
            results.update({'{}_{}'.format(split, k): v for k, v in init_results.items()})
            output_log.update(init_output_log)
        if log_output:
            with open(os.path.join(self.training_args.output_dir, f'eval_log_{self.global_step}.json'), 'w') as fout:
                json.dump(output_log, fout, indent=2)

        logger.info("[---- EVAL ----]")
        for k, v in results.items():
            if k.startswith('dev_avg'):
                results[k] = v * self.dataset_obj.dev_penalty_multiplier
        if self.training_args.task == 'pt_match':
            logger.info(f"Avg Mean Rank: {results['dev_avg_mean_ranks']}")
            logger.info(f"Avg Max Rank: {results['dev_avg_max_ranks']}")
            logger.info(f"Avg HITS@1: {results['dev_avg_hits@1']}")
            logger.info(f"Avg Weak HITS@1: {results['dev_avg_weak_hits@1']}")
            if from_train and results['dev_avg_weak_hits@1'] > self.best_metric:
                self.best_ckpt_step = self.global_step
                self.best_metric = results['dev_avg_weak_hits@1']
                self.weak_best_metric = results['dev_avg_weak_hits@1']
                # also log the best results in a different variable
                best_results = {}
                for k, v in results.items():
                    if k.startswith("dev_avg_"):
                        best_results["best_" + k] = v
                for k, _ in best_results.items():
                    results[k] = best_results[k]
            results['dev_best_metric_step'] = self.best_ckpt_step
            logger.info("Best metric till now...")
            logger.info(f"Avg HITS@1: {self.best_metric}")
            logger.info(f"Avg Weak HITS@1: {self.weak_best_metric}")
        elif self.training_args.task == 'kbc':
            # Account for ignoring queries
                # else:
                #     assert isinstance(v, np.ndarray) or isinstance(v, list)
            logger.info(f"Avg HITS@1: {results['dev_avg_hits@1']}")
            logger.info(f"Avg HITS@3: {results['dev_avg_hits@3']}")
            logger.info(f"Avg HITS@5: {results['dev_avg_hits@5']}")
            logger.info(f"Avg HITS@10: {results['dev_avg_hits@10']}")
            logger.info(f"MRR: {results['dev_avg_rr']}")
            if from_train and results['dev_avg_hits@1'] > self.best_metric:
                self.best_ckpt_step = self.global_step
                self.best_metric = results['dev_avg_hits@1']
                # also log the best results in a different variable
                best_results = {}
                for k, v in results.items():
                    if k.startswith("dev_avg_"):
                        best_results["best_" + k] = v
                for k, _ in best_results.items():
                    results[k] = best_results[k]
            results['dev_best_metric_step'] = self.best_ckpt_step
            logger.info("Best metric till now...")
            logger.info(f"Avg HITS@1: {self.best_metric}")

        if self.training_args.use_wandb:
            if from_train:
                results['global_step'] = self.global_step
            log_results = {k: v for k, v in results.items() if
                           not isinstance(v, list) and not isinstance(v, np.ndarray)}
            wandb.log(log_results, commit=False if from_train else True)
        logger.info("Eval done!, took {} seconds".format(time.time() - st_time))

    @torch.no_grad()
    def predict(self):
        self.model.eval()
        if self.query_encoder is not None:
            self.query_encoder.eval()
        results = {}
        if self.training_args.task == 'pt_match' or self.training_args.task == 'kbc':
            eval_loaders = [("test", self.dataset_obj.test_dataloader)]
        else:
            eval_loaders = None
        for split, loader in eval_loaders:
            init_results, _ = self.evaluate_on_dataloader(loader, log_output=False)
            results.update({'{}_{}'.format(split, k): v for k, v in init_results.items()})

        logger.info("[---- PREDICT ----]")
        # Account for ignoring queries
        for k, v in results.items():
            if k.startswith('test_avg'):
                results[k] = v * self.dataset_obj.test_penalty_multiplier
        if self.training_args.task == 'pt_match':
            logger.info(f"Avg Mean Rank: {results['test_avg_mean_ranks']}")
            logger.info(f"Avg Max Rank: {results['test_avg_max_ranks']}")
            logger.info(f"Avg HITS@1: {results['test_avg_hits@1']}")
            logger.info(f"Avg Weak HITS@1: {results['test_avg_weak_hits@1']}")
        elif self.training_args.task == 'kbc':
                # else:
                #     assert isinstance(v, np.ndarray) or isinstance(v, list)
            print(f"Avg Mean Rank: {results[split + '_avg_mr']}")
            print(f"Avg Mean Reciprocal Rank: {results[split + '_avg_mrr']}")
            print(f"Avg HITS@1: {results[split + '_avg_hits@1']}")
        if self.training_args.use_wandb:
            wandb.log(results)

    def save(self):
        if not os.path.exists(self.training_args.output_dir):
            os.makedirs(self.training_args.output_dir)
        if self.training_args.save_total_limit is not None:
            # Limit the total amount of checkpoints
            past_ckpts = self.saved_ckpts.copy()
            if self.best_ckpt_step in past_ckpts:
                past_ckpts.remove(self.best_ckpt_step)
            if len(past_ckpts) >= self.training_args.save_total_limit:
                # Delete oldest checkpoint
                ckpt_to_delete = past_ckpts[0]
                self.saved_ckpts.remove(ckpt_to_delete)
                ckpt_filenm = os.path.join(self.training_args.output_dir, f"ckpt{ckpt_to_delete}_pytorch_model.bin")
                os.remove(ckpt_filenm)
                opt_filenm = os.path.join(self.training_args.output_dir, f"ckpt{ckpt_to_delete}_optimizer.pt")
                os.remove(opt_filenm)
                if self.query_encoder is not None and self.query_encoder.is_trainable():
                    encoder_dirnm = os.path.join(self.training_args.output_dir, f"ckpt{ckpt_to_delete}")
                    shutil.rmtree(encoder_dirnm)
                opt_filenm = os.path.join(self.training_args.output_dir, f"ckpt{ckpt_to_delete}_args.json")
                os.remove(opt_filenm)

        self.model.eval()
        ckpt_filenm = os.path.join(self.training_args.output_dir, f"ckpt{self.global_step}_pytorch_model.bin")
        with open(ckpt_filenm, 'wb') as fout:
            torch.save(self.model.state_dict(), fout)
        opt_filenm = os.path.join(self.training_args.output_dir, f"ckpt{self.global_step}_optimizer.pt")
        with open(opt_filenm, 'wb') as fout:
            torch.save(self.optimizer.state_dict(), fout)
        if self.query_encoder is not None and self.query_encoder.is_trainable():
            encoder_dirnm = os.path.join(self.training_args.output_dir, f"ckpt{self.global_step}")
            self.query_encoder.save(encoder_dirnm)
        args_file_name = os.path.join(self.training_args.output_dir, f"ckpt{self.global_step}_args.json")
        # save the global step count
        args_to_save = {"global_step": self.global_step}
        with open(args_file_name, "w") as fout:
            json.dump(args_to_save, fout)
        if self.global_step == self.best_ckpt_step:
            # save another copy named as pytorch_model.bin and optimizer.pt
            ckpt_filenm = os.path.join(self.training_args.output_dir, "pytorch_model.bin")
            with open(ckpt_filenm, 'wb') as fout:
                torch.save(self.model.state_dict(), fout)
            opt_filenm = os.path.join(self.training_args.output_dir, "optimizer.pt")
            with open(opt_filenm, 'wb') as fout:
                torch.save(self.optimizer.state_dict(), fout)
            if self.query_encoder is not None and self.query_encoder.is_trainable():
                encoder_dirnm = os.path.join(self.training_args.output_dir, "query_encoder")
                self.query_encoder.save(encoder_dirnm)
            args_filenm = os.path.join(self.training_args.output_dir, "args.json")
            with open(args_filenm, 'w') as fout:
                json.dump(args_to_save, fout)
        self.saved_ckpts.append(self.global_step)
