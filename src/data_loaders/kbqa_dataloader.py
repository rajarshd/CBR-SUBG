from copy import deepcopy
import json
import numpy as np
import os
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataListLoader
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm
from src.global_config import logger
from collections import defaultdict
from src.data_loaders.data_utils import load_data_all_triples
import pickle


class KBQAGraphData(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None, dist_feats=None, seed_nodes_mask=None,
                 label_nodes_mask=None, split=None, ex_id=None, query=None, query_str=None, answers=None, knn_ids=None,
                 penalty=torch.LongTensor([0])):
        super(KBQAGraphData, self).__init__(x, edge_index, edge_attr)
        self.dist_feats = dist_feats
        self.seed_node_ids = seed_nodes_mask
        self.label_node_ids = label_nodes_mask
        self.split = split
        self.ex_id = ex_id
        self.query = query
        self.query_str = query_str
        self.answers = answers
        self.knn_ids = knn_ids
        self.penalty = penalty


class KBQADataLoader:
    def __init__(self, data_dir: str, data_file_suffix: str = '', train_batch_size: int = 16, eval_batch_size: int = 32,
                 add_dist_feature: bool = False, add_inv_edges: bool = False, max_dist: int = 3,
                 downsample_eval_frac: float = 1.0, task: str = "pt_match", dataset_name="webqsp",
                 precomputed_query_encoding_dir: str = None, paths_file_name="paths_1000_len_3.pkl",
                 kb_system_file=None):
        self.data_dir = data_dir
        self.data_file_suffix = "_" + data_file_suffix if data_file_suffix else ''
        self.dataset_name = dataset_name
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.add_dist_feature = add_dist_feature
        self.add_inv_edges = add_inv_edges
        self.max_dist = max_dist
        self.downsample_eval_frac = downsample_eval_frac
        self.dev_subsample_idx = None
        assert 0.0 <= downsample_eval_frac <= 1.0
        if self.downsample_eval_frac < 1.0 and task != 'pt_match':
            logger.warning(f"'downsample_eval_frac' is not implemented for task: {task}")
        self.raw_train_data, self.raw_dev_data, self.raw_test_data = None, None, None
        # KBQA: These masks hold whether an example from the full parition was dropped (due to no subgraph)
        self.raw_train_drop_mask, self.raw_dev_drop_mask, self.raw_test_drop_mask = None, None, None
        self.train_dataset, self.train_dataloader, self.dev_dataset, self.dev_dataloader, \
        self.test_dataset, self.test_dataloader = [], None, [], None, [], None
        self.train_idmap, self.dev_idmap, self.test_idmap = {}, {}, {}
        self.n_entities = None
        self.n_relations = None
        self.n_base_feat, self.n_additional_feat, self.node_feat_dim = None, None, None
        self.full_adj_map = {}
        self.ent2id, self.id2ent, self.rel2id, self.id2rel = dict(), dict(), dict(), dict()
        self.full_edge_index, self.full_edge_attr = None, None
        self.base_feature_matrix = None
        self.lazy_load_ctr = 0
        self.dev_penalty_multiplier, self.test_penalty_multiplier = 1.0, 1.0
        self.task = task
        self.kb_system_file = kb_system_file
        self.load_dataset()
        self.query_enc_train, self.query_enc_dev, self.query_enc_test = None, None, None
        self.precomputed_query_encoding_dir = precomputed_query_encoding_dir
        if self.task == "pt_match":
            if self.precomputed_query_encoding_dir is not None:
                self.load_query_encodings()
        self.all_kg_map = None  # Map of all triples in the knowledge graph. Use this map only for filtering in evaluation.
        self.all_paths = None  # use this instead of traversing the entire subgraph
        if self.task == "kbc":
            file_names = [os.path.join(self.data_dir, f'{split}.txt') for split in ['graph', 'train', 'test', 'dev']]
            self.all_kg_map = load_data_all_triples(*file_names)
            logger.info("Reading graph file {}".format(paths_file_name))
            graph_pkl = os.path.join(self.data_dir, paths_file_name)
            with open(graph_pkl, "rb") as fin:
                self.all_paths = pickle.load(fin)

    def load_dataset(self):
        self.full_edge_index, self.full_edge_attr = [[], []], []
        if self.task == "pt_match":
            if self.dataset_name.lower() == "webqsp" or self.dataset_name.lower() == "cwq" or self.dataset_name.lower() == "metaqa" or self.dataset_name.lower() == "freebaseqa":
                seen_edges = set()
                logger.info("Reading vocab...")
                with open(os.path.join(self.data_dir,
                                       f"entities{self.data_file_suffix}.txt")) as fin:
                    for line_ctr, line in enumerate(fin):
                        line = line.strip()
                        self.ent2id[line] = line_ctr
                        self.id2ent[line_ctr] = line
                with open(os.path.join(self.data_dir,
                                       f"relations{self.data_file_suffix}.txt")) as fin:
                    for line_ctr, line in enumerate(fin):
                        line = line.strip()
                        self.rel2id[line] = line_ctr
                        self.id2rel[line_ctr] = line
                for split in ['train', 'dev', 'test']:
                    logger.info(
                        "Loading {}...".format(os.path.join(self.data_dir,
                                                            f'{split}{self.data_file_suffix}.json')))
                    with open(os.path.join(self.data_dir,
                                           f'{split}{self.data_file_suffix}.json')) as fin:
                        data = json.load(fin)
                        if split == "train":  # train file needs a lot of time to read, so cache it for later
                            self.raw_train_data = data
                    uniq_rels_per_q = []
                    for q_data in tqdm(data):
                        q_subgraph = q_data["subgraph"]["tuples"]
                        rels_per_q = set()
                        for triple in q_subgraph:
                            e1, r, e2 = triple
                            rels_per_q.add(r)
                            if (e1, r, e2) not in seen_edges:
                                self.full_adj_map.setdefault(self.id2ent[e1], {}).setdefault(self.id2rel[r], []).append(
                                    self.id2ent[e2])
                                self.full_edge_index[0].append(e1)
                                self.full_edge_index[1].append(e2)
                                self.full_edge_attr.append(r)
                                seen_edges.add((e1, r, e2))
                        uniq_rels_per_q.append(len(rels_per_q))
                    print("Num rels per question: {}".format(np.mean(uniq_rels_per_q)))
            elif self.dataset_name.lower() == "synthetic":
                if self.kb_system_file is None:
                    with open(os.path.join(self.data_dir, "dset_sampling_config.json")) as fin:
                        dset_config = json.load(fin)
                    self.kb_system_file = dset_config["kb_system"]
                    if not os.path.exists(self.kb_system_file):
                        self.kb_system_file = os.path.join(self.data_dir, os.path.basename(dset_config["kb_system"]))
                with open(self.kb_system_file) as fin:
                    kb_system = json.load(fin)
                if os.path.exists(os.path.join(self.data_dir, "relation_vocab.json")):
                    with open(os.path.join(self.data_dir, "relation_vocab.json")) as fin:
                        self.rel2id = json.load(fin)
                else:
                    self.rel2id = {rel_name: r_ctr for r_ctr, rel_name in enumerate(kb_system["rel_types"])}
                    # write relation vocab to be used later
                    with open(os.path.join(self.data_dir, "relation_vocab.json"), "w") as fout:
                        json.dump(self.rel2id, fout, indent=2)
                self.id2rel = {v: k for k, v in self.rel2id.items()}

                if os.path.exists(os.path.join(self.data_dir, "entity_vocab.json")):
                    with open(os.path.join(self.data_dir, "entity_vocab.json")) as fin:
                        self.ent2id = json.load(fin)
                else:
                    e_ctr = 0
                    for file_name in ["train.json", "dev.json", "test.json"]:
                        with open(os.path.join(self.data_dir, file_name)) as fin:
                            data = json.load(fin)
                        for d in data:
                            rels_per_q = set()
                            entities = d["graph"]["entities"]
                            for e in entities:
                                self.ent2id[e] = e_ctr
                                e_ctr += 1
                    # write entity vocab to be used later
                    with open(os.path.join(self.data_dir, "entity_vocab.json"), "w") as fout:
                        json.dump(self.ent2id, fout, indent=2)
                    self.id2ent = {v: k for k, v in self.ent2id.items()}

                seen_edges = set()
                uniq_rels_per_q = []
                for file_name in ["train.json", "dev.json", "test.json"]:
                    with open(os.path.join(self.data_dir, file_name)) as fin:
                        data = json.load(fin)
                    for d in data:
                        rels_per_q = set()
                        entities = d["graph"]["entities"]
                        adj_map = d["graph"]["adj_map"]
                        for e1, e1_map in adj_map.items():
                            for r, e2_list in e1_map.items():
                                for e2 in e2_list:
                                    if (e1, r, e2) not in seen_edges:
                                        rels_per_q.add(r)
                                        self.full_adj_map.setdefault(e1, {}).setdefault(r, []).append(e2)
                                        self.full_edge_index[0].append(self.ent2id[e1])
                                        self.full_edge_index[1].append(self.ent2id[e2])
                                        self.full_edge_attr.append(self.rel2id[r])
                                        seen_edges.add((e1, r, e2))
                        uniq_rels_per_q.append(len(rels_per_q))
                print("Num rels per question: {}".format(np.mean(uniq_rels_per_q)))
            else:
                kb_filenm = os.path.join(self.data_dir, 'kb.txt')
                with open(kb_filenm) as fin:
                    for line in fin:
                        e1, r, e2 = line.strip().split('|')
                        self.full_adj_map.setdefault(e1, {}).setdefault(r, []).append(e2)
                        # self.full_adj_map.setdefault(e2, {}).setdefault(r + '_inv', []).append(e1)
                        if e1 not in self.ent2id:
                            self.ent2id[e1] = len(self.ent2id)
                        if e2 not in self.ent2id:
                            self.ent2id[e2] = len(self.ent2id)
                        if r not in self.rel2id:
                            self.rel2id[r] = len(self.rel2id)
                        self.full_edge_index[0].append(self.ent2id[e1])
                        self.full_edge_index[1].append(self.ent2id[e2])
                        self.full_edge_attr.append(self.rel2id[r])
        elif self.task == "kbc":
            for split in ['train', 'test', 'dev', 'graph']:
                for line in open(os.path.join(self.data_dir, f'{split}.txt')):
                    e1, r, e2 = line.strip().split('\t')
                    if e1 not in self.ent2id:
                        self.ent2id[e1] = len(self.ent2id)
                    if e2 not in self.ent2id:
                        self.ent2id[e2] = len(self.ent2id)
                    if r not in self.rel2id:
                        self.rel2id[r] = len(self.rel2id)
                    if split == "graph":  # vocab created from all splits, but graph should not have test edges
                        self.full_adj_map.setdefault(e1, {}).setdefault(r, []).append(e2)
                        self.full_edge_index[0].append(self.ent2id[e1])
                        self.full_edge_index[1].append(self.ent2id[e2])
                        self.full_edge_attr.append(self.rel2id[r])
        else:
            logger.warning("Task not implemented!!!!")

        self.n_entities = len(self.ent2id)
        self.n_relations = len(self.rel2id)
        if self.add_inv_edges:
            # rel2id_for_inv_edges = {rel_name + '_inv': rel_idx + self.n_relations
            #                         for rel_name, rel_idx in self.rel2id.items()}
            # self.rel2id.update(rel2id_for_inv_edges)
            inv_rels = []
            for rel, _ in self.rel2id.items():
                rel_inv = rel + "_inv" if not rel.endswith("_inv") else rel[:-4]
                if rel_inv not in self.rel2id:
                    inv_rels.append(rel_inv)
            for rel in inv_rels:
                self.rel2id[rel] = len(self.rel2id)
            seen_edges = set()
            for e1_id, e2_id, r_id in zip(self.full_edge_index[0], self.full_edge_index[1], self.full_edge_attr):
                seen_edges.add((e1_id, r_id, e2_id))
            rev_edge_index, rev_edge_attr = [[], []], []
            for e1_id, e2_id, r_id in zip(self.full_edge_index[0], self.full_edge_index[1], self.full_edge_attr):
                rel_nm = self.id2rel[r_id]
                r_inv_id = self.rel2id[rel_nm + "_inv" if not rel_nm.endswith("_inv") else rel_nm[:-4]]
                if (e2_id, r_inv_id, e1_id) not in seen_edges:
                    seen_edges.add((e2_id, r_inv_id, e1_id))
                    rev_edge_index[0].append(e2_id)
                    rev_edge_index[1].append(e1_id)
                    rev_edge_attr.append(r_inv_id)
            self.full_edge_index[0].extend(rev_edge_index[0])
            self.full_edge_index[1].extend(rev_edge_index[1])
            self.full_edge_attr.extend(rev_edge_attr)
            self.full_adj_map = self.add_inv_edges_to_adj(self.full_adj_map)
            self.n_relations = len(self.rel2id)

        self.id2ent = {v: k for k, v in self.ent2id.items()}
        self.id2rel = {v: k for k, v in self.rel2id.items()}

        self.full_edge_index = torch.LongTensor(self.full_edge_index)
        self.full_edge_attr = torch.LongTensor(self.full_edge_attr)

        self.n_base_feat = self.n_relations if self.add_inv_edges else 2 * self.n_relations
        self.n_additional_feat = self.max_dist + 1 if self.add_dist_feature else 0
        self.node_feat_dim = self.n_base_feat + self.n_additional_feat

        sparse_inds = [[], []]
        for e1, r_dict in self.full_adj_map.items():
            for r, e2_list in r_dict.items():
                rel_idx = self.rel2id[r]
                sparse_inds[0].append(self.ent2id[e1])
                sparse_inds[1].append(rel_idx)
        self.base_feature_matrix = torch.sparse_coo_tensor(sparse_inds, torch.ones(len(sparse_inds[0])),
                                                           (len(self.ent2id), self.node_feat_dim))
        if self.task == "pt_match":
            if self.raw_train_data is None:
                if self.dataset_name.lower() == "webqsp" or self.dataset_name.lower() == "cwq" or self.dataset_name.lower() == "metaqa" or self.dataset_name.lower() == "freebaseqa":
                    with open(os.path.join(self.data_dir, f'train{self.data_file_suffix}.json')) as fin:
                        self.raw_train_data = json.load(fin)
                else:
                    with open(os.path.join(self.data_dir, "train.json")) as fin:
                        self.raw_train_data = json.load(fin)
            logger.info("Creating train id map...")
            item_ctr = 0
            raw_train_data_temp = []
            self.raw_train_drop_mask = []
            for item in self.raw_train_data:
                if "subgraph" in item and len(item["subgraph"]["tuples"]) == 0:
                    logger.warning("!!!! A train datapoint is being removed because of no subgraphs!!!")
                    self.raw_train_drop_mask.append(1)
                    continue
                self.train_idmap[item["id"]] = item_ctr
                raw_train_data_temp.append(item)
                self.raw_train_drop_mask.append(0)
                item_ctr += 1
            # Length of mask should match original dataset split length
            assert len(self.raw_train_data) == len(self.raw_train_drop_mask)
            self.raw_train_drop_mask = torch.BoolTensor(self.raw_train_drop_mask)
            # remove the example we ignored from the raw train data. Will it cause us problems later?
            self.raw_train_data = raw_train_data_temp
            for item in self.raw_train_data:
                self.train_dataset.append(
                    self.convert_rawdata_to_cbrdata(item, 'train', self.add_dist_feature, self.max_dist, lazy=True))
            assert len(self.train_idmap) == len(self.train_dataset)
            self.train_dataloader = DataListLoader(self.train_dataset, self.train_batch_size, shuffle=True)

            if self.dataset_name.lower() == "webqsp" or self.dataset_name.lower() == "cwq" or self.dataset_name.lower() == "metaqa" or self.dataset_name.lower() == "freebaseqa":
                with open(os.path.join(self.data_dir, f'dev{self.data_file_suffix}.json')) as fin:
                    self.raw_dev_data = json.load(fin)
            elif self.dataset_name.lower() == "synthetic":
                with open(os.path.join(self.data_dir, "dev.json")) as fin:
                    self.raw_dev_data = json.load(fin)
            item_ctr = 0
            raw_dev_data_temp = []
            self.raw_dev_drop_mask = []
            dev_penalty_ctr = 0
            for item in self.raw_dev_data:
                if "subgraph" in item and len(item["subgraph"]["tuples"]) == 0:
                    logger.warning("!!!! A DEV datapoint is being removed because of no subgraphs!!!")
                    self.raw_dev_drop_mask.append(1)
                    dev_penalty_ctr += 1
                    continue
                self.dev_idmap[item["id"]] = item_ctr
                raw_dev_data_temp.append(item)
                self.raw_dev_drop_mask.append(0)
                item_ctr += 1
            self.dev_penalty_multiplier = (len(self.raw_dev_data) - dev_penalty_ctr) / (len(self.raw_dev_data))
            # Length of mask should match original dataset split length
            assert len(self.raw_dev_data) == len(self.raw_dev_drop_mask)
            self.raw_dev_drop_mask = torch.BoolTensor(self.raw_dev_drop_mask)
            # remove the example we ignored from the raw dev data. Will it cause us problems later?
            self.raw_dev_data = raw_dev_data_temp
            for item in self.raw_dev_data:
                self.dev_dataset.append(
                    self.convert_rawdata_to_cbrdata(item, 'dev', self.add_dist_feature, self.max_dist, lazy=True))
            assert len(self.dev_idmap) == len(self.dev_dataset)
            if self.downsample_eval_frac < 1.0:
                rng_ = np.random.default_rng(42)
                subsample_sz = int(np.ceil(self.downsample_eval_frac * len(self.raw_dev_data)))
                self.dev_subsample_idx = sorted(rng_.choice(len(self.raw_dev_data), subsample_sz, replace=False))
                subsample_dev_data = [self.dev_dataset[idx_] for idx_ in self.dev_subsample_idx]
                self.dev_dataloader = DataListLoader(subsample_dev_data, self.eval_batch_size)
            else:
                self.dev_dataloader = DataListLoader(self.dev_dataset, self.eval_batch_size)

            if self.dataset_name.lower() == "webqsp" or self.dataset_name.lower() == "cwq" or self.dataset_name.lower() == "metaqa" or self.dataset_name.lower() == "freebaseqa":
                with open(os.path.join(self.data_dir, f'test{self.data_file_suffix}.json')) as fin:
                    self.raw_test_data = json.load(fin)
            else:
                with open(os.path.join(self.data_dir, "test.json")) as fin:
                    self.raw_test_data = json.load(fin)
            item_ctr = 0
            raw_test_data_temp = []
            self.raw_test_drop_mask = []
            test_penalty_ctr = 0
            for item in self.raw_test_data:
                if "subgraph" in item and len(item["subgraph"]["tuples"]) == 0:
                    logger.warning("!!!! A TEST datapoint is being removed because of no subgraphs!!!")
                    self.raw_test_drop_mask.append(1)
                    test_penalty_ctr += 1
                    continue
                self.test_idmap[item["id"]] = item_ctr
                raw_test_data_temp.append(item)
                self.raw_test_drop_mask.append(0)
                item_ctr += 1
            self.test_penalty_multiplier = (len(self.raw_test_data) - test_penalty_ctr) / (len(self.raw_test_data))
            # Length of mask should match original dataset split length
            assert len(self.raw_test_data) == len(self.raw_test_drop_mask)
            self.raw_test_drop_mask = torch.BoolTensor(self.raw_test_drop_mask)
            # remove the example we ignored from the raw test data. Will it cause us problems later?
            self.raw_test_data = raw_test_data_temp
            for item in self.raw_test_data:
                self.test_dataset.append(
                    self.convert_rawdata_to_cbrdata(item, 'test', self.add_dist_feature, self.max_dist, lazy=True))

            assert len(self.test_idmap) == len(self.test_dataset)
            self.test_dataloader = DataListLoader(self.test_dataset, self.eval_batch_size)

        elif self.task == "kbc":
            self.raw_train_data_map, self.raw_dev_data_map, self.raw_test_data_map = defaultdict(list), defaultdict(
                list), defaultdict(list)
            for line in open(os.path.join(self.data_dir, 'train.txt')):
                e1, r, e2 = line.strip().split('\t')
                self.raw_train_data_map[(e1, r)].append(e2)
                self.raw_train_data_map[(e2, r + "_inv")].append(e1)
            dev_penalty_ctr = 0
            for line in open(os.path.join(self.data_dir, 'dev.txt')):
                e1, r, e2 = line.strip().split('\t')
                if e1 not in self.full_adj_map:
                    dev_penalty_ctr += 1
                else:
                    self.raw_dev_data_map[(e1, r)].append(e2)
                if e2 not in self.full_adj_map:
                    dev_penalty_ctr += 1
                else:
                    self.raw_dev_data_map[(e2, r + "_inv")].append(e1)
            logger.warning(f"!!!! {dev_penalty_ctr} DEV queries were removed because of no subgraphs!!!")
            self.dev_penalty_multiplier = len(self.raw_dev_data_map) / (len(self.raw_dev_data_map) + dev_penalty_ctr)
            test_penalty_ctr = 0
            for line in open(os.path.join(self.data_dir, 'test.txt')):
                e1, r, e2 = line.strip().split('\t')
                if e1 not in self.full_adj_map:
                    test_penalty_ctr += 1
                else:
                    self.raw_test_data_map[(e1, r)].append(e2)
                if e2 not in self.full_adj_map:
                    test_penalty_ctr += 1
                else:
                    self.raw_test_data_map[(e2, r + "_inv")].append(e1)
            logger.warning(f"!!!! {test_penalty_ctr} TEST queries were removed because of no subgraphs!!!")
            self.test_penalty_multiplier = len(self.raw_test_data_map) / (
                    len(self.raw_test_data_map) + test_penalty_ctr)
            self.raw_train_data, self.raw_dev_data, self.raw_test_data = [], [], []
            for ctr, ((e1, r), e2_list) in enumerate(self.raw_train_data_map.items()):
                self.raw_train_data.append({"id": (e1, r), "question": r, "seed_entities": [e1], "answer": e2_list})
                self.train_idmap[(e1, r)] = ctr
            for ctr, ((e1, r), e2_list) in enumerate(self.raw_dev_data_map.items()):
                self.raw_dev_data.append({"id": (e1, r), "question": r, "seed_entities": [e1], "answer": e2_list})
                self.dev_idmap[(e1, r)] = ctr
            for ctr, ((e1, r), e2_list) in enumerate(self.raw_test_data_map.items()):
                self.raw_test_data.append({"id": (e1, r), "question": r, "seed_entities": [e1], "answer": e2_list})
                self.test_idmap[(e1, r)] = ctr

            for item in tqdm(self.raw_train_data):
                self.train_dataset.append(self.convert_rawdata_to_cbrdata(item, 'train', self.add_dist_feature,
                                                                          self.max_dist, lazy=True))
            for item in tqdm(self.raw_dev_data):
                self.dev_dataset.append(self.convert_rawdata_to_cbrdata(item, 'dev', self.add_dist_feature,
                                                                        self.max_dist, lazy=True))
            for item in tqdm(self.raw_test_data):
                self.test_dataset.append(self.convert_rawdata_to_cbrdata(item, 'test', self.add_dist_feature,
                                                                         self.max_dist, lazy=True))

            self.train_dataloader = DataListLoader(self.train_dataset, self.train_batch_size, shuffle=True)
            self.dev_dataloader = DataListLoader(self.dev_dataset, self.eval_batch_size, shuffle=False)
            self.test_dataloader = DataListLoader(self.test_dataset, self.eval_batch_size, shuffle=False)

    @staticmethod
    def add_inv_edges_to_adj(adj_map):
        full_adj_map = deepcopy(adj_map)
        for e1, re2_map in adj_map.items():
            for r, e2_list in re2_map.items():
                # r_inv = r + '_inv'
                r_inv = r + "_inv" if not r.endswith("_inv") else r[:-4]
                for e2 in e2_list:
                    if e2 not in full_adj_map: full_adj_map[e2] = {}
                    if r_inv not in full_adj_map[e2]: full_adj_map[e2][r_inv] = []
                    full_adj_map[e2][r_inv].append(e1)
        for e1, re2_map in full_adj_map.items():
            for r in re2_map:
                re2_map[r] = sorted(set(re2_map[r]))
        return full_adj_map

    @staticmethod
    def compute_shortest_distances(edge_index, sources, node_set, max_dist):
        dist_arr = torch.ones(len(node_set), dtype=torch.long) * 1000
        source_mask = edge_index.new_full((len(node_set),), False, dtype=torch.bool)
        source_mask[sources] = True
        explored = source_mask.clone()
        dist_arr[source_mask] = 0
        row, col = edge_index
        for d in range(1, max_dist + 1):
            edge_mask = torch.index_select(source_mask, 0, row)
            source_mask.fill_(False)
            source_mask[col[edge_mask]] = True
            source_mask = source_mask & ~explored
            explored |= source_mask
            assert torch.all(dist_arr[source_mask] > d)
            dist_arr[source_mask] = d

        return dist_arr

    def convert_rawdata_to_cbrdata(self, raw_data: dict, split: str, add_dist_feature: bool = False, max_dist: int = 3,
                                   lazy: bool = False, inplace_obj: KBQAGraphData = None):
        if self.task == "pt_match":
            knn_ids = []
            for kid in raw_data["knn"]:
                if split != 'train' or kid != raw_data["id"]:
                    if kid in self.train_idmap:
                        knn_ids.append(self.train_idmap[kid])
            knn_ids = torch.LongTensor(knn_ids)
        else:
            knn_ids = None
        if split == 'train':
            ex_id = self.train_idmap[raw_data["id"]]
        elif split == 'dev':
            ex_id = self.dev_idmap[raw_data["id"]]
        elif split == 'test':
            ex_id = self.test_idmap[raw_data["id"]]
        else:
            raise ValueError(f"Unknown split {split}")
        ques_str = raw_data["question"] if self.task == "pt_match" and (
                self.dataset_name.lower() == 'webqsp' or self.dataset_name.lower() == 'cwq') else raw_data["id"]

        raw_data["answer"] = list(set(raw_data["answer"]))  # in CWQ, sometimes answers are repeatd causing errors later
        if lazy:
            return KBQAGraphData(None, None, None, None, None, None, split, ex_id, raw_data["id"], ques_str,
                                 raw_data["answer"], knn_ids, torch.LongTensor([0]))

        if self.task == "pt_match":
            if self.dataset_name.lower() in ["webqsp", "cwq", "metaqa", "synthetic", "freebaseqa"]:
                sub_nodes, sub_edge_index, seed_ent_loc, sub_edge_attr = self.get_cached_k_hop_subgraph(raw_data)
            else:
                seed_node_idx = [self.ent2id[e_] for e_ in raw_data["seed_entities"]]
                sub_nodes, sub_edge_index, seed_ent_loc, sub_edge_mask = k_hop_subgraph(node_idx=seed_node_idx,
                                                                                        num_hops=max_dist,
                                                                                        edge_index=self.full_edge_index,
                                                                                        relabel_nodes=True,
                                                                                        flow='target_to_source',
                                                                                        num_nodes=len(self.ent2id))
                sub_edge_attr = self.full_edge_attr[sub_edge_mask]
        elif self.task == "kbc":
            sub_nodes, sub_edge_index, seed_ent_loc, sub_edge_attr = self.get_cached_k_hop_subgraph(raw_data)
        else:
            raise NotImplementedError(f"Task {self.task} is invalid")

        x = sub_nodes
        dist_feats = None
        if add_dist_feature:
            dist_feats = self.compute_shortest_distances(sub_edge_index, seed_ent_loc, sub_nodes, max_dist)
            try:
                assert torch.max(dist_feats) <= max_dist
            except:
                #  in CWQ, few values in the graph are disconnected
                dist_feats[torch.nonzero(dist_feats > max_dist)] = max_dist

        seed_nodes_mask = torch.zeros(x.shape[0], dtype=torch.bool)
        seed_nodes_mask[seed_ent_loc] = 1
        label_nodes_mask = torch.zeros(x.shape[0], dtype=torch.bool)
        ans_node_idx = [self.ent2id[e_] for e_ in raw_data["answer"]]
        penalty = 0
        for aid in ans_node_idx:
            if not torch.any(sub_nodes == aid):
                penalty += 1
            label_nodes_mask[sub_nodes == aid] = 1

        assert len(ans_node_idx) == penalty + label_nodes_mask.sum()
        penalty = torch.LongTensor([penalty])
        if inplace_obj is not None:
            self.lazy_load_ctr += 1
            inplace_obj.x = x
            inplace_obj.edge_index = sub_edge_index
            inplace_obj.edge_attr = sub_edge_attr
            inplace_obj.dist_feats = dist_feats
            inplace_obj.seed_node_ids = seed_nodes_mask
            inplace_obj.label_node_ids = label_nodes_mask
            inplace_obj.penalty = penalty
            return inplace_obj
        else:
            return KBQAGraphData(x, sub_edge_index, sub_edge_attr, dist_feats, seed_nodes_mask, label_nodes_mask,
                                 split, ex_id, raw_data["id"], ques_str, raw_data["answer"], knn_ids, penalty)

    def load_query_encodings(self):
        self.query_enc_train = torch.load(os.path.join(self.precomputed_query_encoding_dir, 'query_enc_train.pt'),
                                          map_location='cpu')
        self.query_enc_train = self.query_enc_train[~self.raw_train_drop_mask]
        assert len(self.train_idmap) == self.query_enc_train.shape[0]
        self.query_enc_dev = torch.load(os.path.join(self.precomputed_query_encoding_dir, 'query_enc_dev.pt'),
                                        map_location='cpu')
        self.query_enc_dev = self.query_enc_dev[~self.raw_dev_drop_mask]
        assert len(self.dev_idmap) == self.query_enc_dev.shape[0]
        self.query_enc_test = torch.load(os.path.join(self.precomputed_query_encoding_dir, 'query_enc_test.pt'),
                                         map_location='cpu')
        self.query_enc_test = self.query_enc_test[~self.raw_test_drop_mask]
        assert len(self.test_idmap) == self.query_enc_test.shape[0]

    def get_cached_k_hop_subgraph(self, query):
        if self.dataset_name.lower() == "webqsp" or self.dataset_name.lower() == "cwq" or self.dataset_name.lower() == "metaqa" or self.dataset_name.lower() == "freebaseqa":
            # find entities in the query subgraph
            all_entities = set(query["subgraph"]["entities"])
            # add seed entities to this set as few questions of CWQ fail. However I dont know how much we can
            # do for those questions if the query entities are missing
            for s_e in query["seed_entities"]:
                all_entities.add(s_e)
            local_vocab = {}
            sub_nodes = []
            sub_edge_indx = [[], []]
            sub_edge_attr = []
            for i in range(len(self.id2ent)):
                if i in all_entities:
                    local_vocab[i] = len(local_vocab)
                    sub_nodes.append(i)
            seen_edges = set()
            for triple in query["subgraph"]["tuples"]:
                e1, r, e2 = triple
                if (e1, r, e2) not in seen_edges:
                    sub_edge_indx[0].append(local_vocab[e1])
                    sub_edge_indx[1].append(local_vocab[e2])
                    sub_edge_attr.append(r)
                    seen_edges.add((e1, r, e2))
                # add inverse edges as well
                if self.add_inv_edges:
                    rel_inv = self.id2rel[r] + "_inv" if not self.id2rel[r].endswith("_inv") else self.id2rel[r][:-4]
                    inv_r = self.rel2id[rel_inv]
                    if (e2, inv_r, e1) not in seen_edges:
                        sub_edge_indx[0].append(local_vocab[e2])
                        sub_edge_indx[1].append(local_vocab[e1])
                        sub_edge_attr.append(inv_r)
                        seen_edges.add((e2, inv_r, e1))

            seed_ent_loc = [local_vocab[s_e] for s_e in query["seed_entities"]]
        elif self.dataset_name.lower() == "synthetic":
            all_entities = set(query["graph"]["entities"])
            for s_e in query["seed_entities"]:
                all_entities.add(s_e)
            local_vocab = {}
            sub_nodes = []
            sub_edge_indx = [[], []]
            sub_edge_attr = []
            for i in range(len(self.id2ent)):
                if self.id2ent[i] in all_entities:
                    local_vocab[self.id2ent[i]] = len(local_vocab)
                    sub_nodes.append(i)
            seen_edges = set()
            adj_map = query["graph"]["adj_map"]
            for e1, e1_map in adj_map.items():
                for r, e2_list in e1_map.items():
                    for e2 in e2_list:
                        if (e1, r, e2) not in seen_edges:
                            sub_edge_indx[0].append(local_vocab[e1])
                            sub_edge_indx[1].append(local_vocab[e2])
                            sub_edge_attr.append(self.rel2id[r])
                            seen_edges.add((e1, r, e2))
                        # add inverse edges as well
                        if self.add_inv_edges:
                            inv_r = self.id2rel[self.rel2id[r + "_inv"]]
                            if (e2, inv_r, e1) not in seen_edges:
                                sub_edge_indx[0].append(local_vocab[e2])
                                sub_edge_indx[1].append(local_vocab[e1])
                                sub_edge_attr.append(self.rel2id[inv_r])
                                seen_edges.add((e2, inv_r, e1))
            seed_ent_loc = []
            for i, ent_id in enumerate(sub_nodes):
                if self.id2ent[ent_id] in query["seed_entities"]:
                    seed_ent_loc.append(i)
        else:
            e1 = query["seed_entities"][0]
            assert self.all_paths is not None
            paths_e1 = self.all_paths[e1]
            all_entities = set()
            sub_edge_indx = [[], []]
            sub_edge_attr = []
            sub_nodes = []
            local_vocab = {}
            # collect all entities in the local subgraph first
            all_entities.add(e1)
            for path in paths_e1:
                for rel, ent in path:
                    all_entities.add(ent)
            # Add entities from one-hop edges
            for rel in self.full_adj_map[e1].keys():
                all_entities.update(self.raw_train_data_map[(e1, rel)])
            for i in range(len(self.id2ent)):
                if self.id2ent[i] in all_entities:
                    local_vocab[self.id2ent[i]] = len(local_vocab)
                    sub_nodes.append(i)
            seen_edges = set()
            for path in paths_e1:
                curr_ent = e1
                for rel, ent in path:
                    if rel.startswith("_"):
                        rel = rel[1:] + "_inv"
                    if (curr_ent, rel, ent) not in seen_edges:
                        sub_edge_indx[0].append(local_vocab[curr_ent])
                        sub_edge_indx[1].append(local_vocab[ent])
                        sub_edge_attr.append(self.rel2id[rel])
                        seen_edges.add((curr_ent, rel, ent))
                    # also add inverse edges
                    if rel.endswith("_inv"):
                        inv_rel = rel[:-4]
                    else:
                        inv_rel = rel + "_inv"
                    if (ent, inv_rel, curr_ent) not in seen_edges:
                        sub_edge_indx[0].append(local_vocab[ent])
                        sub_edge_indx[1].append(local_vocab[curr_ent])
                        sub_edge_attr.append(self.rel2id[inv_rel])
                        seen_edges.add((ent, inv_rel, curr_ent))
                    curr_ent = ent
            # Add one-hop edges
            for rel in self.full_adj_map[e1].keys():
                if rel.endswith("_inv"):
                    inv_rel = rel[:-4]
                else:
                    inv_rel = rel + "_inv"
                for ent in self.raw_train_data_map[(e1, rel)]:
                    if (e1, rel, ent) not in seen_edges:
                        sub_edge_indx[0].append(local_vocab[e1])
                        sub_edge_indx[1].append(local_vocab[ent])
                        sub_edge_attr.append(self.rel2id[rel])
                        seen_edges.add((e1, rel, ent))
                    # also add inverse edges
                    if (ent, inv_rel, e1) not in seen_edges:
                        sub_edge_indx[0].append(local_vocab[ent])
                        sub_edge_indx[1].append(local_vocab[e1])
                        sub_edge_attr.append(self.rel2id[inv_rel])
                        seen_edges.add((ent, inv_rel, e1))
            seed_ent_loc = []
            for i, ent in enumerate(sub_nodes):
                if ent == self.ent2id[e1]:
                    seed_ent_loc.append(i)
                    break
        return torch.LongTensor(sub_nodes), torch.LongTensor(sub_edge_indx), \
               torch.LongTensor(seed_ent_loc), torch.LongTensor(sub_edge_attr)
