from tqdm import tqdm
from collections import defaultdict
import numpy as np
import tempfile
from typing import DefaultDict, List, Tuple, Dict, Set
import pickle


def augment_kb_with_inv_edges(file_name: str) -> None:
    # Create temporary file read/write
    t = tempfile.NamedTemporaryFile(mode="r+")
    # Open input file read-only
    i = open(file_name, 'r')

    # Copy input file to temporary file, modifying as we go
    temp_list = []
    for line in i:
        t.write(line.strip() + "\n")
        e1, r, e2 = line.strip().split("\t")
        temp_list.append((e1, r, e2))
        temp_list.append((e2, "_" + r, e1))

    i.close()  # Close input file
    o = open(file_name, "w")  # Reopen input file writable
    # Overwriting original file with temporary file contents
    for (e1, r, e2) in temp_list:
        o.write("{}\t{}\t{}\n".format(e1, r, e2))
    t.close()  # Close temporary file, will cause it to be deleted
    o.close()


def create_adj_list(file_name: str) -> DefaultDict[str, Tuple[str, str]]:
    out_map = defaultdict(list)
    fin = open(file_name)
    for line_ctr, line in tqdm(enumerate(fin)):
        line = line.strip()
        e1, r, e2 = line.split("\t")
        out_map[e1].append((r, e2))
    return out_map


def load_data(file_name: str) -> DefaultDict[Tuple[str, str], list]:
    out_map = defaultdict(list)
    fin = open(file_name)

    for line in tqdm(fin):
        line = line.strip()
        e1, r, e2 = line.split("\t")
        out_map[(e1, r)].append(e2)

    return out_map


def load_data_all_triples(train_file: str, dev_file: str, test_file: str) -> DefaultDict[Tuple[str, str], list]:
    """
    Returns a map of all triples in the knowledge graph. Use this map only for filtering in evaluation.
    :param train_file:
    :param dev_file:
    :param test_file:
    :return:
    """
    out_map = defaultdict(list)
    for file_name in [train_file, dev_file, test_file]:
        fin = open(file_name)
        for line in tqdm(fin):
            line = line.strip()
            e1, r, e2 = line.split("\t")
            out_map[(e1, r)].append(e2)
    return out_map


def create_vocab(kg_file: str) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
    entity_vocab, rev_entity_vocab = {}, {}
    rel_vocab, rev_rel_vocab = {}, {}
    fin = open(kg_file)
    entity_ctr, rel_ctr = 0, 0
    for line in tqdm(fin):
        line = line.strip()
        e1, r, e2 = line.split("\t")
        if e1 not in entity_vocab:
            entity_vocab[e1] = entity_ctr
            rev_entity_vocab[entity_ctr] = e1
            entity_ctr += 1
        if e2 not in entity_vocab:
            entity_vocab[e2] = entity_ctr
            rev_entity_vocab[entity_ctr] = e2
            entity_ctr += 1
        if r not in rel_vocab:
            rel_vocab[r] = rel_ctr
            rev_rel_vocab[rel_ctr] = r
            rel_ctr += 1
    return entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab


def read_graph(file_name: str, entity_vocab: Dict[str, int], rel_vocab: Dict[str, int]) -> np.ndarray:
    adj_mat = np.zeros((len(entity_vocab), len(rel_vocab)))
    fin = open(file_name)
    for line in tqdm(fin):
        line = line.strip()
        e1, r, _ = line.split("\t")
        adj_mat[entity_vocab[e1.lower()], rel_vocab[r.lower()]] = 1

    return adj_mat


def load_mid2str(mid2str_file: str) -> DefaultDict[str, str]:
    mid2str = defaultdict(str)
    with open(mid2str_file) as fin:
        for line in tqdm(fin):
            line = line.strip()
            try:
                mid, ent_name = line.split("\t")
            except ValueError:
                continue
            if mid not in mid2str:
                mid2str[mid] = ent_name
    return mid2str


def get_unique_entities(kg_file: str) -> Set[str]:
    unique_entities = set()
    fin = open(kg_file)
    for line in fin:
        e1, r, e2 = line.strip().split()
        unique_entities.add(e1)
        unique_entities.add(e2)
    fin.close()
    return unique_entities


def get_entities_group_by_relation(file_name: str) -> DefaultDict[str, List[str]]:
    rel_to_ent_map = defaultdict(list)
    fin = open(file_name)
    for line in fin:
        e1, r, e2 = line.strip().split()
        rel_to_ent_map[r].append(e1)
    return rel_to_ent_map


def get_inv_relation(r: str, dataset_name="nell") -> str:
    if dataset_name == "nell":
        if r[-4:] == "_inv":
            return r[:-4]
        else:
            return r + "_inv"
    else:
        if r[:2] == "__" or r[:2] == "_/":
            return r[1:]
        else:
            return "_" + r


def return_nearest_relation_str(sim_sorted_ind, rev_rel_vocab, rel, k=5):
    """
    helper method to print nearest relations
    :param sim_sorted_ind: sim matrix sorted wrt index
    :param rev_rel_vocab:
    :param rel: relation we want sim for
    :return:
    """
    print("====Query rel: {}====".format(rev_rel_vocab[rel]))
    nearest_rel_inds = sim_sorted_ind[rel, :k]
    return [rev_rel_vocab[i] for i in nearest_rel_inds]


def get_node_degree(file: str) -> Dict[str, Dict[str, int]]:
    fin = open(file)
    degree = {}
    for line in tqdm(fin):
        line = line.strip()
        e1, r, e2 = line.split("\t")
        if e1 not in degree:
            degree[e1] = {}
            degree[e1]["out"] = 1
            degree[e1]["in"] = 0
            degree[e1]["total"] = 1
        else:
            degree[e1]["out"] += 1
            degree[e1]["total"] += 1
        if e2 not in degree:
            degree[e2] = {}
            degree[e2]["in"] = 1
            degree[e2]["out"] = 0
            degree[e2]["total"] = 1
        else:
            degree[e2]["in"] += 1
            degree[e2]["total"] += 1

    for d in degree:
        in_degree = degree[d]["in"]
        out_degree = degree[d]["out"]
        total_degree = degree[d]["total"]
        assert (in_degree + out_degree == total_degree), "Error in calculating degree"

    return degree


def load_data_all_triples(graph_file: str, train_file: str, dev_file: str, test_file: str) -> DefaultDict[
    Tuple[str, str], list]:
    """
    Returns a map of all triples in the knowledge graph. Use this map only for filtering in evaluation.
    :param train_file:
    :param dev_file:
    :param test_file:
    :return:
    """
    out_map = defaultdict(list)
    for file_name in [graph_file, train_file, dev_file, test_file]:
        fin = open(file_name)
        for line in tqdm(fin):
            line = line.strip()
            e1, r, e2 = line.split("\t")
            out_map[(e1, r)].append(e2)
            out_map[(e2, r + "_inv")].append(e1)
    return out_map


def calculate_upper_bound(graph_pkl: str, eval_file: str):
    query_map = defaultdict(list)
    total_queries, total_covered = 0, 0
    all_subgraphs_num_nodes, all_subgraphs_num_edges = [], []
    with open(eval_file) as fin:
        for line in fin:
            e1, r, e2 = line.strip().split("\t")
            query_map[(e1, r)].append(e2)

    with open(graph_pkl, "rb") as fin:
        paths = pickle.load(fin)
    for query, ans_list in tqdm(query_map.items()):
        e1, r = query
        all_entities_in_subgraph = set()
        unique_edges = set()
        all_entities_in_subgraph.add(e1)
        paths_e1 = paths[e1]
        for p in paths_e1:
            parent = e1
            for rel, ent in p:
                unique_edges.add((parent, rel, ent))
                all_entities_in_subgraph.add(ent)
                parent = ent
        # now check how many entities are in the query list to get coverage
        all_subgraphs_num_nodes.append(len(all_entities_in_subgraph))
        all_subgraphs_num_edges.append(len(unique_edges))
        for e2 in ans_list:
            if e2 in all_entities_in_subgraph:
                total_covered += 1
            total_queries += 1
    print("Avg num nodes in subgraph: {}, avg num edges in subgraph: {}".format(np.mean(all_subgraphs_num_nodes),
                                                                                np.mean(all_subgraphs_num_edges)))
    print("Total queries: {}, total covered: {}, %covered: {}".format(total_queries, total_covered,
                                                                      (total_covered / total_queries) * 100))

