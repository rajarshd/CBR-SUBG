import os
from collections import defaultdict
from tqdm import tqdm
import pickle
from numpy.random import default_rng
import numpy as np
import argparse
import wandb

rng = default_rng()
from adaptive_subgraph_collection.adaptive_utils import get_query_entities_and_answers, \
    get_query_entities_and_answers_cwq, execute_kb_query_for_hops, get_query_entities_and_answers_freebaseqa, \
    get_query_entities_and_answers_metaqa, read_metaqa_kb, find_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect subgraphs around entities using CBR")
    parser.add_argument("--train_file", type=str,
                        default='/mnt/nfs/scratch1/rajarshi/cbr-weak-supervision/data_with_mentions/webqsp_data_with_mentions/train.json')
    parser.add_argument("--dataset_name", type=str, default='webqsp')
    parser.add_argument("--output_dir", type=str,
                        default='/mnt/nfs/scratch1/rajarshi/cbr-weak-supervision/subgraphs/webqsp_gold_entities')
    parser.add_argument("--use_gold_entities", action='store_true')
    parser.add_argument("--metaqa_kb_file", type=str, default="/mnt/nfs/scratch1/rajarshi/cbr-weak-supervision/MetaQA-synthetic/3-hop/kb.txt")
    parser.add_argument("--job_id", type=int, default=0)
    parser.add_argument("--total_jobs", type=int, default=1)
    parser.add_argument("--use_wandb", type=int, default=1)
    args = parser.parse_args()
    args.use_wandb = (args.use_wandb == 1)
    if args.use_wandb:
        wandb.init("adaptive-subgraph-collection")

    if args.dataset_name.lower() == 'webqsp':
        qid2qents, qid2answers, qid2gold_chains, qid2q_str = get_query_entities_and_answers(args.train_file,
                                                                                            return_gold_entities=args.use_gold_entities)
    elif args.dataset_name.lower() == 'cwq':
        qid2qents, qid2answers, qid2gold_spqls, qid2q_str = get_query_entities_and_answers_cwq(args.train_file,
                                                                                               return_gold_entities=args.use_gold_entities)
    elif args.dataset_name.lower() == 'freebaseqa':
        qid2qents, qid2answers, qid2gold_spqls, qid2q_str = get_query_entities_and_answers_freebaseqa(args.train_file)

    elif args.dataset_name.lower() == 'metaqa':
        qid2qents, qid2answers, qid2gold_spqls, qid2q_str = get_query_entities_and_answers_metaqa(args.train_file,
                                                                                                  return_gold_entities=args.use_gold_entities)

    if args.dataset_name.lower() == 'metaqa':  # metaqa has its own KB and not full Freebase, hence do not need SPARQL
        # read metaqa KB
        # find 1, 2, 3 hop paths between question entities and answers
        all_subgraphs = defaultdict(list)
        e1_map = read_metaqa_kb(args.metaqa_kb_file)
        qid2qents = [(qid, q_ents) for (qid, q_ents) in sorted(qid2qents.items(), key=lambda item: item[0])]
        job_size = len(qid2qents) / args.total_jobs
        st = args.job_id * job_size
        en = (1 + args.job_id) * job_size
        print("St: {}, En: {}".format(st, en))
        empty_ctr = 0
        all_len = []
        for ctr, (qid, q_ents) in tqdm(enumerate(qid2qents)):
            if st <= ctr < en:
                ans_ents = qid2answers[qid]
                len_q = 0
                for q_ent in q_ents:
                    for ans_ent in ans_ents:
                        paths = find_paths(e1_map, q_ent, ans_ent)
                        all_subgraphs[qid].append({'st': q_ent, 'en': ans_ent, 'chains': paths})
                        len_q += len(paths)
                if len_q == 0:
                    empty_ctr += 1
                all_len.append(len_q)

        print("Empty_ctr: {} out of {} queries".format(empty_ctr, (en - st)))
        out_file = os.path.join(args.output_dir, "{}_train_chains_{}.pkl".format(args.dataset_name.lower(), str(args.job_id)))
        print("Writing file at {}".format(out_file))
        with open(out_file, "wb") as fout:
            pickle.dump(all_subgraphs, fout)

    else:
        all_subgraphs = defaultdict(list)
        all_len = []
        for ctr, (qid, q_ents) in tqdm(enumerate(qid2qents.items())):
            ans_ents = qid2answers[qid]
            for q_ent in q_ents:
                for ans_ent in ans_ents:
                    spql_2_hop = "select distinct ?r1 ?r2 where { " + "ns:" + q_ent + " ?r1 ?e1 . ?e1 ?r2 ns:" + ans_ent + ". }"
                    ret = execute_kb_query_for_hops(spql_2_hop, hop=2)
                    is_exception = ret[1]
                    if not is_exception:
                        all_subgraphs[qid].append({'st': q_ent, 'en': ans_ent, 'chains': ret[0]})
                        all_len.append(len(ret[0]))
                    else:
                        print(spql_2_hop)
                    spql_1_hop = "select distinct ?r1 where { " + "ns:" + q_ent + " ?r1 ns:" + ans_ent + ". }"
                    ret = execute_kb_query_for_hops(spql_1_hop, hop=1)
                    if not is_exception:
                        all_subgraphs[qid].append({'st': q_ent, 'en': ans_ent, 'chains': ret[0]})
                        all_len.append(len(ret[0]))
                    else:
                        print(spql_1_hop)
        # there are some qids for which the (above) query didnt execute because
        # the entities are string literals and the queries above dont work
        # To handle them, issue a special query that look for those strings in the
        # immediate neighborhood. Unfortunately we can only look in the one-hop neighborhood.
        # Doing more that takes way too much per query. Worth asking some SPARQL expert, how
        # to handle such cases.
        print("Number of queries: {}".format(len(all_subgraphs)))
        empty_qids = set()
        for qid, _ in qid2qents.items():
            if qid not in all_subgraphs:
                empty_qids.add(qid)
        for empty_qid in tqdm(empty_qids):
            q_ents = qid2qents[empty_qid]
            answers = qid2answers[empty_qid]
            for q_ent in q_ents:
                for ans_ent in answers:
                    spql_1_hop_literal = "select distinct ?r1 where { " + "ns:" + q_ent + " ?r1 ?e1. FILTER(STR(?e1) = '" + ans_ent + "') }"
                    ret = execute_kb_query_for_hops(spql_1_hop_literal, hop=1)
                    is_exception = ret[1]
                    if not is_exception:
                        all_subgraphs[empty_qid].append({'st': q_ent, 'en': ans_ent, 'chains': ret[0]})
                        all_len.append(len(ret[0]))
                    else:
                        print(spql_1_hop_literal)
        print("Number of queries after executing query for literals: {}".format(len(all_subgraphs)))
        out_file = os.path.join(args.output_dir, "{}_2_hop_train_chains.pkl".format(args.dataset_name.lower()))
        with open(out_file, "wb") as fout:
            pickle.dump(all_subgraphs, fout)
        print("Min: {}, Mean: {}, Median: {}, Max:{}".format(np.min(all_len), np.mean(all_len), np.median(all_len),
                                                             np.max(all_len)))
