import argparse
import shutil

import faiss
import json
import numpy as np
import re
import os
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel
import torch


def get_khop_subgraph(seed_ent, full_kb, k):
    src_nodes = set(seed_ent)
    explored = set()
    new_src = set()
    subgraph = {}
    for step in range(k):
        for s_e in src_nodes:
            subgraph[s_e] = full_kb[s_e]
            for r, e2_list in full_kb[s_e].items():
                new_src.update(e2_list)
        explored.update(src_nodes)
        src_nodes = new_src - explored
        new_src = set()
    return subgraph, explored.union(src_nodes)


def graph_analytics(graph):
    node_set, edge_ctr = set(), 0
    for e1, re2_map in graph.items():
        node_set.add(e1)
        for r, e2_list in re2_map.items():
            node_set.update(e2_list)
            edge_ctr += len(e2_list)
    return len(node_set), edge_ctr


def load_dataset(filenm, full_kb, entity_vocab):
    dset = []
    dset_analytics = [[], [], []]
    with open(filenm) as fin:
        for line_ctr, line in tqdm(enumerate(fin)):
            query = {'id': line_ctr, 'seed_entities': set()}
            q, ans = line.strip().split('\t')
            query['question'] = q
            query['answer'] = ans.split('|')
            for a in query['answer']:
                assert a in entity_vocab

            seed_e = re.findall(r'\[(.*)\]', q)
            for s_e in seed_e:
                if s_e in entity_vocab:
                    query['seed_entities'].add(s_e)
                if s_e.title() in entity_vocab:
                    query['seed_entities'].add(s_e.title())
            query['seed_entities'] = list(query['seed_entities'])
            assert len(query['seed_entities']) > 0
            khop_adj, subgraph_entity_set = get_khop_subgraph(query['seed_entities'], full_kb, 3)
            subgraph_stats = graph_analytics(khop_adj)
            dset_analytics[0].append(subgraph_stats[0])
            dset_analytics[1].append(subgraph_stats[1])
            query['debugging'] = {'missing_ans': []}
            missing_ans = False
            for a in query['answer']:
                if a not in subgraph_entity_set:
                    missing_ans = True
                    query['debugging']['missing_ans'].append(a)
            dset_analytics[2].append(1 if missing_ans else 0)
            dset.append(query)
    return dset, dset_analytics


def encode_str_batch(batch, tokenizer, encoder, device):
    curr_batch = tokenizer(batch, padding=True, return_tensors='pt')
    outputs = encoder(input_ids=curr_batch["input_ids"].to(device),
                      attention_mask=curr_batch["attention_mask"].to(device))
    query_vecs = outputs.pooler_output.cpu().numpy()
    query_vecs /= np.linalg.norm(query_vecs, axis=1, keepdims=True)
    return query_vecs


def add_neighbors(args, case_index, dset, tokenizer, encoder, device, split='eval'):
    query_list = [query['question'] for query in dset]
    with torch.no_grad():
        for idx in trange(int(np.ceil(len(query_list) / args.eval_batch_size))):
            curr_batch = query_list[idx * args.eval_batch_size: (idx + 1) * args.eval_batch_size]
            query_vecs = encode_str_batch(curr_batch, tokenizer, encoder, device)
            case_dist, case_id = case_index.search(query_vecs, args.n_cases + 1 if split == 'train' else args.n_cases)
            if idx == 0:
                print("Sanity check")
                print(case_id[:4])
                print(case_dist[:4])
            for k in range(len(curr_batch)):
                ex_id = idx * args.eval_batch_size + k
                assert 'knn' not in dset[ex_id]
                if split == 'train':
                    dset[ex_id]["knn"] = [int(c) for c in case_id[k] if c != ex_id][:args.n_cases]
                else:
                    dset[ex_id]["knn"] = case_id[k].tolist()


def main(args):
    kb = {}
    entity_vocab, rel_vocab = set(), set()
    with open(os.path.join(args.src_dir, 'kb.txt')) as fin:
        for line in fin:
            e1, r, e2 = line.strip().split('|')
            kb.setdefault(e1, {}).setdefault(r, []).append(e2)
            kb.setdefault(e2, {}).setdefault(r+'_inv', []).append(e1)
            entity_vocab.add(e1)
            entity_vocab.add(e2)
            rel_vocab.add(r)
            rel_vocab.add(r+'_inv')
    full_kb_stats = graph_analytics(kb)
    assert full_kb_stats[0] == len(entity_vocab)
    print(f"KB has {full_kb_stats[0]} entities, {len(rel_vocab)} relations and {full_kb_stats[1]} edges")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoder = AutoModel.from_pretrained(args.model).to(device)
    encoder.eval()

    for split in ['1-hop', '2-hop', '3-hop']:
        print(f"Split: {split}")
        if not os.path.isdir(os.path.join(args.output_dir, split)):
            os.makedirs(os.path.join(args.output_dir, split))

        shutil.copyfile(os.path.join(args.src_dir, 'kb.txt'), os.path.join(args.output_dir, split, 'kb.txt'))

        print("Loading train set")
        train_dset, train_dset_analytics = load_dataset(os.path.join(args.src_dir, split, 'vanilla', 'qa_train.txt'),
                                                        kb, entity_vocab)

        index_path = os.path.join(args.output_dir, split, 'case_index')
        if not os.path.exists(index_path):
            print("Encoding train queries")
            train_queries = [query['question'] for query in train_dset]
            case_index = faiss.IndexFlatIP(encoder.config.hidden_size)
            all_indices = []
            with torch.no_grad():
                for idx in range(int(np.ceil(len(train_queries)/args.eval_batch_size))):
                    curr_batch = train_queries[idx * args.eval_batch_size: (idx + 1) * args.eval_batch_size]
                    context_vecs = encode_str_batch(curr_batch, tokenizer, encoder, device)
                    all_indices.append(context_vecs)
            all_indices = np.concatenate(all_indices)
            print(f"Adding {all_indices.shape[0]} vectors to the index")
            case_index.add(all_indices)
            print(f"Saving index to {index_path}")
            faiss.write_index(case_index, index_path)
        else:
            print("Loading train query index")
            case_index = faiss.read_index(index_path)

        print("Adding nearest neighbors to train queries")
        add_neighbors(args, case_index, train_dset, tokenizer, encoder, device, split='train')

        print("-- Train analytics --")
        print(f"Mean number of nodes in subgraphs: {np.mean(train_dset_analytics[0]): 0.2f} "
              f"({np.mean(train_dset_analytics[0]) / full_kb_stats[0] * 100: 0.2f}% of full KB)")
        print(f"Max number of nodes in subgraphs: {np.max(train_dset_analytics[0])} "
              f"({np.max(train_dset_analytics[0]) / full_kb_stats[0] * 100: 0.2f}% of full KB)")
        print(f"Mean number of edges in subgraphs: {np.mean(train_dset_analytics[1]): 0.2f} "
              f"({np.mean(train_dset_analytics[1]) / full_kb_stats[1] * 100: 0.2f}% of full KB)")
        print(f"Max number of edges in subgraphs: {np.max(train_dset_analytics[1])} "
              f"({np.max(train_dset_analytics[1]) / full_kb_stats[1] * 100: 0.2f}% of full KB)")
        print(f"Queries with answer outside 3-hop subgraph: {np.sum(train_dset_analytics[2])} "
              f"({np.mean(train_dset_analytics[2]) * 100: 0.2f}% of train dataset)")
        with open(os.path.join(args.output_dir, split, 'train.json'), 'w') as fout:
            json.dump(train_dset, fout, indent=2)
        print('---------------')

        print("Loading dev set")
        dev_dset, dev_dset_analytics = load_dataset(os.path.join(args.src_dir, split, 'vanilla', 'qa_dev.txt'),
                                                    kb, entity_vocab)

        print("Adding nearest neighbors to dev queries")
        add_neighbors(args, case_index, dev_dset, tokenizer, encoder, device)

        print("-- Dev analytics --")
        print(f"Mean number of nodes in subgraphs: {np.mean(dev_dset_analytics[0]): 0.2f} "
              f"({np.mean(dev_dset_analytics[0]) / full_kb_stats[0] * 100: 0.2f}% of full KB)")
        print(f"Max number of nodes in subgraphs: {np.max(dev_dset_analytics[0])} "
              f"({np.max(dev_dset_analytics[0]) / full_kb_stats[0] * 100: 0.2f}% of full KB)")
        print(f"Mean number of edges in subgraphs: {np.mean(dev_dset_analytics[1]): 0.2f} "
              f"({np.mean(dev_dset_analytics[1]) / full_kb_stats[1] * 100: 0.2f}% of full KB)")
        print(f"Max number of edges in subgraphs: {np.max(dev_dset_analytics[1])} "
              f"({np.max(dev_dset_analytics[1]) / full_kb_stats[1] * 100: 0.2f}% of full KB)")
        print(f"Queries with answer outside 3-hop subgraph: {np.sum(dev_dset_analytics[2])} "
              f"({np.mean(dev_dset_analytics[2]) * 100: 0.2f}% of dev dataset)")
        with open(os.path.join(args.output_dir, split, 'dev.json'), 'w') as fout:
            json.dump(dev_dset, fout, indent=2)
        print('---------------')

        print("Loading test set")
        test_dset, test_dset_analytics = load_dataset(os.path.join(args.src_dir, split, 'vanilla', 'qa_test.txt'),
                                                      kb, entity_vocab)

        print("Adding nearest neighbors to test queries")
        add_neighbors(args, case_index, test_dset, tokenizer, encoder, device)

        print("-- Test analytics --")
        print(f"Mean number of nodes in subgraphs: {np.mean(test_dset_analytics[0]): 0.2f} "
              f"({np.mean(test_dset_analytics[0]) / full_kb_stats[0] * 100: 0.2f}% of full KB)")
        print(f"Max number of nodes in subgraphs: {np.max(test_dset_analytics[0])} "
              f"({np.max(test_dset_analytics[0]) / full_kb_stats[0] * 100: 0.2f}% of full KB)")
        print(f"Mean number of edges in subgraphs: {np.mean(test_dset_analytics[1]): 0.2f} "
              f"({np.mean(test_dset_analytics[1]) / full_kb_stats[1] * 100: 0.2f}% of full KB)")
        print(f"Max number of edges in subgraphs: {np.max(test_dset_analytics[1])} "
              f"({np.max(test_dset_analytics[1]) / full_kb_stats[1] * 100: 0.2f}% of full KB)")
        print(f"Queries with answer outside 3-hop subgraph: {np.sum(test_dset_analytics[2])} "
              f"({np.mean(test_dset_analytics[2]) * 100: 0.2f}% of test dataset)")
        with open(os.path.join(args.output_dir, split, 'test.json'), 'w') as fout:
            json.dump(test_dset, fout, indent=2)
        print('---------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default='../data/MetaQA/')
    parser.add_argument("--output_dir", type=str, default='../outputs/MetaQA-synthetic/')
    parser.add_argument("--model", type=str, default='roberta-base')
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--index_path", type=str, default=None)
    parser.add_argument("--n_cases", type=int, default=20)
    cli_args = parser.parse_args()
    main(cli_args)
