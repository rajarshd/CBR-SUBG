import json
import os
import numpy as np
import argparse
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel
import torch
import pickle


def mask_query(t):
    query = t["ProcessedQuestion"]
    replaced = False
    if 'Parses' in t:
        for p in t['Parses']:
            if "PotentialTopicEntityMention" in p:
                mention_name = p["PotentialTopicEntityMention"]
                if not isinstance(mention_name, str):
                    continue
                query = query.replace(mention_name, "<mask>")
                replaced = True
                break
    return query, replaced


def prepare_qa_data(args, train_queries, index):
    num_replaced_ctr = 0
    train_file = os.path.join(args.data_dir, "WebQSP.train.json")
    with open(train_file) as fin:
        train = json.load(fin)
    unmasked_train_queries = [t["ProcessedQuestion"] for t in train["Questions"]]

    file_names = [("train", "WebQSP.train.json"), ("test", "WebQSP.test.json")]
    for split, file_name in file_names:
        data_file = os.path.join(args.data_dir, file_name)
        with open(data_file) as fin:
            data = json.load(fin)
        queries = []
        unmasked_queries = []
        for t in data["Questions"]:
            masked_query, replaced = mask_query(t)
            queries.append(masked_query)
            unmasked_queries.append(t["ProcessedQuestion"])
            if replaced:
                num_replaced_ctr += 1
        print("Num queries replaced by mask: {}".format(num_replaced_ctr))
        all_q_dicts = []
        with torch.no_grad():
            for idx in tqdm(range(int(np.ceil(len(queries) / args.eval_batch_size)))):
                curr_batch = queries[idx * args.eval_batch_size: (idx + 1) * args.eval_batch_size]
                query_vecs = encode_str_batch(curr_batch, args.tokenizer, args.encoder, args.device, args.pool_type)
                sim = np.matmul(query_vecs, index.transpose())
                knn_inds = np.argsort(-sim, axis=1)
                knn_inds = knn_inds[:, :100]
                for i in range(knn_inds.shape[0]):
                    nn_train_queries = [train_queries[knn_ind] for knn_ind in knn_inds[i, :20]]
                    nn_unmasked_train_queries = [unmasked_train_queries[knn_ind] for knn_ind in knn_inds[i, :20]]
                    q_dict = {
                        "question": queries[idx * args.eval_batch_size + i],
                        "unmasked_question": unmasked_queries[idx * args.eval_batch_size + i],
                        "nn_train_queries": nn_train_queries,
                        "nn_unmasked_train_queries": nn_unmasked_train_queries
                    }
                    all_q_dicts.append(q_dict)
        out_file = "nn_{}_{}_{}.json".format(split, args.model_type, args.pool_type)
        with open(os.path.join(args.output_dir, out_file), "w") as fout:
            json.dump(all_q_dicts, fout, indent=2)


def main(args):
    if not os.path.exists(args.index_path):
        num_replaced_ctr = 0
        # read train file
        train_file = os.path.join(args.data_dir, "WebQSP.train.json")
        with open(train_file) as fin:
            train = json.load(fin)
        train_queries = []
        for t in train["Questions"]:
            masked_query, replaced = mask_query(t)
            train_queries.append(masked_query)
            if replaced:
                num_replaced_ctr += 1
        print("Num queries replaced by mask: {}".format(num_replaced_ctr))
        all_indices = []
        with torch.no_grad():
            for idx in tqdm(range(int(np.ceil(len(train_queries) / args.eval_batch_size)))):
                curr_batch = train_queries[idx * args.eval_batch_size: (idx + 1) * args.eval_batch_size]
                context_vecs = encode_str_batch(curr_batch, args.tokenizer, args.encoder, args.device, args.pool_type)
                all_indices.append(context_vecs)
        all_indices = np.concatenate(all_indices)
        assert all_indices.shape[0] == len(train_queries)
        save_obj = {"queries": train_queries, "embeddings": all_indices}
        print(f"Saving index to {args.index_path}")
        with open(args.index_path, "wb") as fout:
            pickle.dump(save_obj, fout)
    else:
        print("Loading train vectors from {}".format(args.index_path))
        all_indices = None
        with open(args.index_path, "rb") as fin:
            saved_query_embeddings = pickle.load(fin)
            train_queries = saved_query_embeddings["queries"]
            all_indices = saved_query_embeddings["embeddings"]
        assert all_indices is not None

    prepare_qa_data(args, train_queries, all_indices)


def cls_pooling(model_output, attention_mask):
    return model_output[0][:, 0]


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def encode_str_batch(batch, tokenizer, encoder, device, pool_type="cls_pool"):
    curr_batch = tokenizer(batch, padding=True, return_tensors='pt')
    outputs = encoder(input_ids=curr_batch["input_ids"].to(device),
                      attention_mask=curr_batch["attention_mask"].to(device))
    # query_vecs = outputs.pooler_output.cpu().numpy()
    if pool_type == "cls_pool":
        query_vecs = cls_pooling(outputs, curr_batch["attention_mask"].to(device)).cpu().numpy()
    elif pool_type == "mean_pool":
        query_vecs = mean_pooling(outputs, curr_batch["attention_mask"].to(device)).cpu().numpy()
    query_vecs /= np.linalg.norm(query_vecs, axis=1, keepdims=True)
    return query_vecs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/nfs/scratch1/rajarshi/CBR-KB-QA/WebQSP")
    parser.add_argument("--output_dir", type=str, default="/mnt/nfs/scratch1/rajarshi/cbr-weak-supervision/test")
    parser.add_argument("--pooling", type=str, default="cls_pool", help="cls_pool or mean_pool")
    # parser.add_argument("--model", type=str, default='roberta-base')
    parser.add_argument("--model", type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--index_path", type=str, default=None)
    parser.add_argument("--n_cases", type=int, default=100)
    args = parser.parse_args()
    args.model_type = args.model.replace("/", "_")
    args.pool_type = args.pooling
    if args.index_path is None:
        args.index_path = os.path.join(args.output_dir,
                                       "train_query_vecs_{}_{}.npy".format(args.model_type, args.pool_type))
    print("Load pre-trained encoder model and tokenizers...")
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.tokenizer = AutoTokenizer.from_pretrained(args.model)
    args.encoder = AutoModel.from_pretrained(args.model).to(args.device)
    args.encoder.eval()
    # get the embedding of the train questions and save it
    main(args)
