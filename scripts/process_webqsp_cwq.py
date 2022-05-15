import json
import os
import numpy as np
import argparse
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel
import torch
import pickle
from sentence_transformers import SentenceTransformer


# import faiss

def prepare_qa_data(data_dir, input_file, train_q_vecs, tokenizer, encoder, device, output_file,
                    masked_query_map=None):
    input_file = os.path.join(data_dir, input_file)
    len_tuples = []
    train_qids = []
    train_ques_str = []
    train_file = os.path.join(data_dir, "train_simple.json")
    with open(train_file) as fin:
        for line in fin:
            q_data = json.loads(line)
            train_qids.append(q_data["id"])
            train_ques_str.append(q_data["question"])
    all_q_dicts = []
    all_train_nn_queries = []
    with open(input_file) as fin:
        for line in tqdm(fin):
            q_data = json.loads(line)
            q_id = q_data["id"]
            ques_str = q_data["question"] if masked_query_map is None else masked_query_map[q_id]
            seed_entities = q_data["entities"]
            answers = [ans["kb_id"] for ans in q_data["answers"]]
            subgraph = q_data["subgraph"]
            len_tuples.append(len(subgraph["tuples"]))
            # encode the train question
            with torch.no_grad():
                query_vec = encode_str_batch([ques_str], tokenizer, encoder, device, pool_type)
                sim = np.matmul(query_vec, train_q_vecs.transpose()).squeeze(0)
                knn_inds = np.argsort(-sim)[:100]
                knn_qids = [train_qids[knn_ind] for knn_ind in knn_inds]
                nn_train_queries = [train_ques_str[knn_ind] for knn_ind in knn_inds[:20]]
                if masked_query_map is not None:
                    mask_nn_train_queries = [masked_query_map[knn_ind] for knn_ind in knn_qids[:20]]
            q_dict = {"id": q_id,
                      "seed_entities": seed_entities,
                      "question": q_data["question"],
                      "answer": answers,
                      "knn": knn_qids,
                      "subgraph": subgraph
                      }
            if masked_query_map is not None:
                q_dict.update({"masked_question": masked_query_map[q_id]})
            all_q_dicts.append(q_dict)
            nn_q_dict = {"id": q_id, "question": q_data["question"], "knns": nn_train_queries}
            if masked_query_map is not None:
                nn_q_dict.update({"masked_question": masked_query_map[q_id], "masked_knns": mask_nn_train_queries})
            all_train_nn_queries.append(nn_q_dict)
    print("Avg num tuples in {} file: {:1.2f}".format(input_file, np.mean(len_tuples)))
    with open(os.path.join(data_dir, output_file), "w") as fout:
        json.dump(all_q_dicts, fout, indent=2)
    with open(os.path.join(data_dir, "nn_" + output_file), "w") as fout:
        json.dump(all_train_nn_queries, fout, indent=2)
    print("Output file written to {}".format(os.path.join(data_dir, output_file)))


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
            # sometimes constraints also has some entity mentions in the question. Get those too
            if "Constraints" in p:
                constraints = p["Constraints"]
                for c in constraints:
                    if c["ArgumentType"] == "Entity":
                        entity_name = c.get("EntityName", None)
                        if entity_name is not None and len(entity_name) > 0:
                            entity_name = entity_name.lower()
                            query = query.replace(entity_name, "<mask>")
            if replaced:
                break
    return query, replaced


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/nfs/scratch1/rajarshi/cbr-weak-supervision/CWQ")
    parser.add_argument("--pooling", type=str, default="mean_pool", help="cls_pool or mean_pool")
    parser.add_argument('--use_masked_questions', type=int, default=1)
    # parser.add_argument("--model", type=str, default='roberta-base')
    parser.add_argument("--model", type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--index_path", type=str, default=None)
    parser.add_argument("--n_cases", type=int, default=100)
    args = parser.parse_args()
    model_type = args.model.replace("/", "_")
    pool_type = args.pooling
    args.use_masked_questions = (args.use_masked_questions == 1)
    if args.index_path is None:
        if args.use_masked_questions:
            args.index_path = os.path.join(args.data_dir,
                                           "masked_train_query_vecs_{}_{}.npy".format(model_type, pool_type))
        else:
            args.index_path = os.path.join(args.data_dir, "train_query_vecs_{}_{}.npy".format(model_type, pool_type))

    # get masked questions if use_masked_questions
    masked_query_map, unmasked_query_map = None, None
    if args.use_masked_questions:
        orig_data_dir = "/mnt/nfs/scratch1/rajarshi/CBR-KB-QA/WebQSP"
        file_names = [("train", "WebQSP.train.json"), ("test", "WebQSP.test.json")]
        masked_query_map = {}
        unmasked_query_map = {}
        for split, file_name in file_names:
            num_replaced_ctr = 0
            data_file = os.path.join(orig_data_dir, file_name)
            with open(data_file) as fin:
                data = json.load(fin)
            for t in data["Questions"]:
                q_id = t["QuestionId"]
                masked_query, replaced = mask_query(t)
                masked_query_map[q_id] = masked_query
                unmasked_query_map[q_id] = t["ProcessedQuestion"]
                if replaced:
                    num_replaced_ctr += 1
            print("Num queries replaced by mask: {}".format(num_replaced_ctr))
    # set up encoder model (roberta), tokenizer etc
    print("Load pre-trained encoder model and tokenizers...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoder = AutoModel.from_pretrained(args.model).to(device)
    encoder.eval()
    # get the embedding of the train questions and save it
    all_indices = None
    if not os.path.exists(args.index_path):
        train_file = os.path.join(args.data_dir, "train_simple.json")
        train_dset = []
        with open(train_file) as fin:
            for line in fin:
                line = line.strip()
                train_dset.append(json.loads(line))
        print("Encoding train queries")
        train_ids = [query['id'] for query in train_dset]
        if args.use_masked_questions:
            train_queries = [masked_query_map[t_id] for t_id in train_ids]
        else:
            train_queries = [query['question'] for query in train_dset]
        all_indices = []
        with torch.no_grad():
            for idx in tqdm(range(int(np.ceil(len(train_queries) / args.eval_batch_size)))):
                curr_batch = train_queries[idx * args.eval_batch_size: (idx + 1) * args.eval_batch_size]
                context_vecs = encode_str_batch(curr_batch, tokenizer, encoder, device, pool_type)
                all_indices.append(context_vecs)
        all_indices = np.concatenate(all_indices)
        assert all_indices.shape[0] == len(train_queries)
        save_obj = {"queries": train_queries, "embeddings": all_indices}
        print(f"Saving index to {args.index_path}")
        with open(args.index_path, "wb") as fout:
            pickle.dump(save_obj, fout)
        # np.save(args.index_path, all_indices)
    else:
        print("Loading train vectors from {}".format(args.index_path))
        all_indices = None
        with open(args.index_path, "rb") as fin:
            saved_query_embeddings = pickle.load(fin)
            train_queries = saved_query_embeddings["queries"]
            all_indices = saved_query_embeddings["embeddings"]
        assert all_indices is not None
    print("Processing train data........")
    if args.use_masked_questions:
        out_file = "train_{}_{}_masked.json".format(model_type, pool_type)
    else:
        out_file = "train_{}_{}.json".format(model_type, pool_type)
    prepare_qa_data(args.data_dir, "train_simple.json", all_indices, tokenizer, encoder, device,
                    out_file, masked_query_map)
    print("Processing dev data........")
    if args.use_masked_questions:
        out_file = "dev_{}_{}_masked.json".format(model_type, pool_type)
    else:
        out_file = "dev_{}_{}.json".format(model_type, pool_type)
    prepare_qa_data(args.data_dir, "dev_simple.json", all_indices, tokenizer, encoder, device, out_file,
                    masked_query_map)
    print("Processing test data........")
    if args.use_masked_questions:
        out_file = "test_{}_{}_masked.json".format(model_type, pool_type)
    else:
        out_file = "test_{}_{}.json".format(model_type, pool_type)
    prepare_qa_data(args.data_dir, "test_simple.json", all_indices, tokenizer, encoder, device, out_file,
                    masked_query_map)
