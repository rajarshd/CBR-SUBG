import argparse
import json
import os
import torch
from tqdm import trange

from src.text_handler import QueryEncoder


def encode_queries(raw_data, query_encoder, eval_batch_size, device, output_device):
    n_batches = len(raw_data) // eval_batch_size + \
                (0 if len(raw_data) % eval_batch_size == 0 else 1)
    output_matrix = torch.empty(len(raw_data), query_encoder.get_query_embedding_dim(), dtype=torch.float)
    for b_idx in trange(n_batches):
        offset_st = b_idx * eval_batch_size
        offset_en = (b_idx + 1) * eval_batch_size
        text_batch = [query['question'] for query in raw_data[offset_st:offset_en]]
        batch_output = query_encoder(text_batch=text_batch, device=device).to(output_device)
        output_matrix[offset_st: offset_en] = batch_output
    return output_matrix


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_device = torch.device('cpu')
    query_encoder = QueryEncoder(args.query_encoder_model, args.pooling_type, train_query_encoder=False).to(device)
    query_encoder.eval()

    if args.output_suffix is None:
        args.output_suffix = args.query_encoder_model.replace('/', '_') if os.path.isdir(args.query_encoder_model) \
            else args.query_encoder_model
        args.output_suffix = args.output_suffix + '_' + args.pooling_type
    print(f"Output dir: {os.path.join(args.output_dir, args.output_suffix)}")
    if not os.path.isdir(os.path.join(args.output_dir, args.output_suffix)):
        os.makedirs(os.path.join(args.output_dir, args.output_suffix))

    for seg in ['train', 'dev', 'test']:
        print(f"Processing {seg}...")
        raw_data = []
        with open(os.path.join(args.data_dir, f"{seg}_simple.json")) as fin:
            for line in fin:
                raw_data.append(json.loads(line))
        print(f"Found {len(raw_data)} examples")
        output_encoding = encode_queries(raw_data, query_encoder, args.eval_batch_size, device, output_device)
        output_filenm = os.path.join(args.output_dir, args.output_suffix, f'query_enc_{seg}.pt')
        print(f"Saving output to {output_filenm}...")
        torch.save(output_encoding, output_filenm)
        print(f"Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_suffix", type=str, default=None)
    parser.add_argument("--query_encoder_model", type=str, default='roberta-base')
    parser.add_argument("--pooling_type", type=str, default='pooler', choices=['pooler', 'cls', 'mean_pool'])
    parser.add_argument("--eval_batch_size", type=int, default=16)
    cli_args = parser.parse_args()
    if cli_args.output_dir is None:
        cli_args.output_dir = cli_args.data_dir
    main(cli_args)
