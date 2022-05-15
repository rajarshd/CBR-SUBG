import torch
from transformers import AutoModel, AutoTokenizer


class QueryEncoderBase:
    def get_query_embedding_dim(self):
        raise NotImplemented

    def is_trainable(self):
        raise NotImplemented

    def save(self, save_path):
        raise NotImplemented


class PrecomputedQueryEncoder(torch.nn.Module, QueryEncoderBase):
    def __init__(self, dataset_obj):
        super(PrecomputedQueryEncoder, self).__init__()
        self.query_enc_train = dataset_obj.query_enc_train
        self.query_enc_dev = dataset_obj.query_enc_dev
        self.query_enc_test = dataset_obj.query_enc_test

    def get_query_embedding_dim(self):
        return self.query_enc_train.shape[1]

    def forward(self, ex_ids, split, device, **kwargs):
        output_mat = torch.empty((len(ex_ids), self.get_query_embedding_dim()), dtype=torch.float)
        for e_ctr, (e_id, e_s) in enumerate(zip(ex_ids, split)):
            if e_s == 'train':
                output_mat[e_ctr] = self.query_enc_train[e_id]
            elif e_s == 'dev':
                output_mat[e_ctr] = self.query_enc_dev[e_id]
            elif e_s == 'test':
                output_mat[e_ctr] = self.query_enc_test[e_id]
            else:
                raise ValueError(f'Unhandled split {e_s}')
        return output_mat.to(device)

    def is_trainable(self):
        return False

    def save(self, save_path):
        pass


class QueryEncoder(torch.nn.Module, QueryEncoderBase):
    def __init__(self, query_encoder_model_name_or_path, pooling_type, train_query_encoder=False):
        super(QueryEncoder, self).__init__()
        self.query_tokenizer = AutoTokenizer.from_pretrained(query_encoder_model_name_or_path)
        self.query_encoder = AutoModel.from_pretrained(query_encoder_model_name_or_path)
        self.pooling_type = pooling_type
        self.train_query_encoder = train_query_encoder
        if not train_query_encoder:
            # Free the encoder
            for param_ in self.query_encoder.parameters():
                param_.requires_grad = False

    def get_query_embedding_dim(self):
        return self.query_encoder.config.hidden_size

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, text_batch, device, **kwargs):
        curr_batch = self.query_tokenizer(text_batch, padding=True, truncation=True, return_tensors='pt')
        curr_batch["input_ids"] = curr_batch["input_ids"].to(device)
        curr_batch["attention_mask"] = curr_batch["attention_mask"].to(device)
        outputs = self.query_encoder(input_ids=curr_batch["input_ids"],
                                     attention_mask=curr_batch["attention_mask"])
        if self.pooling_type == 'pooler':
            return outputs.pooler_output
        elif self.pooling_type == 'cls':
            return outputs.last_hidden_state[:, 0]
        else:
            assert self.pooling_type == 'mean_pool'
            return self.mean_pooling(outputs, curr_batch["attention_mask"])

    def is_trainable(self):
        return self.train_query_encoder

    def save(self, save_path):
        if self.train_query_encoder:
            self.query_tokenizer.save_pretrained(save_path)
            self.query_encoder.save_pretrained(save_path)
