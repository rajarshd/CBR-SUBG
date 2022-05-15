import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import RGCNConv
from src.models.rgcn.query_aware_rgcn_layers import QueryAwareRGCNConv, FastQueryAwareRGCNConv
from src.data_loaders.helper import *
import time
from src.models.scoring_fn import TransEScorer
from src.models.rgcn.fast_rgcn_layer import LowMemFastRGCNConv

__all__ = [
    "RGCN",
    "QueryAwareRGCN"
]


class RGCN(torch.nn.Module):
    def __init__(self, model_params, base_feat_mat=None):
        super(RGCN, self).__init__()
        self.model_params = model_params
        self.dropout = torch.nn.Dropout(p=model_params.drop_rgcn)
        if base_feat_mat is None:
            self.embed = torch.nn.Embedding(self.model_params.num_entities, self.model_params.dense_node_feat_dim)
        else:
            base_feat_mat = base_feat_mat.coalesce()
            self.embed = torch.nn.Embedding.from_pretrained(base_feat_mat, freeze=True, sparse=True)
        if model_params.transform_input:
            # base_feat_mat already has additional distance feature dimensions
            additional_dims = self.model_params.n_additional_feat if base_feat_mat is None else 0
            self.input_lin = torch.nn.Linear(self.embed.embedding_dim + additional_dims,
                                             model_params.gcn_dim, bias=True)
            n_input_in_channels = model_params.gcn_dim
        else:
            additional_dims = self.model_params.n_additional_feat if base_feat_mat is None else 0
            n_input_in_channels = self.embed.embedding_dim + additional_dims
        if self.model_params.num_gcn_layers == 0:  # zero layers, no params required
            if model_params.use_scoring_head is not None and model_params.use_scoring_head == 'transe':
                self.scoring_head = TransEScorer(model_params.num_relations, n_input_in_channels)
            elif model_params.use_scoring_head is not None:
                raise ValueError(f"Unsupported scoring function: {model_params.use_scoring_head}")
            else:
                self.scoring_head = None
            self.input_conv = None
            self.conv_layers = None
            return
        if model_params.use_fast_rgcn:
            self.input_conv = LowMemFastRGCNConv(in_channels=n_input_in_channels, out_channels=model_params.gcn_dim,
                                                 num_relations=model_params.num_relations,
                                                 num_bases=model_params.num_bases)
        else:
            self.input_conv = RGCNConv(in_channels=n_input_in_channels, out_channels=model_params.gcn_dim,
                                       num_relations=model_params.num_relations, num_bases=model_params.num_bases)
        self.conv_layers = torch.nn.ModuleList()
        for _ in range(model_params.num_gcn_layers - 1):
            if model_params.use_fast_rgcn:
                self.conv_layers.append(
                    LowMemFastRGCNConv(in_channels=model_params.gcn_dim, out_channels=model_params.gcn_dim,
                                       num_relations=model_params.num_relations, num_bases=model_params.num_bases))
            else:
                self.conv_layers.append(
                    RGCNConv(in_channels=model_params.gcn_dim, out_channels=model_params.gcn_dim,
                             num_relations=model_params.num_relations, num_bases=model_params.num_bases))

        if model_params.use_scoring_head is not None and model_params.use_scoring_head == 'transe':
            self.scoring_head = TransEScorer(model_params.num_relations, model_params.gcn_dim)
        elif model_params.use_scoring_head is not None:
            raise ValueError(f"Unsupported scoring function: {model_params.use_scoring_head}")
        else:
            self.scoring_head = None

    def forward(self, x, edge_index, edge_type, dist_feats=None, *args):
        # pass x through embedding table
        is_odd = False
        if self.model_params.use_sparse_feats:
            if x.shape[0] % 2 == 1:
                y = torch.cat([x, torch.zeros(1, dtype=x.dtype, device=x.device)])
                x_inp = torch.sparse_coo_tensor(torch.vstack([torch.arange(y.shape[0], device=x.device), y]),
                                                torch.cat([torch.ones_like(x), torch.zeros(1, dtype=x.dtype, device=x.device)]),
                                                size=(y.shape[0], self.embed.num_embeddings), dtype=torch.float)
                is_odd = True
            else:
                x_inp = torch.sparse_coo_tensor(torch.vstack([torch.arange(x.shape[0], device=x.device), x]), torch.ones_like(x),
                                               size=(x.shape[0], self.embed.num_embeddings), dtype=torch.float)
            x = torch.sparse.mm(x_inp, self.embed.weight)
        else:
            x = self.embed(x)
        if dist_feats is not None:
            if self.model_params.use_sparse_feats:
                # make dist_feats sparse matrix
                if is_odd:
                    dist_feats = torch.cat([dist_feats, torch.zeros(1, dtype=x.dtype, device=x.device)])
                sparse_ind = torch.vstack(
                    [torch.arange(x.shape[0], device=x.device), self.model_params.n_base_feat + dist_feats])
                dist_feats = torch.sparse_coo_tensor(sparse_ind, torch.ones(x.shape[0]),
                                                     size=(x.shape[0], self.model_params.node_feat_dim), device=x.device)
                x = x + dist_feats
            else:
                dist_feats_mat = torch.zeros(x.shape[0], self.model_params.max_dist + 1, device=x.device)
                dist_feats_mat[:, dist_feats] = 1
                x = torch.cat([x, dist_feats_mat], dim=1)
        if self.model_params.transform_input:
            # x = torch.sparse.mm(x, self.input_lin.weight.T) + self.input_lin.bias
            x = self.input_lin(x)
            x = x[:-1] if is_odd and self.model_params.use_sparse_feats else x
        if self.input_conv is None:  # zero layers, return input
            return x
        out = self.input_conv(x, edge_index, edge_type)
        out = self.dropout(F.relu(out))
        for i, layer in enumerate(self.conv_layers):
            out = layer(out, edge_index, edge_type)
            if i != (len(self.conv_layers)-1):
                out = self.dropout(F.relu(out))
            else:
                out = self.dropout(out)
        return out

    def run_scoring_head(self, head_embed, rel_ids):
        assert self.scoring_head is not None
        return self.scoring_head(head_embed, rel_ids)

class QueryAwareRGCN(torch.nn.Module):
    def __init__(self, model_params, base_feat_mat=None):
        super(QueryAwareRGCN, self).__init__()
        self.model_params = model_params
        if base_feat_mat is None:
            self.embed = torch.nn.Embedding(self.model_params.num_entities, self.model_params.dense_node_feat_dim)
        else:
            base_feat_mat = base_feat_mat.coalesce()
            self.embed = torch.nn.Embedding.from_pretrained(base_feat_mat, freeze=True, sparse=True)
        if model_params.transform_input:
            # base_feat_mat already has additional distance feature dimensions
            additional_dims = self.model_params.n_additional_feat if base_feat_mat is None else 0
            self.input_lin = torch.nn.Linear(self.embed.embedding_dim + additional_dims,
                                             model_params.gcn_dim, bias=True)
            n_input_in_channels = model_params.gcn_dim
        else:
            additional_dims = self.model_params.n_additional_feat if base_feat_mat is None else 0
            n_input_in_channels = self.embed.embedding_dim + additional_dims
        if self.model_params.num_gcn_layers == 0:  # zero layers, no params required
            self.input_conv = None
            self.conv_layers = None
            return
        if model_params.transform_query:
            self.query_proj_lin = torch.nn.Linear(model_params.query_dim, model_params.query_proj_dim, bias=True)
            query_dim = model_params.query_proj_dim
        else:
            query_dim = model_params.query_dim
        if model_params.use_fast_rgcn:
            self.input_conv = FastQueryAwareRGCNConv(in_channels=n_input_in_channels, out_channels=model_params.gcn_dim,
                                                     num_relations=model_params.num_relations,
                                                     query_dim=query_dim,
                                                     query_attn_type=model_params.query_attn_type,
                                                     query_attn_activation=model_params.query_attn_activation,
                                                     num_bases=model_params.num_bases)
        else:
            self.input_conv = QueryAwareRGCNConv(in_channels=n_input_in_channels, out_channels=model_params.gcn_dim,
                                                 num_relations=model_params.num_relations,
                                                 query_dim=query_dim,
                                                 query_attn_type=model_params.query_attn_type,
                                                 query_attn_activation=model_params.query_attn_activation,
                                                 num_bases=model_params.num_bases)
        self.conv_layers = torch.nn.ModuleList()
        for _ in range(model_params.num_gcn_layers - 1):
            if model_params.use_fast_rgcn:
                self.conv_layers.append(
                    FastQueryAwareRGCNConv(in_channels=model_params.gcn_dim, out_channels=model_params.gcn_dim,
                                           num_relations=model_params.num_relations,
                                           query_dim=query_dim,
                                           query_attn_type=model_params.query_attn_type,
                                           query_attn_activation=model_params.query_attn_activation,
                                           num_bases=model_params.num_bases))
            else:
                self.conv_layers.append(
                    QueryAwareRGCNConv(in_channels=model_params.gcn_dim, out_channels=model_params.gcn_dim,
                                       num_relations=model_params.num_relations,
                                       query_dim=query_dim,
                                       query_attn_type=model_params.query_attn_type,
                                       query_attn_activation=model_params.query_attn_activation,
                                       num_bases=model_params.num_bases))

    def forward(self, x, edge_index, edge_type, query_emb, x_batch, edge_batch, dist_feats=None, *args):
        # pass x through embedding table
        is_odd = False
        if self.model_params.use_sparse_feats:
            if x.shape[0] % 2 == 1:
                y = torch.cat([x, torch.zeros(1, dtype=x.dtype, device=x.device)])
                x_inp = torch.sparse_coo_tensor(torch.vstack([torch.arange(y.shape[0], device=x.device), y]),
                                                torch.cat([torch.ones_like(x),
                                                           torch.zeros(1, dtype=x.dtype, device=x.device)]),
                                                size=(y.shape[0], self.embed.num_embeddings), dtype=torch.float)
                is_odd = True
            else:
                x_inp = torch.sparse_coo_tensor(torch.vstack([torch.arange(x.shape[0], device=x.device), x]),
                                                torch.ones_like(x),
                                                size=(x.shape[0], self.embed.num_embeddings), dtype=torch.float)
            x = torch.sparse.mm(x_inp, self.embed.weight)
        else:
            x = self.embed(x)
        if dist_feats is not None:
            if self.model_params.use_sparse_feats:
                # make dist_feats sparse matrix
                if is_odd:
                    dist_feats = torch.cat([dist_feats, torch.zeros(1, dtype=x.dtype, device=x.device)])
                sparse_ind = torch.vstack(
                    [torch.arange(x.shape[0], device=x.device), self.model_params.n_base_feat + dist_feats])
                dist_feats = torch.sparse_coo_tensor(sparse_ind, torch.ones(x.shape[0]),
                                                     size=(x.shape[0], self.model_params.node_feat_dim),
                                                     device=x.device)
                x = x + dist_feats
            else:
                dist_feats_mat = torch.zeros(x.shape[0], self.model_params.max_dist + 1, device=x.device)
                dist_feats_mat[:, dist_feats] = 1
                x = torch.cat([x, dist_feats_mat], dim=1)
        if self.model_params.transform_input:
            x = self.input_lin(x)
            x = x[:-1] if is_odd and self.model_params.use_sparse_feats else x
        elif x.is_sparse:
            x = x.to_dense()
            x = x[:-1] if is_odd and self.model_params.use_sparse_feats else x
        if self.input_conv is None:  # zero layers, return input
            return x
        if self.model_params.transform_query:
            query_emb = self.query_proj_lin(query_emb)
        out = self.input_conv(x, edge_index, edge_type, query_emb, x_batch, edge_batch)
        out = F.relu(out)
        for i, layer in enumerate(self.conv_layers):
            out = layer(out, edge_index, edge_type, query_emb, x_batch, edge_batch)
            if i != (len(self.conv_layers) - 1):
                out = F.relu(out)
        return out
