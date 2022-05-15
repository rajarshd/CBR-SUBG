import torch
from torch_scatter import scatter_sum
from torch_geometric.nn.conv.rgcn_conv import *
from src.models.rgcn.fast_rgcn_layer import LowMemFastRGCNConv


class QueryAwareRGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    .. note::
        This implementation is as memory-efficient as possible by iterating
        over each individual relation type.
        Therefore, it may result in low GPU utilization in case the graph has a
        large number of relations.
        As an alternative approach, :class:`FastRGCNConv` does not iterate over
        each individual type, but may consume a large amount of memory to
        compensate.
        We advise to check out both implementations to see which one fits your
        needs.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
            In case no input features are given, this argument should
            correspond to the number of nodes in your graph.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int, optional): If set to not :obj:`None`, this layer will
            use the basis-decomposition regularization scheme where
            :obj:`num_bases` denotes the number of bases to use.
            (default: :obj:`None`)
        num_blocks (int, optional): If set to not :obj:`None`, this layer will
            use the block-diagonal-decomposition regularization scheme where
            :obj:`num_blocks` denotes the number of blocks to use.
            (default: :obj:`None`)
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 num_relations: int,
                 query_dim: int,
                 query_attn_type: str = '',
                 query_attn_activation: str = 'softmax',
                 num_bases: Optional[int] = None,
                 num_blocks: Optional[int] = None,
                 aggr: str = 'mean',
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable

        super(QueryAwareRGCNConv, self).__init__(aggr=aggr, node_dim=0, **kwargs)

        if num_bases is not None and num_blocks is not None:
            raise ValueError('Can not apply both basis-decomposition and '
                             'block-diagonal-decomposition at the same time.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.query_dim = query_dim
        self.query_attn_type = query_attn_type
        assert self.query_attn_type in ['full', 'dim', 'sep']
        self.query_attn_activation = query_attn_activation
        assert self.query_attn_activation in ['sigmoid', 'softmax']
        self.num_bases = num_bases
        self.num_blocks = num_blocks

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]

        if num_bases is not None:
            self.weight = Parameter(
                torch.Tensor(num_bases, in_channels[0], out_channels))
            self.comp = Parameter(torch.Tensor(num_relations, num_bases))

        elif num_blocks is not None:
            # assert (in_channels[0] % num_blocks == 0
            #         and out_channels % num_blocks == 0)
            # self.weight = Parameter(
            #     torch.Tensor(num_relations, num_blocks,
            #                  in_channels[0] // num_blocks,
            #                  out_channels // num_blocks))
            # self.register_parameter('comp', None)
            raise NotImplementedError("Querry aware attention is currently not implemented with "
                                      "block-diagonal-decomposition")

        else:
            self.weight = Parameter(
                torch.Tensor(num_relations, in_channels[0], out_channels))
            self.register_parameter('comp', None)

        if root_weight:
            self.root = Param(torch.Tensor(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if query_attn_type == 'full':
            self.query_attn_weight = Parameter(torch.Tensor(1, in_channels[0], out_channels, query_dim))
        elif query_attn_type == 'dim':
            self.query_attn_weight_l = Parameter(torch.Tensor(1, in_channels[0], 1))
            self.query_attn_weight_r = Parameter(torch.Tensor(1, 1, out_channels))
            self.query_attn_weight = Parameter(torch.Tensor(in_channels[0] + out_channels, query_dim))
        else:
            # query_attn_type == 'sep'
            self.query_attn_weight = Parameter(torch.Tensor(num_relations, query_dim))
        if self.query_attn_activation == 'sigmoid':
            self.query_attn_bias = Parameter(torch.Tensor(num_relations, 1))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.comp)
        glorot(self.root)
        if self.query_attn_type == 'dim':
            glorot(self.query_attn_weight_l)
            glorot(self.query_attn_weight_r)
        glorot(self.query_attn_weight)
        if self.query_attn_activation == 'sigmoid':
            zeros(self.query_attn_bias)
        zeros(self.bias)

    def compute_relation_attention(self, rel_weights, query_vec):
        """
        :param rel_weights: Relation parameters of shape `[num_relations, in_channels_l, out_channels]`
        :param query_vec: Query embddings of shape `[num_graphs, query_dim]`
        :return: rel_attn: Relation attention per query of shape `[num_relations, query_dim]` normalized over dimensin 0
        """
        # This operation is common to all attention types. Scaling is sqrt(d) is taking inspiration from
        # transformer attention
        query_attn_feat = self.query_attn_weight @ query_vec.t() / torch.tensor([self.query_dim],
                                                                                dtype=query_vec.dtype,
                                                                                device=query_vec.device).sqrt()
        if self.query_attn_type == 'full':
            # query_attn_feat.shape = [1, in_channels[0], out_channels, num_graphs]
            unnorm_rel_attn = torch.sum(rel_weights.unsqueeze(-1) * query_attn_feat, dim=1).sum(dim=1)
        elif self.query_attn_type == 'dim':
            # query_attn_feat.shape = [in_channels[0] + out_channels, num_graphs]
            rel_feat_l = torch.sum(self.query_attn_weight_l * rel_weights, dim=1)
            rel_feat_r = torch.sum(rel_weights * self.query_attn_weight_r, dim=2)
            # rel_feat.shape = [num_relations, in_channels[0] + out_channels]
            rel_feat = torch.cat([rel_feat_l, rel_feat_r], dim=1)
            unnorm_rel_attn = rel_feat @ query_attn_feat
        else:
            # query_attn_type == 'sep'
            # query_attn_feat.shape = [num_relations, num_graphs]
            unnorm_rel_attn = query_attn_feat
        if self.query_attn_activation == 'sigmoid':
            rel_attn = torch.sigmoid(unnorm_rel_attn + self.query_attn_bias)
        elif self.query_attn_activation == 'softmax':
            rel_attn = torch.softmax(unnorm_rel_attn, dim=0)
        else:
            raise NotImplementedError(f"Unsupported activation function: {self.query_attn_activation}")
        return rel_attn

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]], edge_index: Adj, edge_type: OptTensor,
                query_vec: Tensor, x_batch: Tensor, edge_batch: Tensor):
        r"""
        Args:
            x: The input node features. Can be either a :obj:`[num_nodes,
                in_channels]` node feature matrix, or an optional
                one-dimensional node index tensor (in which case input features
                are treated as trainable node embeddings).
                Furthermore, :obj:`x` can be of type :obj:`tuple` denoting
                source and destination node features.
            edge_type: The one-dimensional relation type/index for each edge in
                :obj:`edge_index`.
                Should be only :obj:`None` in case :obj:`edge_index` is of type
                :class:`torch_sparse.tensor.SparseTensor`.
                (default: :obj:`None`)
            query_vec: Query embeddings on shape `[num_graphs, query_dim]`
            x_batch: The result of `follow_batch` which maps every node to it's graph in the batch.
                Expected shape `[num_nodes]` and values in `range(0, num_graphs)`
            edge_batch: The result of `follow_batch` which maps every edge to it's graph in the batch.
                Expected shape `[num_edges]` and values in `range(0, num_graphs)`
        """

        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        # propagate_type: (x: Tensor)
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition =================
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)

        if self.num_blocks is not None:  # Block-diagonal-decomposition =====
            raise NotImplementedError("Querry aware attention is currently not implemented with "
                                      "block-diagonal-decomposition")
            # if x_l.dtype == torch.long and self.num_blocks is not None:
            #     raise ValueError('Block-diagonal decomposition not supported '
            #                      'for non-continuous input features.')
            #
            # for i in range(self.num_relations):
            #     tmp = masked_edge_index(edge_index, edge_type == i)
            #     h = self.propagate(tmp, x=x_l, size=size)
            #     h = h.view(-1, weight.size(1), weight.size(2))
            #     h = torch.einsum('abc,bcd->abd', h, weight[i])
            #     out += h.contiguous().view(-1, self.out_channels)

        else:  # No regularization/Basis-decomposition ========================
            # rel_attn.shape = [num_relations, num_graphs]
            rel_attn = self.compute_relation_attention(weight, query_vec)
            rel_attn_normalizer = scatter_sum(rel_attn[edge_type, edge_batch], edge_index[1], dim=0,
                                              dim_size=x.shape[0])
            if rel_attn_normalizer.min() == 0.0:
                rel_attn_normalizer[rel_attn_normalizer == 0.0] = 1.0

            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)

                if x_l.dtype == torch.long:
                    out += self.propagate(tmp, x=weight[i, x_l], size=size) * rel_attn[i][x_batch].view(-1, 1)
                else:
                    h = self.propagate(tmp, x=x_l, size=size)
                    out = out + (h @ weight[i]) * rel_attn[i][x_batch].view(-1, 1)
            out = out / rel_attn_normalizer.unsqueeze(1)

        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.num_relations)


class FastQueryAwareRGCNConv(QueryAwareRGCNConv):
    r"""See :class:`QueryAwareRGCNConv`."""
    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index: Adj, edge_type: OptTensor, query_vec: Tensor, x_batch: Tensor, edge_batch: Tensor):
        """"""
        self.fuse = False
        assert self.aggr in ['add', 'sum', 'mean']

        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        # propagate_type: (x: Tensor, edge_type: OptTensor)
        out = self.propagate(edge_index, x=x_l, edge_type=edge_type, size=size, query_vec=query_vec, x_batch=x_batch,
                             edge_batch=edge_batch)

        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_type: Tensor, index: Tensor, query_vec: Tensor, x_batch: Tensor,
                edge_batch: Tensor) -> Tensor:
        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition =================
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)

        if self.num_blocks is not None:  # Block-diagonal-decomposition =======
            raise NotImplementedError("Querry aware attention is currently not implemented with "
                                      "block-diagonal-decomposition")
            # if x_j.dtype == torch.long:
            #     raise ValueError('Block-diagonal decomposition not supported '
            #                      'for non-continuous input features.')
            #
            # weight = weight[edge_type].view(-1, weight.size(2), weight.size(3))
            # x_j = x_j.view(-1, 1, weight.size(1))
            # return torch.bmm(x_j, weight).view(-1, self.out_channels)

        else:  # No regularization/Basis-decomposition ========================
            # rel_attn.shape = [num_relations, num_graphs]
            rel_attn = self.compute_relation_attention(weight, query_vec)
            rel_attn_normalizer = scatter_sum(rel_attn[edge_type, edge_batch], index, dim=0,
                                              dim_size=x_batch.shape[0])[index]
            if rel_attn_normalizer.min() == 0.0:
                rel_attn_normalizer[rel_attn_normalizer == 0.0] = 1.0

            if x_j.dtype == torch.long:
                weight_index = edge_type * weight.size(1) + index
                return weight.view(-1, self.out_channels)[weight_index]

            assert index.shape == edge_type.shape

            base_msgs = torch.bmm(x_j.unsqueeze(-2), weight[edge_type]).squeeze(-2)
            base_msgs = base_msgs * rel_attn[edge_type, x_batch[index]].view([-1, 1]) / rel_attn_normalizer.unsqueeze(1)
            return base_msgs

    def aggregate(self, inputs: Tensor, edge_type: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:

        # Compute normalization in separation for each `edge_type`.
        if self.aggr == 'mean':
            # norm = F.one_hot(edge_type, self.num_relations).to(torch.float)
            # norm = scatter(norm, index, dim=0, dim_size=dim_size)[index]
            # norm = torch.gather(norm, 1, edge_type.view(-1, 1))
            # norm = 1. / norm.clamp_(1.)
            # inputs = norm * inputs
            unique_edges, inv_unique_edges = torch.unique(edge_type, return_inverse=True)
            norm = F.one_hot(inv_unique_edges, len(unique_edges)).to(torch.float)
            norm = scatter(norm, index, dim=0, dim_size=dim_size)[index]
            norm = torch.gather(norm, 1, inv_unique_edges.view(-1, 1))
            norm = 1. / norm.clamp_(1.)
            inputs = norm * inputs

        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size)
