import torch
from torch import Tensor
import torch.nn.functional as F
from torch_scatter import scatter
from typing import Optional

from torch_geometric.nn.conv import FastRGCNConv


class LowMemFastRGCNConv(FastRGCNConv):
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
