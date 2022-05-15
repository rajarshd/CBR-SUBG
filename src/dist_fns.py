import torch
from torch.nn import Module
from torch_scatter import scatter_sum, scatter_mean


class L2Dist(Module):
    def __init__(self, stage1_aggr='mean', stage2_aggr='mean', **kwargs):
        super(L2Dist, self).__init__()
        self.stage1_aggr = stage1_aggr if stage1_aggr != 'none' else None
        self.stage2_aggr = stage2_aggr

    def forward(self, pred, target, target_identifiers=None):
        """
        :param pred: Tensor of shape [n_nodes x embed_dim] where n_nodes is the number of nodes in the query graph
        :param target: Tensor of shape [n_pos x embed_dim] where n_pos is the number of positives in all KN graphs
        :param target_identifiers: Tensor of shape [n_pos] identifying which target belongs to which neighbor
        :return: Tensor of shape [n_nodes] with the sum of L2 distance of predicted tensors to the target tensor
        """
        dist_mat = torch.sqrt((target.unsqueeze(1) - pred.unsqueeze(0)).square().sum(dim=-1) + 1e-8)
        # dist_mat.shape : num_pos in KNN graph x num_nodes in query graph
        if self.stage1_aggr is not None:
            assert target_identifiers is not None
            # dist_mat.shape : number of KNN graphs x num_nodes in query graph
            if self.stage1_aggr == 'mean':
                dist_mat = scatter_mean(dist_mat, index=target_identifiers, dim=0)
            elif self.stage1_aggr == 'sum':
                dist_mat = scatter_sum(dist_mat, index=target_identifiers, dim=0)
            else:
                raise NotImplementedError(f'No support for stage 1 aggr: {self.stage1_aggr}')
        if self.stage2_aggr == 'mean':
            return dist_mat.mean(dim=0)
        elif self.stage2_aggr == 'sum':
            return dist_mat.sum(dim=0)
        else:
            raise NotImplementedError(f'No support for stage 2 aggr: {self.stage2_aggr}')


class CosineDist(Module):
    def __init__(self, stage1_aggr='mean', stage2_aggr='mean', **kwargs):
        super(CosineDist, self).__init__()
        self.stage1_aggr = stage1_aggr if stage1_aggr != 'none' else None
        self.stage2_aggr = stage2_aggr

    def forward(self, pred, target, target_identifiers=None):
        """
        :param pred: Tensor of shape [n_nodes x embed_dim] where n_nodes is the number of nodes in the query graph
        :param target: Tensor of shape [n_pos x embed_dim] where n_pos is the number of positives in all KN graphs
        :param target_identifiers: Tensor of shape [n_pos] identifying which target belongs to which neighbor
        :return: Tensor of shape [n_nodes] with the sum of cosine distance of predicted tensors to the target tensor
        """
        # Distance is -1 x cosine similarity
        _x1 = target.unsqueeze(1)
        _x2 = pred.unsqueeze(0)
        dist_mat = - ((_x1 * _x2)/(_x1.norm(dim=-1, keepdim=True) * _x2.norm(dim=-1, keepdim=True) + 1e-8)).sum(dim=-1)
        # dist_mat.shape : num_pos in KNN graph x num_nodes in query graph
        if self.stage1_aggr is not None:
            assert target_identifiers is not None
            # dist_mat.shape : number of KNN graphs x num_nodes in query graph
            if self.stage1_aggr == 'mean':
                dist_mat = scatter_mean(dist_mat, index=target_identifiers, dim=0)
            elif self.stage1_aggr == 'sum':
                dist_mat = scatter_sum(dist_mat, index=target_identifiers, dim=0)
            else:
                raise NotImplementedError(f'No support for stage 1 aggr: {self.stage1_aggr}')
        if self.stage2_aggr == 'mean':
            return dist_mat.mean(dim=0)
        elif self.stage2_aggr == 'sum':
            return dist_mat.sum(dim=0)
        else:
            raise NotImplementedError(f'No support for stage 2 aggr: {self.stage2_aggr}')


class TransEDist(Module):
    def __init__(self, gamma: float, **kwargs):
        super(TransEDist, self).__init__()
        self.gamma = gamma

    def forward(self, pred, target, **kwargs):
        """
        :param pred:
        :param target:
        :return: Tensor of shape [n_nodes]
        """
        sub_emb, rel_emb = pred
        graph_batch = kwargs['graph_batch']
        node_batch_emb = kwargs['node_emb'][graph_batch.x, :]
        num_neighbors = kwargs['num_neigh']
        obj_emb = sub_emb + rel_emb  # Get TransE representations
        x = self.gamma - torch.norm(obj_emb.unsqueeze(1) - node_batch_emb, p=1, dim=2)
        M = torch.zeros(torch.arange(len(num_neighbors)).repeat_interleave(num_neighbors).max() + 1, len(x))
        M[torch.arange(len(num_neighbors)).repeat_interleave(num_neighbors), torch.arange(len(x))] = 1
        M = torch.nn.functional.normalize(M, p=1, dim=1)
#        x = x[batch_offsets, torch.arange(len(batch_offsets))]
        return torch.mm(M, x)

class ContrastiveDist(Module):
    def __init__(self, **kwargs):
        super(ContrastiveDist, self).__init__()

    def forward(self, pred, target, **kwargs):
        """
        :param pred: Tensor of shape [batch_size x embed_dim]
        :param target: Tensor of shape [num_ent x embed_dim]
        :return: Tensor of shape [batch_size X num_entities] with cosine distance of input entites tensors to all entities
        """
        # Works exactly same as the cosine distance calculation without the dim = 0 summation carried on in CosineDist
        _x1 = target.unsqueeze(1)
        _x2 = kwargs['node_emb'].unsqueeze(0)
        dist_mat = - ((_x1 * _x2)/(_x1.norm(dim=-1, keepdim=True) * _x2.norm(dim=-1, keepdim=True) + 1e-8)).sum(dim=-1)
        return dist_mat.sum(0)



