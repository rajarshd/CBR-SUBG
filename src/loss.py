import torch
from torch import Tensor, BoolTensor
from torch.nn import Module
import torch.nn.functional as F
import time
import random

__all__ = [
    "MarginLoss",
    "TXent",
    "DistanceLoss",
    "BCELoss"
]


class BCELoss(Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.bceloss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, label):
        loss = self.bceloss(pred, label)
        loss[label == 0] /= sum(label == 0)
        loss[label == 1] /= sum(label == 1)
        return torch.mean(loss)


class DistanceLoss(Module):
    def __init__(self, gamma: float, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.L2loss = torch.nn.MSELoss()

    def forward(self, pred: Tensor, label: Tensor, obj_emb: Tensor) -> Tensor:
        # Query Loss
        query_loss = self.L2loss(obj_emb[:,1:] - obj_emb[0,:])
        #answer loss
        ans_loss = self.L2loss(pred[:,1:] - pred[0,:])
        return min(self.gamma, -1*(query_loss- ans_loss).mean())


class MarginLoss(Module):
    def __init__(self, margin, **kwargs):
        super(MarginLoss, self).__init__()
        self.margin = margin

    def forward(self, dist_arr: Tensor, labels: BoolTensor) -> Tensor:
        """
        :param dist_arr: Array of shape [n_nodes] containing distances (lower is better as opposed to score)
        :param labels: Labels array of shape [n_nodes] with 1 for positive class and 0 otherwise
        :return: Tensor loss
        """
        margin_mat = dist_arr[labels].unsqueeze(1) - dist_arr[labels.logical_not()].unsqueeze(0)
        return torch.maximum(margin_mat + self.margin, torch.zeros_like(margin_mat)).mean()


class TXent(Module):
    """
    Temperature-scaled Cross-entropy Loss
    """
    def __init__(self, temperature, **kwargs):
        super(TXent, self).__init__()
        self.temperature = temperature

    def forward(self, dist_arr: Tensor, labels: BoolTensor) -> Tensor:
        """
        :param dist_arr: Array of shape [n_nodes] containing distances (lower is better as opposed to score)
        :param labels: Labels array of shape [n_nodes] with 1 for positive class and 0 otherwise
        :return: Tensor loss
        """
        sim = - dist_arr / self.temperature
        assert sim.dim() == 1
        denom_log_sum_exp = torch.logsumexp(sim, dim=0)
        loss = - torch.logsumexp(sim[labels], dim=0) + denom_log_sum_exp
        return loss

