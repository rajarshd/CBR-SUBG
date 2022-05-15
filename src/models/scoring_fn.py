import torch

__all__ = [
    "TransEScorer"
]


class TransEScorer(torch.nn.Module):
    def __init__(self, num_relations, embedding_dim):
        super(TransEScorer, self).__init__()
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.embed = torch.nn.Embedding(self.num_relations, self.embedding_dim)

    def forward(self, head_embed, rel_ids):
        rel_embed = self.embed(rel_ids)
        assert rel_embed.shape[1] == head_embed.shape[1]
        return head_embed + rel_embed
