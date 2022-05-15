from torch.nn import Module
import os
import pickle
import numpy as np
import torch
from torch_geometric.data.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes

__all__ = [
    "NNSubgraphs",
    "NNSubgraphsFromData"
]


class KBCGraphData(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None, e1=None, r=None, label=None, node_mapping=None):
        super(KBCGraphData, self).__init__(x, edge_index, edge_attr)
        self.e1 = e1
        self.r = r
        self.label = label
        self.node_mapping = node_mapping

    def __inc__(self, key, value):
        if 'e1' in key:
            return self.num_nodes
        else:
            return super(KBCGraphData, self).__inc__(key, value)


class NNSubgraphs(Module):
    def __init__(self, data_args, dataset_obj, **kwargs):
        super().__init__()
        self.otf = data_args.otf
        self.dataset_obj = dataset_obj
        self.dropout = False
        if self.otf == False:  # Use pre-calculated paths if on-the-fly subgraph sampling is disabled
            self.subgraphs = self.load_subgraphs(dataset_obj.data_dir)
            self.transformed_subgraph = {}
        else:
            self.max_nodes = data_args.otf_max_nodes
            self.data = Data(edge_index=dataset_obj.edge_index, edge_attr=dataset_obj.edge_type,
                             num_nodes=dataset_obj.num_ent)

    def load_subgraphs(self, data_dir):
        kg_file = os.path.join(data_dir, 'paths_1000_len_3.pkl')
        print('Loading Subgraphs')
        with open(kg_file, 'rb') as f:
            graph = pickle.load(f)
        print("Completed loading subgraphs")
        return graph

    def transform_subgraph(self, graph, key):
        """
        Returns: Dataloader with
        x : list of nodes in graph
        edge_index : Matrix showing which node is attached to which node in graph [2 X num_edges]
        edge_attr : List showing the relation type of edges mentioned in edge_index
        ent2id : Dictionary storing index of nodes
        """
        edges = set()
        ent2id = {}
        x = []
        ent2id[key] = 0
        x.append(self.dataset_obj.ent2id[key])
        for path in graph:
            src_e = key
            for edge in path:
                if edge[1] not in ent2id:
                    ent2id[edge[1]] = len(ent2id)
                    x.append(self.dataset_obj.ent2id[edge[1]])
                dest = ent2id[edge[1]]
                edges.add(
                    (ent2id[src_e], dest, self.dataset_obj.rel2id[edge[0]])
                )
                src_e = edge[1]
        assert len(x) == len(ent2id)
        edges = np.array(list(edges), dtype=np.int64)

        # Randomly drop out edges in subgraphs (regularization)
        if self.dropout:
            to_remove = np.random.choice(edges, int(self.dropout * len(edges)), replace=False)
            edges = np.delete(edges, to_remove)

        edge_attr = torch.from_numpy(edges[:, 2].ravel())  # Only keep relations from unique edges
        edge_index = torch.from_numpy(edges[:, :2]).t()  # Only keep unique edges, transpose to be (2,n)
        x = torch.LongTensor(x)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, ent2id=ent2id)

    ## TODO : Rewrite the on the fly subgraph creation as evaluation time is high for this process
    # Custom subgraph function, returns new mapping of e1 (node in query)
    @staticmethod
    def subgraph(e1, subset, edge_index, edge_attr=None, relabel_nodes=False,
                 num_nodes=None):
        r"""Returns the induced subgraph of :obj:`(edge_index, edge_attr, mask)`
        containing the nodes in :obj:`subset`, and the new mapping of e1.

        Args:
            subset (LongTensor, BoolTensor or [int]): The nodes to keep.
            edge_index (LongTensor): The edge indices.
            edge_attr (Tensor, optional): Edge weights or multi-dimensional
                edge features. (default: :obj:`None`)
            relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
                :obj:`edge_index` will be relabeled to hold consecutive indices
                starting from zero. (default: :obj:`False`)
            num_nodes (int, optional): The number of nodes, *i.e.*
                :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """

        device = edge_index.device

        if isinstance(subset, list) or isinstance(subset, tuple):
            subset = torch.tensor(subset, dtype=torch.long)

        if subset.dtype == torch.bool or subset.dtype == torch.uint8:
            n_mask = subset

            if relabel_nodes:
                n_idx = torch.zeros(n_mask.size(0), dtype=torch.long,
                                    device=device)
                n_idx[subset] = torch.arange(subset.sum().item(), device=device)
        else:
            num_nodes = maybe_num_nodes(edge_index, num_nodes)
            n_mask = torch.zeros(num_nodes, dtype=torch.bool)
            n_mask[subset] = 1

            if relabel_nodes:
                n_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
                n_idx[subset] = torch.arange(subset.size(0), device=device)

        mask = n_mask[edge_index[0]] & n_mask[edge_index[1]]
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask] if edge_attr is not None else None

        if relabel_nodes:
            edge_index = n_idx[edge_index]
            e1_offset = n_idx[e1]
        else:
            e1_offset = e1

        return edge_index, edge_attr, e1_offset

    # noinspection PyTupleAssignmentBalance
    def forward(self, e1: torch.LongTensor, r: torch.LongTensor, labels: torch.LongTensor, remove_query_edge=True,
                **kwargs):
        if self.otf:
            # Return one set of edge_index and edge_type for an anchor entity
            data_list = []
            e1 = e1.flatten()
            for i in range(len(e1)):
                if remove_query_edge:
                    edge = torch.tensor((e1[i].item(), e2[i].item()))
                    correct_node = (self.dataset_obj.edge_index.T == edge).T  # 2 x num_edge
                    correct_relation = self.dataset_obj.edge_type.T == r[i].item()  # num_edge
                    mask = (correct_node[0] & correct_node[1] & correct_relation)
                    edge_index = self.dataset_obj.edge_index[:, ~mask]
                    edge_type = self.dataset_obj.edge_type[~mask]
                else:
                    edge_index = self.dataset_obj.edge_index
                    edge_type = self.dataset_obj.edge_type

                nodes, edge_index_throwaway, mapping, edge_mask = k_hop_subgraph(node_idx=e1[i].flatten(),
                                                                                 num_hops=3,
                                                                                 edge_index=edge_index,
                                                                                 relabel_nodes=False,
                                                                                 flow='target_to_source',
                                                                                 num_nodes=self.dataset_obj.num_ent)
                if len(nodes) > self.max_nodes:
                    nodes = nodes[torch.randperm(len(nodes))[:self.max_nodes]]
                edge_index, edge_attr, e1_offset = self.subgraph(e1=e1[i].flatten(),
                                                                 subset=nodes,
                                                                 edge_index=edge_index,
                                                                 edge_attr=edge_type,
                                                                 relabel_nodes=True,
                                                                 num_nodes=self.dataset_obj.num_ent)
                x = nodes
                label_offset = labels[i][x]
                if len(r) > 1:
                    data_list.append(KBCGraphData(x, edge_index, edge_attr, e1_offset, r[i], label_offset))
                else:
                    data_list.append(KBCGraphData(x, edge_index, edge_attr, e1_offset, r, label_offset))
            return Batch.from_data_list(data_list, follow_batch=['x'])
        else:
            # Return one set of edge_index and edge_type for an anchor entity
            data_list = []
            for e_ctr, e in enumerate(torch.flatten(e1).tolist()):
                k_e = self.dataset_obj.id2ent[e]
                # Not appending neighbours for entities not present in train
                if len(self.subgraphs[k_e]) != 0:
                    if e not in self.transformed_subgraph:
                        self.transformed_subgraph[e] = self.transform_subgraph(self.subgraphs[k_e], k_e)
                    transformed_graph = self.transformed_subgraph[e]
                    e1_offset = torch.LongTensor([transformed_graph.ent2id[k_e]])
                    label_offset = labels[e_ctr][transformed_graph.x]
                    data_list.append(KBCGraphData(transformed_graph.x, transformed_graph.edge_index,
                                                  transformed_graph.edge_attr, e1_offset, r[e_ctr], label_offset))
            return Batch.from_data_list(data_list, follow_batch=['x'])


class NNSubgraphsFromData(Module):
    def __init__(self, dataset_obj, **kwargs):
        super().__init__()
        self.dataset_obj = dataset_obj

    def forward(self, query_and_knn_list, nn_slices, **kwargs):
        for ctr, val in enumerate(query_and_knn_list):
            if val.x is None:
                if val.split == 'train':
                    item = self.dataset_obj.raw_train_data[val.ex_id]
                elif val.split == 'dev':
                    item = self.dataset_obj.raw_dev_data[val.ex_id]
                else:
                    item = self.dataset_obj.raw_test_data[val.ex_id]
                self.dataset_obj.convert_rawdata_to_cbrdata(item, val.split, self.dataset_obj.add_dist_feature,
                                                            self.dataset_obj.max_dist, lazy=False, inplace_obj=val)

        # Logic for eliminating queries (or all its KNNs) with no label nodes in subgraph
        if query_and_knn_list[0].split == "train":  # check if the first query is from train
            new_query_and_knn_list, new_nn_slices = [], [0]
            for ctr, idx in enumerate(nn_slices[:-1]):
                if query_and_knn_list[idx].label_node_ids.sum().item() == 0:
                    # bad query, move on, query does not have a label node
                    continue
                total = 0
                for val in query_and_knn_list[idx + 1:nn_slices[ctr + 1]]:
                    total += val.label_node_ids.sum().item()
                if total == 0:
                    # bad query, move on, since no KNNs have a label node
                    continue
                new_query_and_knn_list.extend(query_and_knn_list[idx:nn_slices[ctr + 1]])
                new_nn_slices.append(len(new_query_and_knn_list))
            query_and_knn_list = new_query_and_knn_list
            nn_slices = new_nn_slices
            assert nn_slices[-1] == len(query_and_knn_list)
            if len(query_and_knn_list) == 0:  # none of the query in the batch made it
                return None, None
        return Batch.from_data_list(query_and_knn_list, follow_batch=['x', 'edge_attr']), nn_slices
