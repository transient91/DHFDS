from functools import partial

import dgl
import torch
from dgllife.model.gnn import *
from dgllife.utils import smiles_to_bigraph
from torch.utils.data import Dataset
from tqdm import tqdm


class SeqDataset(Dataset):
    def __init__(
        self,
        drug1,
        drug2,
        drug2smiles,
        contexts,
        contexts_dict,
        labels,
        dataset_name,
        config_drug_feature,
    ):
        self.drug1 = drug1
        self.drug2 = drug2
        self.contexts = contexts
        self.contexts_dict = contexts_dict
        self.labels = labels
        self.length = len(self.labels)
        self.drug2smiles = drug2smiles
        self.drug2graphs = {}
        self._pre_process(
            smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
            node_featurizer=config_drug_feature['node_featurizer'],
            edge_featurizer=config_drug_feature['edge_featurizer'],
        )

    def _pre_process(self, smiles_to_graph, node_featurizer, edge_featurizer):
        for key, smile in tqdm(self.drug2smiles.items()):
            graph = smiles_to_graph(
                smile, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer
            )
            self.drug2graphs[key] = graph

    def __getitem__(self, idx):
        graph_drug1 = self.drug2graphs[str(self.drug1[idx])]
        graph_drug2 = self.drug2graphs[str(self.drug2[idx])]
        context_feature = self.contexts_dict[self.contexts[idx]]
        label = self.labels[idx]

        return (
            graph_drug1,
            graph_drug2,
            torch.FloatTensor(context_feature),
            torch.FloatTensor([label]),
        )

    def __len__(self):
        return self.length


def collate_molgraphs(data):
    graph_drug1, graph_drug2, context_feature, labels = map(list, zip(*data))

    bg_drug1 = dgl.batch(graph_drug1)
    bg_drug1.set_n_initializer(dgl.init.zero_initializer)
    bg_drug1.set_e_initializer(dgl.init.zero_initializer)

    bg_drug2 = dgl.batch(graph_drug2)
    bg_drug2.set_n_initializer(dgl.init.zero_initializer)
    bg_drug2.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)
    context_feature = torch.stack(context_feature, dim=0)

    return bg_drug1, bg_drug2, context_feature, labels
