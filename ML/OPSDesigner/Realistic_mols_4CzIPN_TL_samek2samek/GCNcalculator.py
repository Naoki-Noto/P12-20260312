# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 16:20:16 2025

@author: noton
"""

import pickle
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.data import Batch, Data
import deepchem as dc

torch.manual_seed(42)

class Net(nn.Module):
    def __init__(self, in_dim=30, hidden=256, out_dim=6):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.conv4 = GCNConv(hidden, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = global_max_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

properties_calculator = Net()
properties_calculator.load_state_dict(torch.load('GCNmodel/model_mt_sc.pth'))
device = torch.device('cpu')
properties_calculator.to(device)

def custom_collate(batch):
    data_list, target_list = zip(*batch)
    batch_data = Batch.from_data_list(data_list)
    batch_target = torch.stack(target_list)
    return batch_data, batch_target


def predict_properties(smiles_list):
    properties_calculator.eval()
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    graphs = featurizer.featurize(smiles_list)
    
    data_list = []
    for g in graphs:
        x = torch.from_numpy(np.asarray(g.node_features, dtype=np.float32))
        ei = np.asarray(g.edge_index)
        if ei.ndim == 2 and ei.shape[0] != 2 and ei.shape[1] == 2:
            ei = ei.T
        ei = torch.from_numpy(ei.astype(np.int64))
        data_list.append(Data(x=x, edge_index=ei))
    batch = Batch.from_data_list(data_list).to(device)

    with torch.no_grad():
        pred_sc = properties_calculator(batch).cpu().numpy().astype(np.float64, copy=False)

    with open('GCNmodel/scaler_mt.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return scaler.inverse_transform(pred_sc)
