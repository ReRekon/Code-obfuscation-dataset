import os
import sys
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import dgl
import numpy as np
import torch as th
from torch.utils.data import DataLoader

# lib_path = os.path.abspath(os.path.join('.'))
# sys.path.append(lib_path)

from ParameterConfig import ParameterConfig
from .SelfAttention import SelfAttention
from .rvnn import BatchTreeEncoder
from .tbcnn import TBCNNEncoder
from .treelstm import ChildSumTreeLSTM
from .utils import NodesDataset, init_network, TokensDataset
from dgl.nn.pytorch import GATConv


class MergeClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, embedding_matrix, node_vec_stg='mean', text_vec_stg=None,
                 device=None, layer_num=2, activation=F.relu, dp_rate=0.5):
        super(MergeClassifier, self).__init__()
        self.node_vec_stg = node_vec_stg
        if node_vec_stg != 'mean':
            x = import_module('gnnmodels.' + node_vec_stg)
            self.config = x.Config(embedding_matrix, hidden_dim, device)
            model = x.Model(self.config).to(device)
            if node_vec_stg != 'Transformer':
                init_network(model)
            self.node_model = model
        self.device = device

        # GCN
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_dim, hidden_dim, activation=activation))
        # hidden layers
        for i in range(layer_num):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=activation))
        # output layer
        self.dropout = nn.Dropout(p=dp_rate)

        # GAT
        self.gat_layers = nn.ModuleList()
        # input projection (no residual)
        head_num = ParameterConfig.HEAD_NUM
        feat_drop = ParameterConfig.GAT_FEAT_DP_RATE
        attn_drop = ParameterConfig.GAT_ATT_DP_RATE

        self.gat_layers.append(GATConv(in_dim, hidden_dim, head_num, feat_drop=feat_drop, attn_drop=attn_drop,
                                       residual=False, activation=activation))

        # hidden layers
        for i in range(1, layer_num):
            self.gat_layers.append(GATConv(hidden_dim * head_num, hidden_dim, head_num, feat_drop=feat_drop,
                                           attn_drop=attn_drop, residual=True, activation=activation))
        # add another graph convolution layer for output projection
        self.gat_layers.append(GATConv(hidden_dim * head_num, hidden_dim, num_heads=1, feat_drop=feat_drop,
                                       attn_drop=attn_drop, residual=True, activation=activation))

        # 输出层
        self.classify = nn.Linear(176, n_classes)
        self.embedding_matrix = embedding_matrix
        self.is_train_mode = True

        y = import_module('gnnmodels.' + text_vec_stg)
        self.config = y.Config(embedding_matrix, hidden_dim, device)
        text_model = y.Model(self.config).to(device)
        if text_vec_stg != 'Transformer':
            init_network(text_model)
        self.text_model = text_model

        # rvnn
        self.encoder = BatchTreeEncoder(embedding_matrix.shape[0] + 1, 128,
                                        128, ParameterConfig.BATCH_SIZE, self.device)

        # Tree-Lstm
        self.tree_lstm = ChildSumTreeLSTM(embedding_matrix.shape[0] + 1, 128, 128,
                                          0.3, self.device)

        # TBCNN
        self.tbcnn = TBCNNEncoder(embedding_matrix.shape[0] + 1, embedding_matrix.shape[1], 128,
                                  ParameterConfig.BATCH_SIZE, self.device)

        # self-attention
        self.self_attention_model = SelfAttention(128)

    def generate_node_vecs(self, g):
        nodes_attrs = g.ndata['w']
        torches = []
        if self.node_vec_stg == 'mean':
            for ins_list in nodes_attrs:
                vec_list = []
                for ins in ins_list:
                    vec = self.embedding_matrix[ins.item()]
                    vec_list.append(vec)
                arr = np.array(vec_list)
                vec = arr.mean(axis=0)
                torches.append(th.tensor(vec))
            return th.stack(torches, 0)
        else:
            nodes_db = NodesDataset(nodes_attrs)
            data_loader = DataLoader(nodes_db, batch_size=self.config.batch_size, shuffle=False)
            node_vec_list = []
            if self.is_train_mode:
                self.node_model.train()
            else:
                self.node_model.eval()
            for iter, batch in enumerate(data_loader):
                batch_out = self.node_model(batch)
                # tensor_batch = th.tensor(batch).to(self.device)
                # batch_out = self.node_model(tensor_batch)
                node_vec_list.append(batch_out)
            return th.cat(node_vec_list, dim=0)

    def generate_token_vecs(self, tokens):
        tokens_db = TokensDataset(tokens)
        data_loader = DataLoader(tokens_db, batch_size=self.config.batch_size, shuffle=False)
        token_vec_list = []
        if self.is_train_mode:
            self.text_model.train()
        else:
            self.text_model.eval()
        for iter, batch in enumerate(data_loader):
            # batch.shape [32, 64]
            batch_out = self.text_model(batch)

            # tensor_batch = th.tensor(batch).to(self.device)
            # batch_out = self.node_model(tensor_batch)
            # print(batch_out.shape)
            token_vec_list.append(batch_out)
        return th.cat(token_vec_list, dim=0)

    def GCN(self, graph, node):
        for i, layer in enumerate(self.layers):
            if i != 0:
                node = self.dropout(node)
            node = layer(graph, node)
        graph.ndata['h'] = node
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(graph, 'h')

        return hg

    def GAT(self, g, h):
        for i, layer in enumerate(self.gat_layers):
            if i != 0:
                # concat on the output feature dimension (dim=1)
                h = th.transpose(h, 0, 1)
                heads = [hd for hd in h]
                h = th.cat(heads, dim=1)
            h = layer(g, h)

        h = h.squeeze(1)
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return hg

    def RVNN(self, x, lens):
        encodes = []
        for fun in x:
            encodes.append(fun)
        return self.encoder(encodes, sum(lens))

    def TreeLSTM(self, x):
        encodes = []
        for fun in x:
            feature_vec1 = self.tree_lstm(fun)
            encodes.append(feature_vec1[1])
        return torch.cat(encodes)

    def TBCNN(self, x, lens):
        encodes = []
        for fun in x:
            encodes.append(fun)
        return self.tbcnn(encodes, sum(lens))

    def averagePooling(self, v1, v2, v3):
        return (v1 + v2 + v3) / 3

    def maxPooling(self, hg, t, encodes):
        merged_tensor = torch.stack((hg, t, encodes), dim=2)
        m, _ = torch.max(merged_tensor, dim=2)
        return m

    def forward(self, x, tk, g):

        h = self.generate_node_vecs(g).float().to(self.device)
        t = self.generate_token_vecs(tk).float().to(self.device)

        # GCN
        hg = self.GCN(g, h)

        # GAT
        # hg = self.GAT(g, h)


        lens = [1 for i in x]

        # RVNN
        encodes = self.RVNN(x, lens)

        # Tree_lstm
        # encodes = self.TreeLSTM(x)

        # TBCNN
        # encodes = self.TBCNN(x, lens)


        # average
        # m = self.averagePooling(hg, t, encodes)

        # max
        # m = self.maxPooling(hg, t, encodes)

        # cat
        # m = torch.cat([hg, t, encodes])

        # SelfAttention
        m = torch.stack([hg, t, encodes], dim=1)
        m = self.self_attention_model(m)

        return self.classify(m)
