#!/usr/bin/env python
# encoding: utf-8
"""
@author: 瑞
@description:  
@return: 
@time: 2023/5/24 14:51
"""
import torch.nn as nn
import torch
from torch.autograd import Variable

class ChildSumTreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, dropout1, device, use_gpu=True, pretrained_weight=None):
        # in_dim is the input dim and mem_dim is the output dim
        super(ChildSumTreeLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, in_dim)
        self.in_dim = in_dim    # embedding size
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.drop = nn.Dropout(dropout1)
        self.device = device
        self.use_gpu = use_gpu
        self.th = torch.cuda if use_gpu else torch
        # pretrained  embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # self.embedding.weight.requires_grad = False
        self.W_c = nn.Linear(in_dim, mem_dim)

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.to(self.device)
        return tensor

    def node_forward(self, inputs, child_c, child_h):
        # inputs = torch.unsqueeze(inputs, 0)
        child_h_sum = torch.sum(child_h, dim=0)
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(self.fh(child_h) + self.fx(inputs).repeat(len(child_h), 1))
        fc = torch.mul(f, child_c)
        c = torch.mul(i, u) + torch.sum(fc, dim=0)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def forward(self, node):
        # node is a sub-tree
        if node[0] != -1:
            current_node = node[0]  # the root of current sub-tree
            children = node[1:]
            num_children = len(children)   # children number of current node
        else:
            return None

        # recursively process its child sub-trees
        tr_stats = [self.forward(children[idx]) for idx in range(num_children)]

        if num_children == 0:
            child_c = self.create_tensor(Variable(torch.zeros(1, self.in_dim)))
            child_h = self.create_tensor(Variable(torch.zeros(1, self.in_dim)))
        else:
            child_c, child_h = zip(*tr_stats)
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
        cur_node_embedding = self.embedding(Variable(self.th.LongTensor([current_node]).to(self.device)))
        cur_state = self.node_forward(cur_node_embedding, child_c, child_h)
        return cur_state