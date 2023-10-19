import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

config = {
    'max_tree_size': 2500,
    'max_children_size': 60,
    'conv_layer_num': 1
}

use_cuda = True if torch.cuda.is_available() else False

def get_tensor(tensor: torch.Tensor):
    if use_cuda:
        return tensor.cuda()
    return tensor

class ConvolutionLayer(nn.Module):
    def __init__(self, config):
        super(ConvolutionLayer, self).__init__()
        self.config = config
        self.conv_num = self.config['conv_layer_num']
        config['output_size'] = self.config['conv_output']
        config['feature_size'] = config['embedding_dim']
        self.conv_nodes = nn.ModuleList([ConvNode(config=config) for _ in range(self.conv_num)])
        # self.conv_node = ConvNode(config=config)

    def forward(self, nodes, children, children_embedding):
        nodes = [
            conv_node(nodes, children, children_embedding)
            for conv_node in self.conv_nodes
        ]
        return torch.cat(nodes, dim=2)
        # return self.conv_node(nodes, children,children_embedding)


class PoolingLayer(nn.Module):
    def __init__(self):
        super(PoolingLayer, self).__init__()

    def forward(self, nodes):
        pooled = torch.max(nodes, dim=1)[0]
        return pooled


class ConvNode(nn.Module):
    def __init__(self, config):
        super(ConvNode, self).__init__()
        self.config = config
        std = 1.0 / math.sqrt(self.config['feature_size'])
        self.w_t = nn.Parameter(
            torch.normal(size=(self.config['feature_size'], self.config['output_size']), std=std, mean=0))
        self.w_l = nn.Parameter(
            torch.normal(size=(self.config['feature_size'], self.config['output_size']), std=std, mean=0))
        self.w_r = nn.Parameter(
            torch.normal(size=(self.config['feature_size'], self.config['output_size']), std=std, mean=0))
        self.conv = nn.Parameter(
            torch.normal(size=(self.config['output_size'],), std=math.sqrt(2.0 / self.config['feature_size']), mean=0))

    def forward(self, nodes, children, children_vectors):
        # nodes is shape (batch_size x max_tree_size x feature_size)
        # children is shape (batch_size x max_tree_size x max_children)

        # children_vectors will have shape
        # (batch_size x max_tree_size x max_children x feature_size)

        # add a 4th dimension to the nodes tensor
        nodes = torch.unsqueeze(nodes, dim=2)
        # tree_tensor is shape
        # (batch_size x max_tree_size x max_children + 1 x feature_size)
        tree_tensor = torch.cat((nodes, children_vectors), dim=2)

        # coefficient tensors are shape (batch_size x max_tree_size x max_children + 1)
        c_t = eta_t(children)
        c_r = eta_r(children, c_t)
        c_l = eta_l(children, c_t, c_r)
        #
        # concatenate the position coefficients into a tensor
        # (batch_size x max_tree_size x max_children + 1 x 3)
        coef = torch.stack((c_t, c_r, c_l), dim=3)

        # stack weight matrices on top to make a weight tensor
        # (3, feature_size, output_size)
        weights = torch.stack((self.w_t, self.w_r, self.w_l), dim=0)

        # combine
        batch_size = children.shape[0]
        max_tree_size = children.shape[1]
        max_children = children.shape[2]

        # reshape for matrix multiplication
        x = batch_size * max_tree_size
        y = max_children + 1
        result = torch.reshape(tree_tensor, (x, y, self.config['feature_size']))
        coef = torch.reshape(coef, (x, y, 3))
        result = torch.transpose(result, 1, 2)
        result = torch.matmul(result, coef)
        result = torch.reshape(result, (batch_size, max_tree_size, 3, self.config['feature_size']))

        # output is (batch_size, max_tree_size, output_size)
        result = torch.tensordot(result, weights, [[2, 3], [0, 1]])

        # output is (batch_size, max_tree_size, output_size)
        return torch.tanh(result + self.conv)


def eta_t(children):
    """Compute weight matrix for how much each vector belongs to the 'top'"""
    # children is shape (batch_size x max_tree_size x max_children)
    batch_size = children.shape[0]
    max_tree_size = children.shape[1]
    max_children = children.shape[2]
    # eta_t is shape (batch_size x max_tree_size x max_children + 1)
    return torch.tile(torch.unsqueeze(torch.concat(
        [get_tensor(torch.ones((max_tree_size, 1))), get_tensor(torch.zeros((max_tree_size, max_children)))],
        dim=1), dim=0,
    ), (batch_size, 1, 1))


def eta_r(children, t_coef):
    """Compute weight matrix for how much each vector belongs to the 'right'"""
    # children is shape (batch_size x max_tree_size x max_children)
    batch_size = children.shape[0]
    max_tree_size = children.shape[1]
    max_children = children.shape[2]

    # num_siblings is shape (batch_size x max_tree_size x 1)
    num_siblings = torch.count_nonzero(children, dim=2).float().reshape(batch_size, max_tree_size, 1)

    # num_siblings is shape (batch_size x max_tree_size x max_children + 1)
    num_siblings = torch.tile(
        num_siblings, (1, 1, max_children + 1)
    )
    # creates a mask of 1's and 0's where 1 means there is a child there
    # has shape (batch_size x max_tree_size x max_children + 1)
    mask = torch.cat(
        [get_tensor(torch.zeros((batch_size, max_tree_size, 1))),
         torch.minimum(children, get_tensor(torch.ones(children.shape)))],
        dim=2
    )

    # child indices for every tree (batch_size x max_tree_size x max_children + 1)
    p = torch.tile(
        torch.unsqueeze(
            torch.unsqueeze(
                get_tensor(torch.arange(-1.0, max_children, 1.0, dtype=torch.float32)),
                dim=0
            ),
            dim=0
        ),
        (batch_size, max_tree_size, 1)
    )
    child_indices = torch.multiply(p, mask)

    # weights for every tree node in the case that num_siblings = 0
    # shape is (batch_size x max_tree_size x max_children + 1)
    t = torch.zeros((batch_size, max_tree_size, 1))
    # t = torch.fill(t, 0.5)
    t = t.fill_(0.5)
    t = get_tensor(t)
    singles = torch.cat(
        [get_tensor(torch.zeros((batch_size, max_tree_size, 1))),
         t,
         get_tensor(torch.zeros((batch_size, max_tree_size, max_children - 1)))],
        dim=2)

    # eta_r is shape (batch_size x max_tree_size x max_children + 1)
    return torch.where(
        num_siblings == 1.0,
        # avoid division by 0 when num_siblings == 1
        singles,
        # the normal case where num_siblings != 1
        torch.multiply((1.0 - t_coef), torch.divide(child_indices, num_siblings - 1.0))
    )


def eta_l(children, coef_t, coef_r):
    """Compute weight matrix for how much each vector belongs to the 'left'"""
    batch_size = children.shape[0]
    max_tree_size = children.shape[1]
    # creates a mask of 1's and 0's where 1 means there is a child there
    # has shape (batch_size x max_tree_size x max_children + 1)
    mask = torch.cat(
        [get_tensor(torch.zeros((batch_size, max_tree_size, 1))),
         torch.minimum(children, get_tensor(torch.ones(children.shape)))],
        dim=2)

    # eta_l is shape (batch_size x max_tree_size x max_children + 1)
    return torch.multiply(
        torch.multiply((1.0 - coef_t), (1.0 - coef_r)), mask
    )


class TBCNNEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, device, use_gpu=True, pretrained_weight=None):
        super(TBCNNEncoder, self).__init__()
        self.batch_size = batch_size
        self.config = config
        self.gpu = use_gpu
        self.device = device
        self.encode_dim = encode_dim
        self.embedding_dim = embedding_dim
        self.th = torch.cuda if use_gpu else torch
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.node_emb_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # pretrained  embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

        self.config['embedding_dim'] = embedding_dim
        self.config['conv_output'] = encode_dim

        self.conv_layer1 = ConvolutionLayer(self.config)
        self.pooling_layer = PoolingLayer()

    def get_zeros(self, num: list):
        zeros = Variable(torch.zeros(num, dtype=torch.int64))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def forward(self, x, bs):
        self.batch_size = bs
        nodes = []
        children_indices = []
        children_nodes = []

        for tree in x:  # x is a list of trees
            tmp_nodes = []
            tmp_children_indices = []  # the children's position in the 'nodes' list
            tmp_children_nodes = []  # the list of children corresponding to each node in the 'node' list
            orgianze_tree(tree, tmp_nodes, tmp_children_nodes, tmp_children_indices, 0)
            nodes.append(tmp_nodes)
            children_indices.append(tmp_children_indices)
            children_nodes.append(tmp_children_nodes)

        # pad and convert to tensor
        # convert to tensor, and then pad will be faster?
        max_tree_size = config['max_tree_size']
        max_children_size = config['max_children_size']
        nodes_extended = []
        for nds in nodes:
            if max_tree_size > len(nds):
                pad_len = max_tree_size - len(nds)
            else:
                pad_len = 0
            nds = torch.tensor(nds, dtype=torch.int64, device=self.device)
            nds = torch.cat((nds, self.get_zeros([pad_len])))
            nodes_extended.append(nds[:max_tree_size])
        nodes = torch.stack(nodes_extended)

        children_indices_extended = []
        for indices in children_indices:
            if max_tree_size > len(indices):
                nodes_pad_len = max_tree_size - len(indices)
            else:
                nodes_pad_len = 0
            indices_extended = []
            for nd_indices in indices:
                if max_children_size > len(nd_indices):
                    children_pad_len = max_children_size - len(nd_indices)
                else:
                    children_pad_len = 0
                nd_indices = torch.tensor(nd_indices, dtype=torch.int64, device=self.device)
                nd_indices = torch.cat((nd_indices, self.get_zeros([children_pad_len])))
                indices_extended.append(nd_indices[:max_children_size])
            # indices_extended = torch.Tensor(indices_extended)
            indices_extended = torch.stack(indices_extended)
            indices_extended = torch.cat((indices_extended, self.get_zeros([nodes_pad_len, max_children_size])))
            children_indices_extended.append(indices_extended[:max_tree_size])
        children_indices = torch.stack(children_indices_extended)

        children_nodes_extended = []
        for nds in children_nodes:
            if max_tree_size > len(nds):
                nodes_pad_len = max_tree_size - len(nds)
            else:
                nodes_pad_len = 0
            nds_extended = []
            for child_nodes in nds:
                if max_children_size > len(child_nodes):
                    children_pad_len = max_children_size - len(child_nodes)
                else:
                    children_pad_len = 0
                child_nodes = torch.tensor(child_nodes, dtype=torch.int64, device=self.device)
                child_nodes = torch.cat((child_nodes, self.get_zeros([children_pad_len])))
                nds_extended.append(child_nodes[:max_children_size])
            # nds_extended = torch.Tensor()
            nds_extended = torch.stack(nds_extended)
            nds_extended = torch.cat((nds_extended, self.get_zeros([nodes_pad_len, max_children_size])))
            children_nodes_extended.append(nds_extended[:max_tree_size])
        children_nodes = torch.stack(children_nodes_extended)

        if use_cuda:  # move the data_raw to GPU
            nodes = nodes.cuda()
            children_nodes = children_nodes.cuda()
            children_indices = children_indices.cuda()

        # encode operations
        nodes_embedding = self.node_emb_layer(nodes)
        children_embedding = self.node_emb_layer(children_nodes)
        conv_result = self.conv_layer1(nodes_embedding, children_indices, children_embedding)
        pooling_result = self.pooling_layer(conv_result)
        return pooling_result


'''
Following are utilities for re-formating the inputs to acceptable format of tbcnn's layers
'''


def flat(nums):
    res = []
    for i in nums:
        if isinstance(i, list):
            res.extend(flat(i))
        else:
            res.append(i)
    return res


def orgianze_tree(tree, nodes, children_nodes, children_indices, cur_location):
    if len(tree) == 0:
        return

    if tree[0] != -1:
        current_node = tree[0]  # the root of current sub-tree
        nodes.append(current_node)
        cur_children_nodes = []
        cur_children_indices = []
        children = tree[1:]
        num_children = len(children)  # children number of current node
        start_locations = []
        for i in range(num_children):
            if children[i][0] != -1:
                cur_children_indices.append(cur_location + 1)
                cur_children_nodes.append(children[i][0])
                start_locations.append(cur_location + 1)
                # orgianze_tree(children[i], nodes, children_nodes, children_indices, cur_location+1)
                child_flatten = flat(children[i])
                step = len(child_flatten)
                cur_location += step

        # if num_children == 0:
        #     cur_children_nodes.append(0)
        #     cur_children_indices.append(0)
        children_nodes.append(cur_children_nodes)
        children_indices.append(cur_children_indices)

        for i in range(num_children):
            if children[i][0] != -1:
                orgianze_tree(children[i], nodes, children_nodes, children_indices, start_locations[i])
