import torch
import torch.nn.functional as F

from ParameterConfig import ParameterConfig


class SelfAttention(torch.nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size

        # 初始化权重矩阵
        self.values = torch.nn.Linear(embed_size, 88, bias=False)
        self.keys = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.queries = torch.nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, tensor):

        # 通过线性层进行投影
        values = self.values(tensor)
        keys = self.keys(tensor)
        queries = self.queries(tensor)

        # 计算注意力分数
        attention = torch.matmul(queries, keys.permute(0, 2, 1))

        attention = attention / (self.embed_size ** 0.5)

        # 使用softmax函数计算注意力权重
        attention = F.softmax(attention, dim=-1)
        # 使用权重对值进行加权平均
        out = torch.matmul(attention, values)

        out = out.view(out.shape[0], -1)

        return out
