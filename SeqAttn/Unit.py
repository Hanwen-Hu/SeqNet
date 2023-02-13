import copy

import torch
import torch.nn as nn

import Attention as a


class Layer(nn.Module):
    def __init__(self, embed_dim, feature_dim, stable=True):
        super().__init__()
        self.encode = nn.Linear(embed_dim, feature_dim)
        self.seq = nn.Linear(embed_dim, feature_dim)
        self.decode = nn.Linear(feature_dim, embed_dim)
        self.attn = a.SingleHeadAttention(embed_dim, stable)

    def forward(self, x):
        feature = self.encode(x)
        sequ = x #self.seq(x)
        x = self.decode(self.attn(sequ, sequ, feature))+x
        x = torch.nn.functional.leaky_relu(x)
        return x


class Coder(nn.Module):
    def __init__(self, layer, layer_num):
        super().__init__()
        self.layer_num = layer_num
        self.layer_list = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            x = self.layer_list[i](x)
        return x


# 最后通过一个向量对所有序列做叠加
class Generator(nn.Module):
    def __init__(self, seq_num, out_dim):
        super().__init__()
        self.vector = nn.Sequential(nn.Linear(seq_num, seq_num),
                                    nn.Linear(seq_num, out_dim))

    def forward(self, x):
        # batch*seq_num*embed_dim
        output = self.vector(x.transpose(1, 2))
        return output
