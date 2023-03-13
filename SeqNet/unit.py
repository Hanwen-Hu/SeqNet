import copy
import torch.nn as nn

import SeqNet.attention as attn


class Layer(nn.Module):
    def __init__(self, sub_seq_len, pattern_num, stable=True):
        super().__init__()
        self.state = stable
        self.encode = nn.Linear(sub_seq_len, pattern_num)
        self.decode = nn.Linear(pattern_num, sub_seq_len)
        self.attn = attn.MultiHeadAttention(sub_seq_len, pattern_num, stable)
        self.fun = nn.PReLU()

    def forward(self, x):
        patterns = self.encode(x)
        x = self.decode(self.attn(x, x, patterns)) + x
        return self.fun(x)

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
        self.selector = nn.Linear(seq_num, out_dim)

    def forward(self, x):
        # batch*seq_num*embed_dim
        return self.selector(x.transpose(1, 2))
