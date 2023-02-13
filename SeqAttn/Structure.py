import torch
import torch.nn as nn

import Tool as t
import Unit as u
from Config import out_channel


class SeqAttn_Alpha(nn.Module):
    def __init__(self, channel, slice_step, slice_len, feature_len, predict_len, layer_num=3):
        super().__init__()
        self.slice_step = slice_step
        self.slice_len = slice_len
        self.embed = nn.Linear(49,16)
        self.predict_len = predict_len
        layer = u.Layer(slice_len, feature_len, stable=True)
        self.coder = u.Coder(layer, layer_num)
        #self.generator = u.Generator((predict_len * 4 - slice_len) // slice_step * channel + channel, 1)
        self.generator = u.Generator(16 , 1)

    # 输入历史序列，维度为batch*embed_dim*dim
    def forward(self, x):
        x = t.slicing(x, self.slice_step, self.slice_len)
        x = self.embed(x.transpose(-1,-2)).transpose(-1, -2)
        x = self.coder(x)
        x = self.generator(x)
        return x


class SeqAttn_Sigma(nn.Module):
    def __init__(self, channel, slice_step, slice_len, feature_len, predict_len):
        super().__init__()
        self.slice_step = slice_step
        self.slice_len = slice_len
        self.predict_len = predict_len
        layer = u.Layer(slice_len, feature_len, stable=False)
        self.coder = u.Coder(layer, 2)
        self.generator = u.Generator((predict_len * 4 - slice_len) // slice_step * channel + channel, 1)

    # 输入历史序列，维度为batch*embed_dim*dim
    def forward(self, x, avg=True):
        # 用均值滤波的方式分割序列为趋势项和残差项，预测残差项
        mean = torch.zeros_like(x)
        if avg:
            mean = torch.mean(x, dim=-2, keepdim=True)
            x = x - mean
        x = t.slicing(x, self.slice_step, self.slice_len)
        #x = self.coder(x)
        x = self.generator(x)
        if avg:
            x = x + mean[:, :, -out_channel:]
        return x
