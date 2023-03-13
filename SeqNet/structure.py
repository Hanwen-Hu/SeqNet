import torch
import torch.nn as nn

import SeqNet.unit as u
from SeqNet.settings import args
from SeqNet.tool import slicing

from config import basic_args as basic


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_alpha = PeriodNet()
        self.model_mu = TrendNet()

    # 输入历史序列，维度为batch*embed_dim*dim
    def forward(self, alpha, mu):
        alpha = self.model_alpha(alpha)
        mu = self.model_mu(mu)
        return alpha, mu


class PeriodNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.predict_len = basic.l_pred
        self.slice_step = (basic.scale - 1) * self.predict_len // (args.slice_num - 1)
        self.embed = nn.Linear(args.slice_num * basic.d_in, args.embed_dim)
        layer = u.Layer(basic.l_pred, args.pattern_dim, stable=True)
        self.coder = u.Coder(layer, args.period_layer)
        self.generator = u.Generator(args.embed_dim, basic.d_out)

    def forward(self, x):
        x = slicing(x, self.slice_step, self.predict_len)
        x = self.embed(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.coder(x)
        return self.generator(x)


class TrendNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.slice_step = (basic.scale - 1) * basic.l_pred // (args.slice_num - 1)
        layer = u.Layer(basic.l_pred, args.pattern_dim, stable=False)
        self.coder = u.Coder(layer, args.trend_layer)
        self.generator = u.Generator(args.embed_dim, basic.d_out)
        self.embed = nn.Linear(args.slice_num * basic.d_in, args.embed_dim)

    def forward(self, x):
        mean = torch.mean(x, dim=-2, keepdim=True)
        x = x - mean
        x = slicing(x, self.slice_step, basic.l_pred)
        x = self.embed(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.coder(x)
        x = self.generator(x)
        return x + mean[:, :, -basic.d_out:]
