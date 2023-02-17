import torch
import torch.nn as nn

import SeqNet.unit as u
import SeqNet.arguments as self_args
from SeqNet.tool import slicing
from config import args

class SeqNet(nn.Module):
    def __init__(self, mode = 0):
        super().__init__()
        self.mode = mode
        self.predict_len = args.predict_len
        self.slice_step = (args.history_scale -1) * self.predict_len // (self_args.slice_num -1)
        self.embed = nn.Linear(self_args.slice_num*args.in_channel, self_args.embed_dim)
        if mode == 0:
            layer = u.Layer(args.predict_len, self_args.pattern_num, stable=True)
        elif mode == 1:
            layer = u.Layer(args.predict_len, self_args.trend_num, stable=False)
        else:
            layer = nn.Linear(args.predict_len, args.predict_len)
        self.coder = u.Coder(layer, self_args.layer_num)
        self.generator = u.Generator(self_args.embed_dim, args.out_channel)
        self.simpler = nn.Linear(args.predict_len * args.in_channel * args.history_scale, args.predict_len * args.out_channel)


    # 输入历史序列，维度为batch*embed_dim*dim
    def forward(self, x):
        #mean = torch.mean(x, dim=-2, keepdim=True)
        #x -= mean
        x = slicing(x, self.slice_step, self.predict_len)
        x = self.embed(x.transpose(-1,-2)).transpose(-1, -2)
        x = self.coder(x)
        x = self.generator(x)
        # x = x.reshape(x.shape[0], -1)
        # x = self.simpler(x)
        # x = x.reshape(x.shape[0], -1, args.out_channel)
        return x #+ mean[:, :, -args.out_channel:]
