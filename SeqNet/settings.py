import argparse
import torch


def args_setting():
    hyper_para = argparse.ArgumentParser()
    hyper_para.add_argument('-period_layer', type=int, default=3)
    hyper_para.add_argument('-trend_layer', type=int, default=3)
    hyper_para.add_argument('-embed_dim', type=int, default=8, help='Number of Features among different Channels')
    hyper_para.add_argument('-pattern_dim', type=int, default=16, help='Number of Fluctuation Patterns')
    hyper_para.add_argument('-slice_num', type=int, default=13, help='Number of sub-sequences in one Channel')
    self_args = hyper_para.parse_args()
    if torch.cuda.is_available():
        self_args.device = torch.device('cuda', 0)
    else:
        self_args.device = torch.device('cpu')
    return self_args


args = args_setting()
