import argparse
import torch

def args_setting():
    hyper_para = argparse.ArgumentParser()
    hyper_para.add_argument('-data_name', type=str, default='Taxi')
    hyper_para.add_argument('-batch_size', type=int, default=64)
    hyper_para.add_argument('-predict_len', type=int, default=288)
    hyper_para.add_argument('-epoch', type=int, default=40)
    hyper_para.add_argument('-learning_rate', type=float, default=1e-3)
    hyper_para.add_argument('-in_channel', type=int, default=1)
    hyper_para.add_argument('-out_channel', type=int, default=1)
    hyper_para.add_argument('-history_scale', type=int, default=4)  # history_len = predict_len * history_scale
    para = hyper_para.parse_args()
    if torch.cuda.is_available():
        para.device = torch.device('cuda', 0)
    else:
        para.device = torch.device('cpu')
    return para

args = args_setting()