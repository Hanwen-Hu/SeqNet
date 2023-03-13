import torch

from SeqNet.settings import args

from config import basic_args as basic

l_history = basic.scale * basic.l_pred
l_window = basic.l_pred * 4 + 1  # 做分解的窗口大小


def decompose(series, k_res=None, k_avg=None):
    series = series.reshape(series.shape[0], -1)
    l_seq, d_seq = series.shape[0], series.shape[1]
    left = torch.ones(l_window // 2, d_seq).to(args.device) * series[:1]
    right = torch.ones(l_window // 2, d_seq).to(args.device) * series[-1:]
    padded = torch.cat([left, series, right], dim=0)
    window_tool = torch.nn.AvgPool1d(l_window, stride=1)
    avg = window_tool(padded.transpose(0, 1)).transpose(0, 1)
    res = series - avg
    if k_res is None:
        k_res = torch.std(res, dim=0, keepdim=True)
        k_avg = torch.std(avg, dim=0, keepdim=True)
        v = k_avg[0, -1]
        k_avg = k_avg / v
        res /= k_res
        avg /= k_avg
    else:
        res /= k_res
        avg /= k_avg
    return torch.cat([res, avg], dim=-1), k_res, k_avg


def slicing(x, slice_step, slice_len):
    # batch*seq_len*channel
    x = x.transpose(1, 2)  # batch*channel*seq_len
    assert (x.shape[-1] - slice_len) % slice_step == 0
    slice_num = (x.shape[-1] - slice_len) // slice_step + 1
    result = torch.zeros(x.shape[0], x.shape[1] * slice_num, slice_len).to(args.device)
    for i in range(slice_num):
        start = i * slice_step
        result[:, i * x.shape[1]:(i + 1) * x.shape[1], :] = x[:, :, start:start + slice_len]
    return result
