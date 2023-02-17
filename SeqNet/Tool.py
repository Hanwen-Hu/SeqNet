# 序列分解
import torch
import numpy as np
from config import args

future_len = args.predict_len
history_len = 4 * future_len
window_len = history_len + 1  # 做分解的窗口大小


def decompose(series):
    series = series.reshape(-1, args.in_channel)
    sequence_len = series.shape[0]
    left = np.ones((window_len // 2, series.shape[1])) * series[:1]
    right = np.ones((window_len // 2, series.shape[1])) * series[-1:]
    # 序列分解
    alpha_series = np.zeros(series.shape)
    sigma_series = np.zeros(series.shape)
    mu_series = np.zeros(series.shape)
    series_c = np.concatenate([left, series, right], axis=0)
    for i in range(sequence_len):
        mu_series[i] = np.mean(series_c[i:i + window_len], axis=0)  # 分解得到mu
    series -= mu_series
    for i in range(sequence_len):
        sigma_series[i] = np.sqrt(
            np.mean(np.power(series[max(0, i - window_len + 1):i + 1], 2), axis=0)) + 1e-5  # 分解得到sigma
        alpha_series[i] = series[i] / sigma_series[i]  # 分解得到alpha
    return alpha_series, sigma_series, mu_series


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
