# 序列分解
import numpy as np
from Config import predict_len

future_len = predict_len
history_len = 4*future_len
window_len = history_len+1  # 做分解的窗口大小


def decompose(series):
    sequence_len = series.shape[0]
    left = np.ones((window_len//2, series.shape[1]))*series[:1]
    right = np.ones((window_len//2, series.shape[1]))*series[-1:]
    # 序列分解
    alpha_series = np.zeros(series.shape)
    sigma_series = np.zeros(series.shape)
    mu_series = np.zeros(series.shape)
    series_c = np.concatenate([left, series, right], axis=0)
    for i in range(sequence_len):
        mu_series[i] = np.mean(series_c[i:i+window_len], axis=0)  # 分解得到mu
    series -= mu_series
    for i in range(sequence_len):
        sigma_series[i] = np.sqrt(np.mean(np.power(series[max(0, i-window_len+1):i+1], 2), axis=0))+1e-5  # 分解得到sigma
        alpha_series[i] = series[i] / sigma_series[i]  # 分解得到alpha
    return alpha_series, sigma_series, mu_series


