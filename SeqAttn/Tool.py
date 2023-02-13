import torch

from Config import device


def slicing(x, slice_step, slice_len):
    # batch*seq_len*channel
    x = x.transpose(1, 2)  # batch*channel*seq_len
    # print(x.shape, slice_len, slice_step)
    assert (x.shape[-1]-slice_len) % slice_step == 0
    slice_num = (x.shape[-1]-slice_len) // slice_step + 1
    result = torch.zeros(x.shape[0], x.shape[1] * slice_num, slice_len).to(device)
    for i in range(slice_num):
        start = i * slice_step
        result[:, i * x.shape[1]:(i + 1) * x.shape[1], :] = x[:, :, start:start+slice_len]
    return result


# def decompose(sequence, pred_len):
#     left = sequence[:, :1, :] * torch.ones(1, pred_len // 2, 1).to(device)
#     right = sequence[:, -1:, :] * torch.ones(1, pred_len - pred_len // 2 - 1,1).to(device)
#     seq = torch.cat([left, sequence, right], dim=1)
#     avg_core = nn.AvgPool1d(kernel_size=pred_len, stride=1, padding=0)
#     trend = avg_core(seq.transpose(-1,-2)).transpose(-1,-2)
#     residual = sequence-trend
#     return trend, residual

# # 标准化两个序列并计算内积
# def cos_similarity(kernel, sequence, step=10, top_k=8):
#     # 首先把kernel标准化为均值为0，长度为1的向量
#     # kernel的维度为batch*1*pred_len
#     avg_k = torch.mean(kernel, dim=-1, keepdim=True)
#     kernel -= avg_k
#     kernel /= torch.sqrt(kernel * kernel)  # 长度归一化
#     # 而后分离出序列的趋势项，趋势的有效区间为预测长度
#     # sequence 的维度是batch*channel*seq_len
#     left = sequence[:, :,1] * torch.ones(1, 1, kernel.shape // 2)
#     right = sequence[:, :,-1:] * torch.ones(1, 1,kernel.shape[-1] - kernel.shape[-1] // 2 - 1)
#     seq = torch.cat([left, sequence, right], dim=-1)
#     avg_s = nn.AvgPool1d(kernel_size=kernel.shape[-1], stride=1, padding=0)
#     seq -= avg_s  # 将历史序列的均值置为0
#     interest = torch.zeros(sequence.shape[0], kernel.shape[1], (sequence.shape[-1] - kernel.shape[-1] + 1) // step)
#     for i in range(0, sequence.shape[-1] - kernel.shape[-1] + 1, step):
#         # 目前interest中存储的是整体序列各个位置与未来序列的内积结果，维度为batch*dim*len
#         interest[:, :, i] = seq[:, -1:, i / step:i * step + kernel.shape[-1]] * kernel
#     interest = torch.mean(interest, dim=0)  # 对interest在batch维度上取平均，以找到统一的attention位置, 1*pred_len
#     _, positions = interest.topk(top_k, dim=-1)  # position维度是1*top_k
#     result = torch.zeros(sequence.shap[0], sequence.shape[1] * top_k, kernel.shape[-1]).to(device)
#     for i in range(top_k):
#         result[:, i*sequence.shape[1]:(i+1)*sequence.shape[1], :] = sequence[:, :, positions[0,i]*step:positions[0,i]*step+kernel.shape[-1]]
#     return result
