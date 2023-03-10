# 定义对序列的注意力机制
import torch
import torch.nn as nn

from SeqNet.settings import args


# 周期事件attention
def corr_attention(query, key, value):
    # batch*head_num*seq_num*pattern_num
    assert query.shape[-1] == key.shape[-1]
    score = torch.matmul(query, key.transpose(-1, -2))  # dot product
    len_q = torch.sqrt(torch.sum(query * query, dim=-1))
    len_k = torch.sqrt(torch.sum(key * key, dim=-1))
    len_mat = torch.matmul(len_q.unsqueeze(-1), len_k.unsqueeze(-2))
    score = score / len_mat  # cosine similarity
    signal = torch.sign(score)
    size = score.shape
    score = torch.abs(score).reshape(-1)
    _, idx = torch.sort(score, descending=True)
    prob = torch.arange(1.0 / idx.shape[0], 1 + 1.0 / idx.shape[0], 1.0 / idx.shape[0]).to(args.device)
    score[idx] = prob
    score = -torch.log(score).reshape(size) * signal
    v_attn = torch.matmul(score, value)
    return v_attn


# 趋势事件attention
def diff_attention(query, key, value):
    # 维度为batch*head_num*seq_num*head_dim
    # 此时query和key均为提取到的序列特征
    assert query.shape[-1] == key.shape[-1]
    query = query - torch.mean(query, dim=-1, keepdim=True)
    query = query.unsqueeze(-2)
    key = key - torch.mean(key, dim=-1, keepdim=True)
    key = key.unsqueeze(-3)
    score = torch.mean((query - key) * (query - key), dim=-1)
    size = score.shape
    score = score.reshape(-1)
    _, idx = torch.sort(score, descending=False)
    prob = torch.arange(1.0 / idx.shape[0], 1 + 1.0 / idx.shape[0], 1.0 / idx.shape[0]).to(args.device)
    score[idx] = prob
    score = -torch.log(score).reshape(size)
    v_attn = torch.matmul(score, value)
    return v_attn


# 定义多头注意力机制，但可能暂时不考虑使用
class MultiHeadAttention(nn.Module):
    def __init__(self, seq_len, pattern_num, state, head_num=1):
        super().__init__()
        assert seq_len % head_num == 0
        assert pattern_num % head_num == 0
        self.state = state
        self.head_num = head_num

    def forward(self, query, key, value):
        # batch*seq_num*embed_dim
        batch_size, seq_num = query.shape[0], query.shape[1]
        query, key, value = [x.reshape(batch_size, seq_num, self.head_num, -1).transpose(1, 2) for x in (query, key, value)]
        if self.state:
            v_out = corr_attention(query, key, value)
        else:
            v_out = diff_attention(query, key, value)
        return v_out.transpose(1, 2).contiguous().reshape(batch_size, seq_num, -1)
