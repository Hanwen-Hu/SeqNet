# 定义对序列的注意力机制
import torch
import torch.nn as nn

from Config import device


def topk_attention(query, key, value):
    assert query.shape[-1] == key.shape[-1]
    score = torch.matmul(query, key.transpose(-1, -2))
    #print(score.shape)
    len_q = torch.sqrt(torch.sum(query*query, dim=-1))
    len_k = torch.sqrt(torch.sum(key*key, dim=-1))
    len_mat = torch.matmul(len_q.unsqueeze(-1), len_k.unsqueeze(-2))
    score = score / len_mat
    val, index = score.topk(score.shape[-1]//3, dim = -1)
    bias, _ = torch.min(val, dim=-1, keepdim=True)
    score[score<bias] = 0
    score[torch.abs(score)<0.6] = 0
    score = score / torch.sum(torch.abs(score), dim=-1, keepdim=True)
    v_attn = torch.matmul(score, value)
    return v_attn


def attention(query, key, value):
    assert query.shape[-1] == key.shape[-1]
    score = torch.matmul(query, key.transpose(-1, -2))/query.shape[-1]
    score = torch.relu(score)
    score = score / (torch.sum(torch.abs(score), dim=-1, keepdim=True)+1e-5)  #torch.softmax(score, dim=-1)#
    v_attn = torch.matmul(score, value)
    return v_attn

def diff_attention(query, key, value):
    # 维度为batch*head_num*seq_num*head_dim
    # 此时query和key均为提取到的序列特征
    assert query.shape[-1] == key.shape[-1]
    query = torch.mean(query, dim=-1, keepdim=True)
    key = torch.mean(key, dim=-1, keepdim=True).transpose(-1,-2)
    score = -torch.log(torch.softmax(-(query - key) * (query - key), dim=-1)+torch.Tensor([1e-3]).to(device))
    score = score / torch.sum(score, dim=-1, keepdim=True)
    v_attn = torch.matmul(score, value)
    return v_attn


def cos_attention(query, key, value):
    # 维度为batch*head_num*seq_num*head_dim
    # 此时query和key均为提取到的序列特征
    assert query.shape[-1] == key.shape[-1]
    # query = query-torch.mean(query, dim=-1, keepdim=True)
    # key = key-torch.mean(key, dim=-1, keepdim=True)
    len_seq, num_seq = query.shape[-1], query.shape[-2]
    score = torch.matmul(query, key.transpose(-1, -2))/query.shape[-1]
    score = torch.relu(score)
    # len_q = torch.sqrt(torch.sum(query * query, dim=-1))
    #len_k = torch.sqrt(torch.sum(key * key, dim=-1))
    #len_mat = torch.matmul(len_q.unsqueeze(-1), len_k.unsqueeze(-2))
    #score = score / (len_mat + torch.Tensor([1e-5]).to(device))
    #score = score / (torch.sum(torch.abs(score), dim=-1, keepdim=True))
    # 修正对角元素，赋予其95%显著区间的相关系数值
    # dia = torch.ones(num_seq).to(device)
    # score = score - torch.diag(dia).unsqueeze(0).unsqueeze(0) * (score - torch.sqrt(torch.Tensor([4 / (len_seq + 2)]).to(device)))
    score = -torch.log(torch.softmax(-(3 - 2) * score * score / (1 - score * score + 1e-3), dim=-1)+torch.Tensor([1e-3]).to(device))  # 计算信息量
    score = score / torch.sum(score, dim=-1, keepdim=True)  # 信息量归一化
    #vv, _ = torch.max(score[0], dim=-1)
    #print(vv)
    v_attn = torch.matmul(score, value)
    return v_attn


# 定义多头注意力机制，但可能暂时不考虑使用
class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim, stable=True, head_num=1):
        super().__init__()
        assert embed_dim % head_num == 0
        self.state = stable
        self.head_num = head_num

    def forward(self, query, key, value):
        # batch*seq_num*embed_dim
        batch_size, seq_num = query.shape[0], query.shape[1]
        query, key, value = [x.reshape(batch_size, seq_num, self.head_num, -1).transpose(1, 2) for x in (query, key, value)]
        if self.state:
            v_out = topk_attention(query, key, value)
        else:
            v_out = diff_attention(query, key, value)
        return v_out.transpose(1, 2).contiguous().reshape(batch_size, seq_num, -1)
