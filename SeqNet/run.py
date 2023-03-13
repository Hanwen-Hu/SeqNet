import time
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt

from SeqNet.tool import decompose
from SeqNet.structure import Model
from SeqNet.settings import args

from config import basic_args as basic


# 从序列按条生成数据
def generate_set(s):
    l_seq = s.shape[0]
    x = torch.zeros([l_seq - basic.l_pred * (basic.scale + 1) + 1, basic.l_pred * basic.scale, s.shape[-1]]).to(args.device)
    y = torch.zeros([l_seq - basic.l_pred * (basic.scale + 1) + 1, basic.l_pred, s.shape[-1]]).to(args.device)
    for i in range(l_seq - basic.l_pred * (basic.scale + 1) + 1):
        x[i] = s[i:i + basic.l_pred * basic.scale]
        y[i] = s[i + basic.l_pred * basic.scale:i + basic.l_pred * (basic.scale + 1)]
    return data.TensorDataset(x, y)


# 加载数据集
def load_data(goal='Train'):
    print('Dataset:', basic.data_name, ' ', 'Predict Length:', basic.l_pred)
    print('----Loading Dataset----')
    seq = np.loadtxt('DataTXT/' + basic.data_name + '_' + goal + '.txt')
    print('--------Success!-------')
    return seq


def train_batch(loader, optimizer, model, loss_fun):
    avg_loss, a_loss, m_loss, s_loss = 0, 0, 0, 0
    batch_num = 0
    for j, (v, l) in enumerate(loader):
        optimizer.zero_grad()
        v_a, v_m = v[:, :, :basic.d_in], v[:, :, basic.d_in:]
        l_a, l_m = l[:, :, basic.d_in - basic.d_out:basic.d_in], l[:, :, -basic.d_out:]
        p_a, p_m = model(v_a, v_m)
        loss = loss_fun(p_a, l_a) + loss_fun(p_m, l_m)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        a_loss += loss_fun(p_a, l_a).item()
        m_loss += loss_fun(p_m, l_m).item()
        batch_num += 1
    return model, avg_loss / batch_num, a_loss / batch_num, m_loss / batch_num


def train():
    seq = torch.Tensor(load_data('Train')).to(args.device)
    seq, k_res, k_avg = decompose(seq)  # 分解后并排合成到一起，训练时不需要输出方差
    model = Model().to(args.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=basic.learning_rate, weight_decay=1e-4)
    print('----Training Starts----')
    print('Epoch:', basic.epoch)
    for i in range(basic.epoch):
        dataset = data.DataLoader(generate_set(seq), batch_size=basic.batch_size, shuffle=True)
        start_time = time.time()
        model, error, error_a, error_m = train_batch(dataset, optimizer, model, criterion)
        end_time = time.time()
        time.sleep(2)
        print('Epoch', i, 'Time', round(end_time - start_time, 2), 'Train Error', error, error_a, error_m)
    torch.save(model, 'Model/SeqNet_' + basic.data_name + '_' + str(basic.l_pred) + '.pth')
    return model, k_res, k_avg


# 测试函数，输入模型和数据集，输出mse和mae
def test(mode='Test', k_res=None, k_avg=None):  # true: the accuracy of total seq; false: the accuracy of each component
    loss, loss1, loss2, loss3, batch_num = [0, 0], 0, 0, 0, 0
    mse_fun = lambda x, y: torch.mean(torch.sqrt(torch.mean((x - y) * (x - y), dim=-2)))
    mae_fun = lambda x, y: torch.mean(torch.abs(x - y))
    seq = torch.Tensor(load_data(mode)).to(args.device)
    model = torch.load('Model/SeqNet_' + basic.data_name + '_' + str(basic.l_pred) + '.pth')
    seq, k_res, k_avg = decompose(seq, k_res, k_avg)
    loader = data.DataLoader(generate_set(seq), batch_size=basic.batch_size, shuffle=False)
    for j, (v, l) in enumerate(loader):
        v_a, v_m = v[:, :, :basic.d_in], v[:, :, basic.d_in:]
        l_a, l_m = l[:, :, basic.d_in - basic.d_out:basic.d_in], l[:, :, -basic.d_out:]
        p_a, p_m = model(v_a, v_m)
        p = p_a * k_res[:, -basic.d_out:].unsqueeze(0) + p_m * k_avg[:, -basic.d_out:]
        l = l_a * k_res[:, -basic.d_out:].unsqueeze(0) + l_m * k_avg[:, -basic.d_out:]
        # if mode == 'Test':
        #     p = p - p[:, 0:1] + l[:, 0:1]
        loss[0] += mse_fun(p, l).item()
        loss[1] += mae_fun(p, l).item()
        batch_num += 1
    return round(loss[0] / batch_num, 4), round(loss[1] / batch_num, 4)


def draw(mode='Test', k_res=None):  # 'all'; 'alpha'; 'mu'; 'sigma'
    model = torch.load('Model/SeqNet_' + basic.data_name + '_' + str(basic.l_pred) + '.pth')
    seq = torch.Tensor(load_data(mode)).to(args.device)
    seq, k_res, k_avg = decompose(seq)
    index = 0
    dataset = generate_set(seq)
    while index >= 0:
        index = int(input("index: "))
        v, l = dataset[index:index + 1]
        v_a, v_m = v[:, :, :basic.d_in], v[:, :, basic.d_in:]
        l_a, l_m = l[:, :, basic.d_in - basic.d_out:basic.d_in], l[:, :, -basic.d_out:]
        p_a, p_m = model(v_a, v_m)
        p = (p_a * k_res[:, -basic.d_out:].unsqueeze(0) + p_m * k_avg[:, -basic.d_out:].unsqueeze(0))[0, :, -1].detach().cpu().numpy()
        l = (l_a * k_res[:, -basic.d_out:].unsqueeze(0) + l_m * k_avg[:, -basic.d_out:].unsqueeze(0))[0, :, -1].detach().cpu().numpy()
        p = p - p[0] + l[0]
        print(v_a.shape, k_res.shape)
        v = (v_a * k_res[:, -basic.d_out:].unsqueeze(0) + v_m * k_avg[:, -basic.d_out:].unsqueeze(0))[0, :, -1].detach().cpu().numpy()
        curve = np.concatenate([v, p], axis=-1)
        label = np.concatenate([v, l], axis=-1)
        plt.figure()
        plt.plot(np.arange(basic.l_pred * (basic.scale + 1)), curve, label='predict')
        plt.plot(np.arange(basic.l_pred * (basic.scale + 1)), label, label='real')
        plt.legend()
        plt.title(basic.data_name + ' ' + str(index) + ' ' + str(basic.l_pred))
        plt.show()
