import time
import numpy as np
import torch.utils.data as data

from SeqNet.arguments import *
from SeqNet.tool import decompose
from SeqNet.structure import SeqNet
from config import args

# 生成训练数据
def generate_set(s):
    seq_len = s.shape[0]
    train_feature = np.zeros([seq_len - args.predict_len * (args.history_scale + 1) + 1, args.predict_len * args.history_scale, s.shape[1]])
    train_label = np.zeros([seq_len - args.predict_len * (args.history_scale +1) + 1, args.predict_len, s.shape[1]])
    for i in range(seq_len - args.predict_len * (args.history_scale +1) + 1):
        train_feature[i] = np.array(s[i:i + args.predict_len * args.history_scale])
        train_label[i] = np.array(s[i + args.predict_len * args.history_scale:i + args.predict_len * (args.history_scale + 1)])
    return data.TensorDataset(torch.Tensor(train_feature).to(args.device), torch.Tensor(train_label).to(args.device))


# 加载数据集并分解
def load_data(goal='Train'):
    print('Dataset:', args.data_name, ' ', 'Predict Length:', args.predict_len)
    print('----Loading Dataset----')
    seq = np.loadtxt('DataTXT/' + args.data_name + '_' + goal + '.txt')
    alpha, mu, sigma = decompose(seq)
    print('--------Success!-------')
    return alpha, mu, sigma


def train_batch(loader, optimizer, model, loss_fun):
    avg_loss = 0
    batch_num = 0
    for j, (v, l) in enumerate(loader):
        optimizer.zero_grad()
        label = l[:, :, -args.out_channel:]  # the last out_dim attributes are the sequence to predict
        predict = model(v[:, :, -args.in_channel:])
        loss = loss_fun(predict, label)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        batch_num += 1
    return model, avg_loss / batch_num


def train(dataset, mode=1):
    dataset = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    if mode == 1:
        model = SeqNet(0).to(args.device)
    else:
        model = SeqNet(1).to(args.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print('----Training Starts----')
    print('Epoch:', args.epoch)
    for i in range(args.epoch):
        start_time = time.time()
        model, error = train_batch(dataset, optimizer, model, criterion)
        end_time = time.time()
        print('Epoch', i, 'Time', round(end_time - start_time, 2), 'Train Error', error)
    dict1 = {1: 'a', 2: 'm', 3: 's'}
    torch.save(model, 'Model/SeqNet_'+dict1[mode]+'_'+args.data_name+'_'+str(args.predict_len)+'.pth')
    return model





# 测试函数，输入模型和数据集，输出mse和mae
def test(mode=0):  #0-all, 1-alpha, 2-mu, 3-sigma
    mse_loss, mae_loss, batch_num = 0, 0, 0
    mse_fun = torch.nn.MSELoss()
    mae_fun = lambda x, y: torch.mean(torch.abs(x - y))
    a, m, s = load_data('Test')
    if mode == 0:
        model1 = torch.load('Model/SeqNet_a_'+args.data_name+'_'+str(args.predict_len)+'.pth')
        model2 = torch.load('Model/SeqNet_m_'+args.data_name+'_'+str(args.predict_len)+'.pth')
        model3 = torch.load('Model/SeqNet_s_'+args.data_name+'_'+str(args.predict_len)+'.pth')
        alpha = data.DataLoader(generate_set(a), batch_size=args.batch_size, shuffle=False)
        mu = data.DataLoader(generate_set(m), batch_size=args.batch_size, shuffle=False)
        sigma = data.DataLoader(generate_set(s), batch_size=args.batch_size, shuffle=False)
        for j, ((v1, l1), (v2, l2), (v3, l3)) in enumerate((alpha, mu, sigma)):
            result = model2(v2[:,:,-args.in_channel:]) + model1(v1[:,:,-args.in_channel:]) * model3(v3[:,:,-args.in_channel:])
            label = l2[:,:,-args.out_channel:] + l1[:,:,-args.out_channel:] * l3[:,:,-args.out_channel:]
            mse_loss += mse_fun(result, label).item()
            mae_loss += mae_fun(result, label).item()
            batch_num += 1
    else:
        dict1 = {1:'a', 2:'m', 3:'s'}
        dict2 = {1:a, 2:m, 3:s}
        model = torch.load('Model/SeqNet_'+dict1[mode]+'_'+ args.data_name+'_'+str(args.predict_len)+'.pth')
        loader = data.DataLoader(generate_set(dict2[mode]), batch_size=args.batch_size, shuffle=False)
        for j, (v, l) in enumerate(loader):
            result = model(v[:,:,-args.in_channel:])
            label = l[:,:,-args.out_channel:]
            mse_loss += mse_fun(result, label).item()
            mae_loss += mae_fun(result, label).item()
            batch_num += 1
    return mse_loss / batch_num, mae_loss / batch_num
