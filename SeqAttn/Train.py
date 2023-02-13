import time

import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data

from Config import *
from Structure import SeqAttn_Alpha, SeqAttn_Sigma
from Decompose import decompose


def generate_set(s):
    # 生成训练数据
    seq_len = s.shape[0]
    train_feature = np.zeros([seq_len-predict_len*5+1, predict_len*4, s.shape[1]])
    train_label = np.zeros([seq_len-predict_len*5+1, predict_len, s.shape[1]])
    for i in range(seq_len-predict_len*5+1):
        train_feature[i] = np.array(s[i:i+predict_len*4])
        train_label[i] = np.array(s[i+predict_len*4:i+predict_len*5])
    return data.TensorDataset(torch.Tensor(train_feature).to(device), torch.Tensor(train_label).to(device))


def load_data(name):
    print('Model: Seq-Attn')
    print('Dataset:', name, ' ', 'Predict Length:', predict_len)
    print('----Loading Dataset----')
    train_seq = np.loadtxt('DataTXT/'+name + '_Train.txt')
    test_seq = np.loadtxt('DataTXT/'+name + '_Test.txt')
    print('----Decomposition Begins----')
    alpha, mu, sigma = decompose(train_seq)
    alpha_, mu_, sigma_ = decompose(test_seq)
    print('----Generating Sets----')
    alpha = generate_set(alpha)
    mu = generate_set(mu)
    sigma = generate_set(sigma)
    alpha_ = generate_set(alpha_)
    #mu_ = generate_set(mu_)
    #sigma_ = generate_set(sigma_)
    print('--------Success!-------')
    return alpha, mu, sigma, alpha_, mu_, sigma_


def train_batch(loader, optimizer, model, loss_fun):
    avg_loss = 0
    batch_num = 0
    for j, (v, l) in enumerate(loader):
        optimizer.zero_grad()
        label = l[:, :, -out_channel:]  # the last out_dim attributes are the sequence to predict
        predict = model(v[:, :, -in_channel:])
        loss = loss_fun(predict, label)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        batch_num += 1
    print(avg_loss / batch_num)
    return model


def train():
    dataset_alpha, dataset_mu, dataset_sigma, test_a, test_m, test_s = load_data(data_name)
    loader_a = data.DataLoader(dataset_alpha, batch_size=batch_size, shuffle=True)
    loader_m = data.DataLoader(dataset_mu, batch_size=batch_size, shuffle=True)
    loader_s = data.DataLoader(dataset_sigma, batch_size=batch_size, shuffle=True)
    print('----Training Starts----')
    print('Epoch:', epoch)
    model_alpha = SeqAttn_Alpha(in_channel, slice_step, slice_len, feature_len, predict_len, layer_num).to(device)
    model_mu = SeqAttn_Sigma(in_channel, slice_step, slice_len, 4, predict_len).to(device)
    model_sigma = SeqAttn_Sigma(in_channel, slice_step, slice_len, 4, predict_len).to(device)
    criterion = torch.nn.MSELoss()
    optimizer_a = torch.optim.Adam(model_alpha.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer_m = torch.optim.Adam(model_mu.parameters(), lr=1e-3)
    optimizer_s = torch.optim.Adam(model_sigma.parameters(), lr=1e-3)
    for i in range(epoch):
        start_time = time.time()
        model_alpha = train_batch(loader_a, optimizer_a, model_alpha, criterion)
        #model_mu = train_batch(loader_m, optimizer_m, model_mu, criterion)
        #model_sigma = train_batch(loader_s, optimizer_s, model_sigma, criterion)
        end_time = time.time()
        test_ax, test_ay = test_a[:]
        #test_mx, test_my = test_m[:]
        #test_sx, test_sy = test_s[:]
        test_ax = model_alpha(test_ax) #* model_sigma(test_sx) + model_mu(test_mx)
        #test_mx = model_mu(test_mx)
        #test_sx = model_sigma(test_sx)
        #test_y = test_ay * test_sy + test_my
        #scale = 8.572438505693652
        loss = torch.mean((test_ax-test_ay[:,:,-1:])*(test_ax-test_ay[:,:,-1:]))
        print('Epoch', i, 'Time', round(end_time-start_time, 2), 'Error', loss.item())
        plt.figure()
        index = 2000
        C, L = dataset_alpha[index:index+1]
        T = model_alpha(C).reshape(-1).detach().cpu().numpy()
        L = L[:, :, -1].reshape(-1).detach().cpu().numpy()
        np.savetxt('ett_sigma_f.txt', T)
        np.savetxt('ett_sigma_t.txt', L)
        plt.plot(np.arange(predict_len), T, label='predict')
        plt.plot(np.arange(predict_len), L, label='real')
        # C = C[:, :, -1].reshape(-1).detach().cpu().numpy()
        # for i in range(4):
        #     plt.plot(np.arange(predict_len), C[i*predict_len:(i+1)*predict_len], label=str(i))
        plt.legend()
        plt.show()

train()
# torch.save(model_mu, 'Result/SeqAttn-'+str(data_name)+str(predict_len)+'.pth')
# time_list = np.array(time_list)
# loss_list = np.array(loss_list)
# np.savetxt('Result/time.txt', time_list)
# np.savetxt('Result/loss.txt', loss_list)

def test():
    a = 1
# for j in range(x_test.shape[1] // batch_size):
#     alp = model_alpha(x_test[0, j * batch_size:(j + 1) * batch_size, :, -in_channel:])
#     alp_l = y_test[0, j * batch_size:(j + 1) * batch_size, :, -out_channel:]
#     mu = model_mu(x_test[1, j * batch_size:(j + 1) * batch_size, :, -in_channel:])
#     mu_l = y_test[1, j * batch_size:(j + 1) * batch_size, :, -out_channel:]
#     sig = model_sigma(x_test[2, j * batch_size:(j + 1) * batch_size, :, -in_channel:])
#     sig_l = y_test[2, j * batch_size:(j + 1) * batch_size, :, -out_channel:]
#     verify = mu + alp * sig
#     real = mu_l + alp_l * sig_l
#     scale = 26.40450525073059
#     error_mse += torch.mean((verify - real) * (verify - real)).item() / scale / scale
#     error_mae += torch.mean(torch.abs(verify - real)).item() / scale
# print('epoch:', i, ' time:', round(end_time - start_time, 2), ' MSE:', error_mse / (x_test.shape[1] // batch_size),
#       'MAE', error_mae / (x_test.shape[1] // batch_size))
# if i % 2 == 0:
#     plt.figure()
#     print(torch.mean((verify[0] - real[0]) * (verify[0] - real[0])))
#     plt.plot(np.arange(predict_len), alp[0].detach().cpu().numpy(), color='red')
#     plt.plot(np.arange(predict_len), alp_l[0].detach().cpu().numpy(), color='blue')
#     plt.show()