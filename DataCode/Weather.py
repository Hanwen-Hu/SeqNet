import numpy as np
import pandas as pd

from SeqNet.settings import *

assert data_name == 'Weather'


# 标准化数据集
def normalize(train, test):
    avg = np.mean(train)
    std = np.std(train)
    train = (train - avg) / std
    test = (test - avg) / std
    return train, test


def load_data(path):
    order = pd.read_csv(path)
    count = np.zeros([order.shape[0], in_channel], dtype=float)
    index = 0
    for i in order.itertuples():
        count[index] = i[2:]
        index += 1
    print('Done')
    return count


# 生成数据集
print('-------Loading Dataset-------')
count = load_data('weather2.csv')
print('-------Generating Train Dataset-------')
proportion = 0.2  # 测试集比例
train_data = count[:-int(count.shape[0] * proportion)]
test_data = count[-int(count.shape[0] * proportion):]
np.savetxt(data_name + '_Train.txt', train_data)
np.savetxt(data_name + '_Test.txt', test_data)
