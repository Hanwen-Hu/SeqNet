import numpy as np
import pandas as pd

from SeqNet.arguments import *

assert data_name == 'ECL'


def load_data(path):
    order = pd.read_csv(path)
    count = np.zeros([order.shape[0], in_channel], dtype=float)
    index = 0
    for i in order.itertuples():
        count[index] = i[2:2 + in_channel]
        index += 1
    print(sum(pd.isna(count)))
    print('Done')
    return count


# 生成数据集
print('-------Loading Dataset-------')
count = load_data('../DataCSV/LD2011_2014.csv')
print(count.shape)
print('-------Generating Train Dataset-------')
proportion = 0.2  # 测试集比例
train_data = count[:-int(count.shape[0] * proportion)]
test_data = count[-int(count.shape[0] * proportion):]
np.savetxt('../DataTXT/'+data_name + '_Train.txt', train_data)
np.savetxt('../DataTXT/'+data_name + '_Test.txt', test_data)
