import numpy as np
import pandas as pd

from SeqNet.settings import *

assert data_name == 'Taxi'


def load_data(path, timestep=5, start_day=0):
    order = pd.read_csv(path)
    assert 60 % timestep == 0
    count = np.zeros((31 * 24 * 60 // timestep,1), dtype=int)
    for i in order.itertuples():
        time, lon, lat = i[2], i[5], i[6]
        day, hour, minute = int(time[8:10]) - 1, int(time[11:13]), int(time[14:16])
        index = (day * 24 + hour) * 60 // timestep + minute // timestep
        count[index] += 1
    print('Done')
    return count


# 生成数据集
print('-------Loading Dataset-------')
count_11 = load_data('data_shanghai_202111.csv', timestep=5, start_day=1)
count_12 = load_data('data_shanghai_202112.csv', timestep=5, start_day=3)
count = np.concatenate([count_11, count_12], axis=0)
print(count.shape)
print('-------Generating Train Dataset-------')
proportion = 0.3  # 测试集比例
train_data = count[:-int(count.shape[0] * proportion)]
test_data = count[-int(count.shape[0] * proportion):]
np.savetxt(data_name + '_Train.txt', train_data)
np.savetxt(data_name + '_Test.txt', test_data)
