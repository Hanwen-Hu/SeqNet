import argparse


# 变量名标识说明
# l:length 表示时间方向上的长度
# d:dimension 表示不同属性方向上的维度
#
def args_setting():
    hyper_para = argparse.ArgumentParser()
    # Basic Model Settings
    hyper_para.add_argument('-batch_size', type=int, default=64)
    hyper_para.add_argument('-epoch', type=int, default=20)
    hyper_para.add_argument('-learning_rate', type=float, default=1e-3)
    # Dataset Settings
    hyper_para.add_argument('-data_name', type=str, default='ETT', help='ETT, ECL, Weather')
    hyper_para.add_argument('-d_out', type=int, default=1)
    hyper_para.add_argument('-scale', type=int, default=4)  # l_history = l_pred * scale
    args = hyper_para.parse_args()
    dims = {'ETT': 7, 'ECL': 10, 'Weather': 16}
    lens = {'ETT': 96, 'ECL': 96, 'Weather': 144}
    args.l_pred = lens[args.data_name]
    args.d_in = dims[args.data_name]
    return args


basic_args = args_setting()
