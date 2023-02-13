import torch

epoch = 20
data_name = 'ETT'  # ETT,Taxi,ECL,Weather
in_channel = 7  # 7,1,10,16
predict_len = 96  # 96,288,96,144
if torch.cuda.is_available():
    device = torch.device('cuda', 1)
else:
    device = torch.device('cpu')
batch_size = 256
layer_num = 2
out_channel = 1
feature_len = 66
slice_len = predict_len
slice_step = slice_len // 2

# 288 1615.7129477107712
# 576 1485.6057142365628
# 1000 1357.01253309703
#ECL8.57244
# 96 26.40450525073059