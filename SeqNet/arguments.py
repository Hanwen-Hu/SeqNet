import torch

# ETT,Taxi,ECL,Weather
 # 7,1,10,16
# 96,288,96,144
layer_num = 1
embed_dim = 4
pattern_num = 64  # number of fluctuation patterns
trend_num = 4  # number of trend features
slice_num = 7  # number of sub-sequences