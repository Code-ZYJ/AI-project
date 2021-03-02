import numpy as np
from torch import nn
import torch
'''
使用前确保最后一维为采样的维度
'''
def sampling(data, size=1500):
    data = torch.from_numpy(data).type(torch.float)
    out = nn.Upsample(size=size)(data)
    out = np.array(out)
    return out

if __name__=='__main__':
    data = np.random.random((1,3,5006))
    sampling(data).shape

