from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
import matplotlib.pyplot as plt
#%%  读取数据集
images=[]
mnist=load_digits()
imgs = mnist.data
imgs = imgs.reshape(imgs.shape[0],-1)

data = torch.from_numpy(imgs).type(torch.float32).cuda()
data = TensorDataset( data,data)
dataloader = DataLoader(data, batch_size=512, shuffle=True)
#%% 构建AE
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64,32),
            nn.Tanh(),
            nn.Linear(32,16),
            nn.Tanh(),
            nn.Linear(16,8),
            nn.Tanh(),
            nn.Linear(8,3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3,8),
            nn.Tanh(),
            nn.Linear(8,16),
            nn.Tanh(),
            nn.Linear(16,32),
            nn.Tanh(),
            nn.Linear(32,64),
            nn.Sigmoid()
        )

    def forward(self,input):
        enc = self.encoder(input)
        output = self.decoder(enc)
        return output,enc

#%% 训练
autoencoder = AE().cuda()
print(autoencoder)
epochs = 4000
opt=torch.optim.Adam(autoencoder.parameters(), lr=0.001)
loss_func = nn.MSELoss()

for epoch in range(epochs):
    for x,y in dataloader:
        y_pred,_ = autoencoder(x)
        loss = loss_func(y_pred,y)   #自训练，x既是输入又是输出
        loss.backward()
        opt.step()
        opt.zero_grad()
    print('epoch: {}  loss: {}'.format(epoch,loss))

#% 结果展示
with torch.no_grad():
    dec,enc = autoencoder(torch.from_numpy(imgs).type(torch.float32).cuda())
    X, Y, Z = enc.data[:, 0].cpu().numpy(), enc[:, 1].cpu().numpy(), enc.data[:, 2].cpu().numpy()

plt.imshow(dec[0].cpu().reshape(8,8))
plt.imshow(imgs[0].reshape(8,8))
ax = plt.subplot(111,projection='3d')
ax.scatter(X,Y,Z)