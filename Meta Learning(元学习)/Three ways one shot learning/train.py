import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dl = resize(imread('./images/DL.jpg'),(200,200)).transpose((2,0,1))
ma = resize(imread('./images/MA.jpg'),(200,200)).transpose((2,0,1))
ml = resize(imread('./images/ML.jpg'),(200,200)).transpose((2,0,1))

#%% 构建数据集
dl = torch.from_numpy(dl*255).type(torch.float32).unsqueeze(dim=0)
ma = torch.from_numpy(ma*255).type(torch.float32).unsqueeze(dim=0)
ml = torch.from_numpy(ml*255).type(torch.float32).unsqueeze(dim=0)
zero = torch.Tensor([[0.]])
one = torch.Tensor([[1.]])

data = np.concatenate((ml,dl,ma))  #后面预测要用的
torch.save(data,'data.pkl')

dataset = [[dl,ma,zero],[dl, dl, one],
           [dl,ml,zero],[ma, ma, one],
           [ma,ml,zero],[ml, ml, one]]

#%% Siamese_network

class Siamese_network(nn.Module):
    def __init__(self):
        super(Siamese_network, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(3,32,5,padding=(2,2)),
                             nn.Tanh(),
                             nn.AvgPool2d(2),
                             nn.Conv2d(32,4,5,padding=(2,2)),
                             nn.Tanh(),
                             nn.MaxPool2d(2),
                             nn.Dropout(0.03),
                             nn.Flatten())
        self.linear = nn.Sequential(nn.Dropout(0.6),
                                    nn.Linear(50*50*4,512),
                                    nn.Tanh(),
                                    nn.Dropout(0.6),
                                    nn.Linear(512,1))
        self.out = nn.Sigmoid()
    def forward(self,img1,img2):
        fla1 = self.conv(img1)
        fla2 = self.conv(img2)
        fla = F.tanh(abs(fla1-fla2))
        out = self.out(self.linear(fla))
        return out

#%% 训练
epochs = 900
model = Siamese_network().to(device)
loss_fn = nn.BCELoss()
optim = torch.optim.Adam(model.parameters(),lr=1e-3)

for epoch in tqdm(range(epochs)):
    for x,y,z in dataset:
        x = x.to(device)
        y = y.to(device)
        label = z.to(device)

        y_pred = model(x,y)
        loss = loss_fn(y_pred,label)
        loss.backward()
        optim.step()
        optim.zero_grad()


#%% 预测
model.eval()
print(model(ml.to(device),ma.to(device)))
print(model(ma.to(device),dl.to(device)))
print(model(ml.to(device),dl.to(device)))

print(model(ml.to(device),ml.to(device)))
print(model(ma.to(device),ma.to(device)))
print(model(dl.to(device),dl.to(device)))

torch.save(model,'sia_net.pt')


#%% 权重可视化
import matplotlib.pyplot as plt
weight = []
for i in model.parameters():
    with torch.no_grad():
        i = i.cpu().numpy()
    print(i.shape)
    weight.append(i)

conv1 = weight[0].reshape(-1,5,5)
conv2 = weight[2].reshape(-1,5,5)

for i in range(len(conv2)):
    plt.imshow(conv1[i],cmap='gray')
    plt.show()