from skimage.io import imread
from skimage.transform import resize
import torch
from torch import nn
from glob import glob
import numpy as np
from torch.utils.data import TensorDataset,DataLoader
import torch.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
keyword = ['cat','dog','fish','mouse','rabbit']
path = glob('./support set/*.jpg')

#%% 构建根据support set进行对应处理
# 给每个图像resize，并打上标签
imgs, labels = [], []
for p in path:
    img = imread(p)
    img = resize(img,(128,128)).transpose((2,0,1))
    imgs.append(img*255)     #resize后将元素归一化了
    for k in keyword:
        if k in p:
            labels.append(k)
            break

# 构建用于输入到网路中的数据集
batch_size=4
anchor, x_pos, x_neg = [], [], []
for i in range(len(labels)):
    for j in range(len(labels)):
        for k in range(len(labels)):
            if labels[i]==labels[j] and labels[i]!=labels[k]:   #anchor与x_pos相同，与x_neg相反
                anchor.append(imgs[i])
                x_pos.append(imgs[j])
                x_neg.append(imgs[k])
anchor = torch.from_numpy(np.array(anchor)).type(torch.float32)
x_pos = torch.from_numpy(np.array(x_pos)).type(torch.float32)
x_neg = torch.from_numpy(np.array(x_neg)).type(torch.float32)

ds = TensorDataset(anchor, x_pos, x_neg)
dl = DataLoader(ds, batch_size=batch_size, shuffle = True)


#%% Siamese_network & Triplet loss
class Siamese_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 7, padding=(3,3)),
                                   nn.Tanh(),
                                   nn.MaxPool2d(2))   #输出[batch,64,64,32]
        self.conv2 = nn.Sequential(nn.Conv2d(32,64, 5, padding=(2,2)),
                                   nn.Tanh(),
                                   nn.MaxPool2d(2))   #输出[batch,32,32,64]
        self.conv3 = nn.Sequential(nn.Conv2d(64,128,3, padding=(1,1)),
                                   nn.Tanh(),
                                   nn.AvgPool2d(2))   #输出[batch,16,16,128]
        self.flatten = nn.Flatten()
        self.hidden = nn.Sequential(nn.Linear(16*16*128,2048),
                                    nn.Linear(2048,256),
                                    nn.Linear(256,32))
        self.out = nn.Linear(32,3)

    def forward(self,anchor, x_pos, x_neg):
        anchor_out = self.out(self.hidden(self.flatten(self.conv3(self.conv2(self.conv1(anchor))))))
        x_pos_out = self.out(self.hidden(self.flatten(self.conv3(self.conv2(self.conv1(x_pos))))))
        x_neg_out = self.out(self.hidden(self.flatten(self.conv3(self.conv2(self.conv1(x_neg))))))
        return anchor_out, x_pos_out,x_neg_out

#%% 训练

model = Siamese_network().to(device)
margin = 4
loss_fn = nn.TripletMarginLoss(margin = margin)

epochs = 100
optim = torch.optim.RMSprop(model.parameters(),lr = 1e-4)
learning_curve = []

for epoch in range(epochs):
    LOSS = 0
    for anchor, x_pos, x_neg in dl:
        anchor = anchor.to(device)
        x_pos = x_pos.to(device)
        x_neg = x_neg.to(device)
        anchor_out, x_pos_out, x_neg_out = model(anchor, x_pos, x_neg)

        loss = loss_fn(anchor_out, x_pos_out,x_neg_out)
        loss.backward()
        optim.step()
        optim.zero_grad()
        with torch.no_grad():
            LOSS += loss
    print('epoch: {} -------- loss: {}'.format(epoch+1, LOSS))
    learning_curve.append(LOSS)
    if LOSS == 0:
        break

#%% 可视化
with torch.no_grad():
    sample_anchor = anchor_out.cpu().numpy()
    sample_pos = x_pos_out.cpu().numpy()
    sample_neg = x_neg_out.cpu().numpy()

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
mpl.rcParams['legend.fontsize'] = 10

plt.plot(learning_curve)
plt.show()

for i in range(4):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(sample_anchor[i,0], sample_anchor[i,1], sample_anchor[i,2],
                label='anchor', marker = '^')
    ax.scatter(sample_pos[i,0], sample_pos[i,1], sample_pos[i,2],
                label='pos', marker = 'v')
    ax.scatter(sample_neg[i,0], sample_neg[i,1], sample_neg[i,2],
                label='neg', marker = 'o')

    # anchor 与 pos 之间的距离
    dis = 0
    for j in range(3):
        dis += (sample_anchor[i,j] - sample_pos[i,j])**2
    ax.plot([sample_anchor[i, 0], sample_pos[i, 0]],
            [sample_anchor[i, 1], sample_pos[i, 1]],
            [sample_anchor[i, 2], sample_pos[i, 2]],
            label = '{:.2f}'.format(np.sqrt(dis)))
    ax.legend()
    # anchor 与 neg 之间的距离
    dis = 0
    for j in range(3):
        dis += (sample_anchor[i,j] - sample_neg[i,j]) ** 2
    ax.plot([sample_anchor[i, 0], sample_neg[i, 0]],
            [sample_anchor[i, 1], sample_neg[i, 1]],
            [sample_anchor[i, 2], sample_neg[i, 2]],
            label = '{:.2f}'.format(np.sqrt(dis)))
    ax.legend()
    plt.show()

#%% 预测
def get_img(path):
    img = imread(path)
    img = resize(img, (128, 128)).transpose((2, 0, 1))
    return img

def predict(img):
    anchor = [img*255] * 5
    anchor = np.array((anchor))
    f_path = './support set/'
    pos_path = ['cat1.jpg','dog1.jpg','fish1.jpg','mouse1.jpg','rabbit1.jpg']
    neg_path = ['dog1.jpg','fish1.jpg','mouse1.jpg','rabbit1.jpg','cat1.jpg']
    x_pos, x_neg = [], []
    for i in range(len(pos_path)):
        x_pos.append(get_img(f_path+pos_path[i]))
        x_neg.append(get_img(f_path+neg_path[i]))
    x_pos, x_neg = np.array(x_pos), np.array(x_neg)
    anchor = torch.from_numpy(anchor).type(torch.float32).to(device)
    x_pos = torch.from_numpy(x_pos).type(torch.float32).to(device)
    x_neg = torch.from_numpy(x_neg).type(torch.float32).to(device)
    #开始预测
    model.eval()
    anchor_out, x_pos_out, x_neg_out = model(anchor,x_pos,x_neg)
    # 找到最为接近的点
    dis = []
    for i in range(len(anchor_out)):
        tmp = 0
        for j in range(anchor_out.shape[1]):
            tmp += (anchor_out[i,j] - x_pos_out[i,j])**2
        dis.append(tmp)
    index = np.array(dis).argmin()
    return keyword[index]


path = './querry/mouse.jpg'
predict(get_img(path))