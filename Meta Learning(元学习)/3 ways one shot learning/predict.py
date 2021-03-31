from skimage.io import imread
from skimage.transform import resize
import torch
import numpy as np
from torch import nn
from skimage.io import imshow

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = torch.load('sia_net.pt')
data = torch.load('data.pkl')
data =torch.from_numpy(data)

def predict(path):
    img = imread(path)
    img = resize(img,(200,200)).transpose((2,1,0))
    input = torch.from_numpy(img * 255).type(torch.float32).unsqueeze(dim=0).to(device)
    net.eval()
    score1 = net(input, data[0].unsqueeze(dim=0).to(device))   # ml
    score2 = net(input, data[1].unsqueeze(dim=0).to(device))   # dl
    score3 = net(input, data[2].unsqueeze(dim=0).to(device))   # ma
    score = torch.cat((score1,score2,score3),dim=-1)
    if torch.argmax(score)==0:
        return '机器学习'
    if torch.argmax(score)==1:
        return '深度学习'
    if torch.argmax(score)==2:
        return '元学习'
    #return nn.Softmax(dim=-1)(torch.cat((score1,score2,score3),dim=-1))


path1 = './validate_img/sdxx.jpg'
path2 = './validate_img/yxx.jpg'
print(predict(path1))
print(predict(path2))