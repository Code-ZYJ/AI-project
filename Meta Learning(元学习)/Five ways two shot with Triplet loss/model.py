import torch
from torch import nn

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
