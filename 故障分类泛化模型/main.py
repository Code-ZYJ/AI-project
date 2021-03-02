import numpy as np

def process_pv(x):
    for i in range(x.shape[0]):   #遍历样本
        for j in range(x.shape[2]):   #遍历PV
            while x[i,1,j] < -360:
                x[i,1,j] += 360
            while x[i,1,j] > 0:
                x[i,1,j] -= 360
    return x
#%% 故障或定位数据整合
xtrain1 = np.load('./故障定位/频响数据训练集.npy')
xtrain1 = xtrain1[:,1:,:]    #去掉了频率这一维
ytrain1 = []
for i in range(1,7):
    ytrain1.append(np.ones(9)*i)
ytrain1 = np.array(ytrain1).reshape(-1)-1

xtest1 = np.load('./故障定位/频响数据测试集.npy')
xtest1 = xtest1[:,1:,:]    #去掉了频率这一维
ytest1 = []
for i in range(1,7):
    ytest1.append(np.ones(9)*i)
ytest1 = np.array(ytest1).reshape(-1)-1

xtrain1 = process_pv(xtrain1)  #处理一下相位角
xtest1 = process_pv(xtest1)

print(xtrain1.shape)
print(ytrain1.shape)
print(xtest1.shape)
print(ytest1.shape)

#%% 故障类型及程度诊断数据整合
F1 = np.load('./故障类型及程度诊断/训练集/F1.npy').transpose((0,2,1))[:,1:,:]    #去掉了频率这一维
F2 = np.load('./故障类型及程度诊断/训练集/F2.npy').transpose((0,2,1))[:,1:,:]    #去掉了频率这一维
F3 = np.load('./故障类型及程度诊断/训练集/F3.npy').transpose((0,2,1))[:,1:,:]    #去掉了频率这一维
F4 = np.load('./故障类型及程度诊断/训练集/F4.npy').transpose((0,2,1))[:,1:,:]    #去掉了频率这一维
F5 = np.load('./故障类型及程度诊断/训练集/F5.npy').transpose((0,2,1))[:,1:,:]    #去掉了频率这一维
F6 = np.load('./故障类型及程度诊断/训练集/F6.npy').transpose((0,2,1))[:,1:,:]    #去掉了频率这一维
test = np.load('./故障类型及程度诊断/验证集/test.npy').transpose((0,2,1))[:,1:,:]   #去掉了频率这一维

xtrain2 = np.concatenate((F1,F2,F3,F4,F5,F6),axis=0)
xtest2 = test

ytrain2 = []
for i in range(1,7):
    ytrain2.append(np.ones(9)*i)
ytrain2 = np.array(ytrain2).reshape(-1)-1

ytest2 = []
for i in range(1,7):
    ytest2.append(np.ones(5)*i)
ytest2 = np.array(ytest2).reshape(-1)-1

xtrain2 = process_pv(xtrain2)  #处理一下相位角
xtest2 = process_pv(xtest2)

print(xtrain2.shape)
print(ytrain2.shape)
print(xtest2.shape)
print(ytest2.shape)

#%% 数据集3
xtrain3 = np.load('./故障分类数据集3/xtrain3.npy')
ytrain3 = np.load('./故障分类数据集3/ytrain3.npy')
xtest3 = np.load('./故障分类数据集3/xtest3.npy')
ytest3 = np.load('./故障分类数据集3/ytest3.npy')

xtrain3 = xtrain3[:,1:,:]
xtest3 = xtest3[:,1:,:]

xtrain3 = process_pv(xtrain3)
xtest3 = process_pv(xtest3)
print(xtrain3.shape)
print(ytrain3.shape)
print(xtest3.shape)
print(ytest3.shape)


# location

train_location = np.concatenate((np.ones(xtrain1.shape[0])*1, np.ones(xtrain2.shape[0])*2, np.ones(xtrain3.shape[0])*3),axis=0)
test_location = np.concatenate((np.ones(xtest1.shape[0])*1, np.ones(xtest2.shape[0])*2, np.ones(xtest3.shape[0])*3),axis=0)



#%% evaluate test

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
y_true = [0,0,0,1,1,1,2,2,2]
y_pred = [0,1,1,2,1,0,0,2,2]
confusion_matrix(y_true=y_true,y_pred=y_pred)
accuracy_score(y_true=y_true,y_pred=y_pred)
print(classification_report(y_true, y_pred, target_names=['fault 1','fault 2','fault 3']))








#%% 以下对数据进行进一步处理与捆绑，使得其能投入到模型中进行训练
'''
1 -> 对所有输入数据采样
2 -> 用 reshape 将 PV 拼接到 dB 后面
3 -> 用TensorData 和 DataLoader 把 x,y,location 拼接
'''
from sampling import sampling     #一次性完成 1 和 2
xtrain1 = -10 * sampling(xtrain1).reshape(xtrain1.shape[0],-1)
xtest1 = -10 * sampling(xtest1).reshape(xtest1.shape[0],-1)
xtrain2 = -10 * sampling(xtrain2).reshape(xtrain2.shape[0],-1)
xtest2 = -10 * sampling(xtest2).reshape(xtest2.shape[0],-1)
xtrain3 = -10 * sampling(xtrain3).reshape(xtrain3.shape[0],-1)
xtest3 = -10 * sampling(xtest3).reshape(xtest3.shape[0],-1)
print(xtrain1.shape);print(xtrain2.shape);print(xtrain3.shape)

# 查看数据样子
import matplotlib.pyplot as plt
plt.plot(xtrain1[0]);plt.show()
plt.plot(xtrain2[0]);plt.show()
plt.plot(xtrain3[0]);plt.show()



# 打包
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
# 把 训练集 和 测试集 分别合并
TRAIN = np.concatenate((xtrain1,xtrain2,xtrain3),axis=0)
TEST = np.concatenate((xtest1,xtest2,xtest3),axis=0)
TRAIN_TARGET = np.concatenate((ytrain1,ytrain2,ytrain3),axis=0).reshape(-1,1)
TEST_TARGET = np.concatenate((ytest1,ytest2,ytest3),axis=0).reshape(-1,1)
TRAIN_LOC = train_location.reshape(-1,1)
TEST_LOC = test_location.reshape(-1,1)
print(TRAIN.shape);print(TEST.shape)
print(TRAIN_TARGET.shape); print(TEST_TARGET.shape)

# 捆绑前把数据转成 Tensor
TRAIN = torch.from_numpy(TRAIN).type(torch.LongTensor)
TEST = torch.from_numpy(TEST).type(torch.LongTensor)
TRAIN_TARGET = torch.from_numpy(TRAIN_TARGET).type(torch.int8)
TEST_TARGET = torch.from_numpy(TEST_TARGET).type(torch.int8)
TRAIN_LOC = torch.from_numpy(TRAIN_LOC).type(torch.int8)
TEST_LOC = torch.from_numpy(TEST_LOC).type(torch.int8)
plt.plot(TRAIN[0]);plt.show();

train_data = TensorDataset(TRAIN, TRAIN_TARGET, TRAIN_LOC)
test_data = TensorDataset(TEST, TEST_TARGET, TEST_LOC)
train_data = DataLoader(train_data,batch_size=2,shuffle=True)
test_data = DataLoader(test_data,batch_size=2,shuffle=True)





#%% 搭建模型
'''
万事俱备，只欠东风！
搭建模型
'''
from build_model import Encoder

class Generalization_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_exact = 4  #这个是注意力的特征提取，这个参数可以改
        '''下面这些参数不能修改'''
        embedding = 16   # Encoder里面的词向量维度
        self.fault_type = 6  #既然所有数据的故障都分了6类，那我就都写一起算了。(如果类别不一样的话这里要改)

        self.encoder = Encoder()
        self.attn = nn.Linear(embedding,self.attn_exact)
        self.f1_layer = nn.Linear(embedding, self.attn_exact * self.fault_type)
        self.f2_layer = nn.Linear(embedding, self.attn_exact * self.fault_type)
        self.f3_layer = nn.Linear(embedding, self.attn_exact * self.fault_type)
        self.softmax_d1 = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self,x_train, loc):
        out = self.encoder(x_train)[0][:,0,:]  #取第0个数开始判断  [batch_size, embedding]
        # 想一下要不把这个 out 给打平，打平的话，上面的参数要改
        if loc==1:
            attention_score = self.attn(out)        # [batch_size, attn_exact]
            attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))   # [batch_size, 1, self.attn_exact]
            f1_value = self.f1_layer(out).view(-1,self.attn_exact, self.fault_type)   # [batch_size, self.attn_exact, self.fault_type]
            f1_out = torch.matmul(attention_score, f1_value).squeeze(1)
        else:
            f1_out = None
        if loc==2:
            attention_score = self.attn(out)  # [batch_size, attn_exact]
            attention_score = self.dropout(
                self.softmax_d1(attention_score).unsqueeze(1))  # [batch_size, 1, self.attn_exact]
            f2_value = self.f2_layer(out).view(-1, self.attn_exact, self.fault_type)  # [batch_size, self.attn_exact, self.fault_type]
            f2_out = torch.matmul(attention_score, f2_value).squeeze(1)
        else:
            f2_out = None
        if loc == 3:
            attention_score = self.attn(out)  # [batch_size, attn_exact]
            attention_score = self.dropout(
                self.softmax_d1(attention_score).unsqueeze(1))  # [batch_size, 1, self.attn_exact]
            f3_value = self.f3_layer(out).view(-1, self.attn_exact, self.fault_type)  # [batch_size, self.attn_exact, self.fault_type]
            f3_out = torch.matmul(attention_score, f3_value).squeeze(1)
        else:
            f3_out = None
        return f1_out, f2_out, f3_out, self.encoder    #把模型也导出来最为最终的 泛化 模型

#%% 训练！！
for x,y in









model = Generalization_model().cuda()
a,b,c = model(x,loc = 1)



model = model.cuda()
for x,y,z in train_data:
    pass
x = x.cuda()
a,b = model(x)