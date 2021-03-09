import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

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
#y_true = [0,0,0,1,1,1,2,2,2]
#y_pred = [0,1,1,2,1,0,0,2,2]
#confusion_matrix(y_true=y_true,y_pred=y_pred)
#accuracy_score(y_true=y_true,y_pred=y_pred)
#print(classification_report(y_true, y_pred, target_names=['fault 1','fault 2','fault 3']))








#%% 以下对数据进行进一步处理与捆绑，使得其能投入到模型中进行训练
'''
1 -> 对所有输入数据采样
2 -> 用 reshape 将 PV 拼接到 dB 后面
3 -> 用TensorData 和 DataLoader 把 x,y,location 拼接
'''
from sampling import sampling     #一次性完成 1 和 2
size=1000
xtrain1 = -10 * sampling(xtrain1,size=size).reshape(xtrain1.shape[0],-1)
xtest1 = -10 * sampling(xtest1,size=size).reshape(xtest1.shape[0],-1)
xtrain2 = -10 * sampling(xtrain2,size=size).reshape(xtrain2.shape[0],-1)
xtest2 = -10 * sampling(xtest2,size=size).reshape(xtest2.shape[0],-1)
xtrain3 = -10 * sampling(xtrain3,size=size).reshape(xtrain3.shape[0],-1)
xtest3 = -10 * sampling(xtest3,size=size).reshape(xtest3.shape[0],-1)
print(xtrain1.shape);print(xtrain2.shape);print(xtrain3.shape)

# 查看数据样子
import matplotlib.pyplot as plt
plt.plot(xtrain1[0]);plt.show()
plt.plot(xtrain2[0]);plt.show()
plt.plot(xtrain3[0]);plt.show()



# 打包方式1
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
# 把 训练集 和 测试集 分别合并
TRAIN = np.concatenate((xtrain1,xtrain2,xtrain3),axis=0)
TEST = np.concatenate((xtest1,xtest2,xtest3),axis=0)
TRAIN_TARGET = np.concatenate((ytrain1,ytrain2,ytrain3),axis=0).reshape(-1)
TEST_TARGET = np.concatenate((ytest1,ytest2,ytest3),axis=0).reshape(-1)
TRAIN_LOC = train_location.reshape(-1)
TEST_LOC = test_location.reshape(-1)
print(TRAIN.shape);print(TEST.shape)
print(TRAIN_TARGET.shape); print(TEST_TARGET.shape)

# 捆绑前把数据转成 Tensor
TRAIN = torch.from_numpy(TRAIN).type(torch.LongTensor)
TEST = torch.from_numpy(TEST).type(torch.LongTensor)
TRAIN_TARGET = torch.from_numpy(TRAIN_TARGET).type(torch.LongTensor)
TEST_TARGET = torch.from_numpy(TEST_TARGET).type(torch.LongTensor)
TRAIN_LOC = torch.from_numpy(TRAIN_LOC).type(torch.LongTensor)
TEST_LOC = torch.from_numpy(TEST_LOC).type(torch.LongTensor)
#plt.plot(TRAIN[0]);plt.show();

train_data = TensorDataset(TRAIN, TRAIN_TARGET, TRAIN_LOC)
test_data = TensorDataset(TEST, TEST_TARGET, TEST_LOC)
train_data = DataLoader(train_data,batch_size=2,shuffle=True)
test_data = DataLoader(test_data,batch_size=1,shuffle=True)





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
        if loc==0:
            attention_score = self.attn(out)        # [batch_size, attn_exact]
            attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))   # [batch_size, 1, self.attn_exact]
            f1_value = self.f1_layer(out).view(-1,self.attn_exact, self.fault_type)   # [batch_size, self.attn_exact, self.fault_type]
            f1_out = torch.matmul(attention_score, f1_value).squeeze(1)
        else:
            f1_out = None
        if loc==1:
            attention_score = self.attn(out)  # [batch_size, attn_exact]
            attention_score = self.dropout(
                self.softmax_d1(attention_score).unsqueeze(1))  # [batch_size, 1, self.attn_exact]
            f2_value = self.f2_layer(out).view(-1, self.attn_exact, self.fault_type)  # [batch_size, self.attn_exact, self.fault_type]
            f2_out = torch.matmul(attention_score, f2_value).squeeze(1)
        else:
            f2_out = None
        if loc == 2:
            attention_score = self.attn(out)  # [batch_size, attn_exact]
            attention_score = self.dropout(
                self.softmax_d1(attention_score).unsqueeze(1))  # [batch_size, 1, self.attn_exact]
            f3_value = self.f3_layer(out).view(-1, self.attn_exact, self.fault_type)  # [batch_size, self.attn_exact, self.fault_type]
            f3_out = torch.matmul(attention_score, f3_value).squeeze(1)
        else:
            f3_out = None
        return f1_out, f2_out, f3_out, self.encoder    #把模型也导出来最为最终的 泛化 模型


class Generalization_model2(nn.Module):
    def __init__(self):
        super().__init__()
        '''下面这些参数不能修改'''
        embedding = 16   # Encoder里面的词向量维度
        self.fault_type = 6  #既然所有数据的故障都分了6类，那我就都写一起算了。(如果类别不一样的话这里要改)

        self.encoder = Encoder()
        self.f1_layer = nn.Linear(embedding, self.fault_type)
        self.f2_layer = nn.Linear(embedding, self.fault_type)
        self.f3_layer = nn.Linear(embedding, self.fault_type)
        self.softmax_d1 = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self,x_train, loc):
        out = self.encoder(x_train)[0][:,-1,:]  #取第0个数开始判断  [batch_size, embedding]
        # 想一下要不把这个 out 给打平，打平的话，上面的参数要改
        if loc == 1:
            f1_out = self.f1_layer(out)
        else:
            f1_out = None
        if loc == 2:
            f2_out = self.f2_layer(out)
        else:
            f2_out = None
        if loc == 3:
            f3_out = self.f3_layer(out)
        else:
            f3_out = None
        return f1_out, f2_out, f3_out,   #把模型也导出来最为最终的 泛化 模型







#%% 训练！！
'''

这种不能批处理的方式虽然节省了存储空间，但运算效率低

#device = 'cpu'
EPOCHS = 1
loss_fn = nn.CrossEntropyLoss()
model = Generalization_model2().to(device)
optim = torch.optim.Adam(model.parameters(),lr=1e-4)
loss_epochs = []

for epoch in range(EPOCHS):
    for i,(x,y,loc) in tqdm(enumerate(train_data)):
        x = x.to(device)
        y = y.to(device)
        loc = loc.to(device)
        model(x, loc)
        y_pred1, y_pred2, y_pred3 = model(x,loc)   #问题点在这里
        if y_pred1 != None:
            loss = 144/54 * loss_fn(y_pred1,y.to(device))
        if y_pred2 != None:
            loss = 144/54 * loss_fn(y_pred2,y.to(device))
        if y_pred3 != None:
            loss = 144/36 * loss_fn(y_pred3,y.to(device))
        loss_epochs.append(loss.item())
        loss.backward()
        optim.step()
        optim.zero_grad()

    print('Epochs:{}-----Loss:{}'.format(epoch,loss))


plt.plot(loss_epochs);plt.show()
'''


#%% 对数据批处理的训练
def add_nan_data(xtrain1,xtrain2,xtrain3,ytrain1,ytrain2,ytrain3):
    max_length = max(len(xtrain1),len(xtrain2),len(xtrain3))
    def add_nan(max_length,xtrain1,ytrain1):   #对数据和标签进行处理
        x1 = np.zeros((max_length, xtrain1.shape[-1]))
        y1 = np.zeros(max_length)
        x1[:len(xtrain1), :] = xtrain1
        y1[:len(ytrain1)] = ytrain1
        if len(xtrain1) == max_length:
            pass
        else:
            x1[len(xtrain1) - max_length:] = None
            y1[len(xtrain1) - max_length:] = None
        return x1,y1

    x1, y1 = add_nan(max_length,xtrain1,ytrain1)
    x2, y2 = add_nan(max_length, xtrain2, ytrain2)
    x3, y3 = add_nan(max_length, xtrain3, ytrain3)
    return  x1,x2,x3,y1,y2,y3

def package_data(xtrain1,xtrain2,xtrain3,ytrain1,ytrain2,ytrain3,batch_size=1):
    # add_nan_data操作(下面度以训练集数据为例)
    xtrain1,xtrain2,xtrain3,ytrain1,ytrain2,ytrain3 = add_nan_data(xtrain1,xtrain2,xtrain3,ytrain1,ytrain2,ytrain3)
    xtrain1 = torch.from_numpy(xtrain1).type(torch.LongTensor)  #转成Tensor
    xtrain2 = torch.from_numpy(xtrain2).type(torch.LongTensor)
    xtrain3 = torch.from_numpy(xtrain3).type(torch.LongTensor)
    ytrain1 = torch.from_numpy(ytrain1).type(torch.LongTensor)
    ytrain2 = torch.from_numpy(ytrain2).type(torch.LongTensor)
    ytrain3 = torch.from_numpy(ytrain3).type(torch.LongTensor)
    # 开始捆绑
    train= TensorDataset(xtrain1,xtrain2,xtrain3,ytrain1,ytrain2,ytrain3)
    train = DataLoader(train,batch_size=batch_size,shuffle=True)
    return train

train = package_data(xtrain1,xtrain2,xtrain3,ytrain1,ytrain2,ytrain3,batch_size=1)
test = package_data(xtest1,xtest2,xtest3,ytest1,ytest2,ytest3)

class batch_process_pretrained_model(nn.Module):
    def __init__(self):
        super().__init__()
        '''下面这些参数不能修改'''
        embedding = 16  # Encoder里面的词向量维度
        hidden_layer = 8
        rate = 0.1
        self.fault_type = 6  # 既然所有数据的故障都分了6类，那我就都写一起算了。(如果类别不一样的话这里要改)
        self.encoder = Encoder()
        self.f1_layer = nn.Sequential(nn.Linear(embedding,self.fault_type))
        self.f2_layer = nn.Sequential(nn.Linear(embedding,self.fault_type))
        self.f3_layer = nn.Sequential(nn.Linear(embedding,self.fault_type))
        #self.softmax_d1 = nn.LogSoftmax(dim=1)

    def forward(self, xtrain1,xtrain2,xtrain3):
        if xtrain1.size(0)>0:
            out1 = self.encoder(xtrain1)[0][:, -1, :]  # 取第0个数开始判断  [batch_size, embedding]
            f1_out = self.f1_layer(out1)
            #f1_out = self.softmax_d1(self.dropout(self.f1_layer(out1)))
        else:
            f1_out = None
        if xtrain2.size(0)>0:
            out2 = self.encoder(xtrain2)[0][:, -1, :]
            f2_out = self.f2_layer(out2)
            #f2_out = self.softmax_d1(self.dropout(self.f2_layer(out2)))
        else:
            f2_out = None
        if xtrain3.size(0)>0:
            out3 = self.encoder(xtrain3)[0][:, -1, :]
            f3_out = self.f2_layer(out3)
            #f3_out = self.softmax_d1(self.dropout(self.f3_layer(out3)))
        else:
            f3_out=None
        return f1_out,f2_out,f3_out


class batch_process_pretrained_model1(nn.Module):
    def __init__(self):
        super().__init__()
        '''下面这些参数不能修改'''
        embedding = 16
        self.attn_exact = 32
        self.fault_type = 6
        self.encoder = Encoder()
        self.attn = nn.Linear(embedding,self.attn_exact)
        self.f1_layer = nn.Linear(embedding,self.attn_exact * self.fault_type)
        self.f2_layer = nn.Linear(embedding,self.attn_exact * self.fault_type)
        self.f3_layer = nn.Linear(embedding,self.attn_exact * self.fault_type)
        self.softmax_d1 = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0)
        self.flatten = nn.Flatten()

    def forward(self, xtrain1, xtrain2, xtrain3):
        if xtrain1.size(0) > 0:
            out = self.encoder(xtrain1)[0]
            attention_score = self.attn(out)
            attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))
            f1_value = self.f1_layer(out).view(-1,self.attn_exact,self.fault_type)
            f1_out = torch.matmul(attention_score,f1_value).squeeze(1)
        else:
            f1_out = None
        if xtrain2.size(0) > 0:
            out = self.encoder(xtrain2)[0][:, 0, :]
            attention_score = self.attn(out)
            attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))
            f2_value = self.f2_layer(out).view(-1, self.attn_exact, self.fault_type)
            f2_out = torch.matmul(attention_score, f2_value).squeeze(1)
        else:
            f2_out = None
        if xtrain3.size(0) > 0:
            out = self.encoder(xtrain3)[0][:,0,:]
            attention_score = self.attn(out)
            attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))
            f3_value = self.f3_layer(out).view(-1,self.attn_exact,self.fault_type)
            f3_out = torch.matmul(attention_score,f3_value).squeeze(1)
        else:
            f3_out = None
        return f1_out, f2_out, f3_out




class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first
    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y

class batch_process_pretrained_model2(nn.Module):
    def __init__(self):
        super().__init__()
        '''下面这些参数不能修改'''
        embedding = 16
        self.attn_exact = 32
        self.fault_type = 6
        self.encoder = Encoder()
        self.f1_layer = nn.Linear(size*2, self.fault_type)
        self.f2_layer = nn.Linear(size*2, self.fault_type)
        self.f3_layer = nn.Linear(size*2, self.fault_type)
        self.softmax_d1 = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)
        self.timedis =  TimeDistributed(nn.Linear(embedding,1),batch_first=True)
        self.ln= nn.LayerNorm(2000)
    def forward(self, xtrain1, xtrain2, xtrain3):
        if xtrain1.size(0) > 0:
            encoder_out = self.encoder(xtrain1)[0]
            out = self.dropout(self.timedis(encoder_out)).squeeze(dim=-1)
            f1_out = self.f1_layer(out)
        else:
            f1_out = None
        if xtrain2.size(0) > 0:
            encoder_out = self.encoder(xtrain2)[0]
            out = self.dropout(self.timedis(encoder_out)).squeeze(dim=-1)
            f2_out = self.f2_layer(out)
        else:
            f2_out = None
        if xtrain3.size(0) > 0:
            encoder_out = self.encoder(xtrain3)[0]
            out = self.dropout(self.timedis(encoder_out)).squeeze(dim=-1)
            f3_out = self.f3_layer(out)
        else:
            f3_out = None
        return f1_out, f2_out, f3_out





class batch_process_pretrained_model3(nn.Module):
    def __init__(self):
        super().__init__()
        '''下面这些参数不能修改'''
        embedding = 16
        self.attn_exact = 32
        self.fault_type = 6
        self.encoder = Encoder()
        self.f1_layer = nn.Linear(2000*16, self.fault_type)
        self.f2_layer = nn.Linear(2000*16, self.fault_type)
        self.f3_layer = nn.Linear(2000*16, self.fault_type)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
    def forward(self, xtrain1, xtrain2, xtrain3):
        if xtrain1.size(0) > 0:
            encoder_out = self.encoder(xtrain1)[0]
            out = self.flatten(encoder_out)
            f1_out = self.dropout(self.f1_layer(out))
        else:
            f1_out = None
        if xtrain2.size(0) > 0:
            encoder_out = self.encoder(xtrain2)[0]
            out = self.flatten(encoder_out)
            f2_out = self.dropout(self.f2_layer(out))
        else:
            f2_out = None
        if xtrain3.size(0) > 0:
            encoder_out = self.encoder(xtrain3)[0]
            out = self.flatten(encoder_out)
            f3_out = self.dropout(self.f3_layer(out))
        else:
            f3_out = None
        return f1_out, f2_out, f3_out



def mask_true_yvalue(ytrain):
    mask = ytrain >= 0
    ytrain = ytrain[mask]
    return ytrain
def mask_true_xvalue(xtrain = xtrain3):
    mask = xtrain[:,0]>=0
    xtrain = xtrain[mask]
    return xtrain





model = batch_process_pretrained_model2().to(device)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.RMSprop(model.parameters(),lr = 1e-4
                            ,weight_decay=15
                            )  #weight_decay=0.01 L2正则化
EPOCHS = 600
#学习率调整
from torch.optim.lr_scheduler import ReduceLROnPlateau     #监听自适应
scheduler = ReduceLROnPlateau(optim,mode='min',factor=0.9,patience=20,threshold=1e-8,)
LOSS=[]
# 一个epoch的测试
for epoch in range(EPOCHS):
    pred1, pred2, pred3 = [], [], []
    target1, target2, target3 = [], [], []
    average_f1,f1_target1,f1_target2,f1_target3=[2],[],[],[]
    for xtrain1,xtrain2,xtrain3,ytrain1,ytrain2,ytrain3 in tqdm(train):
        ytrain1 = mask_true_yvalue(ytrain1).to(device)
        ytrain2 = mask_true_yvalue(ytrain2).to(device)
        ytrain3 = mask_true_yvalue(ytrain3).to(device)
        xtrain1 = mask_true_xvalue(xtrain1).to(device)
        xtrain2 = mask_true_xvalue(xtrain2).to(device)
        xtrain3 = mask_true_xvalue(xtrain3).to(device)
        y1, y2, y3 = model(xtrain1, xtrain2, xtrain3)
        if y1 is not None:
            loss = (144/54)**2 * loss_fn(y1, ytrain1)
            pred1.append(torch.argmax(y1,dim=1).item())   #每个任务的诊断位置
            target1.append(ytrain1.item())
        if y2 is not None:
            loss+= (144/54)**2 * loss_fn(y2, ytrain2)
            pred2.append(torch.argmax(y2).item())
            target2.append(ytrain2.item())
        if y3 is not None:
            loss+= (144/36)**2 * loss_fn(y3, ytrain3)
            pred3.append(torch.argmax(y3).item())
            target3.append(ytrain3.item())
        loss.backward()
        optim.step()
        optim.zero_grad()
        # 每个任务实际的位置
    f1_target1.append(f1_score(target1, pred1, average='macro'))
    f1_target2.append(f1_score(target2, pred2, average='macro'))
    f1_target3.append(f1_score(target3, pred3, average='macro'))
    average_f1.append(f1_target1[-1] + f1_target2[-1] + f1_target3[-1])
    LOSS.append(loss.item())
    print('epoch: {} --- loss: {:.4f} --- average_f1: {:.4f}'.format(epoch,LOSS[-1],average_f1[-1]))
    print('f1_target1: {:.4f} --- f1_target2: {:.4f} --- f1_target3: {:.4f}'.format(f1_target1[-1],f1_target2[-1],f1_target3[-1]))
    print('task1 ACC: {:.4f} --- task2 ACC: {:.4f} --- task3 ACC: {:.4f}'.format(accuracy_score(target1,pred1),
                                                                                 accuracy_score(target2,pred2),
                                                                                 accuracy_score(target3,pred3)))
    #scheduler.step()   #自动梯度下降
    #验证部分，用于保存最佳模型
    target1_test,target2_test,target3_test = [],[],[]
    f1_on_test=[2]
    pred1, pred2, pred3 = [], [], []
    for xtest1,xtest2,xtest3,ytest1,ytest2,ytest3 in test:
        ytest1 = mask_true_yvalue(ytest1).to(device)
        ytest2 = mask_true_yvalue(ytest2).to(device)
        ytest3 = mask_true_yvalue(ytest3).to(device)
        xtest1 = mask_true_xvalue(xtest1).to(device)
        xtest2 = mask_true_xvalue(xtest2).to(device)
        xtest3 = mask_true_xvalue(xtest3).to(device)
        y1, y2, y3 = model(xtest1, xtest2, xtest3)
        if y1 is not None:
            pred1.append(torch.argmax(y1,dim=1).item())   #每个任务的诊断位置
            target1_test.append(ytest1.item())
        if y2 is not None:
            pred2.append(torch.argmax(y2).item())
            target2_test.append(ytest2.item())
        if y3 is not None:
            pred3.append(torch.argmax(y3).item())
            target3_test.append(ytest3.item())
        f1_on_test.append( f1_score(target1_test, pred1, average='macro')+f1_score(target2_test, pred2, average='macro')+f1_score(target3_test, pred3, average='macro'))
    if f1_on_test[-1]+average_f1[-1] > 4.9:
        torch.save(model,'./Model/(L2=60)Epo{} - f1_train{:.2f} - f1_test{:.2f}.pt'.format(epoch,  average_f1[-1], f1_on_test[-1]))
    print('f1_on_train: {:.4f} ------------ f1_on_test: {:.4f}'.format( average_f1[-1],f1_on_test[-1]))
    #学习率调整
    #if f1_on_test[-1] + average_f1[-1] < f1_on_test[-2] + average_f1[-2]:
    scheduler.step(f1_on_test[-1])


plt.figure(dpi=200)
plt.plot(LOSS,'k-',label='loss')
plt.plot(average_f1, 'y-*',label = 'average_f1')
plt.plot(f1_target1, 'r--', label = 'f1_target1')
plt.plot(f1_target2, 'g--', label = 'f1_target2')
plt.plot(f1_target3, 'b--', label = 'f1_target3')
plt.legend()
plt.show()








#%%
'''
pred1, pred2, pred3 = [], [], []
target1, target2, target3 = [], [], []
average_f1,f1_target1,f1_target2,f1_target3=[],[],[],[]
for xtrain1,xtrain2,xtrain3,ytrain1,ytrain2,ytrain3 in tqdm(test):
    ytrain1 = mask_true_yvalue(ytrain1).to(device)
    ytrain2 = mask_true_yvalue(ytrain2).to(device)
    ytrain3 = mask_true_yvalue(ytrain3).to(device)
    xtrain1 = mask_true_xvalue(xtrain1).to(device)
    xtrain2 = mask_true_xvalue(xtrain2).to(device)
    xtrain3 = mask_true_xvalue(xtrain3).to(device)
    y1, y2, y3 = model(xtrain1, xtrain2, xtrain3)
    if y1 is not None:
        loss = 144/54 * loss_fn(y1, ytrain1)
        pred1.append(torch.argmax(y1,dim=1).item())   #每个任务的诊断位置
        target1.append(ytrain1.item())
    if y2 is not None:
        loss+= 144/54 * loss_fn(y2, ytrain2)
        pred2.append(torch.argmax(y2).item())
        target2.append(ytrain2.item())
    if y3 is not None:
        loss+= 144/36 * loss_fn(y3, ytrain3)
        pred3.append(torch.argmax(y3).item())
        target3.append(ytrain3.item())


# 每个任务实际的位置
f1_target1.append(f1_score(target1, pred1, average='macro'))
f1_target2.append(f1_score(target2, pred2, average='macro'))
f1_target3.append(f1_score(target3, pred3, average='macro'))
average_f1.append(f1_target1[-1] + f1_target2[-1] + f1_target3[-1])
LOSS.append(loss.item())
print('epoch: {} --- loss: {:.4f} --- average_f1: {:.4f}'.format(epoch,LOSS[-1],average_f1[-1]))
print('f1_target1: {:.4f} --- f1_target2: {:.4f} --- f1_target3: {:.4f}'.format(f1_target1[-1],f1_target2[-1],f1_target3[-1]))
print('task1 ACC: {:.4f} --- task2 ACC: {:.4f} --- task3 ACC: {:.4f}'.format(accuracy_score(target1,pred1),
                                                                             accuracy_score(target2,pred2),
                                                                             accuracy_score(target3,pred3)))
'''