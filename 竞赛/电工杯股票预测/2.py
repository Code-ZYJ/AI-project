import json
from utils import get_rate
import pandas as pd
import numpy as np
#'''     读取37个公司的股价数据
df = []
for i in range(1,38):
    tmp = pd.read_excel('附件1.xlsx',sheet_name='Sheet0 ({})'.format(i))
    df.append(tmp)
#'''

dict_z = {'南玻A' :  307069.21 ,
        '深圳能源' :  475738.99,
        '东旭蓝天' : 148687.39,
        '方大集团': 107387.42,
        '深赛格':  123565.62,
        '宝鹰股份': 134129.69,
        '东南网架': 103440.22,
        '延华智能' : 71215.30,
        '拓日新能' : 123634.21,
        '中利集团' : 87178.71,
        '亚厦股份' : 133999.65,
        '广田集团': 153727.97,
        '瑞和股份': 37829.20,
        '亚玛顿': 16000.00,
        '永高股份': 123538.39,
        '中装建设': 72144.58,
        '南网能源': 378787.88,
        '特锐德':  104071.07,
        '嘉寓股份':71676.00,
        '东方日升':90135.99,
        '秀强股份':61850.24,
        '海达股份':60123.42,
        '旋极信息':172759.06,
        '中来股份':108962.74,
        '华自科技':25617.15,
        '启迪设计':17448.02,
        '汉嘉设计':22573.83,
        '精工钢构':201287.43,
        '苏美达':130674.94,
        '隆基股份':386639.48,
        '林洋能源':174888.93,
        '明阳智能':195092.87,
        '江河集团':115405.00,
        '中衡设计':27680.77,
        '森特股份':53880.00,
        '芯能科技':50000.00,
        '清源股份':27380,
}

dict_lt = {'南玻A' : 306736.81 ,
        '深圳能源' :  475738.99,
        '东旭蓝天' : 106034.07,
        '方大集团': 107157.21,
        '深赛格':  78466.18,
        '宝鹰股份': 133561.95,
        '东南网架': 95851.10,
        '延华智能' : 71125.66,
        '拓日新能' : 121571.98,
        '中利集团' : 69827.66,
        '亚厦股份' : 133099.12,
        '广田集团': 152929.43,
        '瑞和股份': 30983.80,
        '亚玛顿': 15957.36,
        '永高股份': 112973.09,
        '中装建设': 63666.45,
        '南网能源': 75757.58,
        '特锐德': 94472.35,
        '嘉寓股份': 71676.00,
        '东方日升':89944.59,
        '秀强股份':59313.41,
        '海达股份':48815.19,
        '旋极信息':125369.03,
        '中来股份':77977.89,
        '华自科技':24500.19,
        '启迪设计':15838.22,
        '汉嘉设计':20936.17,
        '精工钢构':201287.43,
        '苏美达':130674.94,
        '隆基股份':386630.92,
        '林洋能源':174888.93,
        '明阳智能':147260.83,
        '江河集团':115405.00,
        '中衡设计':27479.82,
        '森特股份':53880.00,
        '芯能科技':30830.00,
        '清源股份':27380,
}

with open('./网上搜得的信息/各公司总股本（最新）.json','w') as f:
        json.dump(dict_z,f,ensure_ascii=False)
with open('./网上搜得的信息/各公司流通股本（最新）.json','w') as f:
        json.dump(dict_lt,f,ensure_ascii=False)



#%%
# 2019-04-01 调整市值计算
with open('./网上搜得的信息/各公司总股本.json') as f:
        dict_z = json.load(f)
with open('./网上搜得的信息/各公司流通股本.json') as f:
        dict_lt = json.load(f)
fluent = {}
for key in dict_z.keys():
        fluent[key] = dict_z[key] * get_rate(dict_lt[key]/dict_z[key])    #每个股份的调整市值
'''求取每个公司的调整市值（2019-04-01）'''
for i,key in enumerate(fluent.keys()):
        fluent[key] = df[i][df[i]['交易时间'] == '2019-04-01']['收盘价'].values[0] * fluent[key]
'''总调整市值（2019-04-01）'''
total_fluent_3_ = sum([i for i in fluent.values()])



# 2021-05-05 调整市值计算
with open('./网上搜得的信息/各公司总股本（最新）.json') as f:
        dict_z = json.load(f)
with open('./网上搜得的信息/各公司流通股本（最新）.json') as f:
        dict_lt = json.load(f)
fluent = {}
for key in dict_z.keys():
        fluent[key] = dict_z[key] * get_rate(dict_lt[key]/dict_z[key])    #每个股份的调整市值
'''求取每个公司的调整市值（2021-05-06）'''
for i,key in enumerate(fluent.keys()):
        fluent[key] = df[i][df[i]['交易时间'] == '2021-05-06']['收盘价'].values[0] * fluent[key]
'''总调整市值（2021-05-06）'''
total_fluent_2_ = sum([i for i in fluent.values()])



#以 2019-04-01 的股本计算 2021-05-06
with open('./网上搜得的信息/各公司总股本.json') as f:
        dict_z = json.load(f)
with open('./网上搜得的信息/各公司流通股本.json') as f:
        dict_lt = json.load(f)
fluent = {}
for key in dict_z.keys():
        fluent[key] = dict_z[key] * get_rate(dict_lt[key]/dict_z[key])    #每个股份的调整市值
'''求取每个公司的调整市值（2021-05-06）'''
for i,key in enumerate(fluent.keys()):
        fluent[key] = df[i][df[i]['交易时间'] == '2021-05-06']['收盘价'].values[0] * fluent[key]
'''总调整市值（2019-05-06）'''
total_fluent_1_ = sum([i for i in fluent.values()])


'''计算新除数 new_div'''
new_div = total_fluent_3_*total_fluent_2_/total_fluent_1_
# 修正后的板块指数计算方法：  total_fluent_2_/new_div*1000


#
'''2021-05-06到2021-05-27的收盘价'''
date = {}
start_index = df[0][df[0]['交易时间'] == '2021-05-06'].index.values[0]
end_index = df[0][df[0]['交易时间'] == '2021-05-27'].index.values[0]
for i in range(start_index,end_index+1):    #先获取日期
        ans = str(df[0]['交易时间'][i])[0:10]
        date[ans] = []

#下面这段程序获取了问题1时间段内的收盘价，缺失的以0进行填充
for dat in date.keys():
        each_day = []
        for i in range(37):
                try:
                        spj = df[i][df[i]['交易时间']==dat]['收盘价'].values[0]
                except:
                        spj = 0
                each_day.append(spj)
        date[dat] = each_day
with open('37家公司在2021-05-06__2021-05-27的收盘价.json','w') as f:  #以字典保存37个公司的收盘价
        json.dump(date,f,ensure_ascii=False)


#%% 以 2019-04-01 的股本数据做的数据

with open('./网上搜得的信息/各公司总股本.json') as f:
        dict_z = json.load(f)
with open('./网上搜得的信息/各公司流通股本.json') as f:
        dict_lt = json.load(f)
fluent = {}
for key in dict_z.keys():
        fluent[key] = dict_z[key] * get_rate(dict_lt[key]/dict_z[key])    #每个股份的调整市值
'''求取每个公司的调整市值（2021-05-06）'''
final_score = []
fluent_value = {}
for dat in date.keys():
        for i,key in enumerate(fluent.keys()):
                fluent_value[key] = df[i][df[i]['交易时间'] == dat]['收盘价'].values[0] * fluent[key]
        total = sum([i for i in fluent_value.values()])
        final_score.append(total/new_div * 1000)
'''板块指数old'''
final_score_old = final_score



#以 2021-05-27 的最新股本数据修正后的数据
with open('./网上搜得的信息/各公司总股本（最新）.json') as f:
        dict_z = json.load(f)
with open('./网上搜得的信息/各公司流通股本（最新）.json') as f:
        dict_lt = json.load(f)
fluent = {}
for key in dict_z.keys():
        fluent[key] = dict_z[key] * get_rate(dict_lt[key]/dict_z[key])    #每个股份的调整市值
'''求取每个公司的调整市值（2021-05-06）'''
final_score = []
fluent_value = {}
for dat in date.keys():
        for i,key in enumerate(fluent.keys()):
                fluent_value[key] = df[i][df[i]['交易时间'] == dat]['收盘价'].values[0] * fluent[key]
        total = sum([i for i in fluent_value.values()])
        final_score.append(total/new_div * 1000)
'''板块指数new'''
final_score_new = final_score

#%% 作图
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 设置支持负号显示

plt.plot(final_score_old,label = '待修正数据')
plt.plot(final_score_new, label = '修正后数据')
plt.title('模型修正前后板块指数对比')
plt.legend()
plt.show()

final_score_old = np.array(final_score_old)
final_score_new = np.array(final_score_new)

print('原模型得到数据的\n均值： {}\t方差： {}\t标准差： {}\t最值差：{}'
      .format(final_score_old.mean(), final_score_old.var(), final_score_old.std(),
              final_score_old.max() - final_score_old.min()))
print('修正模型得到数据的\n均值： {}\t方差： {}\t标准差： {}\t最值差：{}'
      .format(final_score_new.mean(), final_score_new.var(), final_score_new.std(),
              final_score_new.max() - final_score_new.min()))



######
zf_old = [final_score_old[i+1] - final_score_old[i] for i in range(len(final_score_old)-1)]
zf_new = [final_score_new[i+1] - final_score_new[i] for i in range(len(final_score_new)-1)]

plt.bar(range(1,45,3),zf_old,label = '待修正')
plt.bar(range(2,46,3),zf_new,label = '已修正')
plt.title('模型修正前后板块指数涨幅对比')
plt.legend();plt.show()



























#%%   预测问题： 后续数据准备

'''2020-05-27到2021-05-27的收盘价'''
date = {}
start_index = df[0][df[0]['交易时间'] == '2020-05-27'].index.values[0]
end_index = df[0][df[0]['交易时间'] == '2021-05-27'].index.values[0]
for i in range(start_index,end_index+1):    #先获取日期
        ans = str(df[0]['交易时间'][i])[0:10]
        date[ans] = []

#下面这段程序获取了问题1时间段内的收盘价，缺失的以0进行填充
for dat in date.keys():
        each_day = []
        for i in range(37):
                try:
                        spj = df[i][df[i]['交易时间']==dat]['收盘价'].values[0]
                except:
                        spj = 0
                each_day.append(spj)
        date[dat] = each_day
with open('37家公司在2020-05-27__2021-05-27的收盘价.json','w') as f:  #以字典保存37个公司的收盘价
        json.dump(date,f,ensure_ascii=False)



# 该时间段内板块指数变化情况（修正模型下）
with open('./网上搜得的信息/各公司总股本（最新）.json') as f:
        dict_z = json.load(f)
with open('./网上搜得的信息/各公司流通股本（最新）.json') as f:
        dict_lt = json.load(f)
fluent = {}
for key in dict_z.keys():
        fluent[key] = dict_z[key] * get_rate(dict_lt[key]/dict_z[key])    #每个股份的调整市值
'''时间段内的变化'''
final_score = []
for dat,arr in date.items():
        total = 0
        for i, key in enumerate(fluent.keys()):
                total += arr[i]*fluent[key]
        final_score.append(total/new_div * 1000)
'''板块指数new'''
data_train = np.array(final_score)

plt.figure(figsize=(8,3))
plt.plot(data_train)
plt.title('以修正模型评估的近一年时间内的板块指数');plt.show()


# 时序预测
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

step = 50     #20个数据差不多为1个月的量，拿前一个月的数据预测后一个结果
X,Y = [],[]
for i in range(len(data_train)-step):
        X.append(data_train[i:i+step])
        Y.append(data_train[i+step])
X,Y = torch.from_numpy(np.array(X)),torch.from_numpy(np.array(Y))
batch_size = len(X)
train = TensorDataset(X,Y)
train = DataLoader(train,batch_size=batch_size,shuffle=True,drop_last=True)


'''
class Model(nn.Module):
        def __init__(self,hidden_size=10, step = 20 ,):
                super().__init__()
                self.hidden_size = hidden_size
                self.step = step
                self.gru = nn.GRU(input_size=1,hidden_size=hidden_size,batch_first=True)
                self.out = nn.Linear(in_features=hidden_size * step,out_features=1)
        def forward(self,x,hidden=None):
                gru_out,hidden = self.gru(x,None)
                out = self.out(gru_out.reshape(-1,self.hidden_size * self.step))
                return out,hidden

model = Model().to(device)
optim = torch.optim.RMSprop(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()
epochs = 500

torch.autograd.set_detect_anomaly(True)

hidden = torch.zeros(1,32,10).to(device)
LOSS = []
for epo in range(epochs):
        l=0
        for x,y in train:
                x = x.view(batch_size,-1,1).type(torch.FloatTensor).to(device)
                y = y.view(batch_size).type(torch.FloatTensor).to(device)
                y_pred,hidden = model(x,hidden)
                loss = loss_fn(y_pred,y)
                loss.backward(retain_graph=True)
                optim.step()
                optim.zero_grad()
                with torch.no_grad():
                        l += loss.cpu().numpy()/len(X)
        with torch.no_grad():
                LOSS.append(loss.cpu().numpy())
        print('epochs:{}\t MAE_loss:{:.2f}'.format(epo,l))

'''
import torch.nn.functional as F
#  第二个模型
class Model2(nn.Module):
        def __init__(self):
                super(Model2, self).__init__()
                self.gru1 = nn.GRU(1,128,bidirectional=True,num_layers=3)
                self.out = nn.Sequential(nn.Flatten(),
                                         nn.Linear(128*2*step,640),
                                         nn.Tanh(),
                                         nn.Linear(640,128),
                                         nn.Tanh(),
                                         nn.Linear(128,1))
        def forward(self,x):
                gru_out = self.gru1(x)[0]
                out = F.tanh(self.out(gru_out))
                return out

self = Model2().to(device)
self(x)


model = Model2().to(device)
optim = torch.optim.RMSprop(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()
epochs = 500

torch.autograd.set_detect_anomaly(True)
LOSS = []
for epo in range(epochs):
        l=0
        for x,y in train:
                x = x.view(batch_size,-1,1).type(torch.FloatTensor).to(device)
                y = y.view(batch_size).type(torch.FloatTensor).to(device)
                y_pred = model(x)
                loss = loss_fn(y_pred,y)
                loss.backward(retain_graph=True)
                optim.step()
                optim.zero_grad()
                with torch.no_grad():
                        l += loss.cpu().numpy()/len(X)
        with torch.no_grad():
                LOSS.append(loss.cpu().numpy())
        print('epochs:{}\t MAE_loss:{:.2f}'.format(epo,l))




# 损失函数曲线
plt.figure(figsize=(10,3))
plt.plot(LOSS)
plt.show()

# 预测值
pred = model(X.type(torch.FloatTensor).unsqueeze(-1).to(device))
with torch.no_grad():
        pred = pred.cpu().numpy()
        #Y = Y.cpu().numpy()
plt.plot(pred)
plt.plot(Y)
plt.show()


#%% tensorflow 实现
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU
from tensorflow.keras.optimizers import SGD



step = 50     #20个数据差不多为1个月的量，拿前一个月的数据预测后一个结果
X,Y = [],[]
for i in range(len(data_train)-step):
        X.append(data_train[i:i+step])
        Y.append(data_train[i+step])
X,Y = np.array(X),np.array(Y)
batch_size = len(X)

X = X.reshape(batch_size,step,1)
Y = Y.reshape(-1)


model = Sequential()
# LSTM 第一层
model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
# LSTM 第二层
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
# LSTM 第三层
model.add(LSTM(128))
model.add(Dropout(0.2))
# Dense层
model.add(Dense(units=1))


model.summary()

# 编译训练
model.compile(optimizer='rmsprop', loss='mae',learning_rate = 1e-4)
# 模型训练
history = model.fit(X,Y, epochs=4000)
model.save('model.h5')


pred = model(X)
plt.plot(Y)
plt.plot(tf.reshape(pred,[-1]));
plt.show()

#%% 预测
res = []

data = X[-1].reshape(1,-1,1)
for i in range(60):
        out = model(data).numpy()[0,0]
        data = np.concatenate((data[:,1:,:], out.reshape(1,1,1)), axis=1)
        print(data[0,0,0])
        res.append(out)

# 20个交易日的日移动平均线
plt.figure(figsize=(8,5))
plt.plot(res[:20])
plt.title('5月28日后20个交易日的日移动平均线');plt.show()

# 3周的周移动平均线
res7=[]
for i in range(15-5+1):
        res7.append(sum(res[i:i+5])/5)
plt.figure(figsize=(8,5))
plt.plot(res7)
plt.title('5月28日后3周的周移动平均线');plt.show()

# 2个月的月移动平均线
res30=[]
for i in range(44-20+1):
        res30.append(sum(res[i:i+20])/20)
plt.figure(figsize=(8,5))
plt.plot(res30)
plt.title('5月28日后2个月的月移动平均线');plt.show()
