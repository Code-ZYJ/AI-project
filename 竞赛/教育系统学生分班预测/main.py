#%%
import pandas as pd
import numpy as np
import torch
df = pd.read_csv('students_academic_performance.csv')

def summary():
    for i in df.columns:
        print('column: ({}) ,unique: {}\n'.format(i,df[i].unique()))
summary()

#%% 预处理
def preprocess(columns,value,target):
    df[columns][df[columns]==value]=target

# gender
preprocess('gender','M',0)
preprocess('gender','F',1)

# NationLITy
for i in range(len(df['NationalITy'].unique())):
    preprocess('NationalITy',df['NationalITy'].unique()[i],i)

# PlaceofBirth
for i in range(len(df['PlaceofBirth'].unique())):
    preprocess('PlaceofBirth',df['PlaceofBirth'].unique()[i],i)

# StageID
for i in range(len(df['StageID'].unique())):
    preprocess('StageID',df['StageID'].unique()[i],i)

# GradeID
for i in range(len(df['GradeID'].unique())):
    preprocess('GradeID',df['GradeID'].unique()[i],i)

# SectionID
for i in range(len(df['SectionID'].unique())):
    preprocess('SectionID',df['SectionID'].unique()[i],i)

# Topic
for i in range(len(df['Topic'].unique())):
    preprocess('Topic',df['Topic'].unique()[i],i)

# Semester
for i in range(len(df['Semester'].unique())):
    preprocess('Semester',df['Semester'].unique()[i],i)

# Relation
for i in range(len(df['Relation'].unique())):
    preprocess('Relation',df['Relation'].unique()[i],i)

# ParentAnsweringSurvey
for i in range(len(df['ParentAnsweringSurvey'].unique())):
    preprocess('ParentAnsweringSurvey',df['ParentAnsweringSurvey'].unique()[i],i)

# ParentschoolSatisfaction
for i in range(len(df['ParentschoolSatisfaction'].unique())):
    preprocess('ParentschoolSatisfaction',df['ParentschoolSatisfaction'].unique()[i],i)

# StudentAbsenceDays
for i in range(len(df['StudentAbsenceDays'].unique())):
    preprocess('StudentAbsenceDays',df['StudentAbsenceDays'].unique()[i],i)

# Class
for i in range(len(df['Class'].unique())):
    preprocess('Class',df['Class'].unique()[i],i)

summary()

#%% 构造数据集
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
x = np.array(df.iloc[:,:-1]).astype(np.float32)
y = np.array(df.iloc[:,-1]).astype(np.float32)
# import tensorflow as tf
# y = tf.one_hot(y,3)  利用tensorflow去独热编码感觉方便很多
#y = OneHotEncoder().fit(y).transform(y).toarray()      #利用sklearn独热编码

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

#%% 利用pytorch去识别
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 捆绑数据
x_train = torch.from_numpy(x_train).to(device)
x_test = torch.from_numpy(x_test).to(device)
y_train = torch.from_numpy(y_train).type(torch.float32).to(device)
y_test = torch.from_numpy(y_test).type(torch.float32).to(device)
train = TensorDataset(x_train,y_train)
train = DataLoader(train,shuffle=True,batch_size=128)

#%% 定义模型
# 在pytorch中若模型使用CrossEntropyLoss这个loss函数，则不应该在最后一层再使用softmax进行激活。
model = torch.nn.Sequential(
    torch.nn.Linear(in_features=x_train.size(1),out_features=16),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(in_features=16,out_features=32),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(32),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(in_features=32,out_features=16),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(in_features=16,out_features=8),
    torch.nn.ReLU(),
    torch.nn.Linear(8,3),
).to(device)
lossfn = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(),lr=0.001)

ACC,VAL_ACC=[],[]
LOSS,VAL_LOSS=[],[]
for i in range(5000):
    for x,y in train:
        y_pred = model(x)
        loss = lossfn(y_pred,y.long())
        loss.backward()
        optim.step()
        optim.zero_grad()
    acc = torch.sum((torch.argmax(model(x_train),1)==y_train))/len(x_train)
    val_acc = torch.sum((torch.argmax(model(x_test),1)==y_test))/len(x_test)
    ACC.append(acc)
    VAL_ACC.append(val_acc)
    LOSS.append(loss)
    VAL_LOSS.append(lossfn(model(x_test),y_test.long()))
    print('epoch: {}————————loss：{:.4f}/val_loss: {:.4f}————————acc：{:.4f}/val_acc: {:.4f}'.format(i+1,loss,val_acc,acc,val_acc))

# 可视化
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(LOSS,'-',label='loss')
plt.plot(VAL_LOSS,'--',label='val_loss')
plt.legend()
plt.show()
plt.figure(2)
plt.plot(ACC,'-',label='acc')
plt.plot(VAL_ACC,'--',label='acc')
plt.ylim(0,1)
plt.legend()
plt.show()