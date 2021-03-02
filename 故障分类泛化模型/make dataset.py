import numpy as np

def process_pv(x):
    for i in range(x.shape[0]):   #遍历样本
        for j in range(x.shape[2]):   #遍历PV
            while x[i,1,j] < 360:
                x[i,1,j] += 360
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



#%% evaluate

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
y_true = [0,0,0,1,1,1,2,2,2]
y_pred = [0,1,1,2,1,0,0,2,2]
confusion_matrix(y_true=y_true,y_pred=y_pred)
accuracy_score(y_true=y_true,y_pred=y_pred)
print(classification_report(y_true, y_pred, target_names=['fault 1','fault 2','fault 3']))

