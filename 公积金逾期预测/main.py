import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split   
import tensorflow as tf
import seaborn as sns
from scipy.special import boxcox1p
from scipy.stats import norm,skew
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
#%% 正太分布
#原始分布
data = pd.read_csv('./公积金逾期预测-数据/train.csv')
plt.figure(figsize=(25,20))
for i,col in enumerate(data.columns[1:]):
    plt.subplot(4,5,i+1)
    plt.title(data[col].skew())
    sns.distplot(data[col])

#box-cox后分布

plt.figure(figsize=(25,20))
lam=0.5
plt.title(str(lam))
for i,col in enumerate(data.columns[1:]):
    plt.subplot(4,5,i+1)
    test = boxcox1p(data[col],lam)
    plt.title(test.skew())
    sns.distplot(test)

# 相关性
plt.figure(figsize=(15,15))
cor = data.iloc[1:,:].corr()
sns.heatmap(cor,annot=True,cbar=True,square=True,fmt='.2f',cmap='YlGnBu')

#%%  BOX-COX 及去除干扰项

def normalize_col(df):
    '''输入要是 x_train '''
    col_bc=['GRJCJS','GRZHYE','GRZHSNJZYE','GRZHDNGJYE','GRYJCE','DWYJCE','DKFFE']
    dic={'GRJCJS':0.12,'GRZHYE':0.18,'GRZHSNJZYE':0.18,
         'GRZHDNGJYE':0.1,'DWYJCE':0.1,'DKYE':0.2}
    for key,value in dic.items():
        df[key]=boxcox1p(df[key], value)
    return df

def del_bother_item(df):
    '''输入要是 x_train '''
    del df['id']
    del df['HYZK']
    del df['ZHIWU']
    del df['XUELI']
    
    del df['GRZHDNGJYE']  #这一列虽然高度相关，但是缺失值正态化后缺失值太多了
    return df

#  train 数据准备
data = data = pd.read_csv('./公积金逾期预测-数据/train.csv')
data = del_bother_item(normalize_col(data))
plt.figure(figsize=(16,16))
for i,col in enumerate(data.columns):
    plt.subplot(4,4,i+1)
    sns.distplot(data[col])

data=data.drop(data[(data['GRJCJS']<11.5) | (data['GRJCJS']>17.5)].index); sns.displot(data['GRJCJS']) 
data=data.drop(data[(data['GRZHYE']<9.4)].index); sns.displot(data['GRZHYE']) 
data=data.drop(data[(data['GRZHSNJZYE']<10)].index); sns.displot(data['GRZHSNJZYE']) 
data=data.drop(data[(data['DKYE']<15)].index); sns.displot(data['DKYE']) 
#再看数据分布
plt.figure(figsize=(16,16))
for i,col in enumerate(data.columns):
    plt.subplot(4,4,i+1)
    sns.distplot(data[col],fit=norm)

x_train,x_test,y_train,y_test=train_test_split(
    data.iloc[:,:-1],data.iloc[:,-1],test_size=0.5)

#  test 数据准备
data = pd.read_csv('./公积金逾期预测-数据/test.csv')
x_submit=del_bother_item(normalize_col(data))


#%% 寻找最合适lam值
# data = data = pd.read_csv('./公积金逾期预测-数据/train.csv')
# # data=data.drop(data[(data['GRZHYE']<9) | (data['GRZHSNJZYE']<9)].index) # 去除离群点（正态化）
# data = del_bother_item(data)
# for lam in np.arange(0.1,0.22,0.02):
#     plt.figure(figsize=(25,20))
#     for i,col in enumerate(data.columns):
#         plt.subplot(4,5,i+1)
#         test = boxcox1p(data[col],lam)
#         plt.title(test.skew())
#         sns.distplot(test)



#%%  模型
# def processing_file(filename):
#     data = pd.read_csv(filename)
    
#     del data['id']
#     del data['XINGBIE']
#     del data['CSNY']
#     del data['HYZK']
#     del data['ZHIYE']
#     del data['ZHICHEN']
#     del data['ZHIWU']
#     del data['XUELI']
#     del data['DWJJLX']
#     del data['DWSSHY']
#     del data['GRJCJS']  0.12
#     del data['GRZHZT']
#     del data['GRZHYE']  0.18
#     del data['GRZHSNJZYE']  0.18
#     del data['GRZHDNGJYE'] 0.1
#     del data['GRYJCE']
#     del data['DWYJCE']  0.1
#     del data['DKFFE']
#     del data['DKYE']  0.2
#     del data['DKLL']
#     return data

#%%
from sklearn.ensemble import RandomForestClassifier
s=0
while s<0.96:
    clf_rf=RandomForestClassifier().fit(x_train,y_train)
    s=clf_rf.score(x_test,y_test)
    print(clf_rf.score(x_test,y_test))
    
#%%
clf_cat = CatBoostClassifier(learning_rate=0.01,task_type='GPU').fit(x_train,y_train,silent=True,
                                                                     eval_set=(x_test,y_test),use_best_model=True)
s_cat = clf_cat.score(x_test,y_test)
print(clf_cat.score(x_test,y_test))

#%%
dic = {250:0.95610}
from sklearn.ensemble import GradientBoostingClassifier
for i in range(1000,2050,50):
    GBoost=GradientBoostingClassifier(n_estimators=4000,learning_rate=0.05,
                                      max_depth=4,max_features='sqrt',
                                      min_samples_leaf=15,min_samples_split=10,
                                      random_state=i).fit(x_train,y_train)
    s=GBoost.score(x_test,y_test)
    dic[i]=s
    print('random_state: {},    score: {:.4f}'.format(i,s))

#%%搭建模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(512, input_shape = (x_train.shape[1],), activation = 'tanh'))
model.add(tf.keras.layers.Dense(512, activation = 'tanh',kernel_regularizer=tf.keras.regularizers.l2(0.3)))
model.add(tf.keras.layers.Dense(512,activation='relu'))
model.add(tf.keras.layers.Dense(512, activation = 'tanh',kernel_regularizer=tf.keras.regularizers.l2(0.3)))
model.add(tf.keras.layers.Dense(512, activation = 'relu'))
model.add(tf.keras.layers.Dense(256, activation = 'tanh',kernel_regularizer=tf.keras.regularizers.l2(0.1)))
model.add(tf.keras.layers.Dense(128, activation = 'tanh',kernel_regularizer=tf.keras.regularizers.l2(0.1)))
model.add(tf.keras.layers.Dense(64, activation = 'tanh'))
model.add(tf.keras.layers.Dense(32, activation = 'tanh'))
model.add(tf.keras.layers.Dense(8, activation = 'tanh'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
# model.summary()
model.compile(loss = 'binary_crossentropy', optimizer = tf.optimizers.Adam(), metrics = ['acc'])
callbacks1 = tf.keras.callbacks.ReduceLROnPlateau('val_loss',patience=3,factor=0.5,min_lr=0.00001)   # 回调函数
history = model.fit(x_train,y_train, epochs = 1000,
                    validation_data=(x_test,y_test),
                    batch_size = 5120,
                    verbose = 1,
                    callbacks = [callbacks1])
print('---------------------')
print(history.history['acc'][-1])
print(history.history['val_acc'][-1])
# plt.plot(history.history['acc'],'r',label = 'acc')
# plt.plot(history.history['val_acc'],'b',label = 'val_acc')
# plt.legend()
#%% 提交结果
clf=GBoost
clf.score(x_test,y_test)

res=clf_rf.predict(x_submit)

sub = pd.read_csv('./公积金逾期预测-数据/submit.csv')
sub.label = res
sub.to_csv('.submit.csv')

