import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
#中文显示
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']     
#显示负号
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] =False

#%%
ntr=[np.argmax(x) for x in m.predict(x_train)]
nte=[np.argmax(x) for x in m.predict(x_test)]


# train=m.predict(x_train)
# test= m.predict(x_test)
tr,te=[],[]
for i in range(54):
    tr.append(np.argmax(train[i]))
    te.append(np.argmax(test[i]))
ntr=np.hstack((tr[0:9],tr[27:]))
nte=np.hstack((te[0:9],te[27:]))
#%%

plt.figure(figsize=(9,4.5))
plt.plot(np.arange(0,36),ntr,'--k*',label='真实值')
plt.plot(np.arange(0,36),ntr,'bv',alpha=0.8,label='训练集结果')
plt.plot(np.arange(0,36),nte,'r^',alpha=0.8,label='验证集结果')
plt.legend(loc='lower right')
plt.xlabel('样本数')
plt.ylabel('故障类型')
ax=plt.gca()
ax.spines['right'].set_color('none');ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none');ax.spines['left'].set_color('none')
ax.set_title('故障分类模型在数据集上的表现')
plt.yticks([0,1,2],['首部','中部','尾部'])


#%%
data=np.load('正常数据.npy')
#%%
fault=x_train[-9]


plt.figure(figsize=(5,4))
plt.axes(xscale = "log")
plt.plot(data[0,0,:],data[0,1,:],'r',label='正常幅-频曲线')
plt.plot(data[0,0,:],data[0,2,:],'r--',label='正常相-频曲线')
plt.plot(fault[0],fault[1],'b',lw=0.7,alpha=0.8,label='径向外凹幅-频曲线')
plt.plot(fault[0],fault[2],'--b',lw=0.7,alpha=0.8,label='径向外凹相-频曲线')
plt.legend(loc='lower left')
plt.title('直角坐标')
plt.xlabel('幅度(对数形式)/Hz');plt.ylabel('幅值/dB            相位角/°        ',loc='top')
plt.ylim(-600,0)


plt.figure(figsize=(4,4))
plt.polar(data[0,2,:]*np.pi/180,data[0,1,:],'r*',label='正常')
plt.polar(fault[2,:]*np.pi/180,fault[1,:],'b.',alpha=0.5,label='径向外凹')
plt.legend()
plt.title('极坐标');plt.ylim(-600,0)
plt.xlabel('相位角/rad');plt.ylabel('幅值/dB',loc='bottom')



#%%
title=['匝间短路','轴向扭曲','径向内凹','径向外凹']
loc=[8,30,38,49]
plt.figure(figsize=(9,6))
for i in range(4):
    plt.figure(figsize=(3,3))
    plt.imshow(x_train[loc[i]],cmap='green')
    plt.xticks(());plt.yticks(());
    plt.title(title[i])
