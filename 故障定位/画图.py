import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
train=np.load('频响数据训练集.npy')
test =np.load('频响数据测试集.npy')
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] =False
#%%  处理相位大小
for i in range(54):
    for j in range(2003):
        while train[i,2,j]<-180:
            train[i,2,j]=train[i,2,j]+360
        while test[i,2,j]<-180:
            test[i,2,j]=test[i,2,j]+360
train[:,2]=train[:,2]-180
test[:,2] = test[:,2] -180

np.save('处理后训练集',train)
np.save('处理后测试集',test)
#%%  传统与改进对比
data1=np.load('正常数据改进前.npy')
data2=np.load('正常数据.npy')
plt.figure(figsize=(8,3))
# plt.subplot(211)
plt.figure(figsize=(4,2.5))
ax1=plt.gca();ax1.set_title('幅-频对比')
plt.plot(data1[0,0,:],data1[0,1,:],'k-',alpha=0.5,label='传统')
plt.plot(data2[0,0,:],data2[0,1,:],'k--',alpha=0.5,label='改进后')
plt.xlabel('频率(f)');plt.ylabel('幅值(dB)');
ax1.spines['right'].set_color('none');ax1.spines['top'].set_color('none')
plt.legend();plt.ylim(-600,0);plt.xlim(0,2e5)
# plt.subplot(212)
plt.figure(figsize=(4,2.5))
ax2=plt.gca();ax2.set_title('相-频对比')
plt.plot(data1[0,0,:],data1[0,2,:],'k-',alpha=0.5,label='传统')
plt.plot(data2[0,0,:],data2[0,2,:],'k--',alpha=0.5,label='改进后')
plt.xlabel('频率(f)');plt.ylabel('相位(PV)');
ax2.spines['right'].set_color('none');ax2.spines['top'].set_color('none')
plt.legend();plt.ylim(-360,0);plt.xlim(0,2e5)
plt.show()
# %%  对称位置故障对比（幅频）
data1=np.load('不对称1.npy')
data2=np.load('不对称9.npy')
# data3=np.load('不对称1.npy')
# data4=np.load('不对称9.npy')
plt.figure(figsize=(6,4))
# fig=plt.figure(figsize=(6,4))
plt.plot(data1[0,0,:],data1[0,1,:],'r-',label='首部径向外凹',alpha=0.5)
plt.plot(data2[0,0,:],data2[0,1,:],'b-',label='尾部径向外凹',alpha=0.5)
ax=plt.gca()
ax.spines['right'].set_color('none');ax.spines['top'].set_color('none')
ax.spines['left'].set_color('none');ax.spines['bottom'].set_color('none')
ax.set_title('(改进后)首、尾部对称点径向内凹幅-频曲线',loc='center')
# plt.plot(data3[0,0,:],data3[0,1,:],'k-',label='首部径向外凹',alpha=0.5)
# plt.plot(data4[0,0,:],data4[0,1,:],'k--',label='尾部径向外凹',alpha=0.5)
plt.yticks(());plt.xticks(());plt.legend(loc='lower right')
plt.axes([.4,.5,.45,.35])
plt.plot(data1[0,0,850:1200],data1[0,1,850:1200],'r-',lw=1)
plt.plot(data2[0,0,850:1200],data2[0,1,850:1200],'b-',lw=1)
plt.axes([0.15,0.1,0.3,0.3])
plt.plot(data1[0,0,1700:1770],data1[0,1,1700:1770],'r-',lw=1)
plt.plot(data2[0,0,1700:1770],data2[0,1,1700:1770],'b-',lw=1)
plt.legend()
plt.show()
####



#% 对称位置故障对比（相频）
# data3=np.load('不对称1.npy')
# data4=np.load('不对称9.npy')
plt.figure(figsize=(6,4))
# fig=plt.figure(figsize=(6,4))
plt.plot(data1[0,0,:],data1[0,2,:],'r-',label='首部径向外凹',alpha=0.5)
plt.plot(data2[0,0,:],data2[0,2,:],'b-',label='尾部径向外凹',alpha=0.5)
ax=plt.gca()
ax.spines['right'].set_color('none');ax.spines['top'].set_color('none')
ax.spines['left'].set_color('none');ax.spines['bottom'].set_color('none')
ax.set_title('(改进后)首、尾部对称点径向内凹相-频曲线',loc='center')
# plt.plot(data3[0,0,:],data3[0,1,:],'k-',label='首部径向外凹',alpha=0.5)
# plt.plot(data4[0,0,:],data4[0,1,:],'k--',label='尾部径向外凹',alpha=0.5)
plt.yticks(());plt.xticks(());plt.legend(loc='lower right')
plt.axes([0.3,0.05,0.35,0.25])
plt.plot(data1[0,0,850:1180],data1[0,2,850:1180],'r-',lw=1)
plt.plot(data2[0,0,850:1180],data2[0,2,850:1180],'b-',lw=1)
plt.axes([.6,.55,.3,.3])
plt.plot(data1[0,0,1650:1800],data1[0,2,1650:1800],'r-',lw=1)
plt.plot(data2[0,0,1650:1800],data2[0,2,1650:1800],'b-',lw=1)
plt.legend()
plt.show()
####


#%%
train=np.load('处理后训练集.npy')
test=np.load('处理后测试集.npy')

for i in range(54):
    plt.figure(figsize=(3,3))
    plt.polar(test[i,2][:int(2003/3)]*np.pi/180,test[i,1][:int(2003/3)],'r.')
    plt.polar(test[i,2][int(2003/3):int(2003*2/3)]*np.pi/180,test[i,1][int(2003/3):int(2003*2/3)],'g.')
    plt.polar(test[i,2][int(2003*2/3):]*np.pi/180,test[i,1][int(2003*2/3):],'b.')
    plt.ylim(-250,-30)
    plt.xticks([])
    plt.yticks([])
    
    
   #%%
import cv2
imgs=list()
for i in range(54):
    tmp=cv2.imread('Figure 2020-10-28 130007 ({}).png'.format(i))
    tmp=cv2.resize(tmp, (28,28))
    imgs.append(tmp)
imgs=np.array(imgs)
np.save('train_imgs28x28',imgs)


# %%  对称位置故障对比（幅频）
'''
专利改进图
'''
data1=np.load('对称1.npy')
data2=np.load('对称9.npy')
# data3=np.load('不对称1.npy')
# data4=np.load('不对称9.npy')


plt.figure(figsize=(10,8))
gs=gridspec.GridSpec(3,3)
ax1=plt.subplot(gs[0:2,0:2])
ax1.plot(data1[0,0,:],data1[0,1,:],'k-',label='首部径向内凹',alpha=0.5)
ax1.plot(data2[0,0,:],data2[0,1,:],'k--',label='尾部径向内凹',alpha=0.5)
plt.legend();plt.title('（改进前）首、尾对称点径向内凹幅-频曲线');plt.legend()
ax1.set_xticks(());ax1.set_yticks(())
ax2=plt.subplot(gs[0,2])
ax2.plot(data1[0,0,850:1200],data1[0,1,850:1200],'k-',lw=1)
ax2.plot(data2[0,0,850:1200],data2[0,1,850:1200],'k--',lw=1)
plt.title('两端频率范围的局部放大图')
ax3=plt.subplot(gs[1,2])
ax3.plot(data1[0,0,1700:1770],data1[0,1,1700:1770],'k-',lw=1)
ax3.plot(data2[0,0,1700:1770],data2[0,1,1700:1770],'k--',lw=1)



plt.figure(figsize=(10,8))
gs=gridspec.GridSpec(3,3)
ax1=plt.subplot(gs[0:2,0:2])
ax1.plot(data1[0,0,:],data1[0,2,:],'k-',label='首部径向内凹',alpha=0.5)
ax1.plot(data2[0,0,:],data2[0,2,:],'k--',label='尾部径向内凹',alpha=0.5)
plt.legend();plt.title('（改进前）首、尾对称点径向内凹相-频曲线');plt.legend()
ax1.set_xticks(());ax1.set_yticks(())
ax2=plt.subplot(gs[0,2])
ax2.plot(data1[0,0,850:1200],data1[0,2,850:1200],'k-',lw=1)
ax2.plot(data2[0,0,850:1200],data2[0,2,850:1200],'k--',lw=1)
plt.title('两端频率范围的局部放大图')
ax3=plt.subplot(gs[1,2])
ax3.plot(data1[0,0,1700:1770],data1[0,2,1700:1770],'k-',lw=1)
ax3.plot(data2[0,0,1700:1770],data2[0,2,1700:1770],'k--',lw=1)


#%% 各故障样本图
plt.figure(dpi = 400)
for i in range(54):
    plt.subplot(6,9,i+1)
    plt.imshow(x_train[i].reshape(128,128),cmap = 'gray')
    plt.xticks(());plt.yticks(())
    plt.title(str(i+1))

fault={'匝间短路':5,'线圈断股':10, '金属异物':26,
      '轴向扭曲':36,'径向内凹':41,'径向外凹':53}

for k, v in fault.items():
    plt.figure(dpi=200)
    plt.imshow(x_train[v], cmap='gray')
    plt.title(k)
    plt.xticks(());plt.yticks(())


location = {'首部':20,'中部':33, '尾部':9}
for k, v in location.items():
    plt.figure(dpi=200)
    plt.imshow(x_train[v], cmap='gray')
    plt.title(k)
    plt.xticks(());plt.yticks(())








