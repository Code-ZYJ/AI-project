import numpy as np
import math
zc=np.load('zc.npy')
F=np.load('F.npy')
test=np.load('test.npy')
#%%   函数中y均为故障的特征向量       例：F[0]
def cc(y):
    fz,fm=0,0
    for i in range(5006):
        fz+=zc[i]*y[i]
        fm+=(zc[i]**2) * (y[i]**2)
    score=fz/math.sqrt(fm)
    return score 

def ed(y):
    s=0
    for i in range(5006):
        s+=(zc[i]-y[i])**2
    return math.sqrt(s)

def mod(y):
    return max(zc-y)    

def sse(y):
    s=0
    for i in range(5006):
       s+=(y[i]-zc[i])**2
    return s/5006

def ssre(y):
    s=0
    for i in range(5006):
       s+=(y[i]/zc[i]-1)**2
    return s/5006

def ioad(y):
    s=0
    for i in range(5006):
        s+=abs(zc[i]-y[i])
    return s
    
def ssmmre(y):
    s=0
    for i in range(5006):
        s+=((max(zc[i],y[i])/min(zc[i],y[i]))-1)**2
    return s/5006

def rmse(y):
    s=0
    for i in range(5006):
        s+=(abs(y[i])-abs(zc[i]))/(sum(zc)/5006)
    return s
#%%    训练集
feature1=np.zeros(54)
feature2=np.zeros(54)
feature3=np.zeros(54)
feature4=np.zeros(54)
feature5=np.zeros(54)
feature6=np.zeros(54)
feature7=np.zeros(54)
feature8=np.zeros(54)
for i in range(54):
    feature1[i]=cc(F[i])
    feature2[i]=ed(F[i])
    feature3[i]=mod(F[i])
    feature4[i]=sse(F[i])
    feature5[i]=ssre(F[i])
    feature6[i]=ioad(F[i])
    feature7[i]=ssmmre(F[i])    
    feature8[i]=rmse(F[i])

tmp=[feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8]
tmp=np.array(tmp)
tmp=tmp.T
np.save('feature data',tmp)


#%%     验证集
feature1=np.zeros(30)
feature2=np.zeros(30)
feature3=np.zeros(30)
feature4=np.zeros(30)
feature5=np.zeros(30)
feature6=np.zeros(30)
feature7=np.zeros(30)
feature8=np.zeros(30)
for i in range(30):
    feature1[i]=cc(test[i])
    feature2[i]=ed(test[i])
    feature3[i]=mod(test[i])
    feature4[i]=sse(test[i])
    feature5[i]=ssre(test[i])
    feature6[i]=ioad(test[i])
    feature7[i]=ssmmre(test[i])    
    feature8[i]=rmse(test[i])

tmp=[feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8]
tmp=np.array(tmp)
tmp=tmp.T
np.save('val input',tmp)



#%%  svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import scipy.io as io

xtrain=np.load('x_train.npy')
xtest=np.load('x_test.npy')
ytest=io.loadmat('ytest');ytrain=io.loadmat('ytrain.mat');
ytrainl=ytrain['outbase'][1:,1];    ytestl=ytest['T_test'][:,1] 

#%%
from sklearn import metrics

poly=svm.SVR(kernel='rbf',degree=1,C=1)
poly.fit(xtrain,ytrainl)

pre=poly.predict(xtest)
poly.score(xtest,ytestl)

plt.subplot(211)
plt.plot(ytestl,'r',label='real')
plt.plot(pre,'b',label='predict')
plt.legend()

print('mae:',metrics.mean_absolute_error(ytestl,pre))
print('mse:',metrics.mean_squared_error(ytestl,pre))

plt.subplot(212)
pre=poly.predict(xtrain)
plt.plot(ytrainl,'r',label='real')
plt.plot(pre,'b',label='predict')
plt.legend()
print('mae:',metrics.mean_absolute_error(ytrainl,pre))
print('mse:',metrics.mean_squared_error(ytrainl,pre))


#%%  网格搜索
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV as GS

para_dict={'kernel':['poly','rbf'],
           'C':np.arange(1,10,1),
           'degree':np.arange(1,10,1)
           }

poly=svm.SVR(kernel='poly',C=1,degree=1)

GS=GS(poly,para_dict,cv=10)
GS.fit(xtrain,ytrainl)
print(GS.best_params_)
print(GS.best_score_)

#%%
x=np.vstack((xtrain,xtest))

#训练集输出
ytraint=list()
for i in range(1,7):
    for j in range(9):
        ytraint.append(i)
#测试集输出
ytestt=list()
for i in range(1,7):
    for j in range(5):
        ytestt.append(i)
        
ytraint=np.array(ytraint).reshape(-1,1)
ytestt =np.array(ytestt).reshape(-1,1)
y=np.vstack((ytraint,ytestt))
y=y.reshape(-1)
#%%  分类问题
from sklearn import svm
poly=svm.SVC(kernel='linear',
             degree=1,
             C=1)
poly.fit(xtrain,ytraint)
pre=poly.predict(xtest)
print(poly.score(xtest,ytestt))

plt.plot(ytestt,'r*',label='real')
plt.plot(pre,'b^',label='predict')
plt.yticks([1,2,3,4,5,6,],
           ['Short circuit between turns','Broken coil','Metal foreign body','Axial displacement','Radial inward twist','Radial outward twist'])
plt.xlabel('Sample')
plt.ylabel('Fault type')
ax=plt.gca()
ax.spines['right'].set_color('none')            #将右边的边框设置成无
ax.spines['top'].set_color('none') 
plt.legend()

#%%  网格搜索
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV as GS

para_dict={'kernel':['poly','rbf'],
           'C':np.arange(1,10,1),
           'degree':np.arange(1,10,1)
           }

poly=svm.SVR(kernel='poly',C=1,degree=1)

GS=GS(poly,para_dict,cv=10)
GS.fit(x,y)
print(GS.best_params_)
print(GS.best_score_)
        
        


#%% 获取幅频和相位
import pandas as pd
import matplotlib.pyplot as plt
zc=open('zc.txt','r')
zc=zc.readlines()
ls=[]
for i in zc:
    tmp=i.strip().split(' ')
    ls.append(tmp)    
    
    
db,pv=[],[]
for i in range(1,5007):
    db.append(ls[i][0])
    pv.append(ls[i][2])
    
d,p=[],[]
for i in range(1,1000,50):
    d.append(db[i])
    p.append(pv[i])

for i in range(1,6):
    plt.polar(p[i],d[i],'bo')
    plt.polar(p[i+6],d[i+6],'go')
    plt.polar(p[i+12],d[i+12],'ro')
plt.xticks([])  
plt.yticks([])  
