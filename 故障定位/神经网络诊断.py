#%%  CNN故障定位
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

''' 随机数种子 '''
i=1
tf.random.set_seed(i)
np.random.seed(i)

normal=np.load('正常数据.npy')
x_train=np.load('处理后训练集.npy')
x_test =np.load('处理后测试集.npy')
flocation=np.load('故障位置.npy')
ftype=np.load('故障类型.npy')

for i in range(54):
    x_train[i]=x_train[i]-normal
    x_test[i]=x_test[i]-normal

#处理数据
x_train=x_train[:,:,:2000]
x_test =x_test[:,:,:2000]
x_train=x_train.reshape(-1,75,80,1)
x_test=x_test.reshape(-1,75,80,1)
y_train=np.hstack((ftype,flocation))
y_test =np.hstack((ftype,flocation))

x_test=list(x_test);x_train=list(x_train)
for i in range(54):
    x_test[i]=cv2.resize(x_test[i], (128,128)).reshape(128,128,1)
    x_train[i]=cv2.resize(x_train[i], (128,128)).reshape(128,128,1)
x_test=np.array(x_test);x_train=np.array(x_train)

# 故障类型
y=y_train[:,:6]
train_type=tf.data.Dataset.from_tensor_slices((x_train,y))
train_type=train_type.shuffle(54).batch(32)
# 故障位置
y=y_train[:,6:]
#   变为4各故障的诊断
x_train=np.concatenate((x_train[:9],x_train[27:]))
y=np.concatenate((y[:9],y[27:]))
x_test=np.concatenate((x_test[:9],x_test[27:]))
y_test=np.concatenate((y_test[:9],y_test[27:]))       # 变为4各故障的诊断
train_location=tf.data.Dataset.from_tensor_slices((x_train,y))
train_location=train_location.shuffle(54).batch(32)

#  CNN 故障位置诊断
model=tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64,(3,3), input_shape=x_train.shape[1:],padding='same'))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128,(3,3)))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512,activation='relu'
                                ,kernel_regularizer=tf.keras.regularizers.l2(9)
                                ))
# model.add(tf.keras.layers.Dropout(rate=0.1))
model.add(tf.keras.layers.Dense(256,activation='relu'
                                ))
model.add(tf.keras.layers.Dense(3,activation='softmax'))
model.summary()

lr_early=tf.keras.callbacks.EarlyStopping('val_acc',patience=400)
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau('val_acc',patience=100,factor=0.95,min_lr=0.001)   # 回调函数
lr_checkp=tf.keras.callbacks.ModelCheckpoint('./model/{epoch:02d}-{acc:.4f}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=0,
                                             save_best_only=True, save_weights_only=False, mode='auto', period=1)
import os
import datetime
log_dir=os.path.join('logs',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))  #文件名

tf.optimizers.RMSprop()
model.compile(optimizer=tf.optimizers.Adamax(),
              loss='categorical_crossentropy',    #这个损失函数专门用于解决分类问题
              metrics=['acc'])
history=model.fit(train_location,  #训练集样本
                  epochs=5000, 
                  batch_size=128,
                  validation_data=(x_test,y_test[:,6:]),  #验证集
                   callbacks=[
                        lr_early,
                        lr_reduce,
                        lr_checkp
                       ]
                  )

plt.figure(figsize=(12,4))
plt.plot(history.history['acc'],'k-',lw=0.9,label='训练集迭代精度')
plt.plot(history.history['val_acc'],'k--',lw=0.6,label='验证集迭代精度')
ax=plt.gca()
ax.set_title('故障定位模型学习曲线')
ax.spines['right'].set_color('none')            #右边框去掉
ax.spines['top'].set_color('none')              #顶边框去掉
# ax.spines['left'].set_color('none')            #左边框去掉
# ax.spines['bottom'].set_color('none')              #下边框去掉
plt.legend(loc='lower right') 
plt.ylim(0,1)

m=model
#%%
model.save('model故障定位.h5')

m=tf.keras.models.load_model('model故障定位.h5')

#%%  CNN故障分类
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

normal=np.load('正常数据.npy')
x_train=np.load('处理后训练集.npy')
x_test =np.load('处理后测试集.npy')
flocation=np.load('故障位置.npy')
ftype=np.load('故障类型.npy')

for i in range(54):
    x_train[i]=x_train[i]-normal
    x_test[i]=x_test[i]-normal

#处理数据
x_train=x_train[:,:,:2000]
x_test =x_test[:,:,:2000]
x_train=x_train.reshape(-1,75,80,1)
# cv2.resize(x_train,(128,128))
x_test=x_test.reshape(-1,75,80,1)
y_train=np.hstack((ftype,flocation))
y_test =np.hstack((ftype,flocation))

x_test=list(x_test);x_train=list(x_train)
for i in range(54):
    x_test[i]=cv2.resize(x_test[i], (128,128)).reshape(128,128,1)
    x_train[i]=cv2.resize(x_train[i], (128,128)).reshape(128,128,1)
x_test=np.array(x_test);x_train=np.array(x_train)
# 故障类型
y=y_train[:,:6]
train_type=tf.data.Dataset.from_tensor_slices((x_train,y))
train_type=train_type.shuffle(54).batch(32)
# 故障位置
y=y_train[:,6:]
train_location=tf.data.Dataset.from_tensor_slices((x_train,y))
train_location=train_location.shuffle(54).batch(32)



#  CNN 故障类型诊断
model=tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (6,6), input_shape=x_train.shape[1:],padding='same'))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(64,(3,3)))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64,activation='relu'
                                ,kernel_regularizer=tf.keras.regularizers.l1(0.006)
                                ))
model.add(tf.keras.layers.Dense(64,activation='relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(6,activation='softmax'))
model.summary()

lr_early=tf.keras.callbacks.EarlyStopping('val_acc',patience=400)
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau('val_acc',patience=100,factor=0.8,min_lr=0.0001)   # 回调函数
import os
import datetime
log_dir=os.path.join('logs',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))  #文件名
tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir,histogram_freq=1)  #tensorboard的回调函数


model.compile(optimizer='adam',
              loss='categorical_crossentropy',    #这个损失函数专门用于解决分类问题
              metrics=['acc'])
history=model.fit(train_type,  #训练集样本
                  epochs=2000,                 
                  validation_data=(x_test,y_test[:,:6]),  #验证集
                   callbacks=[
                        lr_early,
                        lr_reduce,
                        tensorboard_callback
                       ]
                  )

plt.figure(figsize=(12,4))
plt.plot(history.history['acc'],'k-',lw=0.9,label='训练集迭代精度')
plt.plot(history.history['val_acc'],'k--',lw=0.6,label='验证集迭代精度')
ax=plt.gca()
ax.set_title('故障分类模型学习曲线')
ax.spines['right'].set_color('none')            #右边框去掉
ax.spines['top'].set_color('none')              #顶边框去掉
# ax.spines['left'].set_color('none')            #左边框去掉
# ax.spines['bottom'].set_color('none')              #下边框去掉
plt.legend(loc='lower right') 
plt.ylim(0,1)

#%% 保存模型
model.save('model故障分类.h5')

#%%
for i in range(54):
    plt.figure()
    plt.imshow(x_train[i],cmap='rainbow')
    



