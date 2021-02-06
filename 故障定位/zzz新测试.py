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
train_location=tf.data.Dataset.from_tensor_slices((x_train,y))
train_location=train_location.shuffle(54).batch(32)


#  CNN 故障位置诊断
model=tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64,(3,3), input_shape=x_train.shape[1:], padding='same'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(3,activation='softmax'))

model.summary()

lr_early=tf.keras.callbacks.EarlyStopping('val_acc',patience=400)
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau('val_acc',patience=100,factor=0.95,min_lr=0.001)   # 回调函数
lr_checkp=tf.keras.callbacks.ModelCheckpoint('./model1/{epoch:02d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=0,
                                             save_best_only=True, save_weights_only=False, mode='auto', period=1)
import os
import datetime
log_dir=os.path.join('logs',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))  #文件名


model.compile(optimizer=tf.optimizers.Adamax(),
              loss='categorical_crossentropy',    #这个损失函数专门用于解决分类问题
              metrics=['acc'])
history=model.fit(train_location,  #训练集样本
                  epochs=500, 
                  batch_size=128,
                  validation_data=(x_test,y_test[:,6:]),  #验证集
                   callbacks=[
                        lr_early,
                        lr_reduce,
                        # lr_checkp
                       ]
                  )

plt.figure(figsize=(9,3))
plt.plot(history.history['acc'],'k-',lw=1,label='训练集迭代精度')
plt.plot(history.history['val_acc'],'k--',lw=0.5,label='验证集迭代精度')
ax=plt.gca()
ax.set_title('故障定位模型学习曲线')
ax.spines['right'].set_color('none')            #右边框去掉
ax.spines['top'].set_color('none')              #顶边框去掉
# ax.spines['left'].set_color('none')            #左边框去掉
# ax.spines['bottom'].set_color('none')              #下边框去掉
plt.legend(loc='lower right') 
plt.ylim(0,1)



#%%
model = tf.keras.Sequential()   #顺序模型
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'))
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(1,activation = 'sigmoid'))