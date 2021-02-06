import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']


train_imgs=np.load('./训练集图像/train_imgs28x28.npy')
test_imgs=np.load('./测试集图像/test_imgs28x28.npy')
ftype=np.load('./故障类型.npy')
flocation=np.load('./故障位置.npy')



#%%  CNN 故障分类

train=tf.data.Dataset.from_tensor_slices((train_imgs,ftype))
train=train.shuffle(16).batch(32)
test=tf.data.Dataset.from_tensor_slices((test_imgs,ftype))
test=test.shuffle(16).batch(32)
    
model=tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(2,(3,3),input_shape=(train_imgs.shape[1:]),activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Conv2D(4,(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.LeakyReLU())
    
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10,activation='relu'
                                ,kernel_regularizer=tf.keras.regularizers.l1(0.03)
                                ))
# model.add(tf.keras.layers.Dropout(rate=0.1))
model.add(tf.keras.layers.Dense(6,activation='softmax'))
    
lr_early=tf.keras.callbacks.EarlyStopping('val_acc',patience=400)
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau('val_acc',patience=100,factor=0.8,min_lr=0.0001)   # 回调函数
    
model.compile(optimizer='adam',
               loss='categorical_crossentropy',    #这个损失函数专门用于解决分类问题
               metrics=['acc'])
history=model.fit(train,  #训练集样本
                  epochs=1000,                  #所有样本训练30次
                  validation_data=test,  #验证集
                  callbacks=[
                           # lr_early,
                           # lr_reduce
                           ]
                            )
plt.figure(figsize=(8,2))
ax1=plt.gca()
plt.plot(history.history['acc'],'--r',label='验证集精度',alpha=0.5)
plt.plot(history.history['val_acc'],'--b',label='训练集精度',alpha=0.5)
plt.legend()
ax1.set_title('训练集与验证集的精度随学习次数的变化')
plt.ylim(0,1)

# ftype_data=history
#%%  CNN 故障分类

train=tf.data.Dataset.from_tensor_slices((train_imgs,flocation))
train=train.shuffle(16).batch(32)
test=tf.data.Dataset.from_tensor_slices((test_imgs,flocation))
test=test.shuffle(16).batch(32)
    
model=tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(4,(3,3),input_shape=(train_imgs.shape[1:]),activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Conv2D(8,(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.LeakyReLU())
    
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(8,activation='relu'
                                ,kernel_regularizer=tf.keras.regularizers.l1(0.1)
                                ))
# model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(3,activation='softmax'))
    
lr_early=tf.keras.callbacks.EarlyStopping('val_acc',patience=400)
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau('val_acc',patience=100,factor=0.8,min_lr=0.0001)   # 回调函数
    
model.compile(optimizer='adam',
               loss='categorical_crossentropy',    #这个损失函数专门用于解决分类问题
               metrics=['acc'])
history=model.fit(train,  #训练集样本
                  epochs=1000,                  #所有样本训练30次
                  validation_data=test,  #验证集
                  callbacks=[
                           # lr_early,
                           # lr_reduce
                           ]
                            )
plt.figure(figsize=(8,2))
ax1=plt.gca()
plt.plot(history.history['acc'],'--r',label='验证集精度',alpha=0.5)
plt.plot(history.history['val_acc'],'--b',label='训练集精度',alpha=0.5)
plt.legend()
ax1.set_title('训练集与验证集的精度随学习次数的变化')
plt.ylim(0,1)

# ftype_data=history
