import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)
print(sys.version_info)
for module in mpl,np,pd,sklearn,tf,keras:
    print(module.__name__,module.__version__)
    
'''
1. preprocessing data
2. build model
    2.1 encoder
    2.2 attention
    2.3 decoder
    2.4 loss & optimizer
3. evaluation
    3.1 given sentence, return translated results
    3.2 visualize results (attention)
'''


#%%  导入数据import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_digits
import cv2
from sklearn.preprocessing import OneHotEncoder

#%%  读取数据集
images=[]
mnist=load_digits()
for i in range(len(mnist.images)):
    img=mnist.images[i]
    img=cv2.resize(img,(28,28))
    images.append(img)
images=np.expand_dims(images, -1)
print(images.shape)
print(mnist.target.shape)
enc = OneHotEncoder(sparse=False)   #标签独热编码
data=mnist.target.reshape(-1,1)
label = enc.fit_transform(data)

#划分训练集
train_img,test_img,train_label,test_label=train_test_split(images,mnist.target,test_size=0.2)  
train_img=np.array(train_img)
test_img =np.array(test_img)
# train=tf.data.Dataset.from_tensor_slices((train_img,train_label))  #把训练集的输入和输出合并的时候一定注意里面是元组
# test =tf.data.Dataset.from_tensor_slices((test_img, test_label))
# batch_size=16
# train=train.repeat().shuffle(1000).batch(batch_size)    #数据处理的最后一步

#%% 建立模型
model=tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(2, (3,3), input_shape=train_img.shape[1:],activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(4,(3,3),activation='relu'))
model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10,activation='softmax'))
model.summary()

#%% 模型编译
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',    #这个损失函数专门用于解决分类问题
              metrics=['acc'])
history=model.fit(train_img,train_label,  #训练集样本
                  epochs=30,                  #所有样本训练30次
                  validation_data=(test_img,test_label)  #验证集
                  )
