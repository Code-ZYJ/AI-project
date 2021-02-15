import tensorflow as tf
import cv2
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

path = glob.glob('./imgs_base/*/*')
imgs=[] #用于存放图像
for p in path:
    img = cv2.imread(p)
    img = cv2.resize(img, (256,256))
    imgs.append(img)
labels = list(np.hstack((np.ones(200),np.zeros(200))).reshape(-1,1))
imgs=(np.array(imgs))
x_train,x_test,y_train,y_test=train_test_split(imgs,labels,test_size=0.5)

train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train = train.shuffle(200).batch(1)
test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test = test.batch(1)

#%%
from tensorflow import keras
from tensorflow.keras import layers

conv_base = keras.applications.VGG19(weights = 'imagenet', include_top = False)
conv_base.summary()         #卷积基

model = tf.keras.Sequential()
model.add(conv_base)
model.add(layers.GlobalAveragePooling2D())  #扁平化 与 Flatten 一样，但效果更好
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))  
conv_base.trainable = False     #使卷积基不可训练
model.summary()

callbacks=tf.keras.callbacks.ModelCheckpoint('./m/{epoch:02d}-{acc:.4f}-{val_acc:.4f}.h5', monitor='val_acc', verbose=0,
                                             save_best_only=True, save_weights_only=False, mode='auto', period=1)


model.compile(optimizer=keras.optimizers.Adam(lr = 0.001),
              loss = 'binary_crossentropy',
              metrics = ['acc'])
history = model.fit(train,
                    epochs=10,
                    validation_data=test,
                    callbacks=[callbacks])
#%%
plt.figure(dpi=150,figsize=(4,3))
plt.plot(history.history['acc'],'r',label='acc')
plt.plot(history.history['val_acc'],'b--',label='val_acc')
plt.legend(loc='lower right')
plt.title('VGG19迁移学习模型效果')
plt.ylim([0.8,1])
ax=plt.gca()
ax.spines['right'].set_color('none') 
ax.spines['top'].set_color('none') 




#%% 提交

model.save('model.h5')
