# 导包
import numpy as np
import tensorflow as tf
from sko.GA import GA
from time import time

# 构建数据集
x_train = np.random.randn(700,10)
y_train = np.random.randn(700,)
x_test = np.random.randn(300,10)
y_test = np.random.randn(300,)

#  定义func的目标函数
def get_model(units):
    model=tf.keras.Sequential([
        tf.keras.layers.Dense(units,input_shape=(10,))
    ])
    return model

def func(p):
    x=p
    model=get_model(x)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.MSE,
                  metrics=['mae'])
    history = model.fit(x_train,y_train,
                        validation_data=(x_test,y_test),
                        epochs=5,
                        verbose=0)
    print(history.history['val_mae'][-1])
    return history.history['val_mae'][-1]

# 寻优
a=time()
ga = GA(func=func, n_dim=1, size_pop=50, max_iter=10, lb=1, ub=20)
best_x, best_y = ga.run()
b=time()
print('best_x:', best_x, '\n', 'best_y:', best_y)
print('耗时:',b-a)