import numpy as np
x_train = np.random.randn(600,15)
y_train1 = np.zeros(300)
y_train2 = np.ones(300)
y_train = np.hstack((y_train1,y_train2))

np.save('x_train.npy',x_train)
np.save('y_train.npy',y_train)

s=np.load("D:/py work/python/PYQT5/x_train.npy")