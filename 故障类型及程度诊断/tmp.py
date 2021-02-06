import numpy as np
import matplotlib.pyplot as plt
from scipy import io

y_train = io.loadmat('ytrain.mat')
x_test = np.loadmat('x_test.mat')

t=np.load('test.npy')
F=np.load('F.npy')
plt.plot(F[0,3500:4300])

plt.figure(1)
for i in range(6):
    plt.plot(F[i],'r--',alpha=0.5)
plt.plot(t[0],'r--',alpha=0.5)
plt.plot(t[1],'g-',alpha=0.5,label='2')
plt.plot(t[2],'r--',alpha=0.5)
plt.plot(t[3],'b-',alpha=0.5,label='4')
plt.plot(t[4],'r--',alpha=0.5)
plt.legend()
plt.xlim([3450,3750]);plt.ylim([-75,0])