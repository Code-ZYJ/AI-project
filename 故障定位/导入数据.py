import glob
import numpy as np

#%%
path=glob.glob('./1/*.txt')
data=list()
for i in range(9):
    F,DB,PV=[],[],[]
    with open(path,'r') as file:
        while True:
            lines=file.readline()
            if not lines:
                break
            f,db,pv=[i for i in lines.strip().split()]
            F.append(f)
            DB.append(db)
            PV.append(pv)
        F=np.array(F[1:])
        DB=np.array(DB[1:])
        PV=np.array(PV[1:])
    F=F.astype('float32').reshape(-1)
    DB=DB.astype('float32').reshape(-1)
    PV=PV.astype('float32').reshape(-1)
    
    data.append(np.vstack((F,DB,PV)))
    print('第{}个样本已导入'.format(i))

data=np.array(data)

#%%
for j in range(2003):
    while data[0,2,j]<-180:
        data[0,2,j]=data[0,2,j]+360
data[:,2]=data[:,2]-180
print(data[:,2].max())
np.save(path[:-4],data)