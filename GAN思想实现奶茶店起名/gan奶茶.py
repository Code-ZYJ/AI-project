import torch
import numpy as np
import  torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 导入数据
with open('奶茶店名.txt','r',encoding='utf-8') as f:
    a = f.read().strip(' ').split('\n\n')
names = []
for name in a:
    names.append(name[5:])

#%% 语料预处理
def isChinese(word):  #判断是否是中文
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def get_vocab(names):
    words=[]  #构建词表
    for name in names:
        for a in name:
            words+=[''.join(a)]
    for i,word in enumerate(words):
        if not isChinese(word):
            words[i]=''
    vocab = dict(enumerate(set(words)))
    vocab[119]='unknow'
    vocab[0]='<PAD>'

    # 处理数据集
    vocab_converse=dict()  #获取反向词表
    for key,value in vocab.items():
        vocab_converse[value]=key
    return vocab,vocab_converse

vocab,vocab_converse = get_vocab(names)

# 获得此表序列最长的长度
max_len=len(names[0])
for name in names:
    if max_len < len(name):
        max_len = len(name)
#创建inputs接受数据
inputs = np.zeros((67,max_len))
for i,name in enumerate(names):
    for j,a in enumerate(name):
        if isChinese(a):
            inputs[i][j] = vocab_converse[a]
        else:
            inputs[i][j] = vocab_converse['unknow']

#%% 构建模型
vocab_size = len(vocab)

batch_size = 16

x_train = torch.from_numpy(inputs).type(torch.LongTensor).to(device)
x_train = TensorDataset(x_train, x_train)
x_train = DataLoader(x_train,batch_size=batch_size,shuffle=True,drop_last=True)
for x,_ in x_train:
    print(x)


# 判别器
class discriminateor(torch.nn.Module):
    def __init__(self,embedding_dim,cnn_units):
        super(discriminateor, self).__init__()
        self.cnn_units = cnn_units
        self.embedding = torch.nn.Embedding(vocab_size,embedding_dim)
        self.Tcnn = torch.nn.Conv2d(1,cnn_units, (2,embedding_dim), stride = 1)  #这个地方输入维度 0：batch 1：channels 2：max_len 3:embedding_dim
        self.flatten = torch.nn.Flatten()
        self.out = torch.nn.Linear(7*cnn_units,1)


    def forward(self,input):  #text:[batch,max_len]
        embedded = self.embedding(input)
        embedded = torch.unsqueeze(embedded,dim = 1)
        cnn_out = torch.tanh(self.Tcnn(embedded))
        fla = self.flatten(cnn_out)
        out = torch.sigmoid(self.out(fla))
        return out
#测试 判别器
out = discriminateor(5,8).to(device)(input=torch.rand(16,8).type(torch.LongTensor).to(device))


# 生成器
class generator(torch.nn.Module):
    def __init__(self,embedding_dim,rnn_units):
        super(generator, self).__init__()
        self.rnn_units = rnn_units
        self.embedding = torch.nn.Embedding(vocab_size,embedding_dim)
        self.gru = torch.nn.GRU(embedding_dim,rnn_units,batch_first=True)
        self.out = torch.nn.Linear(rnn_units,vocab_size)

    def forward(self,rand8):
        embedded = self.embedding(rand8)
        gru_output,_ = (self.gru(embedded,None))
        linear_input = gru_output
        out = torch.tanh(self.out(linear_input))
        out = torch.argmax(F.softmax(out,dim=-1),dim = -1)
        return out

# 测试 生成器
out = generator(5, 16).to(device)((torch.rand(16,8)).type(torch.LongTensor).to(device))

def define_model_and_train(EMBEDDING_DIM,CNN_UNITS,RNN_UNITS):
    EMBEDDING_DIM = int(EMBEDDING_DIM)
    CNN_UNITS = int(CNN_UNITS)
    RNN_UNITS = int(RNN_UNITS)
    # 训练
    EPOCHS=100
    # EMBEDDING_DIM=5
    # CNN_UNITS, RNN_UNITS = 8, 16

    D=discriminateor(embedding_dim=EMBEDDING_DIM,cnn_units=CNN_UNITS).to(device)
    G=generator(embedding_dim=EMBEDDING_DIM,rnn_units=RNN_UNITS).to(device)

    optimD = torch.optim.RMSprop(D.parameters(),lr=1e-3)
    optimG = torch.optim.RMSprop(G.parameters(),lr=1e-3)

    BCELoss = torch.nn.BCELoss().to(device)  #二元交叉熵损失
    CELoss = torch.nn.CrossEntropyLoss()   #交叉熵损失

    # 训练过程
    for epoch in range(EPOCHS):
        for x,_ in x_train:
            #转移数据到GPU上
            rand8=(100*torch.rand(batch_size,8)).type(torch.LongTensor).to(device)
            x.to(device)
            #生成名字
            all_out = G(rand8)
            #先训练判别器
            D.train(), G.eval()
            fake_out = D(all_out.type(torch.LongTensor).to(device))  # 这种形式才能到D里面（Embedding）
            real_out = D(x.type(torch.LongTensor).to(device))
            out_labels = torch.cat((fake_out, real_out), dim=0)  # 输出标签
            trg_labels = torch.cat((torch.zeros(batch_size, 1), torch.ones(batch_size, 1)), dim=0).to(device)  # 目标标签
            loss_D = BCELoss(out_labels, trg_labels)
            loss_D.backward()
            optimD.zero_grad()
            optimD.step()
            #再训练生成器
            D.eval(), G.train()
            loss_G = BCELoss(D(all_out.type(torch.LongTensor).to(device)),
                             torch.ones(batch_size, 1).to(device))  # 这个地方的D已经更新了权重值，不同于上面的 fake_out
            loss_G.backward()
            optimG.zero_grad()
            optimG.step()
        #print('Epoch:{}————判别器损失:{:.6f}————生成器损失:{:.6f}'.format(epoch, loss_D.item(), loss_G.item()))


    # 名字展示
    with torch.no_grad():
        all_out = np.array(all_out.cpu())
    stores_name = []
    for i in range(all_out.shape[0]):
        store_name = []
        for j in range(all_out.shape[1]):
            store_name.append(vocab[all_out[i,j]])
        stores_name.append(store_name)

    return loss_D.item(),loss_G.item(),stores_name,all_out


#%% 试着用优化算法优化
from sko.GA import GA
import time
import random
random.seed(24)
torch.manual_seed(24) # cpu
torch.cuda.manual_seed(24) #gpu
np.random.seed(24)

def func(p):
    EMBEDDING_DIM,CNN_UNITS,RNN_UNITS=p
    lossd,lossg,_,_ = define_model_and_train(EMBEDDING_DIM,CNN_UNITS,RNN_UNITS)
    print('判别器损失:{:.6f}——————生成器损失:{:.6f}'.format(lossd,lossg))
    return  lossd+lossg

ga = GA(func=func,n_dim=3,
        lb=[5,4,4],ub=[10,32,16],
        )
a = time.time()
best_x, best_y = ga.run()
b = time.time()





print('总耗时：',b-a)
print('best EMBEDDING_DIM:{}\nbest CNN_UNITS    :{}\nbest RNN_UNITS    :{}\nALL LOSS: {}'.format(best_x[0],best_x[1],best_x[2],best_y))


#%% 利用优化后参数建模
_,_,name,all_out = define_model_and_train(EMBEDDING_DIM=best_x[0],CNN_UNITS=best_x[1],RNN_UNITS=best_x[2])
print(name)

1