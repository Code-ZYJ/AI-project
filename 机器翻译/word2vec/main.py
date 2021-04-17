import jieba
from gensim.corpora import Dictionary
from corpus.get_data import get_data_from_dict
from gensim.models.word2vec import Word2Vec
import json

en,cn = get_data_from_dict()     # 读取语料数据
len(en) == len(cn)    #看看语料长度是否相等

en = [i.strip('\n').split() for i in en]
cn = [jieba.lcut(i.strip('\n')) for i in cn]


#%% 构建词表

#英文词表
special_token = [['<PAD>','<S>','<E>','<UNK>']]
dct_en = Dictionary(special_token)
dct_en.add_documents(en)
vocab_en = dct_en.token2id
print('英文词表长度',len(vocab_en))
with open('./word2vec/vocab_en.json', 'w', encoding='utf-8') as f:
    json.dump(vocab_en,f,ensure_ascii=False)

#中文词表
special_token = [['<PAD>','<S>','<E>','<UNK>']]
dct_cn = Dictionary(special_token)
dct_cn.add_documents((cn))
vocab_cn = dct_cn.token2id
print('中文词表长度：',len(vocab_cn))
with open('./word2vec/vocab_cn.json', 'w', encoding='utf-8') as f:
    json.dump(vocab_cn,f,ensure_ascii=False)


vocab_size_eng= len(vocab_en)
vocab_size_cn = len(vocab_cn)


# 构建中文反向词表，用于预测
vocab_cn_reverse={}
for key, value in vocab_cn.items():
    vocab_cn_reverse[value] = key
with open('./word2vec/vocab_cn_reverse.json','w',encoding='utf-8') as f:
    json.dump(vocab_cn_reverse,f,ensure_ascii=False)


#%% 根据词表构建索引，word2idx
# max_en:英文idx的最大长度   max_cn:中文idx的最大长度
en,cn = get_data_from_dict()     # 读取语料数据

#观察中英文对应句子长度




idxs_en,idxs_cn = [],[]
max_en,max_cn=0,0

for i in range(len(en)):
    idx_en = [vocab_en[word] for word in en[i].split()]
    idx_cn = [vocab_cn[word]  for word in jieba.lcut(cn[i].strip('\n'))]
    if max_en<len(idx_en):
        max_en=len(idx_en)
    if max_cn < len(idx_cn):
        max_cn=len(idx_cn)
    idxs_en.append(idx_en)
    idxs_cn.append(idx_cn)


#%% word2vec
# word2vec
import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset,DataLoader
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Word2Vec(nn.Module):     # model.embedding 可查看 embedding 输出
    def __init__(self,vocab, idxs, embedding_dim,batch_size=10240):
        super(Word2Vec, self).__init__()
        self.vocab = vocab
        self.idxs = idxs
        self.batch_size = batch_size
        self.embedding_layer = nn.Linear(1,embedding_dim)
        self.out = nn.Linear(embedding_dim,len(vocab))
        self.dataset = self.get_dataset()

    def forward(self,x):
        embedding_out = self.embedding_layer(x)
        out = self.out(embedding_out)
        self.embedding = embedding_out
        return out

    def get_dataset(self):
        trainset = []
        for texts in self.idxs:
            if len(texts) > 25:
                for i in range(1, len(texts) - 1, 2):
                    trainset.append([texts[i], texts[i - 1]])
                    trainset.append([texts[i], texts[i + 1]])
        center_word = [[float(w[0])] for w in trainset]
        background_word = [float(w[1]) for w in trainset]
        center_word = torch.from_numpy(np.array(center_word))
        background_word = torch.from_numpy(np.array(background_word))
        dataset = TensorDataset(center_word,background_word)
        dataset = DataLoader(dataset, batch_size=self.batch_size, drop_last=True)
        return dataset

def train(model,epochs=5):
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(),lr=1e-3)
    for epoch in tqdm(range(epochs)):
        print('正在训练第 {} 个 Epoch'.format(epoch))
        for x,y in model.dataset:
            x = x.type(torch.float).to(device)
            y = y.type(torch.LongTensor).to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
    return model


if __name__ == '__main__':
    model_en = Word2Vec(vocab_en,idxs_en,768,batch_size=512)
    model_en = train(model_en)
    torch.save(model_en,'./word2vec/word_embedding_en.pt')

    model_cn = Word2Vec(vocab_en,idxs_en,768,batch_size=512)
    model_cn = train(model_cn)
    torch.save(model_cn,'./word2vec/word_embedding_cn.pt')
