import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from train.model import Seq2Seq
import json
from tqdm import tqdm
import numpy as np
from word2vec.main import Word2Vec

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

embedding_en = torch.load('./word2vec/word_embedding_en.pt')  #载入训练好的embedding
embedding_cn = torch.load('./word2vec/word_embedding_cn.pt')   # 中文的感觉没必要引入

with open('./word2vec/vocab_en.json','r',encoding='utf-8') as f:  #中文词表
    vocab_en = json.load(f)
with open('./word2vec/vocab_cn.json','r',encoding='utf-8') as f:  #中文词表
    vocab_cn = json.load(f)
with open('./word2vec/vocab_cn_reverse.json','r',encoding='utf-8') as f:
    vocab_cn_reverse = json.load(f)

# 数据集
dataset_en = torch.load('./dataset/tokenizer_en.pkl')
dataset_cn = torch.load('./dataset/tokenizer_cn.pkl')

# 批处理封装
batch_size = 12
dataset = TensorDataset(dataset_en,dataset_cn)
dataset = DataLoader(dataset,batch_size=batch_size,drop_last=True)







def train(model,dataset,epochs,batch_size,device,embedding_dim = 768, max_len_cn = 82):
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(),lr=1e-3)
    dec_input = torch.zeros(batch_size, max_len_cn, embedding_dim).to(device)     #用于每次的decoder部分
    for epoch in range(epochs):
        for en,cn in tqdm(dataset):
            en = en.to(device)
            cn = cn.to(device)
            pred = model(en,dec_input)
            #计算损失
            loss = 0
            for i in range(max_len_cn):
                loss += loss_fn(pred[:,i,:], cn[:,i])
            optim.zero_grad()
            loss.backward()
            optim.step()
    return model


embedding_dim = 768
rnn_unit = 128
len_vocab_en = len(vocab_cn)
model = Seq2Seq(rnn_unit, embedding_en, len_vocab_en).to(device)
max_len_cn = 82

model = train(model,dataset,10,batch_size=batch_size,device=device)




#%% 测试
def predict(model,texts):
    max_len_en = 60
    max_len_cn = 80
    embedding_dim = 768
    #***************进行tokenizer****************
    words = [text.split() for text in texts]
    idxs=[]
    for word in words:
        idxs.append([vocab_en[w] for w in word])
    inputs_idxs = np.zeros((len(idxs),max_len_en+2))
    inputs_idxs[:,0] = vocab_en['<S>']
    for i in range(len(idxs)):
        if len(idxs[i]) < max_len_en:
            inputs_idxs[i][1:len(idxs[i])+1] = idxs[i]
            inputs_idxs[i][len(idxs[i])+1] = vocab_en['<E>']
        if len(idxs[i]) > max_len_en:
            inputs_idxs[i][1:61] = idxs[i][:60]
            inputs_idxs[i][61] = vocab_en['<E>']
    # *******************************************
    input = torch.from_numpy(inputs_idxs).type(torch.long)
    dec_input = torch.zeros(len(input), max_len_cn+2, embedding_dim)
    out = model.cpu()(input, dec_input)
    out = torch.argmax(out,dim=-1)
    out = np.array(out)
    trans_outs = []
    for mini_batch in out:
        trans_outs.append([vocab_cn_reverse[str(idx)] for idx in mini_batch])
    chinese = []
    for trans_out in trans_outs:
        if '<E>' in trans_out:
            chinese.append(''.join(trans_out[trans_out.index('<S>')+1: trans_out.index('<E>')]))
        else:
            chinese.append(''.join(trans_out[trans_out.index('<S>')+1:]))
    return chinese


texts = ['PARIS']
predict(model,texts)