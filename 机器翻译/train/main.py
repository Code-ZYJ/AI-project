import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from train.model import Transformer,PositionalEncoding
import json
from tqdm import tqdm
import numpy as np
import math
#from word2vec.main import Word2Vec

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('./word2vec/vocab_en.json','r',encoding='utf-8') as f:  #中文词表
    vocab_en = json.load(f)
with open('./word2vec/vocab_cn.json','r',encoding='utf-8') as f:  #中文词表
    vocab_cn = json.load(f)
with open('./word2vec/vocab_cn_reverse.json','r',encoding='utf-8') as f:
    vocab_cn_reverse = json.load(f)

# 数据集
enc_input = torch.load('./dataset/enc_input.pkl')
dec_input = torch.load('./dataset/dec_input.pkl')
dec_output = torch.load('./dataset/dec_output.pkl')
enc_input = enc_input[:100]
dec_input = dec_input[:100]
dec_output = dec_output[:100]

# 批处理封装
batch_size = 5
dataset = TensorDataset(enc_input, dec_input, dec_output)
dataset = DataLoader(dataset,batch_size=batch_size,drop_last=True)



class Transformer(nn.Module):
    def __init__(self,d_model):
        super(Transformer, self).__init__()
        self.emb_en = nn.Embedding(len(vocab_en), d_model)
        self.emb_cn = nn.Embedding(len(vocab_en), d_model)
        self.posemb = PositionalEncoding(d_model=d_model, dropout=0)
        self.enclayer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
        self.declayer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8)
        self.encoder = nn.TransformerEncoder(self.enclayer, num_layers=6)
        self.decoder = nn.TransformerDecoder(self.declayer, num_layers=6)
        self.output = nn.Linear(d_model, len(vocab_cn))
    def forward(self, enc_input, dec_input):
        enc_input = self.emb_en(enc_input)    # enc_input: [batch, src_len, d_model]
        dec_input = self.emb_cn(dec_input)    # dec_input: [batch, tgt_len, d_model]
        enc_input = self.posemb(enc_input.transpose(0,1))  # enc_input: [src_len, batch, d_model]
        dec_input = self.posemb(dec_input.transpose(0,1))  # dec_input: [tgt_len, batch, d_model]
        memory = self.encoder(enc_input)      # memory: [src_len, batch, d_model]
        dec_output = self.decoder(dec_input, memory)    # dec_output: [tgt_len, batch, d_model]
        out = self.output(dec_output.transpose(0,1))
        return out
    def get_memory(self,enc_input):
        enc_input = self.emb_en(enc_input)
        enc_input = self.posemb(enc_input.transpose(0, 1))
        memory = self.encoder(enc_input)
        return memory

#%% 训练
model = Transformer(512).to(device)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(),lr=2e-5)
LOSS=[]

for epoch in range(1000):
    for enc_input,dec_input,dec_output in tqdm(dataset):
        enc_input = enc_input.to(device)
        dec_input = dec_input.to(device)
        dec_output = dec_output.to(device)

        out = model(enc_input,dec_input)

        loss=0
        for i in range(batch_size):
            if 0 in dec_output:
                index = dec_output[i].argmin()
                loss += loss_fn(out[i,:index+1],dec_output[i,:index+1])
            else:
                loss += loss_fn(out[i],dec_output[i])
        #loss/=batch_size
        loss.backward()
        optim.step()
        optim.zero_grad()
        with torch.no_grad():
            LOSS.append(loss.cpu())
    print('epo:{}---loss:{:4f}'.format(epoch,loss/batch_size))




#%% 测试

def greedy_decoder(memory):    # memory: [src_len, batch, d_model]
    start_symbol = vocab_cn['<S>']
    dec_input = torch.zeros((1, 81)).type(torch.LongTensor).to(device)
    next_symbol = start_symbol
    for i in range(dec_input.size(1)-1):
        dec_input[0][i] = next_symbol
        dec_output = model.decoder(model.posemb(model.emb_cn(dec_input).transpose(0,1)),memory)
        prob = model.output(dec_output)
        prob = torch.argmax(prob,dim=-1)
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input.type(torch.LongTensor)



def predict(texts):
    model.eval()
    texts = texts.split(' ')
    idxs_en = [vocab_en[text] for text in texts]
    enc_input = np.zeros((1,60))
    try:
        enc_input[:,:len(idxs_en)] = idxs_en
    except:
        print('请将输入的词语控制在60以内')
    enc_input = torch.from_numpy(enc_input).type(torch.LongTensor).to(device)
    memory = model.get_memory(enc_input)
    dec_input = greedy_decoder(memory).to(device)
    out = model(enc_input,dec_input).squeeze(0)
    out = torch.argmax(out,dim=-1)
    with torch.no_grad():
        out = out.cpu()
    res = [vocab_cn_reverse[str(o)] for o in out.numpy()]
    return ''.join(res)



texts = 'PARIS'
predict(texts)





#%% 自己用来测试的

for i in range(1):
    for enc_input,dec_input,dec_output in tqdm(dataset):
        enc_input = enc_input.to(device)
        dec_input = dec_input.to(device)
        dec_output = dec_output.to(device)


for epo in range(270):
    out = model(enc_input, dec_input)
    loss = 0
    '''
    for i in range(batch_size):
        if 0 in dec_output:
            index = dec_output[i].argmin()
            loss += loss_fn(out[i, :index + 1], dec_output[i, :index + 1])
        else:
            loss += loss_fn(out[i], dec_output[i])
    '''
    for i in range(2):
            loss += loss_fn(out[i], dec_output[i])

    loss.backward()
    optim.step()
    optim.zero_grad()
    with torch.no_grad():
        LOSS.append(loss)
        print('epoch',epo,'loss',loss/batch_size)



with torch.no_grad():
    real_out = dec_output.cpu()
    ans = torch.argmax(out[0],dim=-1).cpu()
real_out = np.array(real_out)
ans = np.array(ans)
print(''.join([vocab_cn_reverse[str(i)] for i in real_out[0]]))
print(''.join([vocab_cn_reverse[str(i)] for i in ans]))

for i in range(batch_size):
    print(''.join([vocab_cn_reverse[str(i)] for i in real_out[i]]))