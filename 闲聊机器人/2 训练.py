import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
import json
import numpy as np
from tqdm import tqdm
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start_token = '<S>'
end_token = '<E>'
pad_token = '<PAD>'
unk_token = '<UNK>'
# 词表
with open('vocab.json','r',encoding='utf-8') as f:
	vocab = json.load(f)
# 数据集
enc_inputs = np.load('enc_inputs.npy')
dec_inputs = np.load('dec_inputs.npy')
dec_outputs = np.load('dec_outputs.npy')

enc_inputs = torch.from_numpy(enc_inputs).type(torch.LongTensor)
dec_inputs = torch.from_numpy(dec_inputs).type(torch.LongTensor)
dec_outputs = torch.from_numpy(dec_outputs).type(torch.LongTensor)

# 封装
batch_size = 64
dataset = TensorDataset(enc_inputs,dec_inputs,dec_outputs)
dataset = DataLoader(dataset, batch_size=batch_size, drop_last=True)


#%% 模型
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

'''
def greedy_decoder(memory):  # memory: [src_len, batch, d_model]
	start_symbol = vocab[start_token]
	dec_input = torch.zeros((1, 20)).type(torch.LongTensor).to(device)
	next_symbol = start_symbol
	for i in range(dec_input.size(1) - 1):
		dec_input[0][i] = next_symbol
		dec_output = model.decoder(model.posemb(model.emb_cn(dec_input).transpose(0, 1)), memory)
		prob = model.output(dec_output)
		prob = torch.argmax(prob, dim=-1)
		next_word = prob.data[i]
		next_symbol = next_word.item()
	return dec_input.type(torch.LongTensor)
'''
def greedy_decoder(memory):
	start_symbol = vocab[start_token]
	dec_input = torch.zeros((batch_size, 20)).type(torch.LongTensor).to(device)
	next_symbol = start_symbol
	for i in range(dec_input.size(1)):
		dec_input[:,i] = next_symbol
		dec_output = model.decoder(model.posemb(model.emb_cn(dec_input).transpose(0, 1)), memory)
		prob = model.output(dec_output)
		prob = torch.argmax(prob, dim=0)
		next_word = prob[:,i]
		next_symbol = next_word
	return dec_input.type(torch.LongTensor)

class Transformer(nn.Module):
	def __init__(self, d_model):
		super(Transformer, self).__init__()
		self.emb_en = nn.Embedding(len(vocab), d_model)
		self.emb_cn = nn.Embedding(len(vocab), d_model)
		self.posemb = PositionalEncoding(d_model=d_model, dropout=0)
		self.enclayer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
		self.declayer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8)
		self.encoder = nn.TransformerEncoder(self.enclayer, num_layers=6)
		self.decoder = nn.TransformerDecoder(self.declayer, num_layers=6)
		self.output = nn.Linear(d_model, len(vocab))

	def forward(self, enc_input, dec_input):
		enc_input = self.emb_en(enc_input)  # enc_input: [batch, src_len, d_model]
		dec_input = self.emb_cn(dec_input)  # dec_input: [batch, tgt_len, d_model]
		enc_input = self.posemb(enc_input.transpose(0, 1))  # enc_input: [src_len, batch, d_model]
		dec_input = self.posemb(dec_input.transpose(0, 1))  # dec_input: [tgt_len, batch, d_model]
		memory = self.encoder(enc_input)  # memory: [src_len, batch, d_model]
		dec_output = self.decoder(dec_input, memory)  # dec_output: [tgt_len, batch, d_model]
		out = self.output(dec_output.transpose(0, 1))
		return out

	def get_memory(self, enc_input):
		enc_input = self.emb_en(enc_input)
		enc_input = self.posemb(enc_input.transpose(0, 1))
		memory = self.encoder(enc_input)
		return memory


#%%
model = Transformer(d_model=512).to(device)
optim = torch.optim.RMSprop(model.parameters(),lr=1e-5)
loss_fn = nn.CrossEntropyLoss()
epochs = 50

for epoch in range(epochs):
	loss_val = []
	loss_tmp = 0
	for enc_input,dec_input,dec_output in tqdm(dataset):
		enc_input = enc_input.to(device)
		dec_input = dec_input.to(device)
		dec_output = dec_output.to(device)

		memory = model.get_memory(enc_input)

		dec_input = greedy_decoder(memory).to(device)
		out = model(enc_input,dec_input)
		loss = 0
		'''
		for i in range(batch_size):
			if 0 in dec_output:
				index = dec_output[i].argmin()
				loss += loss_fn(out[i, :index + 1], dec_output[i, :index + 1])
			else:
				loss += loss_fn(out[i], dec_output[i])
		'''
		for i in range(batch_size):
			loss += loss_fn(out[i],dec_output[i])
		loss.backward()
		optim.step()
		optim.zero_grad()
		with torch.no_grad():
			loss_tmp += loss.cpu().numpy()
	loss_val.append(loss_tmp)
	print('Epoch:{}\t\tLoss:{:.4f}'.format(epoch,loss_tmp))

torch.save(model,'model.pt')