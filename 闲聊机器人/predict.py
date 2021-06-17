import torch
from torch import nn
import json
import jieba
import numpy as np
from model import PositionalEncoding,Transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 导入模型
model = torch.load('model.pt')
# 加载词表
with open('vocab.json','r',encoding='utf-8') as f:
	vocab = json.load(f)
vocab_reverse = {}
for key,value in vocab.items():
	vocab_reverse[value]=key

# 特殊标记
start_token = '<S>'
end_token = '<E>'
pad_token = '<PAD>'
unk_token = '<UNK>'

batch_size = 1


def greedy_decoder(memory,bach_size = batch_size):
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


def get_enc_dec(word):
	input = [start_token] + jieba.lcut(word) + [end_token]
	input = np.array([vocab[i] for i in input])
	enc_input = np.zeros((1,20))
	if len(input)<20:
		enc_input[0][:len(input)] = input
	else:
		enc_input[0] = input[:20]
		enc_input[0][-1] = pad_token
	return torch.from_numpy(enc_input).type(torch.LongTensor)


model = model.to(device)
def predict(word):
	enc_input = get_enc_dec(word).to(device)
	memory = model.get_memory(enc_input).to(device)
	dec_input = greedy_decoder(memory,bach_size=1).to(device)
	out = model(enc_input, dec_input).squeeze(0)
	out = torch.argmax(out, dim=-1)
	with torch.no_grad():
		out = out.cpu()
	res = [vocab_reverse[o] for o in out.numpy()]
	res = [i for i in res if i not in [end_token,pad_token]]
	return ''.join(res)

word = '你在吗？'
predict(word)


while True:
	print('我：\t')
	word = input()
	print('机器人：')
	print(predict(word))