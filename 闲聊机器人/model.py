import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
import json
import numpy as np
from tqdm import tqdm
import math

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