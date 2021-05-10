#from word2vec.main import Word2Vec

import torch
from torch import nn
import torch.nn.functional as F
import math
'''
class Encoder(nn.Module):
    def __init__(self,rnn_unit, embedding_en, using_word2vec=False):
        super(Encoder, self).__init__()
        self.using_word2vec = using_word2vec
        self.embedding_dim = 768
        self.w2v = embedding_en
        self.gru = nn.GRU(self.embedding_dim,rnn_unit,batch_first=True)
        self.embedding = nn.Embedding(130122,768)

    def forward(self,batch_data):
        batch = batch_data.size(0)
        self.batch_size = batch     #这个在后面的Decoder可能要用

        if self.using_word2vec:
            dataset = batch_data.view(-1,1)
            self.w2v(dataset.type(torch.float))
            emb_out = self.w2v.embedding
            emb_out = emb_out.view(batch,-1,self.embedding_dim)   #这里实现了利用自己的word2vec做embedding
        else:
            emb_out = self.embedding(batch)

        encoder_out, enc_hidden= self.gru(emb_out)
        return enc_hidden

class Decoder(nn.Module):
    def __init__(self,rnn_unit, len_vocab_en):
        super(Decoder, self).__init__()
        self.embedding_dim = 768
        self.gru = nn.GRU(self.embedding_dim, rnn_unit,batch_first = True)
        self.out = nn.Linear(rnn_unit,len_vocab_en)


    def forward(self, dec_input, enc_hidden):
        gru_out, hidden = self.gru(dec_input, enc_hidden)
        logits = self.out(gru_out)
        return logits



class Seq2Seq(nn.Module):
    def __init__(self,rnn_unit, embedding_en, len_vocab_en, using_word2vec=False):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(rnn_unit, embedding_en,using_word2vec=using_word2vec)
        self.decoder = Decoder(rnn_unit, len_vocab_en)

    def forward(self, batch_data, dec_input):
        enc_hidden = self.encoder(batch_data)
        logits = self.decoder(dec_input, enc_hidden)
        out = F.softmax(logits,dim=-1)
        return logits


'''



#%%
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


def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]



class Transformer(nn.Module):
    def __init__(self, vocab_size_en, vocab_size_cn, embedding_dim, nhead = 4, num_encoder_layers = 2):
        super(Transformer, self).__init__()
        '''
        默认参数设置 =>   注意力头数：16
                        encoder层数：6
                        embedding层数：512
        '''
        self.embedding_en = nn.Embedding(vocab_size_en, embedding_dim)
        self.embedding_cn = nn.Embedding(vocab_size_cn, embedding_dim)
        self.position_encoding = PositionalEncoding(d_model = embedding_dim, dropout=0.1, max_len=5000)
        self.transformer_model = nn.Transformer(nhead=nhead, num_encoder_layers=num_encoder_layers, d_model=embedding_dim)
        self.outlayer = nn.Linear(embedding_dim, vocab_size_cn)
    def forward(self, idxs_en, idxs_cn):   #idxs_en:[batch,src_len]
        src = self.embedding_en(idxs_en)
        tgt = self.embedding_cn(idxs_cn)
        src = self.position_encoding(src)
        tgt = self.position_encoding(tgt)
        # 交换了batch和src_len的顺序，这样才能被transformer_model识别
        src = src.transpose(0,1)             #[batch, src_len, embedding_dim] => [src_len, batch, embedding_dim]
        tgt = tgt.transpose(0,1)
        out = self.transformer_model(src,tgt)
        out = out.transpose(0,1)
        out = self.outlayer(out)
        return out



if __name__ == '__main__':
    self = Transformer(1000,1000,768)
    idxs_en = torch.randn([32,10]).type(torch.LongTensor) + 10
    idxs_cn = torch.randn([32,20]).type(torch.LongTensor) + 10
    out = self(idxs_en,idxs_cn)