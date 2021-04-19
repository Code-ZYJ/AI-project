from word2vec.main import Word2Vec

import torch
from torch import nn
import torch.nn.functional as F


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