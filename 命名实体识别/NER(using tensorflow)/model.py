import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

class NER(keras.models.Model):
    def __init__(self,vocab_size, embedding_dim, rnn_units, n_classes):
        # vocab_size为词表大小，embedding_dim为词向量维度，rnn_units要与类别数相等，n_classes为类别总数
        super(NER, self).__init__()
        rnn_units = rnn_units//2
        self.embedding = keras.layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim)   #最大此表述为5000，输出词向量维度为
        self.bilstm = keras.layers.Bidirectional(keras.layers.LSTM(rnn_units,return_sequences=True))
        self.dense = keras.layers.Dense(n_classes)

    def call(self,input):
        embedd = self.embedding(input)
        lstm_out = self.bilstm(embedd)
        logits = self.dense(lstm_out)
        return logits



class CRF(keras.models.Model):
    def __init__(self):
        super(CRF, self).__init__()

    def train(self,logits, target, input_length):
        input_length = tf.convert_to_tensor(input_length, dtype=tf.int32)
        log_likelihood, tran_paras = tfa.text.crf_log_likelihood(logits, target, input_length)
        pred_sequence, viterbi_score = tfa.text.crf_decode(logits, tran_paras, input_length)
        self.tran_paras=tran_paras
        loss = tf.reduce_sum(-log_likelihood)
        return loss

    def predict(self,logits, input_length):
        input_length = tf.convert_to_tensor(input_length, dtype=tf.int32)
        pred_sequence, viterbi_score = tfa.text.crf_decode(logits, self.tran_paras, input_length)
        return pred_sequence