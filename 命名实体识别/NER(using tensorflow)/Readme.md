### 命名实体识别
做这个之前我先看了一下`pytorch`的官方教程。官方教程里用的时`Bi-LSTM + CRF`的发方法。就算我再学习之前了解了有关`CRF`的知识，我还是觉得太复杂了！  
于是我取查了一下有没有一些官方封装的`CRF`可以直接使用，就找到了`Tensorflow`中有`tensorflow_addons`可以支持`CRF`的调用，于是我看了一下官方文档就自己写了一个基于`Tensorflow`的命名实体识别的流程。  
`model.py`里是我写的两个对象，分别是`NER`和`CRF`  

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

我在`用tensorflow做NER.py`里调用了这两个类，没有数据集，我就自己做了一个非常简单的取示意。