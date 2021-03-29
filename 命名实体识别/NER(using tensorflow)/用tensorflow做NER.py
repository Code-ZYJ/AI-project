#%%
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np
import re


word_src = '工作室里放着笔记本电脑和刚买的手机，寝室里放着平板电脑'
target = 'P Pi Pi O O O E Ei Ei Ei Ei O O O O E Ei O P Pi O O O E Ei Ei Ei'
target = target.split(' ')
# P: 地名开头   Pi: 连续     E: 电子产品开头   Ei:电子产品    O: 其他

def build_vocab(word_src,special_token = True):
    word_src = re.sub('，', ' ', word_src)  # 替换标点符号
    vocab=dict()
    vocab_reverse = dict()
    if special_token:
        vocab[0] = '<PAD>' #补齐标签
        vocab[1] = '<S>'   #开始标签
        vocab[2] = '<E>'   #结束标签
        vocab[3] = '<UNK>' #未知标签
        for i,w in enumerate(set(word_src)):    #构建词典
            vocab[i+4] = w
        for key,value in vocab.items():
            vocab_reverse[value]=key
    if not special_token:
        for i,w in enumerate(set(word_src)):    #构建词典
            vocab[i] = w
        for key,value in vocab.items():
            vocab_reverse[value]=key
    return vocab, vocab_reverse

vocab,vocab_reverse = build_vocab(word_src)   #词表
ner_label = {0:'O', 1:'P', 2:'Pi', 3:'E', 4:'Ei'}
ner_label_reverse = {'O':0, 'P':1, 'Pi':2, 'E':3, 'Ei':4}

# 替换标签


#将输入处理为索引
word_idx=[]
for w in word_src:
    if w not in vocab_reverse:
        w='<UNK>'
    word_idx.append(vocab_reverse[w])
word_idx.insert(0, 1)
word_idx.append(2)
word_idx = np.array(word_idx)
#将标签处理为索引
label = [0]   #与<'S'>对应
for l in target:
    if l != ' ':
        label.append(ner_label_reverse[l])
label.append(0)   #与<'E'>对应
label = np.array(label)
input_length = len(word_idx)

print(len(word_idx)==len(label))
print(word_idx)
print(label)


# 封装数据
data = tf.data.Dataset.from_tensors((word_idx,label,input_length))
data = data.shuffle(1).batch(1)

#%% 训练
from model import NER,CRF
vocab_size = len(vocab)
embedding_dim = 32
rnn_units = 4
n_classes = 5

ner = NER(vocab_size,embedding_dim,rnn_units,n_classes)
crf = CRF()
optim = keras.optimizers.Adam(learning_rate=1e-3)

for i in range(100):
    for input,label,input_length in data:
        with tf.GradientTape() as t:
            logits = ner(input)
            loss = crf.train(logits,label,input_length)
        grads = t.gradient(loss,ner.trainable_variables)
        optim.apply_gradients(zip(grads,ner.trainable_variables))


# 预测
logits = ner(input)
crf.predict(logits,input_length)

print(crf)
print(label)