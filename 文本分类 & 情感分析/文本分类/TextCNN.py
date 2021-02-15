import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import tensorflow.keras.optimizers as optim
import torch.utils.data as Data
print(tf.__version__)    # 2.3.0
print(np.__version__)    #1.16.2
#%% 自创语料
sentences = ['i love you','he loves me','she likes baseball','i hate you','sorry for that','this is awful']
labels = [1,1,1,0,0,0]

seqence_length = len(sentences[0])        #every sentences contains seqence_length(=3) words
num_classes = len(set(labels))  #分类问题 0 或 1

word_lsit=''.join(sentences).split()
vocab=list(set(word_lsit))
word2dix = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)  #词表长度

def make_data(sentences,labels):
    inputs,targets = [],[]
    for sen in sentences:
        inputs.append(sen)
    for out in labels:
        targets.append(out)
    return inputs,targets
input_batch,target_batch=make_data(sentences,labels)

# 这个地方我没按照教程里构建按词表
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
input_batch = [tokenizer.encode_plus(words,add_special_tokens=False) for words in input_batch ]
loader = []
for i in range(len(input_batch)):
    loader.append(input_batch[i]['input_ids'])
input_batch=np.array(loader)
labels=np.array(labels)

tf.compat.v1.disable_eager_execution()  #我也不知道为啥，这个地方加上就能运行了，意思时要关闭Eager模式
embedding_size = 2       # wordeb dim
inputs=tf.keras.Input(shape=(3,))
emb = layers.Embedding(input_dim = vocab_size,
                     output_dim = embedding_size)(inputs)
emb = tf.expand_dims(emb,-1)
conv = layers.Conv2D(3,(2,embedding_size),activation='relu')(emb)
fla = layers.Flatten()(conv)
outputs = layers.Dense(1,activation='sigmoid')(fla)

model = tf.keras.Model(inputs, outputs)
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam())
model.fit(input_batch,labels,epochs=5000)

test='i love money'
test_inputs = tokenizer.encode_plus(test,add_special_tokens=False).input_ids
test=[]
test.append(test_inputs)
model.predict(np.array(test))