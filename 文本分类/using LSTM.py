import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

#%%  读取数据
data=pd.read_csv('Tweets.csv')
data=data[['airline_sentiment','text']]  #提取这两列
print(data.airline_sentiment.unique())  #查看总共有哪些评价
print(data.airline_sentiment.value_counts())  #查看三种评价的计数值
data_p=data[data.airline_sentiment=='positive']  # positive的数据
data_n=data[data.airline_sentiment=='negative']  # negative的数据
data_n=data_n.iloc[:len(data_p)]    #让样本均衡
data=pd.concat([data_n,data_p])     #pos与neg各一半
data=data.sample(len(data))         #乱序
data['review']=(data.airline_sentiment=='positive').astype('int')  #用0与1来代替neg与pos
del data['airline_sentiment']  #删掉这无用的一列

label=data['review']
#%%  tf.keras.layers.Embedding 把文本向量化
# 指标来单词和常见的标点符号
import re
token=re.compile('[A-Za-z]+|[!?,.()]')   #利用re编写token

def reg_text(text):         #规范化文本，保留英文和字符，全部设置为小写
    new_text=token.findall(text)
    new_text=[word.lower() for word in new_text]  #小写（其实 直接new_text.lower()就可以了）
    return new_text

data=data.text.apply(reg_text)  # pandas 的 apply() 函数  相当于 => reg_text(data.text)  apply()里面必须是函数
data=pd.DataFrame(data)
data['review']=label
#把所有单词放到 word_set 中
word_set=set()
for text in data.text:
    for word in text:
        word_set.add(word)
print(len(word_set))

word_list=list(word_set)   # 列表有索引，
print(word_list.index('spending'))

word_index=dict((word,word_list.index(word) + 1 ) for word in word_list)  # 构建字典(从1开始)

#已经构建好索引字典后通过该字典将问Bern转换成数值
#利用apply方法应用正则表达： 将已有的单词读取为数字，不在列表中的变为0
data_ok = data.text.apply(lambda x: [word_index.get(word,0) for word in x])

max_len = max(len(x) for x in data_ok)  #最长的评论
max_word = len(word_set)+1     #一共有多少个词语在字典中

data_ok=keras.preprocessing.sequence.pad_sequences(data_ok.values, maxlen=max_len)    #填充pad
print(data.review.values)  #训练集输出
#%% 模型
inputs = keras.Input(40)
x=keras.layers.Embedding(input_dim=max_word,output_dim=500,input_length=max_len)(inputs)
#x=tf.transpose(x,perm=(0,2,1))
x=keras.layers.LSTM(64)(x)
outputs=keras.layers.Dense(1,activation='sigmoid')(x)
model = keras.Model(inputs=inputs,outputs=outputs)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

history=model.fit(data_ok, data.review.values,
                  epochs=100,
                  batch_size=128,
                  validation_split=0.2  #将数据的20%切分出去作测试数据
                  )
