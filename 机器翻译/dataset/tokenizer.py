#%% 找到并选取最大长度
from corpus.get_data import get_data_from_dict
import jieba
import seaborn as sns
import numpy as np
import json
import torch
en,cn = get_data_from_dict()     # 读取语料数据

en = [i.split() for i in en]
cn = [jieba.lcut(i.strip('\n')) for i in cn]

len_en = [len(e) for e in en]
len_cn = [len(c) for c in cn]

sns.distplot(len_en)   #管擦和中英文序列长度的分布
sns.distplot(len_cn)

max_len_en = 60
max_len_cn = 80

# 载入词表
with open('./word2vec/vocab_en.json', 'r', encoding='utf-8') as f:
    vocab_en = json.load(f)
with open('./word2vec/vocab_cn.json', 'r', encoding='utf-8') as f:
    vocab_cn = json.load(f)

# 根据词表转成 idxs
idxs_en,idxs_cn = [],[]
for i in range(len(en)):
    idx_en = [vocab_en[word] for word in en[i]]
    idx_cn = [vocab_cn[word]  for word in cn[i]]
    idxs_en.append(idx_en)
    idxs_cn.append(idx_cn)

tokenizer_en = np.zeros((len(en),max_len_en + 2))
tokenizer_cn = np.zeros((len(en),max_len_cn + 2))
tokenizer_en[:,0] = vocab_en['<S>']
tokenizer_cn[:,0] = vocab_cn['<S>']

#构建英文的 tokenizer
for i in range(len(en)):
    if len(idxs_en[i])<max_len_en:
        tokenizer_en[i][1:len(idxs_en[i])] = idxs_en[i][:len(idxs_en[i])-1]
        tokenizer_en[i][len(idxs_en[i])] = vocab_en['<E>']     #末尾加 '<E>'
    else:
        tokenizer_en[i][1:max_len_en] = idxs_en[i][:max_len_en-1]
        tokenizer_en[i][max_len_en-1] = vocab_en['<E>']

#构建中文的 tokenizer
for i in range(len(cn)):
    if len(idxs_cn[i])<max_len_cn:
        tokenizer_cn[i][1:len(idxs_cn[i])] = idxs_cn[i][:len(idxs_cn[i])-1]
        tokenizer_cn[i][len(idxs_cn[i])] = vocab_cn['<E>']
    else:
        tokenizer_cn[i][1:max_len_cn] = idxs_cn[i][:max_len_cn-1]
        tokenizer_cn[i][max_len_cn-1] = vocab_cn['<E>']

tokenizer_en = torch.from_numpy(tokenizer_en).type(torch.LongTensor)
tokenizer_cn = torch.from_numpy(tokenizer_cn).type(torch.LongTensor)

torch.save(tokenizer_en, './dataset/tokenizer_en.pkl')
torch.save(tokenizer_cn, './dataset/tokenizer_cn.pkl')