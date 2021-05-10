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

enc_input = np.zeros((len(en),max_len_en))
dec_input = np.zeros((len(cn),max_len_cn + 1))
dec_output= np.zeros((len(cn),max_len_cn + 1))

dec_input[:,0] = vocab_cn['<S>']

#构建英文的 enc_input
for i in range(len(en)):
    if len(idxs_en[i])<max_len_en:
        enc_input[i][:len(idxs_en[i])] = idxs_en[i][:len(idxs_en[i])]
    else:
        enc_input[i] = idxs_en[i][:max_len_en]

#构建中文的 dec_input
for i in range(len(cn)):
    if len(idxs_cn[i])<max_len_cn:
        dec_input[i][1:len(idxs_cn[i])] = idxs_cn[i][:len(idxs_cn[i])-1]
    else:
        dec_input[i][1:max_len_cn] = idxs_cn[i][1:max_len_cn]

#构建中文的 dec_output
for i in range(len(cn)):
    if len(idxs_cn[i])<max_len_cn:
        dec_output[i][:len(idxs_cn[i])] = idxs_cn[i][:len(idxs_cn[i])]
        dec_output[i][len(idxs_cn[i])] = vocab_cn['<E>']
    else:
        dec_output[i][1:max_len_cn] = idxs_cn[i][1:max_len_cn]
        dec_output[i][-1] = vocab_cn['<E>']

enc_input = np.array(enc_input)
dec_input = np.array(dec_input)
dec_output = np.array(dec_output)

enc_input = torch.from_numpy(enc_input).type(torch.LongTensor)
dec_input = torch.from_numpy(dec_input).type(torch.LongTensor)
dec_output = torch.from_numpy(dec_output).type(torch.LongTensor)

torch.save(enc_input, './dataset/enc_input.pkl')
torch.save(dec_input, './dataset/dec_input.pkl')
torch.save(dec_output, './dataset/dec_output.pkl')