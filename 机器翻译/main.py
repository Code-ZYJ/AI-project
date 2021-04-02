import jieba
from gensim.corpora import Dictionary
from corpus.get_data import get_data_from_dict

eng,cn = get_data_from_dict()     # 读取语料数据
len(eng) == len(cn)    #看看语料长度是否相等

eng = [i.strip('\n').split() for i in eng]
cn = [jieba.lcut(i.strip('\n')) for i in cn]


#%% 构建词表

#英文词表
special_token = [['<S>','<E>','<PAD>','<UNK>']]
dct_eng = Dictionary(special_token)
dct_eng.add_documents(eng)
vocab_eng = dct_eng.token2id

#中文词表
special_token = [['<S>','<E>','<PAD>','<UNK>']]
dct_cn = Dictionary(special_token)
dct_cn.add_documents((cn))
vocab_cn = dct_cn.token2id