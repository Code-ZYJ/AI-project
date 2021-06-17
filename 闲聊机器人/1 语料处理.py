import json

with open('faq.json','r',encoding='UTF-8') as f:
	faq = json.load(f)

# 看看语料
question = faq[1]['post']
answer = faq[1]['replies'][0]['content']

question,answer = [],[]
for i in range(len(faq)):
	question.append(faq[i]['post'])
	answer.append(faq[i]['replies'][0]['content'])

#%% jieba分词
import jieba
from gensim.corpora import Dictionary
import matplotlib.pyplot as plt
import seaborn as sns

start_token = '<S>'
end_token = '<E>'
pad_token = '<PAD>'
unk_token = '<UNK>'

enc_input = [jieba.lcut(i) for i in question]
for enc in enc_input:
	enc.insert(0,start_token)
	enc.append(end_token)

dec_input = [jieba.lcut(i) for i in answer]
dec_output = [dec.copy() for dec in dec_input]     #二维列表的深拷贝

for dec in dec_input:
	dec.insert(0,start_token)

for dec in dec_output:
	dec.append(end_token)


# 构建词表
special_token = [[pad_token, unk_token]]
dct = Dictionary(special_token)
dct.add_documents(enc_input)
dct.add_documents(dec_input)
dct = dct.token2id
with open('vocab.json','w',encoding='utf-8') as f:
	json.dump(dct,f,ensure_ascii=False)


# 选择截断长度
length_enc = [len(i) for i in enc_input]
length_dec = [len(i) for i in dec_input]
sns.distplot(length_enc, label = 'enc_input')
sns.distplot(length_dec, label = 'dec_input')
plt.legend()
plt.show()

lenght_cut = 20


#%% 构建数据集
import numpy as np
enc_inputs = np.zeros((len(enc_input), lenght_cut))
dec_inputs = np.zeros((len(dec_input), lenght_cut))
dec_outputs = np.zeros((len(dec_output), lenght_cut))

#载入词典
with open('vocab.json','r',encoding='utf-8') as f:
	vocab = json.load(f)

# 构建数据集函数
def make_data(enc_inputs,enc_input):
	'''
	enc_inputs：需要得到的数据集
	enc_input: 用于word2idx的预料数据
	'''
	for i in range(len(enc_input)):
		for j,enc in enumerate(enc_input[i]):
			enc_inputs[i,j] = vocab[enc]
	return enc_inputs

enc_inputs = make_data(enc_inputs,enc_input)
dec_inputs = make_data(dec_inputs,dec_input)
dec_outputs = make_data(dec_outputs,dec_output)

# 检验处理过程有没有问题
[len(dec_inputs) == len(dec_outputs) for i in range(len(dec_input))].count(False)

# 保存数据
np.save('enc_inputs.npy',enc_inputs)
np.save('dec_inputs.npy',dec_inputs)
np.save('dec_outputs.npy',dec_outputs)