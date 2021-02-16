import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
import pandas as pd
from transformers import AutoTokenizer,AutoModel
from tqdm import tqdm
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
# 下载数据集（没有git的话可以通过这个网址从github上下载，放到当前工作目录下即可）
#!git clone https://github.com/laxmimerit/IMDB-Movie-Reviews-Large-Dataset-50k.git


#%% 数据读取与处理
train_df = pd.read_excel('./IMDB-Movie-Reviews-Large-Dataset-50k/train.xlsx')
test_df = pd.read_excel('./IMDB-Movie-Reviews-Large-Dataset-50k/test.xlsx')
print(train_df.head(),'\n',train_df.shape,'\n')
print(test_df.head(),'\n',test_df.shape)
print(train_df.columns)
print(train_df['Sentiment'].unique())

# 获取并选择每个文本的长度
len_train = [len(str(train_df['Reviews'][i]).split()) for i in range(len(train_df))]
len_test = [len(str(test_df['Reviews'][i]).split()) for i in range(len(test_df))]
sns.set();plt.figure(figsize=(12,7))
plt.subplot(121); sns.distplot(len_train); plt.xlim(0,1200); plt.title("len of train's words")
plt.subplot(122); sns.distplot(len_test); plt.xlim(0,1200); plt.title("len of test's words")
max_len = 400
print(0.5*(np.mean(np.array(len_train)<max_len)+np.mean(np.array(len_test)<max_len)))
#当每句话的长度设置为 500 的时候，涵盖了 91.88% 的数据


#载入 tokenizer 标注器 和 bert 模型
PRETRAINED_MODEL_NAME = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

def preprocessing(df):
    input_ids,token_type_ids,attention_mask,labels=[],[],[],[]
    for i in tqdm(range(len(df)-1)):
        input_tokens = tokenizer.encode_plus( str(df['Reviews'][i]),  # 输入文本
                                              add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                                              max_length=max_len,  # 填充 & 截断长度
                                              pad_to_max_length=True,
                                              return_tensors='pt'  # 返回 pytorch tensors 格式的数据
                                             )
        input_ids.append(input_tokens['input_ids'])
        token_type_ids.append(input_tokens['token_type_ids'])
        attention_mask.append(input_tokens['attention_mask'])
        if df['Sentiment'][i]=='neg':
            labels.append(0)
        else:
            labels.append(1)
    input_ids = torch.cat(input_ids,dim=0)
    token_type_ids = torch.cat(token_type_ids,dim=0)
    attention_mask = torch.cat(attention_mask,dim=0)
    labels = torch.Tensor(labels).reshape(-1,1)
    return input_ids, token_type_ids, attention_mask, labels

# 获取数据
train_inputs,train_token_typeids,train_attenmask,train_labels = preprocessing(train_df)
test_inputs,test_token_typeids,test_attenmask,test_labels = preprocessing(test_df)

# 制作 Dataloader
train_data = TensorDataset(train_inputs, train_token_typeids, train_attenmask, train_labels)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=1,drop_last=True)
test_data = TensorDataset(test_inputs, test_token_typeids, test_attenmask, test_labels)
test_dataloader = DataLoader(test_data, shuffle=True, batch_size=1,drop_last=True)


#%% 定义模型
class Sentiment_cla(nn.Module):
    def __init__(self):
        super(Sentiment_cla, self).__init__()
        self.bert = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME)
        self.linear = nn.Linear(self.bert.config.hidden_size,1)   # 2分类

    def forward(self, input_ids, token_typeids, attention_mask):
        pooler_out = self.bert(input_ids, token_typeids, attention_mask)
        logits = self.linear(pooler_out[1])
        return F.sigmoid(logits)


#%% 模型训练
EPOCHS = 1
cla = Sentiment_cla().to(device)
loss_fn = nn.BCELoss()
optim = torch.optim.RMSprop(cla.parameters(),lr =2e-5)


for epoch in range(EPOCHS):
    for batch in tqdm(train_dataloader):
        input_ids=batch[0].to(device)
        token_typeids=batch[1].to(device)
        attention_mask=batch[2].to(device)
        y=batch[3].to(device)

        pred = cla(input_ids,token_typeids,attention_mask)
        loss = loss_fn(pred,y)
        loss.backward()
        optim.zero_grad()
        optim.step()
    print('第 {} 次训练的损失：{} '.format(epoch,loss))


#%% 预测单句
input = 'i hate this movie. it sucks! what a stupid man couold make this movie'
threshold = 0.5

def predict(input,threshold):
    input_tokens = tokenizer.encode_plus(input,  # 输入文本
                                         add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                                         max_length=max_len,  # 填充 & 截断长度
                                         pad_to_max_length=True,
                                         return_tensors='pt'  # 返回 pytorch tensors 格式的数据
                                         )
    input_ids = input_tokens['input_ids'].to(device)
    token_typeids = input_tokens['token_type_ids'].to(device)
    attention_mask = input_tokens['attention_mask'].to(device)
    pred = cla(input_ids,token_typeids,attention_mask)
    if pred>0.5:
        return 'Postive'
    else: return 'Negative'

predict(input)



#%% 模型评估
def eval(threshold=0.5):   #这个用gpu跑不了，cpu要11个小时
    TP,FN,FP,TN=0,0,0,0
    cla.to(device)
    for test in tqdm(test_dataloader):
        input_ids = test[0].to(device)
        token_typeids = test[1].to(device)
        attention_mask = test[2].to(device)
        y = test[3].to(device)
        pred = cla(input_ids, token_typeids, attention_mask)
        for i in zip(y,pred):
            if i == 1 and pred>threshold:   # 真实为真，预测为真
                TP += 1
            if i == 1 and pred<threshold:   # 真实为真，预测为假
                FN += 1
            if i == 0 and pred>threshold:   # 真实为假，预测为真
                FP += 1
            if i == 0 and pred<threshold:   # 真实为假，预测为假
                TN += 1
    confusion_matrix = [[TP,FN],[FP,TN]]
    accuracy = (TP+TN)/(TP+FN+FP+TN)
    precision = TP/(TP+FP)   # 在所有 预测 为真的样本中，预测准确的概率
    recall = TP/(TP+FN)      # 在所有 真实 为真的样本中，预测准确的概率
    F1 = 2*(precision*recall)/(precision+recall)
    print('混淆矩阵',confusion_matrix)
    print('准确率',accuracy)
    print('精确度',precision)
    print('召回率',recall)
    print('F1_score：',F1)

    # 绘制ROC曲线、计算AUC
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()

eval()