import sys
import torch
from torch import autograd
from torch import nn
from torch import optim

torch.manual_seed(1)

#%%
def argmax(vec):
    # return the argmax as a pytho n int
    # 返回 vec 的 dim为1维度上的最大索引
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    # 将句子转化为ID
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

#Compute log sum exp in a numerically stable way for the forward algorithm
# #前向算法是不断累积之前的结果，这样就会有个缺点
# 指数和累积到一定程度后，会超过计算机浮点值的最大值，变成inf，这样取log后也是inf
# 存为了避免这种情况，用一个合适的值c1ip去提指数和的公因子，这样就不会使某项变得过大而无法计算
# SUM=log（exp（s1）+exp（s2）+...+exp（s100））
# =logfexp（clip）*Texp（s1-clip）+exp（s2-clip）+..…+exp（s100-clip）J}
# =clip+loglexp（s1-clip）+exp（s2-clip）+...+exp（s100-clip）]
# where clip=max
# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

#%%
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim    # word emidbedding dim
        self.hidden_dim = hidden_dim          # Bi-LSTM hidden dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix            # 将tag转换成
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)     #双向的，多以上面的hidden_dim要除2

        # Maps the output of the LSTM into tag space.
        # 将Bi-LSTM提取的特征向量映射到特征空间，即全连接得到的发射分数
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        # 转移矩阵的参数初始化，transitions[i，j]代表的是从第i个tag转移到第i个tag的转移分数
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        # 初始化所有其他tag转移到START_TAG的分数非常小，即不可能由其他tag转移到START
        # TAG初始化srop_TAG转移到所有其他ag的分数非常小，即不可能由sTop_TAG转移到其他tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()   #函数调用，初始化LSTM的参数

    def init_hidden(self):
        # 初始化LSTM的参数
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        # 通过前向算法地推计算
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        #初始化step_0,即START位置的发射分数，START_TAG取0其他位置取 - 1000
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # 将初始化START位置为0的发射分数赋值给 forward_var
        forward_var = init_alphas

        # Iterate through the sentence
        # 迭代整个句子
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep（当前的前向tensor）
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of the previous tag
                # 取出当前tag的发射分数，与之前时间步的tag无关
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to next_tag from i
                # 取出当前tag由之前tag转移过来的转移分数
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the edge (i -> next_tag) before we do log-sum-exp
                # 当前路径的分数：之前时间步分数+转移分数+发射分数
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the scores.
                # 对当前分数取 log-sum-exp
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            # 更新 alphas_t 递推计算下一个时间步
            forward_var = torch.cat(alphas_t).view(1, -1)
        # 考虑最终转移到 STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        # 计算最终分数
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        # 通过Bi-LSTM提取特征
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)   #通过全连接层得到发射分数
        return lstm_feats

    def _score_sentence(self, feats, tags):   #输入feats是全脸基层输出的发射分数，tag就是路径的tag
        # Gives the score of a provided tag
        # 计算给定tag序列的分数，即一条路径的分数
        score = torch.zeros(1)   #初始化
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            # 递推计算路径分数：转移分数+发射分数
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]   #由最后一个tag转移到stop_tag的分数
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        # 初始化viterbi的previous变量
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step（保存当前时间步的回溯指针）
            viterbivars_t = []  # holds the viterbi variables for this step（保存当前时间步的viterbi变量）

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                # 维特比算法记录最优路径时只考虑上一步的分数以及上一步tag转移到当前tag的分数，并不取决于当前的发射分数                        //这里利用了HMM的思想
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            # 更新previous，加上当前tag来源前一步的tag
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)      #这一步得到了当前时间步每个tag的最优路径
            # 回溯指针记录当前时间步各tag来源前一步的tag
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        # 考虑转移到STOP_TAG的转移分数（从最后的STOP_TAG回溯，看由哪条路径转移过来是最大的）
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        # 通过回溯指针解码出最优路径
        best_path = [best_tag_id]
        # best_tag_id最为栈头，反向遍历bptrs_t找到最优路径
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()    #把路径信息犯规来，就得到从开始到最后的了
        return path_score, best_path     #返回：最优路径对应的分数  ，  最优路径

    def neg_log_likelihood(self, sentence, tags):    #实际就是CRFLoss
        # CRF损失函数由两部分组成，真实路径的分数和所有路径的总分数。
        # 真实路径的分数对应是所有路径中分数最高的。
        # log真实路径的分数/log所有可能路径的分数，越大越好，构造 crf loss 函数取反，loss越小越好
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        # 通过BiLSTM提取发射分数
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        # 根据发射分数、转移分数，及viterbi解码找到一条最优路径
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

#%%
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
# 语料数据集
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]
# 字典构建
word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
# label构建
tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
# 训练前见擦汗模型预测结果
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(
        300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!

#%%