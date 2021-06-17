运行流程
===
# `1 语料处理.py`
在这个程序中对 `faq.json` 的语料数据进行了处理。利用 `jieba` 进行分词，`gensim.corpora.Dictionary` 构建中文词表，
选择最适合的截断长度等。最后把数据集通过 `numpy` 保存为`.npy` 文件。
# `2 训练.py`
项目框架为 `torch` 。载入词典和数据集后主要写了三个函数：位置编码、贪婪解码、Transformer模型。先前测试过，用 `enc_input` 和 `dec_input`进行
训练时，虽然模型下降速度很快，但是，事实用于预测时，`dec_input` 实际上是不知道的。而预测中实际上是利用模型的 `encoder` 层先对`enc_input`进行编码，
得到 `memory` 然后再根据这个 `memory` 和已知的 `start_token`(开始标记符)进行异步贪婪解码，得到一个`dec_input`然后再将 `enc_input`和 `dec_input`在输入到
模型中，得到每个词的权重分布，取最大的词进行输出。因此，我将训练和预测过程进行了统一。
# `predict.py`
这个函数加载了 `2. 训练.py` 训练的模型和 `1.语料处理.py` 的词表。对前面的模型和函数重新加载了一遍。然后用死循环实现不停的对话。
结果如下：
![](https://github.com/Code-ZYJ/AI-project/tree/main/%E9%97%B2%E8%81%8A%E6%9C%BA%E5%99%A8%E4%BA%BA/结果.png)
