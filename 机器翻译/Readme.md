# 机器翻译程序运行流程：  
## 第一步
运行`corpus`文件目录中`get_data.py`将原始语料进行处理。得到中英文对应的字典`dict.json`
## 第二步  
运行`word2vec`文件目录中的`main.py`，通过自己构建的`Word2Vec`方法类，实现`Embedding`层的构建。同时构建词表，但是这里要注意的是，用`gensim`构建的词表有点问题，就是`<PAD>`的标签不是0，所以要对两个词表`vocab_en.json`和`vocab_cn.json`人为的改一下。
## 第三步  
运行`dataset`文件目录中的`tokenizer.py`,将文件转换成可及逆行训练的索引，保存为`ctokenizer_en.pkl`和`tokenizer_n.pkl`
## 第四步  
运行`train`文件目录下的`main.py`调用封装的`Seq2Seq`模型。并训练。  

  
   
####
__模型待改进的地方：__
* 没有使用异步序列生成，至使效果变差
* 为引入`Attention`机制
* 训练过程中，没有将`dec_input`引入，导致训练异常艰难
