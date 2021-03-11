# AI-project
我自己做过的一些AI项目
  
    
    
## 项目介绍



### 研究生期间与课题相关的研究
1. [故障定位：](https://github.com/Code-ZYJ/AI-project/tree/main/%E6%95%85%E9%9A%9C%E5%AE%9A%E4%BD%8D)   
  有关于变压器绕组的故障类型与故障位置的诊断。在这个项目中，应用了`深层的卷积神经网络模型`。由于对故障位置的判断难度大于故障类型，因此对于故障定位模型，其参数量非常大。项目中，`L2正则项`与`回调函数`的设置用来是的模型达到一个比较好的诊断效果。但是没有应用`Dropout`，虽然`Dropout`的设置能有效降低过拟合风险，但对于模型的拟合难度也加大了。**目前该项目以发表了一篇发明专利（实审中）以及一篇论文待发表。**
2. [故障类型及程度诊断](https://github.com/Code-ZYJ/AI-project/tree/main/%E6%95%85%E9%9A%9C%E7%B1%BB%E5%9E%8B%E5%8F%8A%E7%A8%8B%E5%BA%A6%E8%AF%8A%E6%96%AD)  
  这是一篇在读研期间比较早期的研究。目的是诊断变压器绕组的故障类型与故障程度。在故障分类上采用了`决策树`；在故障程度诊断上我利用`Tensorflow`搭建了单隐含层的神经网络模型。调参后在训练集上拟合良好。为抑制过拟合现象，我设置了`Dropout`与`L2正则化`。`回调函数`设置了动态学习率使得模型达到较好拟合。**该文章被SCIⅢ区《Energies》录取**  
3. [故障分类泛化模型](https://github.com/Code-ZYJ/AI-project/tree/main/%E6%95%85%E9%9A%9C%E5%88%86%E7%B1%BB%E6%B3%9B%E5%8C%96%E6%A8%A1%E5%9E%8B)  
  这是近期科研相关的研究，概括之，是一个多任务多分类的应用。根据对`Transformer模型`中`Encoder`部分进行参数与模型的部分更改，使其更能使用与FRA(频率响应法)数据相结合。利用硬共享方式与下层任务输出模型相连接。下层输出模型使用`TimDistributed`层对上层输出进行解码(NLP任务中通常对初始维度解码，后来我发现过拟合严重，才转用这种方式)，然后根据目标标签不同设置输出层。训练时使用梯度自调节，`Checkpoints`等方法……
  
### [竞赛](https://github.com/Code-ZYJ/AI-project/tree/main/%E7%AB%9E%E8%B5%9B)
1. [公积金逾期预测](https://github.com/Code-ZYJ/AI-project/tree/main/%E7%AB%9E%E8%B5%9B/%E5%85%AC%E7%A7%AF%E9%87%91%E9%80%BE%E6%9C%9F%E9%A2%84%E6%B5%8B)  
  这是来源于DC竞赛的一个小比赛。这个比赛我将数据正太化，并提出了一部分数据最后用`DNN`进行诊断。效果不佳的原因在于没有预先做特征工程。
2. [教育系统学生分班预测](https://github.com/Code-ZYJ/AI-project/tree/main/%E7%AB%9E%E8%B5%9B/%E6%95%99%E8%82%B2%E7%B3%BB%E7%BB%9F%E5%AD%A6%E7%94%9F%E5%88%86%E7%8F%AD%E9%A2%84%E6%B5%8B)  
  这个主要是我用于练习`pytorch`的一个小练手，项目中做了`Label-Encoding`等预处理，以及一些可视化操作。
3. [螺母螺栓参评质量检测](https://github.com/Code-ZYJ/AI-project/tree/main/%E7%AB%9E%E8%B5%9B/%E8%9E%BA%E6%AF%8D%E8%9E%BA%E6%A0%93%E5%8F%82%E8%AF%84%E8%B4%A8%E9%87%8F%E6%A3%80%E6%B5%8B)  
  这个项目时对螺母螺栓图片的二分类。我用`VGG19`的卷积基做迁移学习省时省力，效果还不错。  
4. NLP中文预训练模型泛化能力挑战赛  
  详细内容正在完善中······
  
### NLP相关  
1. [奶茶店自动起名](https://github.com/Code-ZYJ/AI-project/tree/main/GAN%E6%80%9D%E6%83%B3%E5%AE%9E%E7%8E%B0%E5%A5%B6%E8%8C%B6%E5%BA%97%E8%B5%B7%E5%90%8D)  
  对语料预处理、`GAN`以及文本生成的有一定的了解之后，我基于`Pytorch`实现了一个奶茶店自动起名的一个小案例。从百度上搜索到的奶茶店名利用`GAN`的思想和训练方式，结合文本生成方法实现了奶茶自动起名的案例。但由于预处理中未设置`CLS`标签，使得结果并不是很理想。  
2. [文本分类 & 情感分析](https://github.com/Code-ZYJ/AI-project/tree/main/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%20%26%20%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90)  
  [文本分类](https://github.com/Code-ZYJ/AI-project/tree/main/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%20%26%20%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB)中分别利用`CNN`、`LSTM`对`Tweets.csv`数据集做多分类(10分类)。对Youtube上基于`Pytorch`的`TextCNN`改写成`Tensorflow`实现。  
  [情感分析](https://github.com/Code-ZYJ/AI-project/tree/main/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%20%26%20%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90/%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90)中对IMDB-50k的影评数据集
利用`huggingface`的`Transformers`调用`Bert`预训练模型实现二分类。  
3. [对word2vec模型的归纳](https://github.com/Code-ZYJ/AI-project/tree/main/word2vec)  
  对B站上的`word2vec`模型的讲解进行了归纳总结，并给出了[对模型的看法与理解](https://mp.weixin.qq.com/s?__biz=Mzg5ODU1NDQ3OQ==&mid=2247483750&idx=1&sn=cb69c7568865b5dbd38098a966eef36a&chksm=c0618e66f7160770dd136038ba1c08c149d4caa6c9ca9a9dc319e73b19e40cea665dd7cb2a08&token=1138451613&lang=zh_CN#rd)
4. [seq2seq(Attention)](https://github.com/Code-ZYJ/AI-project/tree/main/seq2seq(Attention)%20%E5%BE%B7%E8%AF%AD---%E3%80%8B%E8%8B%B1%E8%AF%AD)  
  对异步序列到序列模型的整体流程，由内到外，由表及里进行了全面的剖析，从各分模型的输入输出，到各模型的内部数据流，再到训练异步训练过程给出了[详细的解析](https://mp.weixin.qq.com/s?__biz=Mzg5ODU1NDQ3OQ==&mid=2247483781&idx=1&sn=bbfef8670ce24b3c271003dc71ea3641&chksm=c0618e85f7160793cf7b6e855161ca8ddc95120f0c1c1e6968b9c7e5338c03d128b92475ba84&token=1138451613&lang=zh_CN#rd)  
