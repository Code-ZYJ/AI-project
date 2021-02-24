# AI-project
我自己做过的一些AI项目
  
    
    
## 项目介绍



### 研究生期间与课题相关的研究
1. [故障定位：](https://github.com/Code-ZYJ/AI-project/tree/main/%E6%95%85%E9%9A%9C%E5%AE%9A%E4%BD%8D)   
  有关于变压器绕组的故障类型与故障位置的诊断。在这个项目中，应用了`深层的卷积神经网络模型`。由于对故障位置的判断难度大于故障类型，因此对于故障定位模型，其参数量非常大。项目中，`L2正则项`与`回调函数`的设置用来是的模型达到一个比较好的诊断效果。但是没有应用`Dropout`，虽然`Dropout`的设置能有效降低过拟合风险，但对于模型的拟合难度也加大了。**目前该项目以发表了一篇发明专利（实审中）以及一篇论文待发表**。
2. [故障类型及程度诊断](https://github.com/Code-ZYJ/AI-project/tree/main/%E6%95%85%E9%9A%9C%E7%B1%BB%E5%9E%8B%E5%8F%8A%E7%A8%8B%E5%BA%A6%E8%AF%8A%E6%96%AD)  
  这是一篇在读研期间比较早期的研究。目的是诊断变压器绕组的故障类型与故障程度。在故障分类上采用了`决策树`；在故障程度诊断上我利用`Tensorflow`搭建了单隐含层的神经网络模型。调参后在训练集上拟合良好。为抑制过拟合现象，我设置了`Dropout`与`L2正则化`。`回调函数`设置了动态学习率使得模型达到较好拟合。**目前该文章正在SCIⅢ区审稿中**  
  
### [竞赛](https://github.com/Code-ZYJ/AI-project/tree/main/%E7%AB%9E%E8%B5%9B)
1. [公积金逾期预测](https://github.com/Code-ZYJ/AI-project/tree/main/%E7%AB%9E%E8%B5%9B/%E5%85%AC%E7%A7%AF%E9%87%91%E9%80%BE%E6%9C%9F%E9%A2%84%E6%B5%8B)  
  这是来源于DC竞赛的一个小比赛。这个比赛我将数据正太化，并提出了一部分数据最后用`DNN`进行诊断。效果不佳的原因在于没有预先做特征工程。
2. [教育系统学生分班预测](https://github.com/Code-ZYJ/AI-project/tree/main/%E7%AB%9E%E8%B5%9B/%E6%95%99%E8%82%B2%E7%B3%BB%E7%BB%9F%E5%AD%A6%E7%94%9F%E5%88%86%E7%8F%AD%E9%A2%84%E6%B5%8B)  
  这个主要是我用于练习`pytorch`的一个小练手，项目中做了`Label-Encoding`等预处理，以及一些可视化操作。
3. [螺母螺栓参评质量检测](https://github.com/Code-ZYJ/AI-project/tree/main/%E7%AB%9E%E8%B5%9B/%E8%9E%BA%E6%AF%8D%E8%9E%BA%E6%A0%93%E5%8F%82%E8%AF%84%E8%B4%A8%E9%87%8F%E6%A3%80%E6%B5%8B)  
  这个项目时对螺母螺栓图片的二分类。我用`VGG19`的卷积基做迁移学习省时省力，效果还不错。 
4.NLP中文预训练模型泛化能力挑战赛
  这是天池的一个比赛，比赛要求是对三个文档分别实现情感分析、文本分类和语义理解。我根据`baseline`从代码实现到`docker`成绩提交体验了全流程，对其改进正在进行中
  
###
