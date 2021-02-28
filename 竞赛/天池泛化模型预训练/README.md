# 比赛全流程体验
NLP中文预训练模型泛化能力挑战赛

## 训练环境介绍

```
机器信息：NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2
pytorch 版本 1.6.0

机器信息：NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2
pytorch 版本 1.7.1
```

python依赖：
```
pip install transformers
```

## Docker安装（Ubutun）

命令行安装：
```
sudo apt install docker.io
```

验证：
```
docker info
```
![](https://tianchi-public.oss-cn-hangzhou.aliyuncs.com/public/files/forum/160658242933332501606582428585.png)


## 运行过程

1. 下载Bert全权重，下载 https://huggingface.co/bert-base-chinese/tree/main 下载config.json vocab.txt pytorch_model.bin，把这三个文件放进tianchi-multi-task-nlp/bert_pretrain_model文件夹下。

2. 下载比赛数据集，把三个数据集分别放进 `tianchi-multi-task-nlp/tianchi_datasets/数据集名字/` 下面：
  - OCEMOTION/total.csv: http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531841/OCEMOTION_train1128.csv
  - OCEMOTION/test.csv: http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531841/b/ocemotion_test_B.csv
  - TNEWS/total.csv: http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531841/TNEWS_train1128.csv
  - TNEWS/test.csv: http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531841/b/tnews_test_B.csv
  - OCNLI/total.csv: http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531841/OCNLI_train1128.csv
  - OCNLI/test.csv: http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531841/b/ocnli_test_B.csv

文件目录样例：
```
tianchi-multi-task-nlp/tianchi_datasets/OCNLI/total.csv
tianchi-multi-task-nlp/tianchi_datasets/OCNLI/test.csv
```

3. 分开训练集和验证集，默认验证集是各3000条数据，参数可以自己修改：
```
python ./generate_data.py
```
4. 训练模型，一个epoch：
```
python ./train.py
```
会保存验证集上平均f1分数最高的模型到 ./saved_best.pt

5. 用训练好的模型 ./saved_best.pt 生成结果：
```
python ./inference.py
```

6. 打包预测结果。
```
zip -r ./result.zip ./*.json
```
7. 生成Docker并进行提交，参考：https://tianchi.aliyun.com/competition/entrance/231759/tab/174
  - 创建云端镜像仓库：https://cr.console.aliyun.com/
  - 创建命名空间和镜像仓库；
  - 然后切换到`submission`文件夹下，执行下面命令；

  ```
  # 用于登录的用户名为阿里云账号全名，密码为开通服务时设置的密码。
  sudo docker login --username=xxx@mail.com registry.cn-hangzhou.aliyuncs.com

  # 使用本地Dockefile进行构建，使用创建仓库的【公网地址】
  # 如 docker build -t registry.cn-shenzhen.aliyuncs.com/test_for_tianchi/test_for_tianchi_submit:1.0 .
  docker build -t registry.cn-shenzhen.aliyuncs.com/test_for_tianchi/test_for_tianchi_submit:1.0 .
  ```

  输出构建过程：
  ```
  Sending build context to Docker daemon  18.94kB
  Step 1/4 : FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3
  ---> a4cc999cf2aa
  Step 2/4 : ADD . /
  ---> Using cache
  ---> b18fbb4425ef
  Step 3/4 : WORKDIR /
  ---> Using cache
  ---> f5fcc4ca5eca
  Step 4/4 : CMD ["sh", "run.sh"]
  ---> Using cache
  ---> ed0c4b0e545f
  Successfully built ed0c4b0e545f
  ```

  ```
  # ed0c4b0e545f 为镜像id，上面构建过程最后一行
  sudo docker taged0c4b0e545f registry.cn-shenzhen.aliyuncs.com/test_for_tianchi/test_for_tianchi_submit:1.0

  # 提交镜像到云端
  docker push registry.cn-shenzhen.aliyuncs.com/test_for_tianchi/test_for_tianchi_submit:1.0
  ```

8. [比赛提交页面](https://tianchi.aliyun.com/competition/entrance/531865/submission/723)，填写镜像路径+版本号，以及用户名和密码则可以完成提交。


## 比赛改进思路

1. 修改 calculate_loss.py 改变loss的计算方式，从平衡子任务难度以及各子任务类别样本不均匀入手；
2. 修改 net.py 改变模型的结构，加入attention层，或者其他层；
3. 使用 cleanlab 等工具对训练文本进行清洗；
4. 做文本数据增强，或者在预训练时候用其他数据集pretrain；
5. 对训练好的模型再在完整数据集（包括验证集和训练集）上用小的学习率训练一个epoch；
6. 调整bathSize和a_step，变更梯度累计的程度，当前是batchSize=16，a_step=16；
7. 用 chinese-roberta-wwm-ext 作为预训练模型；
