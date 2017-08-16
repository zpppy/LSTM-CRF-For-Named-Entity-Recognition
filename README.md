# LSTM-CRF-For-Named-Entity-Recognition

本项目主要参考《Adversarial Learning for Neural Dialogue Generation》这篇论文，使用Bi-LSTM+CRF模型实现了中文实体命名识别，无需像传统CRF需要进行人为设定特征函数，特征选择等工作。
- 字向量表示阶段；每个字的represention是由两种方式的向量进行融合：①字向量，采用word2vec基于wiki中文语料库进行训练得到的字向量作为fine-tuning；②位置向量，采用jieba分词对每个句子进行分词，其中单个字用“0”表示，词的开头用“1”表示，词的中间用“2”表示，词的末尾用“3”表示。
- 模型构建阶段；采用lstm+crf的架构计算每种seq_tag的分数。分数由两部分组成：①由lstm计算得到的unary_score（一元分数）；②由crf层维护的binary_score（二元分数）；
- 模型训练阶段通过极大对数似然估计更新参数；预测阶段采用维特比算法求除最有可能的 sequence_tag。
- 结果；迭代50次后，模型的评价指标如下：accuracy:  98.53%; precision:  90.31%; recall:  90.09%; FB1:  90.20。（其实到42次的时候模型差不多已经收敛了）。

### 关于crf层的实现及其源码解析可以参考我的个人博客：
http://www.hackerfun.cn/admin/article/article/45/

模型训练：
python main.py --train=True --clean=True

模型预测：
python main.py
