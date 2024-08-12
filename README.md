

# **基于深度学习的句法依存分析模型**

应用技术：Python、Tensorflow、Glove。

项目描述：使用Python实现的句法依存工具。

项目特点：

* 使用Glove对数据集中的语料进行预处理。
* 使用Tensorflow建立Bilstm、MLP、Biaffine对文本中的单个句子进行句法依存分析进行建模。
* 该模型在英文数据集Conll上的UAS可以达到93.79%，LAS可以达到92.67%。
* 该模型在中文数据集CTB8.0上的UAS可以达到86.26%，LAS可以达到84.32%，优于“A Graph-based Model for Joint Chinese Word Segmentation and Dependency Parsing”中提到的模型。

# Reference

[代码](https://github.com/dwdb/dependency-parser)、[论文](https://arxiv.org/abs/1611.01734)参考链接。
