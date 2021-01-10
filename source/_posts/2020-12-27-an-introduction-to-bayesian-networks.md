title: An Introduction to Bayesian Networks
date: 2020-12-27 18:54:04
tags: Causal Inference, AI
---

冬日晴好, 下午看完了论文, 对Bayesian Network是什么有了系统的了解.论文是causalnex工具里提到的
Stephenson, Todd Andrew. An introduction to Bayesian network theory and usage. No. REP_WORK. IDIAP, 2000.

该论文主要论述了以下几点:
+ What is Bayesian network
+ Inference Bayesian network: junction tree algorithm
+ Learning Bayesian Network
+ Applications
	+ Automatic Speech Recognition: Dynamic Bayesian Network
	+ Computer troubleshooting
	+ Medical diagnosis


<!--more-->

# 1.What is Bayesian Network(BN)
A directed acyclic graph(DAG) with probability distribution for each variable
这是一个交叉领域, 涉及概率论和图论, 主要可以应用于因果推断, 其优势是:
+ 可以引入专家经验
+ 通过图结构化简联合概率分布求解

下图是Hackerman解释BN时用的信用欺诈网络:
![bn01](https://www.cyanny.com/2020/12/27/an-introduction-to-bayesian-networks/bn01.jpg)


# 2.Inference Bayesian Network
一个示例如下, 对欺诈模型进行条件概率求解时, 可借助BN进行化简, 这是一个离散变量的例子.
![bn02](https://www.cyanny.com/2020/12/27/an-introduction-to-bayesian-networks/bn02.jpg)

其他常见的推断方法包括
![bn03](https://www.cyanny.com/2020/12/27/an-introduction-to-bayesian-networks/bn03.jpg)

作者在论文中, 重点讲述了Junction Tree Method. 该算法通过将图进行Moralize和Triangulate转化为Join Tree进行推断
![bn04](https://www.cyanny.com/2020/12/27/an-introduction-to-bayesian-networks/bn04.jpg)


# 3.Learning Bayesian Network
需要关心如下四种场景
![bn05](https://www.cyanny.com/2020/12/27/an-introduction-to-bayesian-networks/bn05.jpg)


# 4.Applications
一个实用案例, 在windows95中, 采用了BN进行printer的异常检测
![bn06](https://www.cyanny.com/2020/12/27/an-introduction-to-bayesian-networks/bn06.jpg)


BN在目前的机器学习中, 应该是计算复杂度高, 应用范围不像深度学习这么广, 而因果推理上, 样本量可以不用很大, 会有不错的应用效果.



