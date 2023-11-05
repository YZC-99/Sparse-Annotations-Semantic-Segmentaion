import torch
import torch.nn as nn

'''
Batch Norm 有4个参数：
可学习：Beta,Gamma
不可学习：Moving Avg(mean)，Moving Avg(Var)

除此之外还有EMA超参数： alpha：momentum，默认为0.1

batch_norm = nn.BatchNorm2d(8) 这句执行后，batch_norm的
mean:tensor([0., 0., 0., 0., 0., 0., 0., 0.])
var:tensor([1., 1., 1., 1., 1., 1., 1., 1.])

输入数据input是一个(4,8,64,64)全为1的tensor，执行这一句：
batch_norm(input)后：
mean:tensor([0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000])
var:tensor([0.9000, 0.9000, 0.9000, 0.9000, 0.9000, 0.9000, 0.9000, 0.9000])

为什么要探究这个，因为论文：Semi-Supervised Semantic Segmentation with Pixel-Level Contrastive Learning from a Class-wise Memory Bank
使用了Teacher—Student模型，他的代码中，使用了标记数据和无标记数据对Teacher进行了无梯度的前馈，猜想是为了更新Batchnorm的参数
请问群里有做半监督的朋友吗？
我遇到一个问题想请教一下，对于Teacher-Student的半监督架构，我在看一篇论文源码的时候发现：
该论文将labeled_data和unlabeled_data在teacher处于训练模式下关闭梯度并进行前馈，teacher在推理pseudo label的时候是采用正常的验证模式。
但他的论文中并未提到关于为什么要把labeled_data和unlabeled_data馈入到处于训练模式的teacher。
朋友们有关于这个
'''

input = torch.ones(4,8,64,64)
batch_norm = nn.BatchNorm2d(8)
batch_norm = nn.Batch
print(batch_norm)
out = batch_norm(input)
out = batch_norm(input)
print(batch_norm)