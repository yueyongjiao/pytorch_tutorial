#-*-coding:utf-8-*-
from __future__ import print_function
import torch as t
t.__version__
import numpy as np

print("hello world!")

x = t.Tensor(5,3)
print(x)
print("Everything is ok")
x = t.Tensor([[1,2],[3,4]])
print(x)
x = t.rand(5,3)
print(x)
print(x.size())
print(x.size()[1])
print(x.size(1))
y = t.rand(5,3)
print(x+y)
print(t.add(x,y))
#Tensor 转为 numpy
b = x.numpy()
print(b)
#numpy 转为 Tensor
a = np.ones(5)
b = t.from_numpy(a)
print(b)
