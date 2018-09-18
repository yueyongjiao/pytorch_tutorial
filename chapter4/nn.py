#-*-coding:utf-8-*-
import torch as t
from torch import nn
from torch.autograd import Variable as V
import numpy as np

class Linear(nn.Module): # 继承nn.Module
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__() # 等价于nn.Module.__init__(self)
        self.w = nn.Parameter(t.randn(in_features, out_features))
        self.b = nn.Parameter(t.randn(out_features))

    def forward(self, x):
        x = x.mm(self.w)  # x.@(self.w)
        print(x.size())
        return x + self.b.expand_as(x)


layer = Linear(4, 3)
input = V(t.randn(2, 4))
print(input.size())
output = layer(input)
print(output)

#
# for name, parameter in layer.named_parameters():
#     print(name, parameter)
#
# a = np.array([[1, 2], [3, 4], [5, 6]])
# b = np.array([[1, 2], [3, 4]])
# a1 = V(t.from_numpy(a))
# b1 = V(t.from_numpy(b))
#
# y = a1.mm(b1)
# print(y)
