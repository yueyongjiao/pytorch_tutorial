#-*-coding:utf-8-*-
import torch as t
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		#卷积层
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.conv1 = nn.Conv2d(6, 16, 5)
		#全连接层
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,10)

	def forward(self, x):
		# 卷积 -> 激活 -> 池化
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		# reshape，‘-1’表示自适应
		x = x.view(x.size()[0], -1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
net = Net()
print(net)

params = list(net.parameters())
print(len(params))

for name,parameters in net.named_parameters():
    print(name,':',parameters.size())

test = t.randn(1, 1, 4, 4)
print("test: ", test)
input = t.randn(1, 1, 32, 32)
out = Net(input)
print(out.size())

