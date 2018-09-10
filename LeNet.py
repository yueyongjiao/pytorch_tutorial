#-*-coding:utf-8-*-
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
	def __init__(self):
		# nn.Module子类的函数必须在构造函数中执行父类的构造函数
		# 下式等价于nn.Module.__init__(self)
		super(Net, self).__init__()

		# 卷积层 '1'表示输入图片为单通道, '6'表示输出通道数，'5'表示卷积核为5*5
		self.conv1 = nn.Conv2d(1, 6, 5)
		# 卷积层
		self.conv2 = nn.Conv2d(6, 16, 5)
		# 仿射层/全连接层，y = Wx + b
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

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
    print(name, ':', parameters.size())

input = t.randn(1, 1, 32, 32)
out = net(input)
print(out.size())


net.zero_grad()# 所有参数的梯度清零
out.backward(t.ones(1, 10))# 反向传播

output = net(input)
target = 1.0 * t.arange(0, 10).view(1, 10)
criterion = nn.MSELoss()
print("output:", output)
print(output.type())

print("target:", target)
print(target.type())
target = target.type(t.FloatTensor)
print(target.type())
loss = criterion(output, target)
print(loss)# loss是个scalar


# 运行.backward，观察调用之前和调用之后的grad
net.zero_grad()# 把net中所有可学习参数的梯度清零
print('反向传播之前 conv1.bias的梯度')
print(net.conv1.bias.grad)
loss.backward()
print('反向传播之后 conv1.bias的梯度')
print(net.conv1.bias.grad)
optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
