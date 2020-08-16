# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 21:58:06 2020

@author: benja
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        #input 32 x 32
        self.conv1 = nn.Conv2d(1, 6, 3) #1 input channel, 6 output, 3x3 square convolution
        # 28 x 28 -> subsampling 14 x 14
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 10 x 10 -> subsampling 5 x 5 ??????
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2)) #Max pooling with (2,2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
net = Net()
params = list(net.parameters())
my_input = torch.randn(1, 1, 32, 32)
out = net(my_input)
print(out)

target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(out, target)
print(loss)

net.zero_grad()

#print('conv1.bias.grad before backward')
#print(net.conv1.bias.grad)
#
#loss.backward()
#
#print('conv1.bias.grad after backward')
#print(net.conv1.bias.grad)
#
#learning_rate = 0.01
#for f in net.parameters():
#    f.data.sub_(f.grad.data * learning_rate)
    
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
loss.backward()
optimizer.step()

