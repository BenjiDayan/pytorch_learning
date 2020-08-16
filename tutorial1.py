# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 20:54:56 2020

@author: benja
"""

import torch

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

a = torch.randn(2,2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
b = (a * a).sum()

out.backward() # equivalent to out.backward(torch.tensor(1.))
print(x.grad) # this is d out / d x


x = torch.randn(3, requires_grad=True)
y = x* 2
while y.data.norm() < 1000:
    y = y * 2
    
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)