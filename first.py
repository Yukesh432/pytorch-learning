import torch.nn as nn
import torch.nn.functional as F
import torch

# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1= nn.Conv2d(1,20,5)
#         self.conv2= nn.Conv2d(20,20,5)

#     def forward(self, x):
#         x= F.relu(self.conv1(x))
#         return F.relu(self.conv2(x))
    

# @torch.no_grad()
# def init_weights(m):
#     print("........................")
#     print(m)
#     print("................................")
#     if type(m)== nn.Linear:
#         m.weight.fill_(1.0)
#         # print(".................................")
#         print(m.weight)
#         print("....................................")

# net= nn.Sequential(nn.Linear(2,2), nn.Linear(2,2))

# net.apply(init_weights)


l= nn.Linear(7,5)
k= nn.Linear(5,1)
net= nn.Sequential(l,l,k)

for idx, m in enumerate(net.modules()):
    print(idx, '->', m)
    print(type(net.modules))
