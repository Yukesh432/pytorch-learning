import torch
import numpy as np

'''For exaample i take a function y= x^2
The derivative(dy/dx) of it is y= 2x....backpropagation
'''
# x= torch.tensor(4.0, requires_grad=True)

# y= x**2
# y.backward()  #doing backpropagation i.e dy/dx= 2x


# print(x.grad)  #since x= 4, the gradient dy/dx= 2x at x= 4 is given by x.grad

print(40* "---")

lst= [[2.,3.,1.], [4.,5.,3.], [7.,6.,4.]]
torch_input= torch.tensor(lst, requires_grad=True)
y= torch_input**2
print(torch_input)

y= y.mean()
y.backward()
print(torch_input.grad)