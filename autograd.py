'''For exaample i take a function y= x^2
The derivative(dy/dx) of it is y= 2x....backpropagation

x= torch.tensor(4.0, requires_grad=True)

y= x**2
y.backward()  #doing backpropagation i.e dy/dx= 2x


print(x.grad)  #since x= 4, the gradient dy/dx= 2x at x= 4 is given by x.grad

References: https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
'''
import torch
import numpy as np

# Simple one-layer neural network with input x, parameter w and b , and loss function.

x= torch.ones(5)   #input tensor
y= torch.zeros(3)  #expected output

w= torch.randn(5,3, requires_grad= True)

b= torch.randn(3, requires_grad=True)

z= torch.matmul(x, w) +b

loss= torch.nn.functional.binary_cross_entropy_with_logits(z, y)


# In this network, w and b are parameters, which we need to optimize. Thus, we need to be able
# to compute the gradients of loss function with respect to those variables. In order to do that,
# we set the requires_grad property of those tensors.

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss= {loss.grad_fn}")

# Computing gradients
# To optimize weights of the paramenters w, and b, we need to compute the derivative of loss function w.r.t. parameters w & b
# wee need d(loss)/dw and d(loss)/db under fixed values of x and y. 
# To compute derivates we call .backward() function and retrieve w.grad and b.grad

loss.backward()
print(w.grad)
print(b.grad)

'''
# Note:: 
a. We can only obtain grad properties for the leaf nodes of the computational graph, which have "requres_grad= True". 
   For all other nodes in out graph, gradients will not be available 
b. We can only perform gradient calculations using backward once on a given graph, for performance reasons. If we
   need to do several backward calls on the same graph, we need to pass retain_graph=True to the backward call.
   '''

# Disabling Gradient Tracking

# there are cases when we dont have to use gradient computation, like in case of Forward propagation...
# we have trained out model and just want to apply to some input data.
# We can stop tracking computation by surrounding our computation code with torch.no_grad()

z= torch.matmul(x, w) +b
print(x.requires_grad)

with torch.no_grad():
    z= torch.matmul(x, w)+b

print(z.requires_grad)

#Why do we wnat to disable gradient tracking??
# To mark some parameters in our neural network as frozen parameters
# To spped up computations when you are only doing forward pass, because computation on tensors
# that do not track gradients would be more efficinet























# print(40* "---")

# lst= [[2.,3.,1.], [4.,5.,3.], [7.,6.,4.]]
# torch_input= torch.tensor(lst, requires_grad=True)
# y= torch_input**2
# print(torch_input)

# y= y.mean()
# y.backward()
# print(torch_input.grad)