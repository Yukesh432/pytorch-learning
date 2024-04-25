import torch
import torch.nn as nn

# defining embedding layer with n vocab size and m vector embedding
n= 5
m= 10

# here pytorch created a lookup table called "embedding" which contains
# n=5 number of rows and m=10 number of columns
# the values of these embedding vectors are initialized randomly from uniform distribution
#  i.e 
embedding= nn.Embedding(n,m)
print(embedding)