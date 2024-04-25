import torch
import torch.nn as nn

# defining embedding layer with n vocab size and m vector embedding
n= 5
m= 10

embedding= nn.Embedding(n,m)
print(embedding)