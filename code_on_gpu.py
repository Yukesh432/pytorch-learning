import torch

# var1= torch.FloatTensor([1., 2., 3., 4.])

var1= torch.FloatTensor([1., 2., 3., 4.]).cuda()

print(var1.device)