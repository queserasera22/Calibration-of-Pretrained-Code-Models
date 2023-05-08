import torch

a = torch.Tensor([[0,0,0,0,0,0],[1,1] ])

print(a)

print(a.view(-1, 4))
print(a)