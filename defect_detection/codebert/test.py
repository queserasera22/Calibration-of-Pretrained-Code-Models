import torch

a = torch.tensor([[0.6,0.4],[0.2, 0.7]])

print(a)

print(a.argmax(dim=1))


print(a[:, 0] > 0.5)

b = a[:, 0]
print(b > 0.5)

c = (b > 0.5) * b + (b <= 0.5) * (1 - b)

print(c)