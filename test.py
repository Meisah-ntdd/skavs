import torch
data = torch.randn(2, 2, 32000)
print(data.shape)
data = data.reshape(-1, data.size()[-1])
print(data.shape)