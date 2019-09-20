import torch
import torch.nn as nn
import torch.nn.functional as F

t4d = torch.randn(2, 5, 3)
print(t4d)

# indices = torch.tensor([[0,0],[1,0]])
print(torch.gather(t4d, 1, torch.tensor([[0,1],[1]])))
print(torch.index_select(t4d, 1, torch.LongTensor([0])))