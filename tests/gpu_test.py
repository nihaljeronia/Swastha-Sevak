import torch

print(torch.cuda.is_available())
print(next(model.parameters()).is_cuda) 