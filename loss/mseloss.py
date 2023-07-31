import torch
def mesloss(x,y):
    return torch.mean((x - y) ** 2)